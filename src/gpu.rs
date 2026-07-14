//! Optional GPU (wgpu compute) acceleration for the wideband sliding-DDC
//! sweep in [`crate::detector::AnalogFpvDetector::detect_from_iq`].
//!
//! The sweep's dominant cost is the per-probe DDC mixer + FIR pass over
//! the *entire* input batch, run sequentially once per probe on the CPU
//! (`detector.rs`'s `ddc_and_decimate`) — for a 1M-sample batch at 50
//! MSPS with ~9 probes, that's ~9 full sequential passes over 1M
//! samples. [`GpuAnalog::sweep`] batches all of that into one dispatch
//! covering every probe × output-sample pair (see
//! `src/shaders/ddc_decimate.wgsl` for the per-kernel math).
//!
//! ## Phase precision
//!
//! Each output sample needs the mixer phasor at its anchor input index
//! `i0 = out_idx * decimation_factor`, which can be ~1e6 for a 1M-sample
//! batch. Computing `phase = phase_adv * f32(i0)` directly in the shader
//! loses on the order of 0.1-0.4 radians of absolute accuracy at that
//! magnitude (f32 has ~24 bits of mantissa). An earlier version of this
//! module tried to route around that with a second GPU pass that built
//! the phasor via `out_len` chained recursive steps (stepping by
//! `decimation_factor` samples at a time) — empirically, that still
//! drifted by several degrees by mid-buffer (verified via
//! `detector::tests::gpu_ddc_matches_cpu_ddc_and_decimate`, which compares
//! directly against `ddc_and_decimate`): a long chain of f32
//! multiply-and-renormalize steps accumulates real angular error even
//! though Newton renormalisation keeps its *magnitude* pinned at 1.
//!
//! The fix here is simpler and more accurate: compute the anchor phasor
//! for every `(probe, out_idx)` directly, on the CPU, in `f64`
//! (`build_phase_table`) — `f64`'s ~52-bit mantissa keeps `phase_adv *
//! i0` accurate to a few times `1e-10` radians even at `i0` ~ 1e6, and
//! there's no accumulation to drift because each value is computed
//! independently rather than recursively. This table is `n_probes *
//! out_len` `f64` multiply-and-wrap operations — a small fraction of the
//! `n_probes * total_iq_len * num_taps` work the GPU shader does, so
//! computing it on the CPU doesn't undercut the offload. The GPU kernel
//! then only ever has to walk this anchor backward by up to `num_taps`
//! (63) small, bounded steps to reach any tap in its window — negligible
//! additional error (see `ddc_decimate.wgsl`'s doc comment).
//!
//! ## What stays on the CPU, unchanged
//!
//! Everything downstream of the decimated IQ this produces — `fm_demod`,
//! the classification FFT, harmonic-comb checks, and the cepstrum gate.
//! Those operate on arbitrary-length buffers (not power-of-two, so a GPU
//! FFT library doesn't apply) and carry the delicate PAL/NTSC bin math;
//! keeping them on CPU means [`GpuAnalog`] only has to reproduce
//! [`crate::detector::AnalogFpvDetector`]'s existing `ddc_and_decimate`
//! output, not the classification logic itself.
//!
//! [`GpuAnalog`] is `Send + Sync` and meant to be constructed once and
//! shared (via `Arc`) across every worker thread — unlike
//! [`crate::detector::AnalogFpvDetector`] itself, which holds a
//! `RefCell<FftPlanner>` and stays `!Sync`/per-worker.

use crate::ddc::{DEFAULT_FIR_TAPS, design_fir_taps};
use num_complex::Complex;
use wgpu::util::DeviceExt;

/// GPU compute handle for the wideband sweep's batched DDC. Build once
/// with [`Self::try_new`] and share via `Arc` across detector instances.
pub struct GpuAnalog {
    device: wgpu::Device,
    queue: wgpu::Queue,
    decimate_pipeline: wgpu::ComputePipeline,
    decimate_bgl: wgpu::BindGroupLayout,
    poll_thread: Option<std::thread::JoinHandle<()>>,
    poll_shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Mirrors `MAX_TAPS` in `ddc_decimate.wgsl` — headroom above
/// `DEFAULT_FIR_TAPS` (63), the only tap count the sweep actually uses.
const SHADER_MAX_TAPS: usize = 96;

/// Build the `(re, im)` mixer phasor at every `(probe, out_idx)` anchor
/// sample, entirely in `f64` — see the module doc comment for why this
/// replaced an all-GPU recursive design. Layout matches what
/// `ddc_decimate.wgsl` expects: `n_probes` blocks of `out_len` `vec2<f32>`
/// entries each.
fn build_phase_table(
    offsets_hz: &[f64],
    sample_rate: u32,
    decimation_factor: u32,
    out_len: u32,
) -> Vec<[f32; 2]> {
    const TWO_PI: f64 = std::f64::consts::TAU;
    let sample_rate = sample_rate as f64;
    let mut table = Vec::with_capacity(offsets_hz.len() * out_len as usize);
    for &offset_hz in offsets_hz {
        let phase_adv = -TWO_PI * offset_hz / sample_rate;
        for k in 0..out_len as u64 {
            let i0 = k * decimation_factor as u64;
            let raw = phase_adv * i0 as f64;
            // Wrap to (-pi, pi] before the f64->f32 cast so the small
            // wrapped value — not the large raw product — is what loses
            // precision going to f32 (negligible at this magnitude).
            let wrapped = raw - TWO_PI * (raw / TWO_PI).round();
            table.push([wrapped.cos() as f32, wrapped.sin() as f32]);
        }
    }
    table
}

impl GpuAnalog {
    fn create_bgl(
        device: &wgpu::Device,
        label: &str,
        entries: &[(u32, wgpu::BufferBindingType)],
    ) -> wgpu::BindGroupLayout {
        let wgpu_entries: Vec<_> = entries
            .iter()
            .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &wgpu_entries,
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        label: &str,
        source: &str,
        bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(bgl)],
            immediate_size: 0,
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Attempt to acquire a GPU adapter and build the sweep pipeline.
    /// Returns `None` on any failure (no adapter, driver rejects the
    /// device request, ...) — callers should fall back to the CPU sweep,
    /// exactly like `fpv_drone_dji::gpu_front_end::GpuFrontEnd::try_new`.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok()?;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let poll_shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let poll_thread = {
            let d = device.clone();
            let shutdown = std::sync::Arc::clone(&poll_shutdown);
            std::thread::spawn(move || {
                while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = d.poll(wgpu::PollType::Poll);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            })
        };

        let read_only = wgpu::BufferBindingType::Storage { read_only: true };
        let read_write = wgpu::BufferBindingType::Storage { read_only: false };
        let uniform = wgpu::BufferBindingType::Uniform;

        let decimate_bgl = Self::create_bgl(
            &device,
            "decimate_bgl",
            &[
                (0, uniform),
                (1, read_only),
                (2, read_only),
                (3, read_only),
                (4, read_only),
                (5, read_write),
            ],
        );
        let decimate_pipeline = Self::create_pipeline(
            &device,
            "decimate",
            include_str!("shaders/ddc_decimate.wgsl"),
            &decimate_bgl,
        );

        Some(Self {
            device,
            queue,
            decimate_pipeline,
            decimate_bgl,
            poll_thread: Some(poll_thread),
            poll_shutdown,
        })
    }

    /// Batched DDC + decimate across every probe in one dispatch.
    /// `decimation_factor` and `cutoff_hz` are shared across all probes
    /// (as in the CPU sweep — see `AnalogFpvDetector::ddc_and_decimate`,
    /// where only `freq_offset` varies per probe within one sweep call);
    /// `offsets_hz[i]` is probe `i`'s mixing frequency. Returns one
    /// decimated `Complex<f32>` buffer per probe, in the same order as
    /// `offsets_hz`, matching `ddc_and_decimate`'s output shape closely
    /// (see the module doc comment for the phase-precision approach that
    /// keeps this accurate across the whole buffer, not just near the
    /// start).
    ///
    /// Panics on a GPU buffer-map failure, matching
    /// `fpv_drone_dji::gpu_front_end::GpuFrontEnd`'s convention: the
    /// caller (orecchiette's per-batch worker loop) already runs under
    /// `catch_unwind`, so a GPU hiccup drops one batch rather than
    /// crashing the process.
    pub fn sweep(
        &self,
        iq_data: &[Complex<f32>],
        sample_rate: u32,
        offsets_hz: &[f64],
        decimation_factor: usize,
        cutoff_hz: f32,
    ) -> Vec<Vec<Complex<f32>>> {
        let n = iq_data.len();
        let n_probes = offsets_hz.len();
        if n == 0 || n_probes == 0 {
            return vec![Vec::new(); n_probes];
        }

        let decimation_factor = decimation_factor.max(1) as u32;
        let out_len = n.div_ceil(decimation_factor as usize) as u32;
        let num_taps = DEFAULT_FIR_TAPS.min(SHADER_MAX_TAPS) as u32;

        let taps = design_fir_taps(cutoff_hz, sample_rate, DEFAULT_FIR_TAPS);
        let offsets_f32: Vec<f32> = offsets_hz.iter().map(|&o| o as f32).collect();
        let phase_table = build_phase_table(offsets_hz, sample_rate, decimation_factor, out_len);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct DecimateConfig {
            sample_rate: f32,
            decimation_factor: u32,
            num_taps: u32,
            total_iq_len: u32,
            out_len: u32,
            n_probes: u32,
        }

        let decimate_config = DecimateConfig {
            sample_rate: sample_rate as f32,
            decimation_factor,
            num_taps,
            total_iq_len: n as u32,
            out_len,
            n_probes: n_probes as u32,
        };

        let offsets_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("offsets_hz"),
                contents: bytemuck::cast_slice(&offsets_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let taps_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("taps"),
                contents: bytemuck::cast_slice(&taps),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let phase_table_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("phase_table"),
                contents: bytemuck::cast_slice(&phase_table),
                usage: wgpu::BufferUsages::STORAGE,
            });
        // SAFETY: `num_complex::Complex<f32>` is `#[repr(C)]` with two
        // consecutive `f32` fields, so reinterpreting a `&[Complex<f32>]`
        // as raw bytes matching WGSL's `vec2<f32>` layout is sound — the
        // same pattern `fpv_drone_dji::gpu_front_end` already uses for
        // its own `Complex32` IQ uploads.
        let input_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(iq_data.as_ptr() as *const u8, n * 8) };
        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input_iq"),
                contents: input_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let decimate_config_buf =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("decimate_config"),
                    contents: bytemuck::cast_slice(&[decimate_config]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let output_size = (n_probes as u64) * (out_len as u64) * 8;
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_iq"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let create_bg = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindingResource]| {
            let wgpu_entries: Vec<_> = entries
                .iter()
                .enumerate()
                .map(|(i, r)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: r.clone(),
                })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &wgpu_entries,
            })
        };

        let decimate_bg = create_bg(
            &self.decimate_bgl,
            &[
                decimate_config_buf.as_entire_binding(),
                taps_buf.as_entire_binding(),
                offsets_buf.as_entire_binding(),
                phase_table_buf.as_entire_binding(),
                input_buf.as_entire_binding(),
                output_buf.as_entire_binding(),
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.decimate_pipeline);
            cpass.set_bind_group(0, &decimate_bg, &[]);
            let total_threads = n_probes as u32 * out_len;
            cpass.dispatch_workgroups(total_threads.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);

        let _ = self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let mut results = Vec::with_capacity(n_probes);
        for p in 0..n_probes {
            let base = p * out_len as usize * 2;
            let mut buf = Vec::with_capacity(out_len as usize);
            for k in 0..out_len as usize {
                let re = floats[base + k * 2];
                let im = floats[base + k * 2 + 1];
                buf.push(Complex::new(re, im));
            }
            results.push(buf);
        }
        drop(data);
        staging_buf.unmap();

        results
    }
}

impl Drop for GpuAnalog {
    fn drop(&mut self) {
        self.poll_shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.poll_thread.take() {
            let _ = handle.join();
        }
    }
}
