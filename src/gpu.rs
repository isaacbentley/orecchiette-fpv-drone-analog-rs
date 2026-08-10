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

/// GPU compute handle for the wideband sweep's batched DDC. Build once
/// with [`Self::try_new`] and share via `Arc` across detector instances.
///
/// Concurrent [`Self::sweep`] calls serialise on the internal buffer
/// pool's mutex — deliberate: per-batch buffer creation was the
/// dominant host-side cost, and one in-flight sweep at a time matches
/// how the detector fleet actually uses a shared GPU.
pub struct GpuAnalog {
    device: wgpu::Device,
    queue: wgpu::Queue,
    decimate_pipeline: wgpu::ComputePipeline,
    decimate_bgl: wgpu::BindGroupLayout,
    /// Grow-only persistent buffers reused across sweeps (uploads go
    /// through `queue.write_buffer`); recreating six buffers per batch
    /// cost more host time than the dispatch itself.
    pool: std::sync::Mutex<BufferPool>,
    poll_thread: Option<std::thread::JoinHandle<()>>,
    poll_shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Default)]
struct BufferPool {
    taps: Option<(wgpu::Buffer, u64)>,
    offsets: Option<(wgpu::Buffer, u64)>,
    phase_table: Option<(wgpu::Buffer, u64)>,
    input: Option<(wgpu::Buffer, u64)>,
    output: Option<(wgpu::Buffer, u64)>,
    staging: Option<(wgpu::Buffer, u64)>,
    config: Option<wgpu::Buffer>,
}

/// A `BindingResource` covering exactly `size` bytes of `buf` from
/// offset 0 — pooled buffers are bound by live region, never capacity.
fn sized_binding(buf: &wgpu::Buffer, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: buf,
        offset: 0,
        size: std::num::NonZeroU64::new(size),
    })
}

/// Fetch `slot`'s buffer, recreating it when `size` outgrows the stored
/// capacity. Grow-only, with 2× headroom on growth so a slowly growing
/// workload doesn't recreate every batch.
fn ensure_buffer(
    device: &wgpu::Device,
    slot: &mut Option<(wgpu::Buffer, u64)>,
    size: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> wgpu::Buffer {
    match slot {
        Some((buf, cap)) if *cap >= size => buf.clone(),
        _ => {
            let cap = size.next_power_of_two().max(256);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cap,
                usage,
                mapped_at_creation: false,
            });
            *slot = Some((buf.clone(), cap));
            buf
        }
    }
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
        entry_point: &str,
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
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Attempt to acquire a GPU adapter and build the sweep pipeline.
    /// Returns `None` on any failure (no adapter, driver rejects the
    /// device request, ...) — callers should fall back to the CPU sweep.
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            // wgpu 30 additions (e.g. `apply_limit_buckets`) keep their
            // defaults — this handle only ever runs one compute pipeline.
            ..Default::default()
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
            "main",
            &decimate_bgl,
        );

        Some(Self {
            device,
            queue,
            decimate_pipeline,
            decimate_bgl,
            pool: std::sync::Mutex::new(BufferPool::default()),
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
    /// Panics on a GPU buffer-map failure. Callers that must survive a
    /// GPU hiccup (e.g. a long-running batch worker) should wrap the
    /// call in `catch_unwind` so a failed map drops one batch rather
    /// than crashing the process — this crate's `detect_from_iq` does
    /// not do that itself.
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

        // ── Pooled buffers: grow-only, reused across sweeps ─────────
        // The lock is held through readback, deliberately serialising
        // concurrent sweeps — see the struct doc.
        let storage_up = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mut pool = self.pool.lock().expect("GPU buffer pool poisoned");

        let offsets_buf = ensure_buffer(
            &self.device,
            &mut pool.offsets,
            (offsets_f32.len() * 4) as u64,
            storage_up,
            "offsets_hz",
        );
        self.queue
            .write_buffer(&offsets_buf, 0, bytemuck::cast_slice(&offsets_f32));

        let taps_buf = ensure_buffer(
            &self.device,
            &mut pool.taps,
            (taps.len() * 4) as u64,
            storage_up,
            "taps",
        );
        self.queue
            .write_buffer(&taps_buf, 0, bytemuck::cast_slice(&taps));

        let phase_table_buf = ensure_buffer(
            &self.device,
            &mut pool.phase_table,
            (phase_table.len() * 8) as u64,
            storage_up,
            "phase_table",
        );
        self.queue
            .write_buffer(&phase_table_buf, 0, bytemuck::cast_slice(&phase_table));

        // `Complex<f32>` is Pod under num-complex's `bytemuck` feature
        // (two consecutive `f32`s, matching WGSL's `vec2<f32>` layout),
        // so this reinterpret needs no `unsafe`.
        let input_bytes: &[u8] = bytemuck::cast_slice(iq_data);
        let input_buf = ensure_buffer(
            &self.device,
            &mut pool.input,
            input_bytes.len() as u64,
            storage_up,
            "input_iq",
        );
        self.queue.write_buffer(&input_buf, 0, input_bytes);

        let decimate_config_buf = pool
            .config
            .get_or_insert_with(|| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("decimate_config"),
                    size: std::mem::size_of::<DecimateConfig>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .clone();
        self.queue.write_buffer(
            &decimate_config_buf,
            0,
            bytemuck::cast_slice(&[decimate_config]),
        );

        let output_size = (n_probes as u64) * (out_len as u64) * 8;
        let output_buf = ensure_buffer(
            &self.device,
            &mut pool.output,
            output_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "output_iq",
        );
        let staging_buf = ensure_buffer(
            &self.device,
            &mut pool.staging,
            output_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "output_staging",
        );

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

        // Bind exactly the live region of each pooled buffer, not the
        // grown capacity: `as_entire_binding` on a capacity-rounded
        // pool buffer both inflates what counts against
        // `max_storage_buffer_binding_size` and hands the shader's
        // runtime-sized arrays stale tail elements.
        let decimate_bg = create_bg(
            &self.decimate_bgl,
            &[
                decimate_config_buf.as_entire_binding(),
                sized_binding(&taps_buf, (taps.len() * 4) as u64),
                sized_binding(&offsets_buf, (offsets_f32.len() * 4) as u64),
                sized_binding(&phase_table_buf, (phase_table.len() * 8) as u64),
                sized_binding(&input_buf, input_bytes.len() as u64),
                sized_binding(&output_buf, output_size),
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

        // Pooled staging buffer may be larger than this sweep's output —
        // map exactly the live region.
        let slice = staging_buf.slice(..output_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        rx.recv().unwrap().unwrap();

        // wgpu 30: get_mapped_range is fallible; a failed map is the
        // same "drop this batch via panic + caller catch_unwind" case
        // as a map_async error (see the fn doc).
        let data = slice
            .get_mapped_range()
            .expect("GPU staging buffer map failed");
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
