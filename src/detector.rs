use crate::types::{DetectionResult, SignalType};
use num_complex::Complex;
use rustfft::{FftPlanner, num_complex::Complex as FftComplex};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::f32::consts::PI;

/// Target decimated rate for each sliding-DDC probe, and the boundary
/// between the narrowband fast path and the wideband sweep. The sweep's
/// 5 MHz step + 5 MHz edge margin only yields a valid probe grid when
/// `sample_rate > 10 MHz`, so captures at or below this rate take the
/// single-shot baseband path instead.
const WIDEBAND_TARGET_RATE_HZ: u32 = 10_000_000;

pub trait FpvDetector {
    /// Detect signals from raw I/Q data (more expensive but high confidence)
    fn detect_from_iq(
        &self,
        iq_data: &[Complex<f32>],
        center_freq: u64,
        sample_rate: u32,
    ) -> Vec<DetectionResult>;
}

pub struct AnalogFpvDetector {
    pub energy_threshold_db: f32,
    pub min_bandwidth: u32,
    pub max_bandwidth: u32,
    /// Floor on `detect_sync_pulses`'s reported confidence before a hit
    /// becomes a `DetectionResult`. `detect_sync_pulses` returns 0.8 for a
    /// clean harmonic-comb match at the exact PAL/NTSC line rate, but only
    /// 0.6 for its weaker fallback path ("periodic but couldn't
    /// disambiguate PAL from NTSC"). Both still pass the cepstrum
    /// structural gate, but a strong, spectrally broad, genuinely
    /// periodic interferer (cellular OFDM symbol/frame timing is the
    /// classic case) can produce a cepstral peak convincing enough to
    /// clear that gate through the 0.6 path without being real H-sync.
    /// Filtering below 0.7 keeps the 0.8 path (clean harmonic match)
    /// while dropping the 0.6 fallback — unless the VBI confirm stage
    /// (see [`crate::vbi::confirm_field_sync`]) promotes it to 0.75 by
    /// finding a real periodic field structure underneath.
    pub min_confidence: f32,
    /// When `true`, an 0.8-confidence harmonic-comb match spanning at
    /// least 2.5 field periods with *zero* confirmed vertical-sync
    /// groups is demoted to 0.6 (and filtered out by the default
    /// `min_confidence`). Default `false`: the crate's own line-rate-
    /// only test fixtures (`make_fm_sync_iq`, `make_pal_pulse_train`)
    /// are exactly this shape and would fail if this were on
    /// unconditionally. Enable it once real-world false-positive data
    /// justifies trusting VBI absence as disqualifying, not just VBI
    /// presence as reassuring.
    pub demote_unconfirmed_video: bool,
    planner: RefCell<FftPlanner<f32>>,
    /// Cached Hann window, keyed by length. `detect_sync_pulses` runs on
    /// every capture block at one steady length, so a single slot hits
    /// ~100% and saves a `cosf` per sample per call (65 k of them at the
    /// default block size).
    hann: RefCell<Option<(usize, Vec<f32>)>>,
    /// Reused demodulation buffer for `detect_sync_pulses` — a fresh
    /// ~256 KB `Vec` per probe per batch was pure allocator churn.
    demod_scratch: RefCell<Vec<f32>>,
    /// Cached sweep FIR design, keyed by `(sample_rate, cutoff_hz)`.
    /// Every probe in a sweep shares one design; only the mixing
    /// offset differs.
    sweep_taps: RefCell<Option<(u32, u32, Vec<f32>)>>,
    /// Optional GPU handle for the wideband sweep's batched DDC (see
    /// [`crate::gpu::GpuAnalog`], behind the `gpu` feature). `None` — the
    /// default — runs the existing sequential-per-probe CPU sweep
    /// unchanged. Meant to be shared via `Arc` across every worker's
    /// detector instance (unlike the detector itself, which stays
    /// per-worker because of `planner`'s `RefCell`); build one with
    /// [`Self::with_gpu`].
    #[cfg(feature = "gpu")]
    gpu: Option<std::sync::Arc<crate::gpu::GpuAnalog>>,
}

impl Default for AnalogFpvDetector {
    fn default() -> Self {
        Self {
            energy_threshold_db: 3.0, // 3dB above noise floor (FM video is wideband, lower SNR per bin)
            min_bandwidth: 1_000_000, // 1 MHz
            max_bandwidth: 30_000_000, // 30 MHz (FM video can be ~20 MHz wide)
            min_confidence: 0.7,
            demote_unconfirmed_video: false,
            planner: RefCell::new(FftPlanner::new()),
            hann: RefCell::new(None),
            demod_scratch: RefCell::new(Vec::new()),
            sweep_taps: RefCell::new(None),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }
}

#[cfg(feature = "gpu")]
impl AnalogFpvDetector {
    /// Build a detector that offloads the wideband sweep's DDC to `gpu`.
    /// `gpu` is typically one process-wide [`crate::gpu::GpuAnalog`]
    /// shared across every worker's detector via `Arc::clone` — building
    /// a `GpuAnalog` opens a GPU device, so callers should construct it
    /// once, not per detector.
    pub fn with_gpu(gpu: std::sync::Arc<crate::gpu::GpuAnalog>) -> Self {
        Self {
            gpu: Some(gpu),
            ..Self::default()
        }
    }
}

/// Time-domain PAL vs NTSC disambiguator for use when the FFT bin
/// resolution is too coarse to separate the two line rates (109 Hz
/// gap; typical first-packet FFTs give 380+ Hz/bin at 25 MSPS).
///
/// Reads the FM-demodulated baseband (smoothed by a ~0.5 µs moving
/// average), walks it looking for sync tips below a robust percentile
/// threshold (see [`crate::levels`]'s `robust_sync_threshold` — DC- and
/// click-outlier-invariant, unlike the global-minimum threshold it
/// replaced), computes the median inter-tip interval, and converts to a
/// line frequency. PAL
/// = 15625 Hz, NTSC = 15734 Hz. Returns `None` when we can't find
/// enough sync tips for a confident median (< 8 intervals) or when
/// the median falls within ±30 Hz of the midpoint, which would make
/// either answer arbitrary.
///
/// Time-domain pulse counting is rate-agnostic because the gap between
/// adjacent sync tips (~ 1600 samples at 25 MSPS) is comfortably larger
/// than the rate uncertainty (~ ±5 samples for crystal-grade clocks).
fn classify_pal_ntsc_time_domain(demod: &[f32], sample_rate: u32) -> Option<SignalType> {
    // ~ 200 µs minimum capture — one full PAL line is ~ 64 µs, so
    // we want at least a few lines. At 25 MSPS that's 5000 samples.
    let min_window = sample_rate as usize / 5_000;
    if demod.len() < min_window {
        return None;
    }
    // Limit the scan window to ~ 5 ms; that's plenty of lines
    // (≈ 78 PAL lines / 80 NTSC lines) and keeps the function
    // cheap on long input buffers.
    let scan_len = ((sample_rate as f32 * 5_000e-6) as usize).min(demod.len());
    // Smooth with the same ~0.5 µs moving average every other pulse
    // scan in the crate uses (levels.rs / vbi.rs) — un-smoothed FM
    // click noise otherwise dominates the extrema thresholded below.
    let ma_win = ((sample_rate as f32 * 0.5e-6) as usize).max(1);
    let smoothed = crate::levels::moving_average(&demod[..scan_len], ma_win);
    // Percentile threshold (p2 + 0.25·(p50−p2)), not `global_min · 0.3`:
    // the global minimum of a low-CNR record is a single FM click far
    // below any real sync tip, which put the threshold beneath the tips
    // and left only clicks detectable — exactly the regime this
    // disambiguator exists for. Percentiles are also DC-invariant, so a
    // demod offset from a tuning error no longer breaks classification.
    let threshold = crate::levels::robust_sync_threshold(&smoothed)?;
    // The scan skip after each accepted tip. 30 µs clears the pulse
    // comfortably without being able to skip past the *next* line's tip.
    let min_gap = (sample_rate as f32 * 30e-6) as usize;
    // Interval bounds for the line-period measurement: full lines only.
    // Both standards sit in 63.5–64.0 µs, so 55–75 µs is generous —
    // while excluding the ~32 µs HALF-line spacing of the equalizing and
    // broad pulses in every vertical blanking interval. The old 30 µs
    // floor admitted those: on a short record the VBI is a large
    // fraction of everything scanned, and eleven 320-sample half-line
    // intervals dragging on the median was half of how a clean PAL
    // record got called NTSC (the other half is the consensus test
    // below).
    let min_line = (sample_rate as f32 * 55e-6) as usize;
    let max_line = (sample_rate as f32 * 75e-6) as usize;
    let scan_len = smoothed.len();
    let mut sync_positions: Vec<usize> = Vec::with_capacity(128);
    let mut i = 0;
    let mut run_widths: Vec<usize> = Vec::with_capacity(128);
    while i < scan_len {
        if smoothed[i] < threshold {
            let run_start = i;
            let mut local_min_idx = i;
            let mut local_min_val = smoothed[i];
            while i < scan_len && smoothed[i] < threshold {
                if smoothed[i] < local_min_val {
                    local_min_val = smoothed[i];
                    local_min_idx = i;
                }
                i += 1;
            }
            run_widths.push(i - run_start);
            sync_positions.push(local_min_idx);
            i = local_min_idx + min_gap;
        } else {
            i += 1;
        }
    }
    // Pulse-shape gate: interval regularity alone cannot tell a sync
    // train from a plain tone AT the line rate — a 15,734 Hz sinusoid
    // produces intervals every bit as consistent as real NTSC, and the
    // consensus test below would pass it. What a tone cannot fake is the
    // duty cycle. An H-sync tip holds the signal below threshold for
    // ~4.7 µs of a 64 µs line; a sinusoid dipping under this same
    // threshold (p2 + 0.25·(p50 − p2), i.e. three quarters of the way
    // down its swing) stays below it for ~2·acos⁻¹ spans ≈ 14.6 µs per
    // cycle. The median width is robust to the VBI's outliers in both
    // directions — equalizing pulses run ~2.4 µs and broad pulses
    // ~27 µs, but both are a small minority of a record's runs — so an
    // 8 µs cut sits in clean air between sync (≈4.7) and tone (≈14.6).
    let max_sync_run = (sample_rate as f32 * 8e-6) as usize;
    if run_widths.is_empty() {
        return None;
    }
    run_widths.sort_unstable();
    if run_widths[run_widths.len() / 2] > max_sync_run {
        return None;
    }
    // Need at least 8 inter-tip intervals to land the median on a
    // PAL/NTSC decision. Anything less and crystal jitter dominates.
    let mut intervals: Vec<usize> = Vec::with_capacity(sync_positions.len().saturating_sub(1));
    for w in sync_positions.windows(2) {
        let gap = w[1] - w[0];
        if gap >= min_line && gap <= max_line {
            intervals.push(gap);
        }
    }
    if intervals.len() < 8 {
        return None;
    }
    intervals.sort_unstable();
    let median = intervals[intervals.len() / 2] as f64;
    if median <= 0.0 {
        return None;
    }
    // Consensus, then precision. A record that is really showing its
    // line rate produces near-identical intervals — the carrier-centred
    // probe of the case this guards read 640 twenty-seven times out of
    // twenty-seven. A probe looking at the signal from the wrong centre
    // (or at noise) produces a broad smear whose median lands wherever
    // the contamination pushes it; one such smear medianed to 627 =
    // 15,943 Hz, inside the plausible-rate gate and closer to NTSC, and
    // a clean PAL record was confidently misclassified. Requiring
    // two-thirds of the intervals within ±1% of the median accepts
    // every real lock and rejects every smear we measured.
    //
    // The period then comes from the MEAN of the consensus cluster, not
    // the integer median: at a 10 MHz probe rate PAL and NTSC line
    // periods are 640.0 and 635.6 samples — 4.4 apart — so the median's
    // whole-sample quantisation is a ±25 Hz error against a 109 Hz
    // decision, and the mean recovers the sub-sample rate.
    let tol = (median * 0.01).max(1.0);
    let cluster: Vec<f64> = intervals
        .iter()
        .map(|&g| g as f64)
        .filter(|g| (g - median).abs() <= tol)
        .collect();
    if cluster.len() * 3 < intervals.len() * 2 {
        return None;
    }
    let period = cluster.iter().sum::<f64>() / cluster.len() as f64;
    let line_hz = sample_rate as f64 / period;
    // PAL = 15625, NTSC = 15734, midpoint = 15679.5. Reject if we're
    // within ±30 Hz of the midpoint — that's the "we genuinely
    // can't tell" zone given typical jitter.
    const PAL_HZ: f64 = 15625.0;
    const NTSC_HZ: f64 = 15734.0;
    const MIDPOINT_HZ: f64 = (PAL_HZ + NTSC_HZ) / 2.0;
    // Reject medians far from BOTH standards outright. On FM-demodulated
    // noise the dips below threshold occur essentially continuously, so
    // the `min_gap` skip makes the median interval collapse to just
    // above `min_gap` (30 µs → ~33 kHz "line rate"). Without this bound
    // that absurd rate still classified as NTSC — it merely had to be
    // *closer* to 15734 than to 15625 — which is exactly how empty-band
    // noise got reported as a video signal on live captures. Real
    // crystal error on a VTX is a few Hz; ±~250 Hz is already generous.
    if !(15_400.0..=16_000.0).contains(&line_hz) {
        return None;
    }
    if (line_hz - MIDPOINT_HZ).abs() < 30.0 {
        return None;
    }
    if (line_hz - PAL_HZ).abs() < (line_hz - NTSC_HZ).abs() {
        Some(SignalType::AnalogVideoPal)
    } else {
        Some(SignalType::AnalogVideoNtsc)
    }
}

impl AnalogFpvDetector {
    pub fn new(energy_threshold_db: f32) -> Self {
        Self {
            energy_threshold_db,
            ..Default::default()
        }
    }

    /// Classify one capture block: harmonic-comb line-rate detection,
    /// cepstrum structural gate, and the VBI confirm stage.
    pub fn detect_sync_pulses(
        &self,
        iq_data: &[Complex<f32>],
        sample_rate: u32,
    ) -> (SignalType, f32) {
        self.detect_sync_pulses_inner(iq_data, sample_rate, None)
    }

    /// Like [`Self::detect_sync_pulses`], but classifies against a
    /// noncoherently averaged magnitude spectrum accumulated in
    /// `integrator` under `key_hz` (typically the absolute tuned/probe
    /// frequency in Hz). Successive batches of the same signal add
    /// coherently in the comb bins while noise averages down ~1/√N —
    /// several dB of sensitivity that a single 68 ms batch cannot
    /// provide. Time-domain PAL/NTSC disambiguation and the VBI confirm
    /// stage still run on the *current* batch (they need the waveform,
    /// not the spectrum).
    pub fn detect_sync_pulses_integrated(
        &self,
        iq_data: &[Complex<f32>],
        sample_rate: u32,
        key_hz: i64,
        integrator: &mut SpectralIntegrator,
    ) -> (SignalType, f32) {
        self.detect_sync_pulses_inner(iq_data, sample_rate, Some((integrator, key_hz)))
    }

    fn detect_sync_pulses_inner(
        &self,
        iq_data: &[Complex<f32>],
        sample_rate: u32,
        integration: Option<(&mut SpectralIntegrator, i64)>,
    ) -> (SignalType, f32) {
        let n = iq_data.len();
        // `sample_rate == 0` would make `bin_hz` zero and the line-rate bin
        // indices (`15625 / bin_hz`) evaluate to `Inf as usize == usize::MAX`,
        // overflowing `bin + range` in `get_peak_energy` (a debug panic).
        // A zero-rate capture carries no meaning anyway — reject it up front.
        if n < 2048 || sample_rate == 0 {
            return (SignalType::Unknown, 0.0);
        }

        // FM demodulation: instantaneous frequency via arg(z[n] * conj(z[n-1])).
        // Use the single shared implementation in `demod::fm_demod_into` so
        // the discriminator never diverges between the detection and decode
        // paths, refilling the reused scratch buffer.
        let mut demod_guard = self.demod_scratch.borrow_mut();
        crate::demod::fm_demod_into(iq_data, &mut demod_guard);
        let demod: &[f32] = &demod_guard;
        let demod_len = demod.len();
        let avg_demod = demod.iter().sum::<f32>() / demod_len as f32;
        let mut var = 0.0f32;
        for &d in demod {
            let diff = d - avg_demod;
            var += diff * diff;
        }
        var /= demod_len as f32;
        if var < 1e-6 {
            return (SignalType::Unknown, 0.0);
        }

        if demod_len < 2 {
            return (SignalType::Unknown, 0.0);
        }

        // Zero-pad the transform to the next power of two. `fm_demod`
        // returns `n - 1` samples, so the default 65536-sample capture
        // block lands on a 65535-point FFT — and 65535 = 3·5·17·257,
        // whose 257 factor drops rustfft onto Rader's algorithm at ~6×
        // the cost of the 65536-point radix-4 NEON path (1009 µs vs
        // 168 µs measured). One zero sample buys that back.
        //
        // Padding adds no information: it sinc-interpolates the same
        // spectrum onto a finer bin grid. True resolution stays
        // `sample_rate / demod_len` (the Rayleigh limit of the record
        // length), so every decision below that depends on *resolution*
        // rather than on bin indexing is pinned to `demod_len` — see
        // `search_range` and `bins_distinct`.
        let fft_len = demod_len.next_power_of_two();
        let fft = self.planner.borrow_mut().plan_fft_forward(fft_len);
        let mut buffer: Vec<FftComplex<f32>> = vec![FftComplex { re: 0.0, im: 0.0 }; fft_len];

        // Hann spans the real samples only; the pad tail stays zero.
        // (The window is ~0 at the edges, so the join is continuous.)
        {
            let mut cache = self.hann.borrow_mut();
            if cache.as_ref().is_none_or(|(n, _)| *n != demod_len) {
                let denom = (demod_len - 1) as f32;
                *cache = Some((
                    demod_len,
                    (0..demod_len)
                        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / denom).cos()))
                        .collect(),
                ));
            }
            let w = &cache.as_ref().unwrap().1;
            for i in 0..demod_len {
                buffer[i].re = (demod[i] - avg_demod) * w[i];
            }
        }

        fft.process(&mut buffer);

        // Magnitude spectrum — everything below runs on amplitudes,
        // which is also what makes cross-batch integration possible:
        // magnitudes average noncoherently (noise drops ~1/√N while the
        // sync comb stays put), whereas complex bins would need
        // impossible phase coherence across batches.
        let fresh_mags: Vec<f32> = buffer.iter().map(|c| c.norm()).collect();
        let mags: &[f32] = match integration {
            Some((integrator, key_hz)) => integrator.accumulate(key_hz, &fresh_mags),
            None => &fresh_mags,
        };

        let bin_hz = sample_rate as f32 / fft_len as f32;
        let bin_pal = (15625.0 / bin_hz).round() as usize;
        let bin_ntsc = (15734.0 / bin_hz).round() as usize;

        // ±1 bin of the *unpadded* grid, expressed in padded bins, so the
        // peak search covers the same absolute Hz window it always has
        // regardless of how much padding was added. Rounded, not floored
        // or ceiled: `fft_len / demod_len` is in [1, 2), so integer
        // division would pin this to 1 and silently shrink the window for
        // a heavily-padded record, while `div_ceil` would pin it to 2 and
        // double the window for the barely-padded 65535→65536 case that
        // the default block size actually hits.
        let search_range = ((fft_len as f32 / demod_len as f32).round() as usize).max(1);

        // The PAL and NTSC line rates are 109 Hz apart, so on all but the
        // longest records their bins are near neighbours. A search window
        // that reaches the *other* standard's bin returns that standard's
        // peak, and both measurements collapse to the same number —
        // whereupon `pal_energy > ntsc_energy * 1.2` and its mirror are
        // both false and the signal is discarded as Unknown, however
        // strong it is.
        //
        // That was the dead zone: a clean carrier missed at 13 of 30
        // capture lengths at 61.44 MSPS, in bands rather than at random,
        // because whether the windows collided depended on where
        // `demod_len` fell relative to the next power of two.
        //
        //   len     padded sep   radius   reaches other bin   result
        //   196608      1          1            yes           identical → miss
        //   262144      1          1            yes           identical → miss
        //   327680      2          2            yes           identical → miss
        //   393216      2          1            no            distinct  → hit
        //
        // Clamping the radius to one bin short of the separation keeps
        // each window on its own line rate. `saturating_sub` handles the
        // degenerate `sep == 0` case (both rates in one bin) by collapsing
        // to a single-bin read; `bins_distinct` below rejects that record
        // for spectral discrimination anyway.
        let bin_sep = bin_ntsc.abs_diff(bin_pal);
        let disc_range = search_range.min(bin_sep.saturating_sub(1));

        let pal_energy = self.get_peak_energy(mags, bin_pal, disc_range);
        let ntsc_energy = self.get_peak_energy(mags, bin_ntsc, disc_range);

        // Floor at bin 1 so the DC bin (nonzero even after mean-subtraction,
        // because the Hann window has nonzero mean) never enters the noise
        // estimate on coarse/short FFTs where round(500/bin_hz) would be 0.
        let noise_start_bin = ((500.0 / bin_hz).round() as usize).max(1);
        let noise_end_bin = fft_len / 2;
        // Median, not mean: this bin range also contains the signal's own
        // video spectrum and harmonic comb, and a mean lets a real signal
        // raise its own detection floor — costing sensitivity exactly on
        // the weak signals the floor exists to find. The comb occupies a
        // small fraction of the bins, so the median tracks the true noise
        // level (and for pure Rayleigh noise it sits within ~6% of the
        // mean, so thresholds barely move on empty bands).
        let noise_floor = if noise_end_bin > noise_start_bin {
            let mut noise_mags: Vec<f32> = mags[noise_start_bin..noise_end_bin].to_vec();
            crate::levels::median(&mut noise_mags)
        } else {
            1e-6
        };

        // 25x the noise floor, not 5x. Measured against a live 61.44 MSPS
        // capture of 2.4 GHz with nothing airborne: the false positives had
        // a line-rate fundamental only 5-7x the floor, i.e. the old
        // threshold sat *inside* the noise tail — and a harmonic needs only
        // 10% of the fundamental and 2.5x the floor to count, so two
        // scattered peaks were enough to claim video. Real video is nowhere
        // near: a healthy VTX measures 400x to 100,000x, and even the
        // weakest case the crate intends to catch (the cross-batch
        // integration path) still clears ~45x. 25x sits between the two
        // with margin on both sides — 3.6x above the worst observed false
        // positive, 1.8x below the weakest real detection.
        let thresh_strong = noise_floor * 25.0;

        const N_HARMONICS: usize = 5;
        const HARMONIC_RATIO: f32 = 0.1;
        const PAL_LINE_HZ: f32 = 15625.0;
        const NTSC_LINE_HZ: f32 = 15734.0;
        let line_bin = bin_ntsc.max(bin_pal);
        let mut pal_harmonics = 0u32;
        let mut ntsc_harmonics = 0u32;
        let max_bin = noise_end_bin;
        if line_bin > 0 {
            // A harmonic is judged only relative to its own fundamental.
            // There used to be an absolute `noise_floor * 2.5` floor here
            // as well, but raising `thresh_strong` to 25x made it inert:
            // a fundamental that clears 25x puts `fundamental * 0.1` above
            // 2.5x by construction, so the absolute term could never bind
            // for any signal that reaches classification.
            let pal_thresh = pal_energy * HARMONIC_RATIO;
            let ntsc_thresh = ntsc_energy * HARMONIC_RATIO;
            for k in 2..=N_HARMONICS {
                let kf = k as f32;
                let hb_pal = (kf * PAL_LINE_HZ / bin_hz).round() as usize;
                let hb_ntsc = (kf * NTSC_LINE_HZ / bin_hz).round() as usize;
                if hb_pal < max_bin {
                    let e = self.get_peak_energy(mags, hb_pal, search_range);
                    if e > pal_thresh {
                        pal_harmonics += 1;
                    }
                }
                if hb_ntsc < max_bin {
                    let e = self.get_peak_energy(mags, hb_ntsc, search_range);
                    if e > ntsc_thresh {
                        ntsc_harmonics += 1;
                    }
                }
            }
        }
        let collide_harmonics = pal_harmonics.max(ntsc_harmonics);
        if std::env::var_os("ANALOG_PROBE").is_some()
            && (pal_energy.max(ntsc_energy) > thresh_strong)
        {
            eprintln!(
                "PROBE floor={:.3e} pal={:.3e}({:.1}x) ntsc={:.3e}({:.1}x) pal_h={} ntsc_h={}",
                noise_floor,
                pal_energy,
                pal_energy / noise_floor.max(1e-12),
                ntsc_energy,
                ntsc_energy / noise_floor.max(1e-12),
                pal_harmonics,
                ntsc_harmonics
            );
        }

        let mut sig_type = SignalType::Unknown;
        let mut conf = 0.0;

        // Pinned to the *unpadded* grid: this asks "is our record long
        // enough to actually resolve 15625 from 15734 Hz?", which is a
        // Rayleigh-limit question about `demod_len`. Zero-padding
        // interpolates but resolves nothing, so deciding this on the
        // padded grid would let two merely-interpolated bins claim a
        // separation the data cannot support — and misclassify PAL as
        // NTSC or vice versa.
        let true_bin_hz = sample_rate as f32 / demod_len as f32;
        let true_sep = (15734.0f32 / true_bin_hz).round() - (15625.0f32 / true_bin_hz).round();
        let bins_distinct = true_sep >= 2.0;

        if bins_distinct {
            if pal_energy > thresh_strong && pal_energy > ntsc_energy * 1.2 && pal_harmonics >= 2 {
                sig_type = SignalType::AnalogVideoPal;
                conf = 0.8;
            } else if ntsc_energy > thresh_strong
                && ntsc_energy > pal_energy * 1.2
                && ntsc_harmonics >= 2
            {
                sig_type = SignalType::AnalogVideoNtsc;
                conf = 0.8;
            }
        } else {
            // FFT bin resolution (`bin_hz`) is too coarse to resolve
            // PAL (15625 Hz) from NTSC (15734 Hz) — they're only 109 Hz
            // apart, but at 25 MSPS with a 65 k chunk `bin_hz` is ≈ 381
            // Hz, so both line rates fold into the same bin. We've
            // confirmed the signal IS analog FPV (`hline_energy` clears
            // the strong-noise floor and we see ≥ 2 harmonics), so
            // disambiguate the two standards in the time domain by
            // measuring the median sync-tip interval directly on the
            // demodulated record. This avoids needing a 20-ms FFT
            // (which we don't have because the first packet is 2.6 ms).
            let hline_energy = pal_energy.max(ntsc_energy);
            if hline_energy > thresh_strong && collide_harmonics >= 2 {
                let time_domain_class = classify_pal_ntsc_time_domain(demod, sample_rate);
                match time_domain_class {
                    Some(SignalType::AnalogVideoPal) => {
                        sig_type = SignalType::AnalogVideoPal;
                        conf = 0.8;
                    }
                    Some(SignalType::AnalogVideoNtsc) => {
                        sig_type = SignalType::AnalogVideoNtsc;
                        conf = 0.8;
                    }
                    _ => {
                        // Time-domain median was inconclusive (too
                        // few sync tips, or median fell exactly
                        // between the two standards). Hold the
                        // `AnalogVideoUnknown` answer rather than
                        // commit to one.
                        sig_type = SignalType::AnalogVideoUnknown;
                        conf = 0.6;
                    }
                }
            }
        }

        // ---- Cepstrum structural gate ----
        // If the harmonic classifier found a candidate, verify it
        // structurally via the cepstrum.  Multi-tone interferers
        // (Wi-Fi beacons, BT hopping) can fool the harmonic check
        // but never produce the sharp quefrency peak that a true
        // periodic pulse train does.
        if sig_type != SignalType::Unknown {
            let candidate_line_hz = match sig_type {
                SignalType::AnalogVideoPal => PAL_LINE_HZ,
                SignalType::AnalogVideoNtsc => NTSC_LINE_HZ,
                _ => PAL_LINE_HZ, // AnalogVideoUnknown — check PAL as proxy
            };
            if !self.verify_cepstrum(mags, sample_rate, candidate_line_hz, demod_len) {
                sig_type = SignalType::Unknown;
                conf = 0.0;
            }
        }

        // ---- VBI confirm stage ----
        // The harmonic-comb + cepstrum checks above only ever see a
        // ~2-8 ms slice of one FFT's worth of data, so they can only
        // confirm the *line rate* is present, not that it belongs to a
        // real interlaced field structure. A field is ~16.7 ms (NTSC) /
        // 20 ms (PAL), and orecchiette's batches are ~68 ms, so multiple
        // complete vertical syncs are usually available in the same
        // slice already handed to this function — checking for them
        // costs one more pulse scan and turns "plausible line-rate
        // comb" into "confirmed periodic field structure", which is
        // essentially unfakeable by a non-video interferer.
        if sig_type.is_analog_video()
            && let Some(levels) = crate::levels::estimate_sync_levels(demod, sample_rate)
        {
            let is_pal_hint = match sig_type {
                SignalType::AnalogVideoPal => Some(true),
                SignalType::AnalogVideoNtsc => Some(false),
                _ => None,
            };
            let evidence = crate::vbi::confirm_field_sync(demod, sample_rate, &levels, is_pal_hint);
            conf =
                apply_vbi_confidence_tier(sig_type, conf, &evidence, self.demote_unconfirmed_video);
        }

        (sig_type, conf)
    }

    fn get_peak_energy(&self, mags: &[f32], bin: usize, range: usize) -> f32 {
        let end = (bin + range).min(mags.len() / 2);
        // Clamp start to end: for a `bin` past Nyquist (only reachable at
        // pathologically low sample rates where the line-rate bin exceeds
        // fft_len/2) `start` could otherwise exceed `end`, panicking the
        // inclusive slice below.
        let start = bin.saturating_sub(range).min(end);
        mags[start..=end].iter().copied().fold(0.0f32, f32::max)
    }

    /// Cepstrum-based structural verification for H-sync pulse trains.
    ///
    /// A true H-sync signal is a narrow rectangular pulse train whose
    /// power spectrum is a harmonic comb.  The cepstrum (IFFT of the
    /// log-power spectrum) transforms that comb into a single sharp
    /// "quefrency" peak at the fundamental period — something a multi-
    /// frequency interference pattern cannot mimic.
    ///
    /// The power-spectrum and log passes are written as tight branchless
    /// loops over contiguous `f32` slices — LLVM auto-vectorises them to
    /// 4-wide NEON (AArch64) or SSE/AVX (x86_64) at `opt-level ≥ 2`.
    /// The IFFT is handled by `rustfft` which uses platform SIMD
    /// internally.
    ///
    /// Returns `true` if the cepstral peak-to-median ratio at the
    /// expected quefrency exceeds a threshold.
    /// `record_len` is the number of *real* (unpadded) samples behind
    /// `fft_buffer` — see the zero-padding note in `detect_sync_pulses`.
    /// Quefrency spacing is `1/sample_rate` no matter how much the
    /// spectrum was padded, so the peak's index is unaffected; but
    /// padding extends the quefrency *range* with interpolation
    /// artefact the record cannot support. Everything below is bounded
    /// to `record_len / 2` so that tail can't dilute the median noise
    /// floor and inflate the peak-to-median ratio this gate turns on.
    fn verify_cepstrum(
        &self,
        mags: &[f32],
        sample_rate: u32,
        candidate_line_hz: f32,
        record_len: usize,
    ) -> bool {
        let fft_len = mags.len();
        if fft_len < 64 {
            return true; // too short for meaningful cepstrum
        }
        let half = (record_len / 2).min(fft_len / 2);

        // Expected quefrency (in samples) for the line rate.
        let expected_q = sample_rate as f32 / candidate_line_hz;
        let q_idx = expected_q.round() as usize;
        if q_idx < 2 || q_idx >= half {
            return true; // can't measure at this resolution
        }

        // ---- Step 1: log-power spectrum ----
        // Written as a single pass over the magnitude spectrum (which
        // may be a cross-batch average — see `SpectralIntegrator`).
        // The inner loop is branchless: `m*m + eps` then `ln()`. LLVM
        // SLP-vectorises the multiply; the `ln` call is scalar but
        // dominates only at very large FFT sizes where the IFFT cost
        // already exceeds it.
        const EPSILON: f32 = 1e-12;
        let mut log_power: Vec<FftComplex<f32>> = Vec::with_capacity(fft_len);
        for &m in mags {
            let power = m * m + EPSILON;
            log_power.push(FftComplex {
                re: power.ln(),
                im: 0.0,
            });
        }

        // ---- Step 2: IFFT → real cepstrum ----
        let ifft = self.planner.borrow_mut().plan_fft_inverse(fft_len);
        ifft.process(&mut log_power);

        // Normalise IFFT output (rustfft doesn't normalise).
        let scale = 1.0 / fft_len as f32;

        // ---- Step 3: extract real cepstrum magnitudes ----
        // Only need the first half (positive quefrencies), bounded to
        // the record's own support — see the fn doc.
        // Written as a branchless multiply — auto-vectorises.
        let mut cepstrum_mag: Vec<f32> = Vec::with_capacity(half);
        for val in log_power.iter().take(half) {
            let v = val.re * scale;
            cepstrum_mag.push(v.abs());
        }

        // ---- Step 4: peak search around expected quefrency ----
        // ±2% tolerance band, minimum ±2 bins.
        let tolerance = ((q_idx as f32 * 0.02).ceil() as usize).max(2);
        let search_start = q_idx.saturating_sub(tolerance);
        let search_end = (q_idx + tolerance).min(half - 1);

        // Branchless max reduction.
        let mut peak_val = 0.0f32;
        for &val in cepstrum_mag.iter().take(search_end + 1).skip(search_start) {
            // Branchless: compiler emits `fmax` on AArch64.
            peak_val = peak_val.max(val);
        }

        // ---- Step 5: median of cepstrum for noise floor ----
        // O(n) quickselect for the middle order statistic instead of a
        // full O(n log n) sort. This runs per detection probe that
        // clears the harmonic gate, and `cepstrum_mag` can be ~125K
        // elements on a wideband sweep. The peak was already extracted
        // above, so we can reorder the buffer in place.
        let mid = cepstrum_mag.len() / 2;
        cepstrum_mag.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let median = cepstrum_mag[mid];

        // ---- Step 6: threshold check ----
        // A real pulse train produces a cepstral peak 5–20× above the
        // median. The bottom of that range is also where a strong,
        // spectrally broad, genuinely periodic *non-video* interferer can
        // land (cellular OFDM symbol/frame timing is the classic case) —
        // wide enough energy across the sweep clears the harmonic-comb gate
        // at enough probes that a 5× cepstral peak stops being reliably
        // diagnostic. Sitting further up the documented real-signal range
        // trades a little sensitivity on very weak/distant VTX signals for
        // rejecting that class of false positive.
        const CEPSTRAL_RATIO_THRESHOLD: f32 = 7.0;
        let ratio = if median > 1e-10 {
            peak_val / median
        } else {
            peak_val * 1e10 // effectively infinite if median is zero
        };

        ratio >= CEPSTRAL_RATIO_THRESHOLD
    }

    /// Mix `freq_offset` Hz down to baseband then decimate to
    /// `target_rate`, returning the isolated complex baseband.
    ///
    /// Implemented as a [`crate::ddc::StreamingDDC`] (63-tap
    /// Blackman-windowed-sinc FIR, > 50 dB stopband attenuation)
    /// followed by integer-stride decimation. The previous shape used
    /// a length-N boxcar (`sum/N`) which had a sinc magnitude response
    /// — under the FM threshold effect that let adjacent-band energy
    /// leak through and synthesise spurious harmonic content in the
    /// discriminator output (DESIGN.md §6 item 1). The proper FIR
    /// closes that gap at the cost of one extra allocation per probe.
    ///
    /// ## Cutoff choice
    ///
    /// Cutoff sits at `target_rate / 3` (≈ 3.33 MHz for the 10 MHz
    /// target). Two competing forces pin it down:
    ///
    /// 1. **Adjacent-probe contamination.** The wideband sweep uses a
    ///    5 MHz step; an on-tune signal at one probe lands 5 MHz off
    ///    centre at the next probe. With a Nyquist cutoff
    ///    (`target_rate / 2 = 5 MHz`) that signal sits exactly at the
    ///    FIR's −6 dB point and still produces enough discriminator
    ///    click-noise to fool the harmonic-consistency check (see the
    ///    `harmonic_check_rejects_pure_tone_via_wideband_sweep` test).
    ///    With `target_rate / 3`, the 5 MHz-off signal lands
    ///    ≈ 76 % through the Blackman FIR's transition band (which
    ///    has full width ≈ 5.5 · Fs / N ≈ 4.4 MHz at this sample-
    ///    rate / tap-count combo). True > 50 dB stopband begins
    ///    around 5.5 MHz off; at 5 MHz off attenuation is roughly
    ///    30–40 dB. The test verifies that's enough to suppress the
    ///    click-noise harmonics in practice.
    /// 2. **Coverage gap at probe boundaries.** Pushing the cutoff
    ///    too tight (e.g. `target_rate / 4 = 2.5 MHz` = `step / 2`)
    ///    puts a signal at the exact midpoint between two probes
    ///    (2.5 MHz from each) at the FIR's −6 dB point on *both*
    ///    sides — a 4× power loss with no fallback. `target_rate / 3`
    ///    leaves 2.5 MHz-off signals at ~0 dB (well inside the
    ///    passband), so no detection blind spot.
    ///
    /// ## Performance notes
    ///
    /// `StreamingDDC::process_decimated` already computes the FIR only
    /// on decimation-aligned samples (the mixer and delay-line writes
    /// are per-sample and irreducible in this structure), so there is
    /// no pending "polyphase" win here — an earlier note claiming a
    /// ~5× speed-up predated that gating. The FIR *design* is cached
    /// across probes and batches (`sweep_taps`), so per-probe
    /// construction costs one 63-`f32` copy, not a `sin_cos` loop.
    fn ddc_and_decimate(
        &self,
        iq_data: &[Complex<f32>],
        sample_rate: u32,
        freq_offset: f32,
        target_rate: u32,
    ) -> Vec<Complex<f32>> {
        let decimation_factor = Self::decimation_factor(sample_rate, target_rate);
        let cutoff_hz = (target_rate as f32) / 3.0;
        let taps = self.cached_sweep_taps(sample_rate, cutoff_hz);
        let mut ddc = crate::ddc::StreamingDDC::from_designed_taps(freq_offset, sample_rate, taps);
        ddc.process_decimated(iq_data, decimation_factor)
    }

    /// The sweep's shared FIR design, cached by `(sample_rate,
    /// cutoff_hz)` — every probe in a sweep differs only in mixing
    /// offset.
    fn cached_sweep_taps(&self, sample_rate: u32, cutoff_hz: f32) -> Vec<f32> {
        let key = (sample_rate, cutoff_hz.to_bits());
        let mut cache = self.sweep_taps.borrow_mut();
        if let Some((rate, cut, taps)) = cache.as_ref()
            && (*rate, *cut) == key
        {
            return taps.clone();
        }
        let taps =
            crate::ddc::design_fir_taps(cutoff_hz, sample_rate, crate::ddc::DEFAULT_FIR_TAPS);
        *cache = Some((key.0, key.1, taps.clone()));
        taps
    }

    /// Integer decimation factor `ddc_and_decimate` actually divides by.
    /// Shared with [`Self::decimated_rate`] so a probe's assumed sample
    /// rate can never drift from what the DDC really produced.
    fn decimation_factor(sample_rate: u32, target_rate: u32) -> usize {
        (sample_rate / target_rate).max(1) as usize
    }

    /// The *true* rate of a `ddc_and_decimate(..., target_rate)` output.
    /// `sample_rate / decimation_factor` is not necessarily exactly
    /// `target_rate` — integer division truncates whenever `sample_rate`
    /// isn't an exact multiple of it (e.g. 25 MSPS / 10 MHz truncates
    /// the factor to 2, giving 12.5 MHz actual output, not 10 MHz).
    /// Passing the wrong assumed rate into `detect_sync_pulses`
    /// corrupts every frequency-derived computation in there (FFT bin
    /// width, line-rate bin indices, harmonic bins) by the same
    /// fraction, which silently broke detection on any wideband-sweep
    /// capture rate that wasn't a clean multiple of `target_rate`.
    fn decimated_rate(sample_rate: u32, target_rate: u32) -> u32 {
        sample_rate / Self::decimation_factor(sample_rate, target_rate) as u32
    }
}

/// Noncoherent cross-batch spectrum accumulator for the `_integrated`
/// detection entry points.
///
/// Keyed by absolute frequency (rounded to a 1 MHz bucket, so probe-grid
/// jitter between sweeps lands in the same bucket), each bucket holds a
/// running average of the FM-demod magnitude spectrum. A real signal's
/// sync comb sits in the same bins batch after batch while noise
/// magnitudes average toward their mean, so after N batches the
/// comb-to-floor contrast improves ~√N — several dB of detection
/// sensitivity that no single ~68 ms batch can provide. The averaging is
/// a cumulative mean up to `window` batches, then an EWMA with
/// `α = 1/window`, so stale signals fade rather than pinning the bucket
/// forever.
///
/// One integrator per capture chain: buckets assume a consistent FFT
/// length per frequency, and a changed capture configuration should
/// start fresh (see [`Self::reset`]).
pub struct SpectralIntegrator {
    window: u32,
    max_buckets: usize,
    buckets: HashMap<i64, SpectralBucket>,
    /// First-insertion order, for cheap oldest-bucket eviction at the
    /// `max_buckets` cap (a runaway hop list shouldn't grow memory
    /// unboundedly — each bucket holds a full magnitude spectrum).
    insertion_order: VecDeque<i64>,
}

struct SpectralBucket {
    mags: Vec<f32>,
    count: u32,
}

impl SpectralIntegrator {
    /// `window` is the effective averaging depth in batches (clamped to
    /// ≥ 1). 4 is a good default: ~6 dB of noise-variance reduction at
    /// ~4 batches of latency for a signal to reach full sensitivity.
    pub fn new(window: u32) -> Self {
        Self {
            window: window.max(1),
            max_buckets: 256,
            buckets: HashMap::new(),
            insertion_order: VecDeque::new(),
        }
    }

    /// Drop all accumulated state (retune, capture-config change).
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.insertion_order.clear();
    }

    /// Fold `fresh` into the bucket for `key_hz` and return the
    /// averaged spectrum. A length mismatch (changed capture length for
    /// this frequency) restarts the bucket from `fresh`.
    fn accumulate(&mut self, key_hz: i64, fresh: &[f32]) -> &[f32] {
        let key = key_hz.div_euclid(1_000_000);
        if !self.buckets.contains_key(&key) {
            if self.insertion_order.len() >= self.max_buckets
                && let Some(evict) = self.insertion_order.pop_front()
            {
                self.buckets.remove(&evict);
            }
            self.insertion_order.push_back(key);
            self.buckets.insert(
                key,
                SpectralBucket {
                    mags: fresh.to_vec(),
                    count: 1,
                },
            );
            return &self.buckets[&key].mags;
        }
        let bucket = self.buckets.get_mut(&key).expect("checked above");
        if bucket.mags.len() != fresh.len() {
            bucket.mags.clear();
            bucket.mags.extend_from_slice(fresh);
            bucket.count = 1;
        } else {
            bucket.count = (bucket.count + 1).min(self.window);
            let alpha = 1.0 / bucket.count as f32;
            for (avg, &x) in bucket.mags.iter_mut().zip(fresh) {
                *avg += alpha * (x - *avg);
            }
        }
        &bucket.mags
    }
}

/// Frequency window within which two detections are treated as the same
/// signal. Roughly one FM-video channel width.
const DEDUP_BW_HZ: f64 = 25e6;

/// Merge detections that fall within [`DEDUP_BW_HZ`] of each other,
/// keeping the strongest (highest confidence, then highest RSSI) member
/// of each group. `results` must already be sorted by frequency.
///
/// Each group is compared against an **immutable anchor** — the first
/// result that opened it — not against the running strongest member.
/// Replacing the kept entry also moves the frequency the *next* result
/// is compared against, which makes evenly-spaced detections chain: the
/// standard 5.8 GHz band plans space channels ~19–20 MHz apart (band F
/// is 5740 / 5760 / 5780 / 5800 …), comfortably inside this 25 MHz
/// window, so three simultaneous VTXs one channel apart collapsed into a
/// single detection — each merge dragged the comparison point up to the
/// next channel, and a whole band could fold into one hit. Same defect,
/// same fix, as the sweep clustering in `detect_from_iq` (see its
/// `anchor_freq` note).
///
/// Factored out of `detect_from_iq` so the grouping rule is directly
/// unit-testable, matching [`apply_vbi_confidence_tier`]'s rationale.
fn dedup_by_frequency(results: Vec<DetectionResult>) -> Vec<DetectionResult> {
    let mut deduped: Vec<DetectionResult> = Vec::new();
    let mut anchor_hz = 0.0f64;
    for r in results {
        if let Some(last) = deduped.last_mut()
            && (r.frequency_hz as f64 - anchor_hz).abs() < DEDUP_BW_HZ
        {
            if r.confidence > last.confidence
                || (r.confidence == last.confidence && r.rssi_dbm > last.rssi_dbm)
            {
                *last = r;
            }
            continue;
        }
        anchor_hz = r.frequency_hz as f64;
        deduped.push(r);
    }
    deduped
}

/// The confidence-tier decision the VBI confirm stage applies, factored
/// out as a pure function of `(sig_type, conf, evidence, demote_flag)`
/// so it's directly unit-testable against synthetic
/// [`crate::vbi::FieldSyncEvidence`] values — constructing a real IQ
/// signal that's simultaneously "confirmable VBI structure" *and*
/// "genuinely PAL/NTSC-ambiguous" (the promote case) is a contrived
/// combination in practice, since a real, well-formed line rate is
/// exactly what lets the harmonic/time-domain classifiers resolve the
/// standard confidently in the first place.
///
/// - **Boost**: an 0.8+ (strong-path) hit with confirmed field-sync
///   structure — two or more groups spaced a real field period apart,
///   or one group when the slice is too short to possibly contain a
///   second — becomes 0.95.
/// - **Promote**: an `AnalogVideoUnknown` (standard-ambiguous, 0.6) hit
///   with confirmed structure becomes 0.75, clearing the default 0.7
///   floor — the slice is definitely analog video, just not tagged
///   PAL vs NTSC.
/// - **Demote** (opt-in via `demote_unconfirmed_video`): an 0.8+ hit
///   spanning at least 2.5 field periods with *zero* confirmed groups
///   drops to 0.6.
fn apply_vbi_confidence_tier(
    sig_type: SignalType,
    conf: f32,
    evidence: &crate::vbi::FieldSyncEvidence,
    demote_unconfirmed: bool,
) -> f32 {
    let short_slice = evidence.slice_field_periods < 2.2;
    let confirmed =
        (evidence.groups >= 2 && evidence.spacing_ok) || (evidence.groups >= 1 && short_slice);

    if conf >= 0.8 && confirmed {
        0.95
    } else if sig_type == SignalType::AnalogVideoUnknown && confirmed {
        0.75
    } else if demote_unconfirmed
        && conf >= 0.8
        && evidence.slice_field_periods >= 2.5
        && evidence.groups == 0
    {
        0.6
    } else {
        conf
    }
}

impl AnalogFpvDetector {
    /// [`FpvDetector::detect_from_iq`] with cross-batch spectral
    /// integration (see [`SpectralIntegrator`]). Differences from the
    /// single-shot path:
    ///
    /// - every wideband probe is demodulated, transformed, and
    ///   accumulated — the per-probe energy gate no longer skips
    ///   classification, because a signal below this batch's energy
    ///   floor is exactly the one integration exists to find (cost:
    ///   ~one FFT per probe per batch instead of only gate-passers);
    /// - classification thresholds run against the accumulated average
    ///   spectrum for each probe's frequency bucket, so sensitivity
    ///   grows over successive batches at the same tuning.
    pub fn detect_from_iq_integrated(
        &self,
        iq_data: &[Complex<f32>],
        center_freq: u64,
        sample_rate: u32,
        integrator: &mut SpectralIntegrator,
    ) -> Vec<DetectionResult> {
        self.detect_from_iq_impl(iq_data, center_freq, sample_rate, Some(integrator))
    }

    fn detect_from_iq_impl(
        &self,
        iq_data: &[Complex<f32>],
        center_freq: u64,
        sample_rate: u32,
        mut integration: Option<&mut SpectralIntegrator>,
    ) -> Vec<DetectionResult> {
        let n = iq_data.len();
        if n < 2048 {
            return vec![];
        }

        let nan_count = iq_data
            .iter()
            .filter(|s| !s.re.is_finite() || !s.im.is_finite())
            .count();
        let sanitized_iq;
        let iq_data = if nan_count > 0 {
            log::warn!(
                "Sanitized {} non-finite samples (NaN/Inf) to zero in Analog processing",
                nan_count
            );
            sanitized_iq = iq_data
                .iter()
                .map(|s| {
                    if s.re.is_finite() && s.im.is_finite() {
                        *s
                    } else {
                        Complex::new(0.0, 0.0)
                    }
                })
                .collect::<Vec<_>>();
            &sanitized_iq[..]
        } else {
            iq_data
        };

        let mut final_results = Vec::new();

        // Fast path for narrow-band / already-baseband signals. The
        // threshold is the wideband target rate (10 MHz), not
        // `min_bandwidth` (3 MHz): the sliding-DDC grid below uses a
        // fixed 5 MHz step and 5 MHz edge margin, so it needs
        // `half_bw > margin`, i.e. `sample_rate > 10 MHz`, to produce a
        // non-degenerate set of probe positions. Below that the grid
        // collapsed to zero/one probe (a 5-8 MHz capture got no
        // coverage at all), so anything ≤ 10 MHz is treated as a single
        // baseband slice and classified directly.
        if sample_rate <= WIDEBAND_TARGET_RATE_HZ {
            let (sig_type, conf) = match integration.as_deref_mut() {
                Some(integ) => self.detect_sync_pulses_integrated(
                    iq_data,
                    sample_rate,
                    center_freq as i64,
                    integ,
                ),
                None => self.detect_sync_pulses(iq_data, sample_rate),
            };
            if sig_type != SignalType::Unknown {
                // Measured mean power like every other path — this used
                // to be a hardcoded -50.0, which fed a constant into the
                // dedup RSSI tiebreak and the scanner's reported levels.
                // The epsilon guards log10(0); unreachable in practice
                // (the classifier's variance gate rejects silence before
                // this) but harmless insurance against a -inf.
                let energy: f32 = iq_data
                    .iter()
                    .map(|s| s.re * s.re + s.im * s.im)
                    .sum::<f32>()
                    / iq_data.len() as f32;
                final_results.push(DetectionResult {
                    channel: None,
                    frequency_hz: center_freq,
                    confidence: conf,
                    rssi_dbm: 10.0 * (energy + 1e-12).log10(),
                    bandwidth_hz: sample_rate,
                    signal_type: sig_type,
                });
            }
            final_results.retain(|r| {
                r.bandwidth_hz >= self.min_bandwidth
                    && r.bandwidth_hz <= self.max_bandwidth
                    && r.confidence >= self.min_confidence
            });
            return final_results;
        }

        // Sliding DDC probe: sweep the entire capture bandwidth in 5 MHz steps,
        // FM-demodulate at each position, and look for H-sync line rate in FFT.
        // No channel table or FFT blob finder needed — finds signals at ANY
        // frequency with proper clustering.
        {
            let target_rate = WIDEBAND_TARGET_RATE_HZ;
            let step_hz = 5_000_000.0f64;
            let half_bw = sample_rate as f64 / 2.0;
            let margin = step_hz;
            let scan_start = -half_bw + margin;
            let scan_end = half_bw - margin;
            // Inclusive endpoint: the loop below visits offsets
            // `scan_start + step·{0..n_steps-1}`. `scan_end - scan_start`
            // is an exact multiple of `step_hz` (both are 5 MHz grids),
            // so a bare `ceil` produced a top probe at `scan_end - step`
            // — leaving the top ~5 MHz of the capture with no probe
            // centre. The `+ 1` lands the last probe on `scan_end`
            // itself; its `target_rate/3` passband stays inside Nyquist.
            let n_steps = ((scan_end - scan_start) / step_hz).round() as usize + 1;

            // A 10–25 MHz capture yields only 1–3 probe positions — too
            // few for the percentile noise floor below to mean anything
            // (`sorted_e[len/4]` was 0.0 for < 4 probes, which let the
            // strongest probe through the gate on a completely empty
            // band, every batch). A video signal is ~20 MHz wide anyway,
            // so at these rates it spans the whole capture: classify the
            // capture as one baseband slice at the tuned centre instead,
            // exactly like the ≤ 10 MHz fast path. This also reports ONE
            // stable frequency (the centre) rather than 2-3 fake probe
            // offsets that deduped into separate detections downstream.
            if n_steps < 4 {
                let (sig_type, conf) = match integration.as_deref_mut() {
                    Some(integ) => self.detect_sync_pulses_integrated(
                        iq_data,
                        sample_rate,
                        center_freq as i64,
                        integ,
                    ),
                    None => self.detect_sync_pulses(iq_data, sample_rate),
                };
                if sig_type != SignalType::Unknown {
                    let energy: f32 = iq_data
                        .iter()
                        .map(|s| s.re * s.re + s.im * s.im)
                        .sum::<f32>()
                        / iq_data.len() as f32;
                    final_results.push(DetectionResult {
                        channel: None,
                        frequency_hz: center_freq,
                        confidence: conf,
                        rssi_dbm: 10.0 * energy.log10(),
                        bandwidth_hz: sample_rate,
                        signal_type: sig_type,
                    });
                }
                final_results.retain(|r| {
                    r.bandwidth_hz >= self.min_bandwidth
                        && r.bandwidth_hz <= self.max_bandwidth
                        && r.confidence >= self.min_confidence
                });
                return final_results;
            }

            // First pass: measure energy at each probe position. Track the
            // *actual* rate each probe was decimated to alongside it (see
            // `decimated_rate`'s doc) rather than re-deriving/assuming it
            // later — that assumption used to silently diverge from
            // reality whenever `sample_rate` wasn't an exact multiple of
            // `target_rate`.
            //
            // `target_rate_prime` is the same for every probe in this
            // call (only the mixing offset varies), matching the choice
            // the pre-GPU per-probe loop below made independently each
            // iteration — hoisted out so the batched GPU path can pass
            // one shared `decimation_factor`/`cutoff_hz` for the whole
            // sweep.
            let target_rate_prime = if sample_rate > target_rate * 2 {
                target_rate
            } else {
                sample_rate
            };
            let isolated_rate = Self::decimated_rate(sample_rate, target_rate_prime);
            let offsets: Vec<f64> = (0..n_steps)
                .map(|step| scan_start + step as f64 * step_hz)
                .collect();

            let mut probes: Vec<(f64, f32, Vec<Complex<f32>>, u32)> = Vec::with_capacity(n_steps);

            #[cfg(feature = "gpu")]
            let gpu_decimated: Option<Vec<Vec<Complex<f32>>>> = self.gpu.as_ref().map(|gpu| {
                let decimation_factor = Self::decimation_factor(sample_rate, target_rate_prime);
                let cutoff_hz = target_rate_prime as f32 / 3.0;
                gpu.sweep(iq_data, sample_rate, &offsets, decimation_factor, cutoff_hz)
            });
            #[cfg(not(feature = "gpu"))]
            let gpu_decimated: Option<Vec<Vec<Complex<f32>>>> = None;

            if let Some(decimated) = gpu_decimated {
                // Batched GPU sweep: one DDC dispatch for every probe,
                // instead of `n_steps` sequential CPU passes over the
                // whole input.
                for (&offset_hz, isolated_iq) in offsets.iter().zip(decimated) {
                    let energy: f32 = isolated_iq
                        .iter()
                        .map(|s| s.re * s.re + s.im * s.im)
                        .sum::<f32>()
                        / isolated_iq.len() as f32;
                    probes.push((offset_hz, energy, isolated_iq, isolated_rate));
                }
            } else {
                for &offset_hz in &offsets {
                    let isolated_iq = self.ddc_and_decimate(
                        iq_data,
                        sample_rate,
                        offset_hz as f32,
                        target_rate_prime,
                    );
                    let energy: f32 = isolated_iq
                        .iter()
                        .map(|s| s.re * s.re + s.im * s.im)
                        .sum::<f32>()
                        / isolated_iq.len() as f32;
                    probes.push((offset_hz, energy, isolated_iq, isolated_rate));
                }
            }

            // Noise floor: 25th percentile of probe energies (robust to FM
            // signals covering a large fraction of the bandwidth).
            // `n_steps >= 4` is guaranteed by the narrow-capture return
            // above, so the percentile is always meaningful here.
            let mut sorted_e: Vec<f32> = probes.iter().map(|p| p.1).collect();
            sorted_e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let noise_floor = sorted_e[sorted_e.len() / 4];
            let max_energy = sorted_e.last().copied().unwrap_or(0.0);
            let multiplier = 10.0f32.powf(self.energy_threshold_db / 10.0);
            // The threshold used to be capped at `max_energy * 0.5`
            // ("don't exclude the peak"), but on a FLAT band — empty
            // spectrum, every probe ≈ the same noise energy — that cap
            // pulled the threshold BELOW the noise floor, so the
            // strongest probe (i.e. some noise window) was analysed on
            // every batch. A real signal clears floor + threshold_db on
            // its own; a band where nothing does is a band with nothing
            // in it. The one case the cap legitimately served — a floor
            // of digital silence (0.0), where floor × multiplier gates
            // nothing — keeps a half-max rescue so a lone strong signal
            // over silence is still analysed.
            let energy_thresh = if noise_floor > 0.0 {
                noise_floor * multiplier
            } else {
                max_energy * 0.5
            };

            // Collect all positive detections from the sweep
            let mut sweep_hits: Vec<(f64, f32, SignalType, f32)> = Vec::new(); // (freq_hz, energy, type, conf)
            for (offset_hz, energy, isolated_iq, isolated_rate) in &probes {
                // Single-shot mode: the energy gate limits classification
                // cost to probes that plausibly hold a signal. Integrated
                // mode classifies (and accumulates) EVERY probe — a
                // signal below this batch's energy floor is exactly the
                // one integration exists to find, and the harmonic +
                // cepstrum + VBI gates still protect against false
                // positives on the averaged spectrum.
                if integration.is_none() && *energy <= energy_thresh {
                    continue;
                }
                let (sig_type, conf) = match integration.as_deref_mut() {
                    Some(integ) => {
                        let key = center_freq as i64 + *offset_hz as i64;
                        self.detect_sync_pulses_integrated(isolated_iq, *isolated_rate, key, integ)
                    }
                    None => self.detect_sync_pulses(isolated_iq, *isolated_rate),
                };
                if sig_type != SignalType::Unknown {
                    let freq_hz = center_freq as f64 + offset_hz;
                    sweep_hits.push((freq_hz, *energy, sig_type, conf));
                }
            }

            // Cluster hits: group detections within 25 MHz (FM video BW),
            // keep the strongest member. Each cluster tracks an immutable
            // `anchor_freq` (the first hit's centre) separately from the
            // strongest member's `(freq, energy, sig, conf)`. The earlier
            // shape compared each new hit against the previous cluster's
            // *strongest member* and then overwrote the anchor when the
            // member updated — for evenly-spaced hits at 0/20/40/60/80
            // MHz that chained the whole sweep into one 80-MHz-wide
            // cluster, because every 20-MHz step landed inside the
            // 25-MHz window relative to the *previous* anchor.
            sweep_hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            const CLUSTER_BW_HZ: f64 = 25e6;
            // Tuple element 0 is the immutable anchor frequency used for
            // grouping; elements 1..=4 are the strongest member's
            // (freq, energy, sig_type, conf).
            let mut clusters: Vec<(f64, f64, f32, SignalType, f32)> = Vec::new();
            for hit in &sweep_hits {
                if let Some(last) = clusters.last_mut()
                    && (hit.0 - last.0).abs() < CLUSTER_BW_HZ
                {
                    // Same cluster — the anchor (last.0) stays fixed, and
                    // the remaining fields split by what each is *for*.
                    //
                    // Frequency and energy follow the strongest member:
                    // the probe with the most energy is looking straight
                    // at the carrier, so its centre is the best position
                    // estimate.
                    //
                    // Classification follows the most CONFIDENT member.
                    // These can genuinely differ: with a ±5 MHz FM
                    // deviation, sync tips swing to the passband edge of
                    // the carrier-centred probe, whose time-domain
                    // disambiguation then fails consensus and returns
                    // AnalogVideoUnknown at 0.6 — while the probe one
                    // step off-centre sees the tips mid-passband and
                    // resolves PAL at 0.8 from twenty-seven identical
                    // intervals. Letting energy pick the classification
                    // too meant the strongest probe's Unknown shadowed
                    // its siblings' confident answer, and the cluster
                    // then died at the min-confidence filter: a clean,
                    // strong signal reported as nothing at all.
                    if hit.1 > last.2 {
                        last.1 = hit.0;
                        last.2 = hit.1;
                    }
                    if hit.3 > last.4 {
                        last.3 = hit.2;
                        last.4 = hit.3;
                    }
                    continue;
                }
                clusters.push((hit.0, hit.0, hit.1, hit.2, hit.3));
            }

            for (_anchor, freq_hz, energy, sig_type, conf) in clusters {
                // Sweep clusters are already deduped within 25 MHz, and
                // the final pass below merges anything that still
                // overlaps, so we can push each cluster directly.
                final_results.push(DetectionResult {
                    channel: None,
                    frequency_hz: freq_hz as u64,
                    confidence: conf,
                    rssi_dbm: 10.0 * energy.log10(),
                    bandwidth_hz: target_rate,
                    signal_type: sig_type,
                });
            }
        }
        // Final dedup: merge any results within 20 MHz, keep strongest
        final_results.sort_by(|a, b| {
            (a.frequency_hz as f64)
                .partial_cmp(&(b.frequency_hz as f64))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut deduped = dedup_by_frequency(final_results);

        deduped.retain(|r| {
            r.bandwidth_hz >= self.min_bandwidth
                && r.bandwidth_hz <= self.max_bandwidth
                && r.confidence >= self.min_confidence
        });

        deduped
    }
}

impl FpvDetector for AnalogFpvDetector {
    fn detect_from_iq(
        &self,
        iq_data: &[Complex<f32>],
        center_freq: u64,
        sample_rate: u32,
    ) -> Vec<DetectionResult> {
        self.detect_from_iq_impl(iq_data, center_freq, sample_rate, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(freq_hz: u64, confidence: f32, rssi_dbm: f32) -> DetectionResult {
        DetectionResult {
            channel: None,
            frequency_hz: freq_hz,
            confidence,
            rssi_dbm,
            bandwidth_hz: 10_000_000,
            signal_type: SignalType::AnalogVideoPal,
        }
    }

    #[test]
    fn dedup_does_not_chain_a_whole_band_into_one_detection() {
        // Band F, four simultaneous VTXs 20 MHz apart: 5740 / 5760 / 5780
        // / 5800. Each is within the 25 MHz merge window of its immediate
        // *neighbour*, so comparing against the running strongest member
        // walked the comparison point up the band and folded all four into
        // a single detection — an 8-channel band would collapse to one hit.
        // Anchoring the comparison bounds each group to one 25 MHz window,
        // so the span is covered by ceil(80/25) = 2 groups, not 1.
        //
        // This does NOT separate adjacent channels, and cannot: FM video is
        // ~20 MHz wide, so two VTXs one channel apart genuinely overlap and
        // the 25 MHz window is sized for that signal bandwidth. Telling
        // those apart needs bandwidth-aware merging, not a narrower window.
        let out = dedup_by_frequency(vec![
            det(5_740_000_000, 0.8, -50.0),
            det(5_760_000_000, 0.9, -40.0),
            det(5_780_000_000, 0.95, -30.0),
            det(5_800_000_000, 0.85, -35.0),
        ]);
        assert_eq!(
            out.len(),
            2,
            "expected the 80 MHz span to need 2 anchored groups, got {:?}",
            out.iter().map(|r| r.frequency_hz).collect::<Vec<_>>()
        );
        // Each surviving group reports its own strongest member.
        assert_eq!(out[0].frequency_hz, 5_760_000_000);
        // 5800 (0.85) loses to 5780 (0.95) inside the second group.
        assert_eq!(out[1].frequency_hz, 5_780_000_000);
    }

    #[test]
    fn dedup_group_width_is_bounded_by_the_window_not_the_input_span() {
        // Eight detections spanning 140 MHz, each 20 MHz from the last.
        // Unanchored, every one merges into its predecessor and the whole
        // sweep returns a single detection. Anchored, the result count
        // grows with the span.
        let hits: Vec<DetectionResult> = (0..8)
            .map(|i| det(5_700_000_000 + i * 20_000_000, 0.8, -50.0))
            .collect();
        let out = dedup_by_frequency(hits);
        assert!(
            out.len() >= 4,
            "140 MHz of detections collapsed to {} group(s) — chaining regression",
            out.len()
        );
    }

    #[test]
    fn dedup_still_merges_genuine_duplicates_and_keeps_the_strongest() {
        // Two probes landing on one real signal a few MHz apart: still one
        // detection, and it must be the higher-confidence one.
        let out = dedup_by_frequency(vec![
            det(5_800_000_000, 0.8, -50.0),
            det(5_805_000_000, 0.95, -30.0),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].frequency_hz, 5_805_000_000);
        assert_eq!(out[0].confidence, 0.95);

        // Equal confidence -> stronger RSSI wins.
        let out = dedup_by_frequency(vec![
            det(5_800_000_000, 0.8, -50.0),
            det(5_802_000_000, 0.8, -20.0),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].rssi_dbm, -20.0);
    }

    /// Generate a synthetic FM-modulated PAL H-sync pulse train.
    ///
    /// Produces IQ data whose FM demodulation yields a clear rectangular
    /// waveform at the PAL line rate (15625 Hz).  The baseband deviation
    /// is ±1.0 radian/sample — strong enough that the FM-demod arg()
    /// output has significant harmonic content.
    fn make_pal_pulse_train(sample_rate: u32, num_lines: usize) -> Vec<Complex<f32>> {
        let line_rate = 15625.0f32;
        let spl = (sample_rate as f32 / line_rate).round() as usize;
        let sync_tip = (sample_rate as f32 * 4.7e-6) as usize;
        let total = spl * num_lines + 1;

        // Baseband: sync tip at −1.0, blanking at +1.0.
        let mut bb = vec![1.0f32; total];
        for line in 0..num_lines {
            let s = line * spl;
            for i in 0..sync_tip.min(total.saturating_sub(s)) {
                bb[s + i] = -1.0;
            }
        }

        // FM-modulate.
        let mut phase = 0.0f32;
        let mut iq = Vec::with_capacity(total);
        for &b in &bb {
            let (s, c) = phase.sin_cos();
            iq.push(Complex::new(c, s));
            phase += b;
        }
        iq
    }

    /// Stage-parity check for GPU Phase 2: the batched GPU DDC
    /// (`GpuAnalog::sweep`) must reproduce `ddc_and_decimate`'s decimated
    /// IQ closely enough that downstream classification (unchanged, CPU,
    /// operating on this output) sees the same signal. Compares directly
    /// against the private `ddc_and_decimate`/`decimation_factor`
    /// helpers rather than going through `detect_from_iq`, so a
    /// divergence here points straight at the GPU kernels rather than
    /// requiring a classification-level failure to notice it.
    ///
    /// Skips gracefully with no adapter. Only meaningfully validates
    /// Metal-specific numerics when run on the dev Mac — see the crate's
    /// GPU acceleration plan for why CI's lavapipe backend isn't a
    /// substitute for that.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_ddc_matches_cpu_ddc_and_decimate() {
        let Some(gpu) = crate::gpu::GpuAnalog::try_new() else {
            eprintln!("No GPU adapter found; skipping gpu_ddc_matches_cpu_ddc_and_decimate");
            return;
        };

        let sample_rate = 50_000_000u32;
        let iq = make_pal_pulse_train(sample_rate, 2000);
        let target_rate = WIDEBAND_TARGET_RATE_HZ;
        let offset_hz = -10_000_000.0f64;
        let decimation_factor = AnalogFpvDetector::decimation_factor(sample_rate, target_rate);
        let cutoff_hz = target_rate as f32 / 3.0;

        let cpu_det = AnalogFpvDetector::default();
        let cpu_out = cpu_det.ddc_and_decimate(&iq, sample_rate, offset_hz as f32, target_rate);

        // Two consecutive sweeps: the second reuses the persistent
        // buffer pool, so this also guards the pooled-upload path.
        for label in ["fresh-buffers", "pooled-rerun"] {
            let gpu_out = gpu.sweep(&iq, sample_rate, &[offset_hz], decimation_factor, cutoff_hz);
            let gpu_probe = &gpu_out[0];

            assert_eq!(
                cpu_out.len(),
                gpu_probe.len(),
                "{label}: GPU/CPU decimated length mismatch"
            );

            let mut sq_err = 0.0f64;
            let mut sq_ref = 0.0f64;
            for (c, g) in cpu_out.iter().zip(gpu_probe.iter()) {
                let d_re = (c.re - g.re) as f64;
                let d_im = (c.im - g.im) as f64;
                sq_err += d_re * d_re + d_im * d_im;
                sq_ref += (c.re as f64).powi(2) + (c.im as f64).powi(2);
            }
            let rel_rms = (sq_err / sq_ref.max(1e-12)).sqrt();
            assert!(
                rel_rms < 0.01,
                "{label}: GPU DDC output diverges from CPU by {:.4}% RMS (tolerance 1%)",
                rel_rms * 100.0
            );
        }
    }

    /// A large decimation factor (smaller pooled output than the
    /// earlier tests') must still match the CPU reference — also
    /// exercises pool rebinding at a different output geometry.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_sweep_matches_cpu_at_large_decimation() {
        let Some(gpu) = crate::gpu::GpuAnalog::try_new() else {
            eprintln!("No GPU adapter found; skipping");
            return;
        };
        let sample_rate = 50_000_000u32;
        let iq = make_pal_pulse_train(sample_rate, 500);
        let decimation_factor = 40usize;
        let cutoff_hz = 400_000.0f32;
        let offset_hz = -3_000_000.0f64;

        let mut ddc = crate::ddc::StreamingDDC::new(offset_hz as f32, sample_rate, cutoff_hz);
        let cpu_out = ddc.process_decimated(&iq, decimation_factor);
        let gpu_out = gpu.sweep(&iq, sample_rate, &[offset_hz], decimation_factor, cutoff_hz);
        let gpu_probe = &gpu_out[0];
        assert_eq!(cpu_out.len(), gpu_probe.len());
        let mut sq_err = 0.0f64;
        let mut sq_ref = 0.0f64;
        for (c, g) in cpu_out.iter().zip(gpu_probe.iter()) {
            sq_err += ((c.re - g.re) as f64).powi(2) + ((c.im - g.im) as f64).powi(2);
            sq_ref += (c.re as f64).powi(2) + (c.im as f64).powi(2);
        }
        let rel_rms = (sq_err / sq_ref.max(1e-12)).sqrt();
        assert!(rel_rms < 0.01, "fallback diverges: {:.4}%", rel_rms * 100.0);
    }

    /// Timing harness for the GPU sweep (not asserted — run with
    /// `cargo test --features gpu gpu_sweep_timing -- --ignored --nocapture`).
    /// Historical baseline on an Apple-silicon GPU: ~15 ms/sweep for
    /// 1 M samples × 11 probes at D = 5 — the number that condemned the
    /// workgroup-tiling experiment (see the shader's note).
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn gpu_sweep_timing() {
        let Some(gpu) = crate::gpu::GpuAnalog::try_new() else {
            eprintln!("No GPU adapter found; skipping");
            return;
        };
        let sample_rate = 50_000_000u32;
        let iq = make_pal_pulse_train(sample_rate, 300); // ~1 M samples, a realistic large batch
        let offsets: Vec<f64> = (0..11).map(|i| -20e6 + i as f64 * 5e6).collect();
        let decimation_factor =
            AnalogFpvDetector::decimation_factor(sample_rate, WIDEBAND_TARGET_RATE_HZ);
        let cutoff_hz = WIDEBAND_TARGET_RATE_HZ as f32 / 3.0;
        // Warm-up (pipeline + pool growth), then timed runs.
        let _ = gpu.sweep(&iq, sample_rate, &offsets, decimation_factor, cutoff_hz);
        let t0 = std::time::Instant::now();
        let reps = 5;
        for _ in 0..reps {
            let _ = gpu.sweep(&iq, sample_rate, &offsets, decimation_factor, cutoff_hz);
        }
        eprintln!("sweep: {:?}/sweep", t0.elapsed() / reps);
    }

    #[test]
    fn cepstrum_passes_real_pal_pulse_train() {
        // Use 500 lines at 10 MSPS → ~320K samples.  That gives
        // bin_hz ≈ 31 Hz, enough to resolve PAL harmonics and
        // produce a clear cepstral peak.
        let sr = 10_000_000u32;
        let iq = make_pal_pulse_train(sr, 500);
        let det = AnalogFpvDetector::new(-20.0);
        let (sig, conf) = det.detect_sync_pulses(&iq, sr);
        assert!(
            sig != SignalType::Unknown,
            "PAL pulse train rejected; sig={sig:?}, conf={conf}"
        );
        assert!(conf > 0.0);
    }

    /// A `sample_rate` of 0 must not panic (it would make the line-rate
    /// bin index `Inf as usize` and overflow `bin + range`). Both the
    /// direct classifier and the full `detect_from_iq` entry point must
    /// degrade to "nothing" instead.
    #[test]
    fn zero_sample_rate_does_not_panic() {
        let iq = vec![Complex::new(0.5, -0.5); 4096];
        let det = AnalogFpvDetector::new(-20.0);
        assert_eq!(det.detect_sync_pulses(&iq, 0).0, SignalType::Unknown);
        assert!(det.detect_from_iq(&iq, 5_800_000_000, 0).is_empty());
    }

    #[test]
    fn cepstrum_rejects_pure_noise() {
        let sr = 10_000_000u32;
        let n = 200_000;
        let mut iq = Vec::with_capacity(n);
        let mut seed = 42u64;
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let re = (seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let im = (seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
            iq.push(Complex::new(re, im));
        }
        let det = AnalogFpvDetector::new(-20.0);
        let (sig, _) = det.detect_sync_pulses(&iq, sr);
        assert_eq!(sig, SignalType::Unknown, "noise should be Unknown");
    }

    /// Verify `verify_cepstrum` directly: a harmonic comb passes,
    /// a flat spectrum fails.
    #[test]
    fn verify_cepstrum_unit_test() {
        let sr = 10_000_000u32;
        let line_hz = 15625.0f32;
        let fft_len = 8192;
        let det = AnalogFpvDetector::new(-20.0);

        // Build a synthetic magnitude spectrum with a harmonic comb at
        // the line rate — simulates what a real pulse train would
        // produce.
        let bin_hz = sr as f32 / fft_len as f32;
        let mut mags = vec![0.001f32; fft_len];
        for k in 1..=10 {
            let bin = (k as f32 * line_hz / bin_hz).round() as usize;
            if bin < fft_len / 2 {
                // Strong peak at this harmonic.
                mags[bin] = 100.0;
                // Mirror.
                mags[fft_len - bin] = 100.0;
            }
        }

        assert!(
            det.verify_cepstrum(&mags, sr, line_hz, fft_len),
            "harmonic comb should pass cepstrum check"
        );

        // Flat spectrum — no periodic structure.
        let flat = vec![1.0f32; fft_len];
        assert!(
            !det.verify_cepstrum(&flat, sr, line_hz, fft_len),
            "flat spectrum should fail cepstrum check"
        );
    }

    /// Synthetic baseband (already FM-demodulated) with sync tips at
    /// the requested line rate. Used to validate
    /// `classify_pal_ntsc_time_domain` in isolation from the FM-demod
    /// step. The waveform is +1 between tips, dipping to -1 for a
    /// 4.7 µs sync pulse — same shape the real demod sees.
    fn make_synthetic_demod(sample_rate: u32, line_hz: f32, num_lines: usize) -> Vec<f32> {
        let spl = (sample_rate as f32 / line_hz).round() as usize;
        let sync_tip = (sample_rate as f32 * 4.7e-6) as usize;
        let total = spl * num_lines;
        let mut bb = vec![1.0f32; total];
        for line in 0..num_lines {
            let s = line * spl;
            for i in 0..sync_tip.min(total.saturating_sub(s)) {
                bb[s + i] = -1.0;
            }
        }
        bb
    }

    #[test]
    fn time_domain_disambig_picks_pal_at_15625_hz() {
        let sr = 25_000_000u32;
        let demod = make_synthetic_demod(sr, 15625.0, 80);
        let class = classify_pal_ntsc_time_domain(&demod, sr);
        assert_eq!(class, Some(SignalType::AnalogVideoPal));
    }

    #[test]
    fn time_domain_disambig_picks_ntsc_at_15734_hz() {
        let sr = 25_000_000u32;
        let demod = make_synthetic_demod(sr, 15734.0, 80);
        let class = classify_pal_ntsc_time_domain(&demod, sr);
        assert_eq!(class, Some(SignalType::AnalogVideoNtsc));
    }

    /// Regression: the classifier's threshold must survive FM-click
    /// outliers. Physically, `fm_demod` output is bounded to ±π, and a
    /// weak signal's sync tips sit well inside that (−0.4·rpv ≈ −0.5
    /// at 5 MHz deviation / 25 MSPS) — so the old `global_min · 0.3`
    /// threshold latched onto a −π click, landed at ≈ −0.94, below
    /// every real tip, and the disambiguator saw only clicks. The
    /// smoothed percentile threshold ignores single-sample clicks
    /// entirely.
    #[test]
    fn time_domain_disambig_survives_click_outliers() {
        let sr = 25_000_000u32;
        // Scale the ±1 synthetic to the real convention: tips at −0.5,
        // blanking at 0.
        let mut demod: Vec<f32> = make_synthetic_demod(sr, 15625.0, 80)
            .into_iter()
            .map(|v| (v - 1.0) * 0.25)
            .collect();
        // Single-sample ±π FM clicks at an aperiodic stride so they
        // can't masquerade as a line rate.
        let mut k = 917usize;
        while k < demod.len() {
            demod[k] = -std::f32::consts::PI;
            k += 917 + (k % 131);
        }
        assert_eq!(
            classify_pal_ntsc_time_domain(&demod, sr),
            Some(SignalType::AnalogVideoPal)
        );
    }

    /// The percentile threshold is DC-invariant: a demod offset from a
    /// tuning error must not break classification (the old
    /// zero-referenced threshold required sync tips below zero).
    #[test]
    fn time_domain_disambig_is_dc_invariant() {
        let sr = 25_000_000u32;
        let mut demod = make_synthetic_demod(sr, 15734.0, 80);
        for v in &mut demod {
            *v += 1.5; // tips now at +0.5, blanking at +2.5 — all positive
        }
        assert_eq!(
            classify_pal_ntsc_time_domain(&demod, sr),
            Some(SignalType::AnalogVideoNtsc)
        );
    }

    /// Weak-signal sanity: a full standards-shaped NTSC signal buried
    /// in complex AWGN must still be detected as analog video. The
    /// noise level here (σ = 0.35 per I/Q component against a
    /// unit-amplitude carrier, ≈ 6 dB CNR in the capture bandwidth)
    /// sits well below clean-signal conditions; it exercises the
    /// median-based FFT noise floor and the percentile sync
    /// thresholds together.
    #[test]
    fn detects_video_under_awgn() {
        let sample_rate = 15_360_000u32;
        let cfg = synth_config(false, sample_rate);
        let mut iq = generate_iq(&cfg, 2, 0.0);
        let mut seed = 0xC0FFEE42u64;
        let gauss = |s: &mut u64| -> f32 {
            // Irwin-Hall (sum of 12 uniforms) approximate N(0,1).
            let mut acc = 0.0f32;
            for _ in 0..12 {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                acc += (*s >> 11) as f32 / (1u64 << 53) as f32;
            }
            acc - 6.0
        };
        for z in iq.iter_mut() {
            z.re += 0.35 * gauss(&mut seed);
            z.im += 0.35 * gauss(&mut seed);
        }
        let det = AnalogFpvDetector::default();
        let (sig_type, conf) = det.detect_sync_pulses(&iq, sample_rate);
        assert!(
            sig_type.is_analog_video(),
            "expected analog video under AWGN, got {sig_type:?} (conf {conf})"
        );
    }

    #[test]
    fn time_domain_disambig_returns_none_on_too_few_tips() {
        let sr = 25_000_000u32;
        // Only 3 lines = 2 intervals < the 8-interval minimum.
        let demod = make_synthetic_demod(sr, 15625.0, 3);
        let class = classify_pal_ntsc_time_domain(&demod, sr);
        assert_eq!(class, None);
    }

    #[test]
    fn time_domain_disambig_returns_none_on_midpoint_rate() {
        let sr = 25_000_000u32;
        // Exactly between PAL and NTSC — neither answer is honest.
        let demod = make_synthetic_demod(sr, 15679.5, 80);
        let class = classify_pal_ntsc_time_domain(&demod, sr);
        assert_eq!(class, None);
    }

    /// Regression: a pulse train far from BOTH line rates must return
    /// `None`, not "whichever standard is closer". FM-demodulated noise
    /// produces dips continuously, so the scan's `min_gap` skip makes
    /// the median interval collapse to ≈ 30 µs (~33 kHz) — which used
    /// to classify as NTSC purely because 33 kHz is nearer 15734 than
    /// 15625. That was the engine behind empty-band false positives on
    /// live captures.
    #[test]
    fn time_domain_disambig_rejects_rates_far_from_both_standards() {
        let sr = 25_000_000u32;
        // ~28.6 kHz pulse train — a plausible noise/seam artifact rate,
        // nowhere near a real video line rate.
        let demod = make_synthetic_demod(sr, 28_600.0, 200);
        let class = classify_pal_ntsc_time_domain(&demod, sr);
        assert_eq!(class, None);
    }

    /// A 10–25 MHz capture (too narrow for a meaningful probe sweep)
    /// must classify the whole slice and report the TUNED CENTRE, not
    /// a probe-grid offset. At 15.36 MSPS the old sweep had exactly two
    /// probes (centre −2.68 / +2.32 MHz) whose fake frequencies deduped
    /// into separate downstream detections of the same transmitter.
    #[test]
    fn narrow_wideband_capture_reports_center_frequency() {
        let sr = 15_360_000u32;
        // 64 lines ≈ 63 k samples → bin_hz ≈ 244 Hz, so the PAL and
        // NTSC line-rate bins collide and classification goes through
        // the time-domain disambiguator — the same shape a live 15.36
        // MSPS batch (~65 k samples) takes.
        let iq = make_pal_pulse_train(sr, 64);
        let det = AnalogFpvDetector::new(3.0);
        let results = det.detect_from_iq(&iq, 5_800_000_000, sr);
        assert_eq!(results.len(), 1, "expected exactly one detection");
        assert_eq!(results[0].frequency_hz, 5_800_000_000);
        assert_eq!(results[0].signal_type, SignalType::AnalogVideoPal);
    }

    /// Regression: a flat, empty band at a narrow-wideband rate must
    /// yield NO detections. The old <4-probe path set `noise_floor` to
    /// 0.0 and capped the threshold at `max_energy * 0.5`, so the
    /// strongest noise window was analysed on every single batch —
    /// which, combined with the unbounded time-domain classifier,
    /// reported phantom PAL/NTSC video on live empty-spectrum captures.
    #[test]
    fn flat_noise_band_at_narrow_wideband_rate_yields_nothing() {
        let sr = 15_360_000u32;
        let n = 150_000;
        let mut iq = Vec::with_capacity(n);
        let mut seed = 0xDEADBEEFu64;
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let re = (seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let im = (seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
            // Scale to the low amplitudes an int16-quantised empty band
            // actually delivers.
            iq.push(Complex::new(re * 1e-3, im * 1e-3));
        }
        let det = AnalogFpvDetector::default();
        let results = det.detect_from_iq(&iq, 3_650_000_000, sr);
        assert!(
            results.is_empty(),
            "empty band produced detections: {results:?}"
        );
    }

    // ── Phase 3: VBI confirm confidence tiers ──────────────────────────

    use crate::synthetic::{SyntheticVideoConfig, TestPattern, generate_iq};
    use crate::vbi::{FieldParity, FieldSyncEvidence};

    fn evidence(groups: usize, spacing_ok: bool, slice_field_periods: f32) -> FieldSyncEvidence {
        FieldSyncEvidence {
            groups,
            spacing_ok,
            slice_field_periods,
        }
    }

    #[test]
    fn tier_boosts_strong_confirmed_hit() {
        let e = evidence(2, true, 4.0);
        assert_eq!(
            apply_vbi_confidence_tier(SignalType::AnalogVideoNtsc, 0.8, &e, false),
            0.95
        );
    }

    #[test]
    fn tier_boosts_single_group_on_a_short_slice() {
        // Too short to possibly contain a second group -- one confirmed
        // group is all the evidence that could exist.
        let e = evidence(1, false, 1.5);
        assert_eq!(
            apply_vbi_confidence_tier(SignalType::AnalogVideoPal, 0.8, &e, false),
            0.95
        );
    }

    #[test]
    fn tier_does_not_boost_a_single_group_on_a_long_slice() {
        // Long enough that a real periodic vsync should have produced a
        // second confirmable group; one alone isn't enough evidence.
        let e = evidence(1, false, 4.0);
        assert_eq!(
            apply_vbi_confidence_tier(SignalType::AnalogVideoNtsc, 0.8, &e, false),
            0.8
        );
    }

    #[test]
    fn tier_promotes_standard_ambiguous_hit_when_confirmed() {
        let e = evidence(2, true, 4.0);
        assert_eq!(
            apply_vbi_confidence_tier(SignalType::AnalogVideoUnknown, 0.6, &e, false),
            0.75
        );
    }

    #[test]
    fn tier_leaves_standard_ambiguous_hit_alone_when_unconfirmed() {
        let e = evidence(0, false, 4.0);
        assert_eq!(
            apply_vbi_confidence_tier(SignalType::AnalogVideoUnknown, 0.6, &e, false),
            0.6
        );
    }

    #[test]
    fn tier_demotes_unconfirmed_strong_hit_only_when_flag_is_set() {
        let e = evidence(0, false, 3.0);
        assert_eq!(
            apply_vbi_confidence_tier(SignalType::AnalogVideoNtsc, 0.8, &e, false),
            0.8
        );
        assert_eq!(
            apply_vbi_confidence_tier(SignalType::AnalogVideoNtsc, 0.8, &e, true),
            0.6
        );
    }

    #[test]
    fn tier_does_not_demote_a_short_slice_even_with_the_flag_set() {
        // Too short to expect a second group -- absence of one isn't
        // evidence of anything.
        let e = evidence(0, false, 1.5);
        assert_eq!(
            apply_vbi_confidence_tier(SignalType::AnalogVideoNtsc, 0.8, &e, true),
            0.8
        );
    }

    fn synth_config(is_pal: bool, sample_rate: u32) -> SyntheticVideoConfig {
        SyntheticVideoConfig {
            sample_rate,
            is_pal,
            deviation_hz: 5e6,
            pattern: TestPattern::Bars,
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        }
    }

    #[test]
    fn full_field_ntsc_signal_boosts_to_0_95_via_detect_sync_pulses() {
        let sample_rate = 15_360_000u32;
        let cfg = synth_config(false, sample_rate);
        // 2 fields: enough for two field-period-spaced confirmed groups.
        let iq = generate_iq(&cfg, 2, 0.0);
        let det = AnalogFpvDetector::default();
        let (sig_type, conf) = det.detect_sync_pulses(&iq, sample_rate);
        assert_eq!(sig_type, SignalType::AnalogVideoNtsc);
        assert_eq!(conf, 0.95, "expected the VBI-confirmed boost");
    }

    #[test]
    fn line_rate_comb_without_vbi_structure_stays_at_0_8_by_default() {
        let sr = 15_360_000u32;
        // Line-rate-only comb (no equalizing/broad-pulse groups) --
        // exactly the shape make_fm_sync_iq / make_pal_pulse_train
        // fixtures already exercise elsewhere in this module.
        let iq = make_pal_pulse_train(sr, 800); // ~51 ms, > 2.5 field periods
        let det = AnalogFpvDetector::default();
        let (sig_type, conf) = det.detect_sync_pulses(&iq, sr);
        assert_eq!(sig_type, SignalType::AnalogVideoPal);
        assert_eq!(conf, 0.8, "unconfirmed comb should be untouched by default");
    }

    #[test]
    fn line_rate_comb_demotes_below_the_floor_when_flag_enabled() {
        let sr = 15_360_000u32;
        let iq = make_pal_pulse_train(sr, 800); // ~51 ms, > 2.5 field periods
        let det = AnalogFpvDetector {
            demote_unconfirmed_video: true,
            ..AnalogFpvDetector::default()
        };
        let (sig_type, conf) = det.detect_sync_pulses(&iq, sr);
        assert_eq!(sig_type, SignalType::AnalogVideoPal);
        assert_eq!(conf, 0.6, "unconfirmed comb should demote once opted in");
        assert!(
            det.min_confidence > conf,
            "demoted confidence should fail the default floor"
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::synthetic::{SyntheticVideoConfig, TestPattern, generate_iq};
    use crate::vbi::FieldParity;

    fn awgn(iq: &mut [Complex<f32>], sigma: f32, seed: &mut u64) {
        let gauss = |s: &mut u64| -> f32 {
            // Irwin-Hall approximate N(0,1): sum of 12 uniforms − 6.
            let mut acc = 0.0f32;
            for _ in 0..12 {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                acc += (*s >> 11) as f32 / (1u64 << 53) as f32;
            }
            acc - 6.0
        };
        for z in iq.iter_mut() {
            z.re += sigma * gauss(seed);
            z.im += sigma * gauss(seed);
        }
    }

    /// The headline property of [`SpectralIntegrator`]: at σ = 1.2 per
    /// I/Q component against a unit carrier, no single batch detects
    /// the signal (0/4 across four independent noise realizations —
    /// calibrated empirically with margin: single-shot is already 0/4
    /// at σ = 1.2 and integration still detects cleanly at σ = 1.5),
    /// but four noncoherently averaged batches classify it at full 0.8
    /// confidence. That is the several-dB sensitivity gain the
    /// integrator exists for.
    #[test]
    fn integration_detects_where_single_batches_cannot() {
        let sample_rate = 15_360_000u32;
        let cfg = SyntheticVideoConfig {
            sample_rate,
            is_pal: false,
            deviation_hz: 5e6,
            pattern: TestPattern::Bars,
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        };
        let clean = generate_iq(&cfg, 2, 0.0);
        let sigma = 1.2f32;
        let det = AnalogFpvDetector::default();

        // Single-shot: every one of the four noisy batches must fail —
        // this pins the "too weak for one batch" premise.
        for b in 0..4u64 {
            let mut iq = clean.clone();
            let mut seed = 0x1111u64.wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
            awgn(&mut iq, sigma, &mut seed);
            let (st, conf) = det.detect_sync_pulses(&iq, sample_rate);
            assert!(
                !st.is_analog_video(),
                "batch {b}: single-shot unexpectedly detected {st:?} (conf {conf}) — \
                 recalibrate sigma upward"
            );
        }

        // Integrated: the same four batches through one accumulator.
        let mut integ = SpectralIntegrator::new(4);
        let mut last = (SignalType::Unknown, 0.0f32);
        for b in 0..4u64 {
            let mut iq = clean.clone();
            let mut seed = 0x1111u64.wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
            awgn(&mut iq, sigma, &mut seed);
            last = det.detect_sync_pulses_integrated(&iq, sample_rate, 5_800_000_000, &mut integ);
        }
        assert!(
            last.0.is_analog_video() && last.1 >= 0.7,
            "integrated detection should succeed, got {:?} conf {:.2}",
            last.0,
            last.1
        );
    }

    #[test]
    fn integrator_resets_bucket_on_length_change_and_caps_bucket_count() {
        let mut integ = SpectralIntegrator::new(4);
        // Average of two batches at the same key.
        let a = vec![1.0f32; 64];
        let b = vec![3.0f32; 64];
        integ.accumulate(5_800_000_000, &a);
        let avg = integ.accumulate(5_800_000_000, &b).to_vec();
        assert!(
            (avg[0] - 2.0).abs() < 1e-6,
            "cumulative mean, got {}",
            avg[0]
        );

        // Length change restarts the bucket rather than mixing shapes.
        let c = vec![7.0f32; 32];
        let after = integ.accumulate(5_800_000_000, &c).to_vec();
        assert_eq!(after.len(), 32);
        assert!((after[0] - 7.0).abs() < 1e-6);

        // Bucket cap: hammering more distinct frequencies than
        // max_buckets must evict, not grow unboundedly.
        let tiny = vec![0.0f32; 8];
        for k in 0..300i64 {
            integ.accumulate(k * 1_000_000, &tiny);
        }
        assert!(integ.buckets.len() <= integ.max_buckets);
    }
}

#[cfg(test)]
mod dead_zone_tests {
    use super::*;
    use crate::synthetic::{SyntheticVideoConfig, TestPattern, generate_iq};
    use crate::vbi::FieldParity;

    /// A strong, clean carrier must not be missed because of where the
    /// record length happens to fall.
    ///
    /// It used to be. `get_peak_energy` searched ±`search_range` padded
    /// bins around each line rate, and PAL and NTSC are only 109 Hz
    /// apart — so whenever that radius reached the other standard\'s bin,
    /// both reads returned the same peak. `pal_energy > ntsc_energy * 1.2`
    /// and its mirror are then both false, and an unmissable signal was
    /// discarded as `Unknown`. Whether the windows collided depended on
    /// where `demod_len` fell relative to the next power of two, so the
    /// misses arrived in bands: at 61.44 MSPS, 13 of 30 block-multiple
    /// lengths failed, nine of them consecutively.
    ///
    /// These three lengths are the ones measured to collide at 25 MSPS,
    /// covering both ways it happened — a padded separation of one bin
    /// with radius 1, and of two bins with radius 2:
    ///
    /// ```text
    ///   len     padded sep   radius   reaches other bin
    ///   196608      1          1            yes
    ///   262144      1          1            yes
    ///   327680      2          2            yes
    /// ```
    ///
    /// Kept to one sample rate and three lengths on purpose: this runs in
    /// a debug build, where a sweep over every rate meant million-sample
    /// FFTs and minutes of wall clock.
    #[test]
    fn colliding_line_rate_windows_do_not_discard_a_strong_signal() {
        let sample_rate = 25_000_000u32;
        let cfg = SyntheticVideoConfig {
            sample_rate,
            is_pal: false,
            deviation_hz: 5e6,
            pattern: TestPattern::Bars,
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        };
        let full = generate_iq(&cfg, 8, 0.0);
        let det = AnalogFpvDetector::default();

        for len in [196_608usize, 262_144, 327_680] {
            assert!(len <= full.len(), "test signal too short for len={len}");
            let found = det.detect_from_iq(&full[..len], 5_800_000_000, sample_rate);
            assert!(
                !found.is_empty(),
                "a clean NTSC carrier at band centre was discarded at len={len} \
                 ({:.2} ms) — the PAL and NTSC peak-search windows are \
                 overlapping again, so both energies read identical and \
                 neither can clear the other by the required 1.2x",
                len as f64 / sample_rate as f64 * 1e3
            );
            // Reporting the opposite standard would be worse than
            // reporting nothing: the viewer would decode with the wrong
            // line geometry.
            assert!(
                !found
                    .iter()
                    .any(|d| d.signal_type == SignalType::AnalogVideoPal),
                "NTSC signal reported as PAL at len={len}"
            );
        }
    }

    /// A short clean PAL record must come back as PAL — the case that
    /// once came back as NTSC, then as nothing, for three stacked
    /// reasons, each fixed separately:
    ///
    /// 1. The time-domain disambiguator admitted the ~32 µs HALF-line
    ///    spacing of VBI equalizing/broad pulses (its interval floor was
    ///    30 µs), and took a bare median over whatever survived. On this
    ///    record an off-centre probe's smear medianed to 627 samples =
    ///    15,943 Hz — inside the plausible-rate gate and closer to NTSC.
    ///    Intervals are now bounded to full lines (55–75 µs), two thirds
    ///    of them must agree within ±1% before the classifier will
    ///    answer, and the rate comes from the consensus cluster's mean
    ///    rather than a whole-sample median (at a 10 MHz probe rate the
    ///    two standards are 4.4 samples apart, so integer quantisation
    ///    alone is a ±25 Hz error against a 109 Hz decision).
    /// 2. With a ±5 MHz deviation, sync tips swing to the passband edge
    ///    of the carrier-centred probe — the strongest one — whose
    ///    disambiguation therefore fails consensus and returns
    ///    `AnalogVideoUnknown` at 0.6, while its off-centre siblings
    ///    read the tips cleanly and answer PAL at 0.8.
    /// 3. Sweep clustering let the strongest member decide the
    ///    classification, so that Unknown shadowed the siblings' PAL and
    ///    the cluster died at the min-confidence filter. Classification
    ///    now follows the most confident member; frequency still follows
    ///    the strongest.
    #[test]
    fn short_pal_record_resolves_as_pal_not_ntsc() {
        let sample_rate = 40_000_000u32;
        let cfg = SyntheticVideoConfig {
            sample_rate,
            is_pal: true,
            deviation_hz: 5e6,
            pattern: TestPattern::Bars,
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        };
        let full = generate_iq(&cfg, 6, 0.0);
        let det = AnalogFpvDetector::default();
        let found = det.detect_from_iq(&full[..131_072], 5_800_000_000, sample_rate);
        assert!(
            !found
                .iter()
                .any(|d| d.signal_type == SignalType::AnalogVideoNtsc),
            "3.3 ms of clean PAL classified as NTSC: {found:?}"
        );
        assert!(
            found
                .iter()
                .any(|d| d.signal_type == SignalType::AnalogVideoPal),
            "3.3 ms of clean PAL not resolved as PAL: {found:?}"
        );
        let hit = found
            .iter()
            .find(|d| d.signal_type == SignalType::AnalogVideoPal)
            .unwrap();
        assert!(
            (hit.frequency_hz as f64 - 5_800e6).abs() < 3e6,
            "PAL hit localised to {} Hz, expected ~5.8 GHz",
            hit.frequency_hz
        );
    }
}
