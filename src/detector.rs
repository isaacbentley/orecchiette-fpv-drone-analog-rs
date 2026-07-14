use crate::types::{DetectionResult, SignalType};
use num_complex::Complex;
use rustfft::{FftPlanner, num_complex::Complex as FftComplex};
use std::cell::RefCell;
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
/// Reads the FM-demodulated baseband, walks it looking for sync tips
/// (local minima below 30 % of the global minimum), computes the
/// median inter-tip interval, and converts to a line frequency. PAL
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
    let slice = &demod[..scan_len];
    let global_min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    if !global_min.is_finite() || global_min >= 0.0 {
        return None;
    }
    let threshold = global_min * 0.3;
    // Sync gap bounds: 30–100 µs covers both NTSC (63.5 µs) and PAL
    // (64.0 µs) with comfortable margin.
    let min_gap = (sample_rate as f32 * 30e-6) as usize;
    let max_gap = (sample_rate as f32 * 100e-6) as usize;
    let mut sync_positions: Vec<usize> = Vec::with_capacity(128);
    let mut i = 0;
    while i < scan_len {
        if slice[i] < threshold {
            let mut local_min_idx = i;
            let mut local_min_val = slice[i];
            while i < scan_len && slice[i] < threshold {
                if slice[i] < local_min_val {
                    local_min_val = slice[i];
                    local_min_idx = i;
                }
                i += 1;
            }
            sync_positions.push(local_min_idx);
            i = local_min_idx + min_gap;
        } else {
            i += 1;
        }
    }
    // Need at least 8 inter-tip intervals to land the median on a
    // PAL/NTSC decision. Anything less and crystal jitter dominates.
    let mut intervals: Vec<usize> = Vec::with_capacity(sync_positions.len().saturating_sub(1));
    for w in sync_positions.windows(2) {
        let gap = w[1] - w[0];
        if gap >= min_gap && gap <= max_gap {
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
    let line_hz = sample_rate as f64 / median;
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

    /// Wrapper that calls `detect_sync_pulses_inner` and then applies
    /// the cepstrum structural gate.  If the cepstrum check fails,
    /// the classification is downgraded to `Unknown`.
    pub fn detect_sync_pulses(
        &self,
        iq_data: &[Complex<f32>],
        sample_rate: u32,
    ) -> (SignalType, f32) {
        self.detect_sync_pulses_with_cepstrum(iq_data, sample_rate)
    }

    fn detect_sync_pulses_with_cepstrum(
        &self,
        iq_data: &[Complex<f32>],
        sample_rate: u32,
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
        // Use the single shared implementation in `demod::fm_demod` so the
        // discriminator never diverges between the detection and decode paths.
        let demod = crate::demod::fm_demod(iq_data);
        let demod_len = demod.len();
        let avg_demod = demod.iter().sum::<f32>() / demod_len as f32;
        let mut var = 0.0f32;
        for &d in &demod {
            let diff = d - avg_demod;
            var += diff * diff;
        }
        var /= demod_len as f32;
        if var < 1e-6 {
            return (SignalType::Unknown, 0.0);
        }

        let fft_len = demod_len;
        let fft = self.planner.borrow_mut().plan_fft_forward(fft_len);
        let mut buffer: Vec<FftComplex<f32>> = vec![FftComplex { re: 0.0, im: 0.0 }; fft_len];

        for i in 0..fft_len {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_len - 1) as f32).cos());
            buffer[i].re = (demod[i] - avg_demod) * window;
        }

        fft.process(&mut buffer);

        let bin_hz = sample_rate as f32 / fft_len as f32;
        let bin_pal = (15625.0 / bin_hz).round() as usize;
        let bin_ntsc = (15734.0 / bin_hz).round() as usize;

        let search_range = 1;

        let pal_energy = self.get_peak_energy(&buffer, bin_pal, search_range);
        let ntsc_energy = self.get_peak_energy(&buffer, bin_ntsc, search_range);

        // Floor at bin 1 so the DC bin (nonzero even after mean-subtraction,
        // because the Hann window has nonzero mean) never enters the noise
        // estimate on coarse/short FFTs where round(500/bin_hz) would be 0.
        let noise_start_bin = ((500.0 / bin_hz).round() as usize).max(1);
        let noise_end_bin = fft_len / 2;
        let mut noise_sum = 0.0;
        let mut noise_count = 0;

        if noise_end_bin > noise_start_bin {
            for c in &buffer[noise_start_bin..noise_end_bin] {
                noise_sum += c.norm();
                noise_count += 1;
            }
        }

        let noise_floor = if noise_count > 0 {
            noise_sum / noise_count as f32
        } else {
            1e-6
        };

        let thresh_strong = noise_floor * 5.0;
        let thresh_weak = noise_floor * 2.5;

        const N_HARMONICS: usize = 5;
        const HARMONIC_RATIO: f32 = 0.1;
        const PAL_LINE_HZ: f32 = 15625.0;
        const NTSC_LINE_HZ: f32 = 15734.0;
        let line_bin = bin_ntsc.max(bin_pal);
        let mut pal_harmonics = 0u32;
        let mut ntsc_harmonics = 0u32;
        let max_bin = noise_end_bin;
        if line_bin > 0 {
            let pal_thresh = pal_energy * HARMONIC_RATIO;
            let ntsc_thresh = ntsc_energy * HARMONIC_RATIO;
            for k in 2..=N_HARMONICS {
                let kf = k as f32;
                let hb_pal = (kf * PAL_LINE_HZ / bin_hz).round() as usize;
                let hb_ntsc = (kf * NTSC_LINE_HZ / bin_hz).round() as usize;
                if hb_pal < max_bin {
                    let e = self.get_peak_energy(&buffer, hb_pal, search_range);
                    if e > pal_thresh && e > thresh_weak {
                        pal_harmonics += 1;
                    }
                }
                if hb_ntsc < max_bin {
                    let e = self.get_peak_energy(&buffer, hb_ntsc, search_range);
                    if e > ntsc_thresh && e > thresh_weak {
                        ntsc_harmonics += 1;
                    }
                }
            }
        }
        let collide_harmonics = pal_harmonics.max(ntsc_harmonics);

        let mut sig_type = SignalType::Unknown;
        let mut conf = 0.0;

        let bins_distinct = bin_pal != bin_ntsc;

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
                let time_domain_class = classify_pal_ntsc_time_domain(&demod, sample_rate);
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
            if !self.verify_cepstrum(&buffer, sample_rate, candidate_line_hz) {
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
            && let Some(levels) = crate::levels::estimate_sync_levels(&demod, sample_rate)
        {
            let is_pal_hint = match sig_type {
                SignalType::AnalogVideoPal => Some(true),
                SignalType::AnalogVideoNtsc => Some(false),
                _ => None,
            };
            let evidence =
                crate::vbi::confirm_field_sync(&demod, sample_rate, &levels, is_pal_hint);
            conf =
                apply_vbi_confidence_tier(sig_type, conf, &evidence, self.demote_unconfirmed_video);
        }

        (sig_type, conf)
    }

    fn get_peak_energy(&self, buffer: &[FftComplex<f32>], bin: usize, range: usize) -> f32 {
        let end = (bin + range).min(buffer.len() / 2);
        // Clamp start to end: for a `bin` past Nyquist (only reachable at
        // pathologically low sample rates where the line-rate bin exceeds
        // fft_len/2) `start` could otherwise exceed `end`, panicking the
        // inclusive slice below.
        let start = bin.saturating_sub(range).min(end);
        buffer[start..=end]
            .iter()
            .map(|c| c.norm())
            .fold(0.0f32, f32::max)
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
    fn verify_cepstrum(
        &self,
        fft_buffer: &[FftComplex<f32>],
        sample_rate: u32,
        candidate_line_hz: f32,
    ) -> bool {
        let fft_len = fft_buffer.len();
        if fft_len < 64 {
            return true; // too short for meaningful cepstrum
        }

        // Expected quefrency (in samples) for the line rate.
        let expected_q = sample_rate as f32 / candidate_line_hz;
        let q_idx = expected_q.round() as usize;
        if q_idx < 2 || q_idx >= fft_len / 2 {
            return true; // can't measure at this resolution
        }

        // ---- Step 1: log-power spectrum ----
        // Written as a single pass over the complex FFT buffer.
        // The inner loop is branchless: `re*re + im*im + eps` then
        // `ln()`.  LLVM SLP-vectorises the multiply-add to 4-wide
        // NEON/SSE; the `ln` call is scalar but dominates only at
        // very large FFT sizes where the IFFT cost already exceeds
        // it.
        const EPSILON: f32 = 1e-12;
        let mut log_power: Vec<FftComplex<f32>> = Vec::with_capacity(fft_len);
        for c in fft_buffer {
            let power = c.re * c.re + c.im * c.im + EPSILON;
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
        // Only need the first half (positive quefrencies).
        // Written as a branchless multiply — auto-vectorises.
        let half = fft_len / 2;
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
    /// ## Known follow-ups
    ///
    /// The implementation below runs the FIR for *every* input
    /// sample even though only every `decimation_factor`-th output is
    /// kept; a polyphase decimating FIR would only compute the
    /// retained outputs for a ~5× per-probe speed-up. The
    /// `StreamingDDC` is also re-constructed per probe per packet
    /// (which re-runs the tap-design `sin_cos` loop), so caching the
    /// designed taps in `AnalogFpvDetector` is another easy win.
    /// Neither matters yet at our current packet sizes; both are
    /// tracked under the multi-mode wire-up item.
    fn ddc_and_decimate(
        iq_data: &[Complex<f32>],
        sample_rate: u32,
        freq_offset: f32,
        target_rate: u32,
    ) -> Vec<Complex<f32>> {
        let decimation_factor = Self::decimation_factor(sample_rate, target_rate);
        let cutoff_hz = (target_rate as f32) / 3.0;
        let mut ddc = crate::ddc::StreamingDDC::new(freq_offset, sample_rate, cutoff_hz);
        ddc.process_decimated(iq_data, decimation_factor)
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

impl FpvDetector for AnalogFpvDetector {
    fn detect_from_iq(
        &self,
        iq_data: &[Complex<f32>],
        center_freq: u64,
        sample_rate: u32,
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
            let (sig_type, conf) = self.detect_sync_pulses(iq_data, sample_rate);
            if sig_type != SignalType::Unknown {
                final_results.push(DetectionResult {
                    channel: None,
                    frequency_hz: center_freq,
                    confidence: conf,
                    rssi_dbm: -50.0,
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
                let (sig_type, conf) = self.detect_sync_pulses(iq_data, sample_rate);
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
                    let isolated_iq = Self::ddc_and_decimate(
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
                if *energy <= energy_thresh {
                    continue;
                }
                let (sig_type, conf) = self.detect_sync_pulses(isolated_iq, *isolated_rate);
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
                    // Same cluster — update the strongest-member fields
                    // only; the anchor (last.0) stays fixed.
                    if hit.1 > last.2 {
                        last.1 = hit.0;
                        last.2 = hit.1;
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
        let mut deduped: Vec<DetectionResult> = Vec::new();
        for r in final_results {
            if let Some(last) = deduped.last_mut()
                && (r.frequency_hz as f64 - last.frequency_hz as f64).abs() < 25e6
            {
                if r.confidence > last.confidence
                    || (r.confidence == last.confidence && r.rssi_dbm > last.rssi_dbm)
                {
                    *last = r;
                }
                continue;
            }
            deduped.push(r);
        }

        deduped.retain(|r| {
            r.bandwidth_hz >= self.min_bandwidth
                && r.bandwidth_hz <= self.max_bandwidth
                && r.confidence >= self.min_confidence
        });

        deduped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let cpu_out =
            AnalogFpvDetector::ddc_and_decimate(&iq, sample_rate, offset_hz as f32, target_rate);
        let gpu_out = gpu.sweep(&iq, sample_rate, &[offset_hz], decimation_factor, cutoff_hz);
        let gpu_probe = &gpu_out[0];

        assert_eq!(
            cpu_out.len(),
            gpu_probe.len(),
            "GPU/CPU decimated length mismatch"
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
            "GPU DDC output diverges from CPU by {:.4}% RMS (tolerance 1%)",
            rel_rms * 100.0
        );
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
        use rustfft::num_complex::Complex as FftC;

        let sr = 10_000_000u32;
        let line_hz = 15625.0f32;
        let fft_len = 8192;
        let det = AnalogFpvDetector::new(-20.0);

        // Build a synthetic FFT buffer with a harmonic comb at the
        // line rate — simulates what a real pulse train would produce.
        let bin_hz = sr as f32 / fft_len as f32;
        let mut fft_buf = vec![FftC { re: 0.001, im: 0.0 }; fft_len];
        for k in 1..=10 {
            let bin = (k as f32 * line_hz / bin_hz).round() as usize;
            if bin < fft_len / 2 {
                // Strong peak at this harmonic.
                fft_buf[bin] = FftC { re: 100.0, im: 0.0 };
                // Mirror.
                fft_buf[fft_len - bin] = FftC { re: 100.0, im: 0.0 };
            }
        }

        assert!(
            det.verify_cepstrum(&fft_buf, sr, line_hz),
            "harmonic comb should pass cepstrum check"
        );

        // Flat spectrum — no periodic structure.
        let flat_buf = vec![FftC { re: 1.0, im: 0.0 }; fft_len];
        assert!(
            !det.verify_cepstrum(&flat_buf, sr, line_hz),
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
