//! Sync-tip / blanking level and FM-deviation estimation.
//!
//! Deliberately independent of sync lock: [`estimate_fm_deviation`] only
//! needs order statistics (percentiles) and a width-gated pulse scan, so
//! it works before the reconstructor or detector has found a single
//! H-sync pulse. That matters because the reconstructor's own vsync
//! threshold is *derived from* the assumed deviation
//! (`-0.3 · 2π·dev/fs`) — an estimator that itself required sync lock
//! would be circular.

/// Fraction of full FM-deviation swing that separates sync-tip level
/// from blanking level. This is the crate-wide convention already
/// implicit in [`crate::video::FrameReconstructor`]'s AGC (which scales
/// active video assuming a `0.4 · radians_per_volt` sync-to-blank
/// swing) and its vsync threshold (`-0.3 · radians_per_volt`, inside
/// that same swing). Using the identical constant here means a deviation
/// estimate derived from a measured swing is self-consistent with every
/// other threshold in the crate by construction — once
/// `set_fm_deviation(estimate)` is applied, sync tips land at exactly
/// `-SYNC_TO_BLANK_FRACTION · radians_per_volt`.
pub const SYNC_TO_BLANK_FRACTION: f32 = 0.4;

/// Minimum capture length required to estimate levels/deviation: ~5 ms,
/// enough for a comfortable multiple of [`MIN_PULSES`] (a busy field
/// delivers ~1 H-sync-class pulse every line period, so even 5 ms
/// yields on the order of 60-80 candidates at typical video line
/// rates). The real protection against a bad estimate is the pulse
/// count / bimodality gates further down, not this floor — it only
/// needs to rule out pathologically short inputs. Deliberately kept
/// under ~9 ms: that's the shortest duration at which
/// `detect_sync_pulses_with_cepstrum`'s FFT bins can still fail to
/// resolve PAL from NTSC (see its `AnalogVideoUnknown` path), and the
/// VBI-confirm stage needs to run on exactly those slices to promote a
/// standard-ambiguous hit — a higher floor here would make that
/// promotion unreachable by construction.
const MIN_ESTIMATE_SAMPLES_DIVISOR: u32 = 200; // sample_rate / 200 = 5 ms

/// Minimum number of surviving sync-like pulses required before trusting
/// the estimate. Real video delivers hundreds of pulses in 40 ms; a
/// handful surviving is far more consistent with noise coincidentally
/// dipping below threshold than with a genuine pulse train.
const MIN_PULSES: usize = 50;

/// Absolute sync-tip-to-blanking level pair, in the same
/// radians-per-sample units [`crate::demod::fm_demod`] outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyncLevels {
    pub sync_tip: f32,
    pub blanking: f32,
}

impl SyncLevels {
    /// The measured swing, always positive for a valid estimate
    /// (`blanking` sits above `sync_tip` in the FM-demod convention
    /// used throughout this crate — see [`crate::video`]).
    pub fn swing(&self) -> f32 {
        self.blanking - self.sync_tip
    }
}

/// Result of [`estimate_fm_deviation`]: the recovered peak FM
/// deviation, the absolute levels it was derived from, and how many
/// pulses contributed (useful telemetry / confidence signal for
/// callers).
#[derive(Debug, Clone, Copy)]
pub struct DeviationEstimate {
    pub deviation_hz: f32,
    pub levels: SyncLevels,
    pub n_pulses: usize,
}

#[inline]
pub(crate) fn moving_average(data: &[f32], win: usize) -> Vec<f32> {
    let mut out = Vec::new();
    moving_average_into(data, win, &mut out);
    out
}

/// [`moving_average`] into a caller-owned buffer, for per-row hot loops
/// (the reconstructor's anti-alias pass runs once per rendered line).
#[inline]
pub(crate) fn moving_average_into(data: &[f32], win: usize, out: &mut Vec<f32>) {
    out.clear();
    if win <= 1 || data.len() < win {
        out.extend_from_slice(data);
        return;
    }
    out.reserve(data.len());
    let mut acc = 0.0f32;
    for (i, &v) in data.iter().enumerate() {
        acc += v;
        if i >= win {
            acc -= data[i - win];
        }
        let n = (i + 1).min(win) as f32;
        out.push(acc / n);
    }
}

/// Median of a slice via `select_nth_unstable` (O(n), no full sort).
/// Returns 0.0 for an empty slice — callers only call this on
/// already-length-checked data.
pub(crate) fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    values[mid]
}

/// Robust sync-detection threshold for a smoothed demod slice:
/// `p2 + 0.25·(p50 − p2)`, with the percentiles taken on a
/// 16×-decimated copy for speed — the same construction
/// [`estimate_fm_deviation`] uses (see its algorithm notes). The 2nd
/// percentile is a brightness/DC-invariant stand-in for "pure sync
/// tip" that a handful of FM-click outliers cannot drag around, unlike
/// a global minimum. Returns `None` when the slice is too short or has
/// no downward spread (`p50 <= p2`) — i.e. nothing sync-like to find.
pub(crate) fn robust_sync_threshold(smoothed: &[f32]) -> Option<f32> {
    let mut decimated: Vec<f32> = smoothed.iter().step_by(16).copied().collect();
    if decimated.len() < 16 {
        return None;
    }
    // Both order statistics come from the same buffer: select_nth
    // doesn't need sorted input, so the p2 partition can't corrupt the
    // later median selection — no defensive clone needed.
    let p2_idx = (((decimated.len() as f32) * 0.02) as usize).min(decimated.len() - 1);
    decimated.select_nth_unstable_by(p2_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let p2 = decimated[p2_idx];
    let p50 = median(&mut decimated);
    if !(p2.is_finite() && p50.is_finite()) || p50 <= p2 {
        return None;
    }
    Some(p2 + 0.25 * (p50 - p2))
}

/// Median absolute deviation around `center`.
fn mad(values: &[f32], center: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut abs_devs: Vec<f32> = values.iter().map(|v| (v - center).abs()).collect();
    median(&mut abs_devs)
}

/// Envelope-based CNR estimate for a constant-envelope (FM) capture,
/// in dB. FM carries no amplitude information, so the envelope of the
/// (band-limited) signal is carrier + noise only: for a Rice-
/// distributed envelope at moderate-to-high CNR, `mean(|z|)² /
/// (2·var(|z|))` recovers `A²/(2σ²)` — the CNR in the filtered
/// bandwidth. Accurate to ~1–2 dB above ≈5 dB CNR; below that the Rice
/// small-signal regime biases it optimistic, which is fine for its
/// job: a cheap link-quality meter to drive adaptive processing
/// (integration depth, denoise blending) and telemetry. Returns `None`
/// on degenerate input (too short, zero/non-finite variance).
pub fn estimate_cnr_db(iq: &[num_complex::Complex<f32>]) -> Option<f32> {
    if iq.len() < 256 {
        return None;
    }
    // Single pass, one sqrt per sample; f64 accumulators keep the
    // sum-of-squares variance form safe from cancellation even at high
    // CNR (envelope nearly constant).
    let n = iq.len() as f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for z in iq {
        let m = z.norm() as f64;
        sum += m;
        sum_sq += m * m;
    }
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);
    if !(mean.is_finite() && var.is_finite()) || var <= 0.0 || mean <= 0.0 {
        return None;
    }
    Some((10.0 * (mean * mean / (2.0 * var)).log10()) as f32)
}

/// Estimate absolute sync-tip / blanking levels. A thin wrapper over
/// [`estimate_fm_deviation`] for callers that only need the levels
/// (e.g. a reconstructor fallback path), not the Hz conversion.
pub fn estimate_sync_levels(demod: &[f32], sample_rate: u32) -> Option<SyncLevels> {
    estimate_fm_deviation(demod, sample_rate).map(|d| d.levels)
}

/// Estimate peak FM deviation (Hz) directly from the demodulated
/// waveform, with no sync lock required.
///
/// Algorithm (see module + crate docs for the rationale):
/// 1. Smooth with a ~0.5 µs moving average to suppress FM click noise.
/// 2. Take the 2nd and 50th percentile of the smoothed signal (via a
///    decimated copy, for speed) as robust, brightness-invariant
///    stand-ins for "pure sync tip" and "typical mid-signal level".
/// 3. Threshold at `p2 + 0.25·(p50 − p2)` and scan for below-threshold
///    runs 1.5–32 µs wide (covers equalizing/H-sync through serrated
///    broad pulses; rejects clicks and long dropouts).
/// 4. For each surviving run, take the median of its interior samples
///    as the tip level, and the median of a +1.0…+3.0 µs window after
///    it as the porch/blanking level (this window lands on blanking
///    level for every pulse class, including the brief serration after
///    a broad pulse).
/// 5. Take the median swing (porch − tip) across all pulses, with a
///    3×MAD outlier gate, then require the population to be clearly
///    bimodal (swing > 20×MAD) before trusting it — a flat/noisy
///    signal produces a small, noisy swing that this rejects.
/// 6. Convert via `deviation_hz = swing · fs / (2π · SYNC_TO_BLANK_FRACTION)`.
pub fn estimate_fm_deviation(demod: &[f32], sample_rate: u32) -> Option<DeviationEstimate> {
    if sample_rate == 0 {
        return None;
    }
    let min_len = (sample_rate / MIN_ESTIMATE_SAMPLES_DIVISOR) as usize;
    if demod.len() < min_len || demod.len() < 64 {
        return None;
    }
    if demod.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let fs = sample_rate as f32;
    let ma_win = ((fs * 0.5e-6) as usize).max(1);
    let smoothed = moving_average(demod, ma_win);

    // Percentile threshold from a decimated copy — see
    // `robust_sync_threshold` (shared with the PAL/NTSC time-domain
    // classifiers, which used to threshold on a fragile global
    // minimum instead).
    let threshold = robust_sync_threshold(&smoothed)?;

    let min_width = ((fs * 1.5e-6) as usize).max(1);
    let max_width = (fs * 32e-6) as usize;
    let porch_lo = (fs * 1.0e-6) as usize;
    let porch_hi = (fs * 3.0e-6) as usize;

    let mut tips: Vec<f32> = Vec::new();
    let mut porches: Vec<f32> = Vec::new();
    let n = smoothed.len();
    let mut i = 0usize;
    while i < n {
        if smoothed[i] < threshold {
            let start = i;
            while i < n && smoothed[i] < threshold {
                i += 1;
            }
            let end = i; // exclusive
            let width = end - start;
            if width >= min_width && width <= max_width {
                let mut interior: Vec<f32> = smoothed[start..end].to_vec();
                let tip = median(&mut interior);
                let plo = (end + porch_lo).min(n);
                let phi = (end + porch_hi).min(n);
                if phi > plo {
                    let mut porch_window: Vec<f32> = smoothed[plo..phi].to_vec();
                    let porch = median(&mut porch_window);
                    tips.push(tip);
                    porches.push(porch);
                }
            }
        } else {
            i += 1;
        }
    }

    if tips.len() < MIN_PULSES {
        return None;
    }

    let swings: Vec<f32> = tips
        .iter()
        .zip(porches.iter())
        .map(|(t, p)| p - t)
        .collect();
    let mut swings_sorted = swings.clone();
    let swing_median = median(&mut swings_sorted);
    let swing_mad = mad(&swings, swing_median).max(1e-9);

    // Keep only (tip, porch) pairs whose swing survives the 3xMAD gate.
    let mut kept_tips = Vec::new();
    let mut kept_porches = Vec::new();
    let mut kept_swings = Vec::new();
    for ((t, p), s) in tips.iter().zip(porches.iter()).zip(swings.iter()) {
        if (s - swing_median).abs() <= 3.0 * swing_mad {
            kept_tips.push(*t);
            kept_porches.push(*p);
            kept_swings.push(*s);
        }
    }
    if kept_tips.len() < MIN_PULSES {
        return None;
    }

    let mut swing_final_sorted = kept_swings.clone();
    let swing = median(&mut swing_final_sorted);
    let swing_final_mad = mad(&kept_swings, swing).max(1e-12);

    // Bimodality sanity: tip and porch populations must be clearly
    // separated relative to their own scatter, or this is noise that
    // happened to straddle the percentile threshold, not real sync
    // structure.
    if swing <= 0.0 || swing <= 20.0 * swing_final_mad {
        return None;
    }

    let mut tip_sorted = kept_tips.clone();
    let sync_tip = median(&mut tip_sorted);
    let mut porch_sorted = kept_porches.clone();
    let blanking = median(&mut porch_sorted);

    let radians_per_volt = swing / SYNC_TO_BLANK_FRACTION;
    let deviation_hz = radians_per_volt * fs / (2.0 * std::f32::consts::PI);

    if !(500_000.0..=25_000_000.0).contains(&deviation_hz) {
        return None;
    }

    Some(DeviationEstimate {
        deviation_hz,
        levels: SyncLevels { sync_tip, blanking },
        n_pulses: kept_tips.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_short_input_returns_none() {
        assert!(estimate_fm_deviation(&[], 15_360_000).is_none());
        assert!(estimate_fm_deviation(&[0.0; 100], 15_360_000).is_none());
    }

    #[test]
    fn zero_sample_rate_returns_none() {
        assert!(estimate_fm_deviation(&[0.0; 100_000], 0).is_none());
    }

    #[test]
    fn non_finite_input_returns_none() {
        let mut data = vec![0.0f32; 100_000];
        data[500] = f32::NAN;
        assert!(estimate_fm_deviation(&data, 15_360_000).is_none());
    }

    #[test]
    fn flat_signal_returns_none() {
        let data = vec![0.1f32; 400_000];
        assert!(estimate_fm_deviation(&data, 15_360_000).is_none());
    }

    #[test]
    fn pure_noise_returns_none() {
        let sr = 15_360_000u32;
        let n = 400_000;
        let mut seed = 12345u64;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        assert!(estimate_fm_deviation(&data, sr).is_none());
    }

    use crate::synthetic::{SyntheticVideoConfig, TestPattern, generate_fields};
    use crate::vbi::FieldParity;

    fn synth_config(deviation_hz: f32, sample_rate: u32) -> SyntheticVideoConfig {
        SyntheticVideoConfig {
            sample_rate,
            is_pal: false,
            deviation_hz,
            pattern: TestPattern::Bars,
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        }
    }

    /// Deviation/sample-rate pairs kept well inside the FM Nyquist
    /// margin (peak baseband excursion, white at +100 IRE = 1.0×
    /// radians_per_volt, comfortably under ~1.2 rad/sample) — see
    /// `synthetic::generate_iq`'s doc for why. `generate_fields` itself
    /// has no such constraint (it's not modulated), but the pairs are
    /// still chosen to resemble physically plausible captures.
    const ACCURACY_CASES: &[(f32, u32)] = &[
        (1.0e6, 10_000_000),
        (1.0e6, 15_360_000),
        (2.5e6, 15_360_000),
        (1.0e6, 20_000_000),
        (3.0e6, 20_000_000),
        (3.0e6, 50_000_000),
        (8.0e6, 50_000_000),
        (5.0e6, 100_000_000),
        (17.0e6, 100_000_000),
    ];

    #[test]
    fn estimator_recovers_deviation_within_5_percent() {
        for &(dev, fs) in ACCURACY_CASES {
            let cfg = synth_config(dev, fs);
            let data = generate_fields(&cfg, 3);
            let est = estimate_fm_deviation(&data, fs)
                .unwrap_or_else(|| panic!("dev={dev} fs={fs}: expected an estimate"));
            let err = (est.deviation_hz - dev).abs() / dev;
            assert!(
                err < 0.05,
                "dev={dev} fs={fs}: estimated {} ({}% error)",
                est.deviation_hz,
                err * 100.0
            );
            assert!(est.n_pulses >= MIN_PULSES);
        }
    }

    #[test]
    fn estimator_is_invariant_to_dc_offset() {
        let mut cfg = synth_config(5.0e6, 15_360_000);
        cfg.dc_offset = 0.6;
        let data = generate_fields(&cfg, 3);
        let est = estimate_fm_deviation(&data, cfg.sample_rate).expect("expected an estimate");
        let err = (est.deviation_hz - cfg.deviation_hz).abs() / cfg.deviation_hz;
        assert!(
            err < 0.05,
            "estimated {} with dc_offset applied",
            est.deviation_hz
        );
    }

    #[test]
    fn estimator_tolerates_realistic_noise() {
        let mut cfg = synth_config(5.0e6, 15_360_000);
        // ~15% of the sync-to-blank swing.
        let radians_per_volt =
            2.0 * std::f32::consts::PI * cfg.deviation_hz / cfg.sample_rate as f32;
        cfg.noise_sigma = 0.15 * SYNC_TO_BLANK_FRACTION * radians_per_volt;
        let data = generate_fields(&cfg, 4);
        let est = estimate_fm_deviation(&data, cfg.sample_rate)
            .expect("expected an estimate under noise");
        let err = (est.deviation_hz - cfg.deviation_hz).abs() / cfg.deviation_hz;
        assert!(
            err < 0.10,
            "estimated {} under noise ({}% error)",
            est.deviation_hz,
            err * 100.0
        );
    }

    #[test]
    fn envelope_cnr_estimate_tracks_true_cnr() {
        // Unit carrier + complex AWGN at known per-component sigma:
        // true CNR = 1/(2·sigma²).
        let mut seed = 0xFEEDu64;
        let gauss = |s: &mut u64| -> f32 {
            let mut acc = 0.0f32;
            for _ in 0..12 {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                acc += (*s >> 11) as f32 / (1u64 << 53) as f32;
            }
            acc - 6.0
        };
        for &(sigma, true_db) in &[(0.1f32, 16.99f32), (0.2, 10.97), (0.35, 6.1)] {
            let iq: Vec<num_complex::Complex<f32>> = (0..200_000)
                .map(|i| {
                    let phase = 0.37 * i as f32; // arbitrary FM-ish rotation
                    num_complex::Complex::from_polar(1.0, phase)
                        + num_complex::Complex::new(
                            sigma * gauss(&mut seed),
                            sigma * gauss(&mut seed),
                        )
                })
                .collect();
            let est = estimate_cnr_db(&iq).expect("estimate expected");
            assert!(
                (est - true_db).abs() < 1.5,
                "sigma={sigma}: estimated {est:.1} dB vs true {true_db:.1} dB"
            );
        }
        assert!(estimate_cnr_db(&[]).is_none());
    }

    #[test]
    fn estimate_sync_levels_matches_deviation_estimate_levels() {
        let cfg = synth_config(5.0e6, 15_360_000);
        let data = generate_fields(&cfg, 3);
        let levels = estimate_sync_levels(&data, cfg.sample_rate).expect("expected levels");
        let full = estimate_fm_deviation(&data, cfg.sample_rate).expect("expected estimate");
        assert_eq!(levels, full.levels);
        assert!(levels.swing() > 0.0);
    }
}
