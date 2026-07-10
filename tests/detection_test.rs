use num_complex::Complex;
use orecchiette_fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
use orecchiette_fpv_drone_analog_rs::types::SignalType;
use std::f32::consts::PI;

// ── Helper: generate FM-modulated IQ from a baseband sync signal ──────────

/// Create IQ data containing an FM-modulated video sync signal at the given
/// line rate (15625 Hz for PAL, 15734 Hz for NTSC). The baseband is a simple
/// square wave at the sync rate, FM-modulated with the given deviation.
fn make_fm_sync_iq(
    sample_rate: u32,
    n_samples: usize,
    line_rate: f32,
    deviation: f32,
    carrier_offset: f32,
) -> Vec<Complex<f32>> {
    let mut iq = Vec::with_capacity(n_samples);
    let mut phase = 0.0f32;
    let offset_adv = 2.0 * PI * carrier_offset / sample_rate as f32;

    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        // Square wave at line rate: sync tip (-0.4) vs blanking (0.0) with
        // active video (0.5). Duty cycle ~7% sync, ~93% active.
        let sync_phase = (t * line_rate).fract();
        let baseband = if sync_phase < 0.07 { -0.4 } else { 0.5 };

        let inst_freq = baseband * deviation;
        let phase_adv = 2.0 * PI * inst_freq / sample_rate as f32;
        phase += phase_adv + offset_adv;
        if phase > PI {
            phase -= 2.0 * PI;
        }
        if phase < -PI {
            phase += 2.0 * PI;
        }

        iq.push(Complex::new(phase.cos(), phase.sin()));
    }
    iq
}

// ── Unit tests: detect_sync_pulses (narrowband, already at baseband) ──────

#[test]
fn detect_sync_pulses_pal_narrowband() {
    let detector = AnalogFpvDetector::default();
    let sample_rate = 1_000_000u32;
    let n = 65536;

    let iq = make_fm_sync_iq(sample_rate, n, 15625.0, 75_000.0, 0.0);
    let (sig_type, conf) = detector.detect_sync_pulses(&iq, sample_rate);

    assert_ne!(
        sig_type,
        SignalType::Unknown,
        "Should detect PAL sync at 1 MSPS"
    );
    assert_eq!(sig_type, SignalType::AnalogVideoPal);
    assert!(
        conf >= 0.5,
        "Confidence should be at least 0.5, got {}",
        conf
    );
}

#[test]
fn detect_sync_pulses_ntsc_narrowband() {
    let detector = AnalogFpvDetector::default();
    let sample_rate = 1_000_000u32;
    let n = 65536;

    let iq = make_fm_sync_iq(sample_rate, n, 15734.0, 75_000.0, 0.0);
    let (sig_type, conf) = detector.detect_sync_pulses(&iq, sample_rate);

    assert_ne!(
        sig_type,
        SignalType::Unknown,
        "Should detect NTSC sync at 1 MSPS"
    );
    assert_eq!(sig_type, SignalType::AnalogVideoNtsc);
    assert!(
        conf >= 0.5,
        "Confidence should be at least 0.5, got {}",
        conf
    );
}

#[test]
fn detect_sync_pulses_noise_only_returns_unknown() {
    use rand::RngExt;
    let detector = AnalogFpvDetector::default();
    let sample_rate = 1_000_000u32;
    let n = 16384;

    let mut rng = rand::rng();
    let iq: Vec<Complex<f32>> = (0..n)
        .map(|_| Complex::new(rng.random_range(-0.1..0.1), rng.random_range(-0.1..0.1)))
        .collect();

    let (sig_type, _conf) = detector.detect_sync_pulses(&iq, sample_rate);
    assert_eq!(
        sig_type,
        SignalType::Unknown,
        "Noise should not trigger detection"
    );
}

// ── Integration tests: full detect_from_iq pipeline ───────────────────────

#[test]
fn detect_from_iq_single_ntsc_narrowband() {
    // Narrowband fast path: sample_rate < min_bandwidth (10 MHz)
    let detector = AnalogFpvDetector::default();
    let sample_rate = 1_000_000u32;
    let n = 65536;

    let iq = make_fm_sync_iq(sample_rate, n, 15734.0, 75_000.0, 0.0);
    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);

    assert!(
        !results.is_empty(),
        "Should detect NTSC via narrowband path"
    );
    assert_eq!(results[0].signal_type, SignalType::AnalogVideoNtsc);
    assert_eq!(results[0].frequency_hz, 5_800_000_000);
}

#[test]
fn detect_from_iq_single_pal_narrowband() {
    let detector = AnalogFpvDetector::default();
    let sample_rate = 1_000_000u32;
    let n = 65536;

    let iq = make_fm_sync_iq(sample_rate, n, 15625.0, 75_000.0, 0.0);
    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);

    assert!(!results.is_empty(), "Should detect PAL via narrowband path");
    assert_eq!(results[0].signal_type, SignalType::AnalogVideoPal);
}

#[test]
fn detect_from_iq_wideband_sliding_ddc_finds_offset_signal() {
    // Wideband path: 50 MSPS capture with a single video signal at -15 MHz
    let detector = AnalogFpvDetector::default();
    let sample_rate = 50_000_000u32;
    let n = 262_144;
    let offset_hz = -15_000_000.0f32;

    let iq = make_fm_sync_iq(sample_rate, n, 15734.0, 5_000_000.0, offset_hz);
    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);

    assert!(
        !results.is_empty(),
        "Sliding DDC should find signal at -15 MHz offset"
    );
    // Should detect near the offset frequency
    let detected_freq = results[0].frequency_hz;
    let expected_freq = 5_800_000_000u64 - 15_000_000;
    let diff_mhz = (detected_freq as f64 - expected_freq as f64).abs() / 1e6;
    assert!(
        diff_mhz < 10.0,
        "Detected frequency {} should be within 10 MHz of expected {}, diff was {} MHz",
        detected_freq,
        expected_freq,
        diff_mhz
    );
}

#[test]
fn detect_from_iq_wideband_two_signals_found() {
    // Two FM video signals separated by 50 MHz in a 100 MSPS capture.
    // This mirrors the synthetic_fpv test file scenario.
    let detector = AnalogFpvDetector::default();
    let sample_rate = 100_000_000u32;
    let n = 262_144;

    let iq_ntsc = make_fm_sync_iq(sample_rate, n, 15734.0, 5_000_000.0, -25_000_000.0);
    let iq_pal = make_fm_sync_iq(sample_rate, n, 15625.0, 5_000_000.0, 25_000_000.0);

    // Superpose the two signals
    let iq: Vec<Complex<f32>> = iq_ntsc
        .iter()
        .zip(iq_pal.iter())
        .map(|(a, b)| a + b)
        .collect();

    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);

    assert!(
        results.len() >= 2,
        "Should detect at least 2 signals, got {}",
        results.len()
    );

    // Check that we have detections near both carrier frequencies
    let freqs: Vec<u64> = results.iter().map(|r| r.frequency_hz).collect();
    let has_lower = freqs
        .iter()
        .any(|&f| (f as f64 - 5_775_000_000.0).abs() < 15e6);
    let has_upper = freqs
        .iter()
        .any(|&f| (f as f64 - 5_825_000_000.0).abs() < 15e6);

    assert!(
        has_lower,
        "Should detect signal near 5775 MHz, got {:?}",
        freqs
    );
    assert!(
        has_upper,
        "Should detect signal near 5825 MHz, got {:?}",
        freqs
    );
}

#[test]
fn detect_from_iq_empty_signal_returns_nothing() {
    let detector = AnalogFpvDetector::default();
    let sample_rate = 20_000_000u32;
    let n = 262_144;

    // Pure noise at -80 dBm level
    use rand::RngExt;
    let mut rng = rand::rng();
    let iq: Vec<Complex<f32>> = (0..n)
        .map(|_| {
            Complex::new(
                rng.random_range(-0.001..0.001),
                rng.random_range(-0.001..0.001),
            )
        })
        .collect();

    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);
    assert!(
        results.is_empty(),
        "Noise-only input should produce no detections, got {}",
        results.len()
    );
}

#[test]
fn detect_from_iq_cw_tone_not_classified_as_video() {
    // A pure CW tone should NOT be classified as analog video
    let detector = AnalogFpvDetector::default();
    let sample_rate = 20_000_000u32;
    let n = 262_144;

    let iq: Vec<Complex<f32>> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f = 3_000_000.0f32; // 3 MHz tone
            Complex::new((2.0 * PI * f * t).cos(), (2.0 * PI * f * t).sin())
        })
        .collect();

    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);
    // A CW tone has no H-sync structure — it must not classify as
    // any analog-video variant (NTSC, PAL, or AnalogVideoUnknown).
    let video_results: Vec<_> = results
        .iter()
        .filter(|r| r.signal_type.is_analog_video())
        .collect();
    assert!(
        video_results.is_empty(),
        "CW tone should not be classified as analog video, got {} detections",
        video_results.len()
    );
}

// ── Clustering tests ──────────────────────────────────────────────────────

#[test]
fn results_are_clustered_within_25mhz() {
    // Two signals 50 MHz apart should produce exactly 2 results, not more
    let detector = AnalogFpvDetector::default();
    let sample_rate = 100_000_000u32;
    let n = 262_144;

    let iq_a = make_fm_sync_iq(sample_rate, n, 15734.0, 5_000_000.0, -25_000_000.0);
    let iq_b = make_fm_sync_iq(sample_rate, n, 15734.0, 5_000_000.0, 25_000_000.0);

    let iq: Vec<Complex<f32>> = iq_a.iter().zip(iq_b.iter()).map(|(a, b)| a + b).collect();

    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);

    assert_eq!(
        results.len(),
        2,
        "Two signals 50 MHz apart should produce exactly 2 clustered results, got {}",
        results.len()
    );
}

// ── AnalogVideoUnknown variant: bin-collision case ────────────────────────

/// On a wideband sweep, the per-probe FFT operates on ~10000 samples
/// at the 10 MSPS decimated rate, giving a per-bin resolution of
/// ~1 kHz. That's much wider than the 109 Hz spacing between the PAL
/// (15625 Hz) and NTSC (15734 Hz) line rates, so both standards land
/// in the same FFT bin and the detector cannot tell them apart from
/// the H-sync energy alone. The detector falls back to the time-domain
/// sync-tip gap measurement, which successfully resolves the standard.
#[test]
fn wideband_sweep_time_domain_fallback_identifies_ntsc_when_bins_collide() {
    let detector = AnalogFpvDetector::default();
    let sample_rate = 50_000_000u32;
    let n = 262_144;
    // Inject a single FM-modulated video signal with NTSC line rate
    // at -15 MHz offset. The sweep's coarse FFT resolution will lump
    // both NTSC and PAL bins together.
    let iq = make_fm_sync_iq(sample_rate, n, 15734.0, 5_000_000.0, -15_000_000.0);

    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);

    assert!(!results.is_empty(), "Should detect injected analog video");
    let r = &results[0];
    assert_eq!(
        r.signal_type,
        SignalType::AnalogVideoNtsc,
        "Wideband sweep time-domain fallback should correctly identify NTSC, got {:?}",
        r.signal_type,
    );
    assert!(
        r.signal_type.is_analog_video(),
        "AnalogVideoUnknown should still pass the is_analog_video() gate",
    );
}

/// FM-modulate a *sinusoidal* baseband at the requested line rate.
/// This produces a clean tone at the line-rate bin in the FM-
/// demodulator output, but NO harmonics (the modulating waveform
/// has no harmonic content). Distinguishes real video (which is a
/// 7 %-duty-cycle pulse train, rich in harmonics) from a narrowband
/// interferer that happens to land at the H-sync frequency.
fn make_fm_sinusoidal_iq(
    sample_rate: u32,
    n_samples: usize,
    modulation_hz: f32,
    deviation: f32,
    carrier_offset: f32,
) -> Vec<Complex<f32>> {
    let mut iq = Vec::with_capacity(n_samples);
    let mut phase = 0.0f32;
    let offset_adv = 2.0 * PI * carrier_offset / sample_rate as f32;
    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let baseband = (2.0 * PI * modulation_hz * t).sin();
        let inst_freq = baseband * deviation;
        let phase_adv = 2.0 * PI * inst_freq / sample_rate as f32;
        phase += phase_adv + offset_adv;
        if phase > PI {
            phase -= 2.0 * PI;
        }
        if phase < -PI {
            phase += 2.0 * PI;
        }
        iq.push(Complex::new(phase.cos(), phase.sin()));
    }
    iq
}

/// A sinusoidal FM modulation at exactly the NTSC line rate looks
/// like H-sync energy in the line-rate bin but has no harmonics.
/// The harmonic-consistency check in `detect_sync_pulses` should
/// reject this — analog video is a pulse train with rich harmonic
/// structure, a pure tone is not.
///
/// Uses the narrowband fast path (sample_rate < min_bandwidth =
/// 3 MSPS) to exercise the harmonic check in isolation.
#[test]
fn harmonic_check_rejects_pure_tone_at_line_rate() {
    let detector = AnalogFpvDetector::default();
    let sample_rate = 1_000_000u32;
    let n = 65536;

    let iq = make_fm_sinusoidal_iq(sample_rate, n, 15734.0, 50_000.0, 0.0);

    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);
    let video_hits: Vec<_> = results
        .iter()
        .filter(|r| r.signal_type.is_analog_video())
        .collect();
    assert!(
        video_hits.is_empty(),
        "Sinusoidal modulation at the H-sync frequency should NOT \
         classify as analog video — no harmonic structure to support \
         a pulse-train hypothesis. Got: {video_hits:?}",
    );
}

/// Same harmonic-rejection test but on the **wideband sweep** path
/// (50 MSPS, signal at −15 MHz offset). v0.4.27 had to use the
/// narrowband fast path for this because the boxcar decimator
/// aliased the strong on-band signal into adjacent probes and
/// synthesised spurious harmonic content; v0.4.29 swapped the boxcar
/// for the proper StreamingDDC FIR, which closes that gap.
#[test]
fn harmonic_check_rejects_pure_tone_via_wideband_sweep() {
    let detector = AnalogFpvDetector::default();
    let sample_rate = 50_000_000u32;
    let n = 262_144;
    let iq = make_fm_sinusoidal_iq(sample_rate, n, 15734.0, 50_000.0, -15_000_000.0);

    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);
    let video_hits: Vec<_> = results
        .iter()
        .filter(|r| r.signal_type.is_analog_video())
        .collect();
    assert!(
        video_hits.is_empty(),
        "Wideband sweep with a pure FM sinusoid at the H-sync \
         frequency should NOT classify as analog video — the \
         StreamingDDC FIR is supposed to keep adjacent-probe \
         aliasing from synthesising harmonic content. Got: {video_hits:?}",
    );
}

/// `SignalType::is_analog_video()` should accept the three video
/// variants and reject everything else.
#[test]
fn is_analog_video_accepts_only_video_variants() {
    assert!(SignalType::AnalogVideoNtsc.is_analog_video());
    assert!(SignalType::AnalogVideoPal.is_analog_video());
    assert!(SignalType::AnalogVideoUnknown.is_analog_video());
    assert!(!SignalType::Unknown.is_analog_video());
    assert!(!SignalType::WidebandDigital.is_analog_video());
    assert!(!SignalType::NarrowbandInterference.is_analog_video());
}

#[test]
fn test_bandwidth_and_energy_threshold_filtering() {
    let mut detector = AnalogFpvDetector::default();
    detector.min_bandwidth = 5_000_000; // Filter out detections with bandwidth < 5 MHz

    let sample_rate = 1_000_000u32; // narrowband, bandwidth = 1 MHz
    let n = 65536;
    let iq = make_fm_sync_iq(sample_rate, n, 15625.0, 75_000.0, 0.0);
    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);

    // Since 1 MHz < 5 MHz, it should be filtered out!
    assert!(
        results.is_empty(),
        "Result should be filtered out because bandwidth 1 MHz < min_bandwidth 5 MHz"
    );

    // But if we lower min_bandwidth to 500_000 (500 kHz), it should succeed
    detector.min_bandwidth = 500_000;
    let results = detector.detect_from_iq(&iq, 5_800_000_000, sample_rate);
    assert!(!results.is_empty(), "Result should not be filtered out");
}
