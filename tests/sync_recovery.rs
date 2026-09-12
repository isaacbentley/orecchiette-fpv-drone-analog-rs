//! Integration tests for NTSC/PAL timing recovery, independent timing vectors,
//! and baseline characterization of sync tracking and impairment handling.

use orecchiette_fpv_drone_analog_rs::synthetic::{
    ExtendedSyntheticConfig, ImpairmentSchedule, SyntheticVideoConfig, TestPattern,
    generate_extended_fixture,
};
use orecchiette_fpv_drone_analog_rs::vbi::{FieldParity, PulseKind};
use orecchiette_fpv_drone_analog_rs::video::FrameReconstructor;

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Independent Timing Vectors (ITU-R BT.470 Tables 1 & 2)
// ═══════════════════════════════════════════════════════════════════════════════
// These values are specified directly from the standard without deriving them
// from any library helper or internal constant.

#[test]
fn independent_ntsc_pal_rational_frequencies() {
    // ITU-R BT.470 Table 1:
    // NTSC line frequency = 15,750,000 / 1001 Hz
    let ntsc_line_hz: f64 = 15_750_000.0 / 1001.0;
    assert!((ntsc_line_hz - 15_734.265734265735f64).abs() < 1e-9);

    // PAL line frequency = 15,625 Hz exactly
    let pal_line_hz: f64 = 15_625.0;
    assert_eq!(pal_line_hz, 15_625.0);

    // Half-lines per field (525 / 2 = 262.5 for NTSC; 625 / 2 = 312.5 for PAL)
    let ntsc_field_lines: f64 = 525.0 * 0.5;
    let pal_field_lines: f64 = 625.0 * 0.5;
    assert_eq!(ntsc_field_lines, 262.5);
    assert_eq!(pal_field_lines, 312.5);

    // Subcarrier cycles per line
    // NTSC: 455 / 2 = 227.5
    let ntsc_fsc_per_line: f64 = 455.0 / 2.0;
    assert_eq!(ntsc_fsc_per_line, 227.5);
    let ntsc_fsc_hz: f64 = ntsc_line_hz * ntsc_fsc_per_line;
    assert!((ntsc_fsc_hz - 3579545.4545f64).abs() < 0.1);

    // PAL: 1135 / 4 + 1 / 625 = 283.7516
    let pal_fsc_per_line: f64 = 1135.0 / 4.0 + 1.0 / 625.0;
    assert_eq!(pal_fsc_per_line, 283.7516);
    let pal_fsc_hz: f64 = pal_line_hz * pal_fsc_per_line;
    assert!((pal_fsc_hz - 4_433_618.75f64).abs() < 1e-6);
}

#[test]
fn integer_rounding_error_accumulation() {
    // At 15.36 MSPS, exact nominal NTSC line period is:
    let fs: f64 = 15_360_000.0;
    let ntsc_line_hz: f64 = 15_750_000.0 / 1001.0;
    let exact_period: f64 = fs / ntsc_line_hz;
    assert!((exact_period - 976.2133333333334f64).abs() < 1e-9);

    // If initialized at integer-rounded 976:
    let rounded_period: f64 = 976.0;
    let per_line_error: f64 = exact_period - rounded_period;
    assert!((per_line_error - 0.2133333333334f64).abs() < 1e-9);

    // Over 240 active lines in a field, uncorrected integer period drifts by:
    let drift_samples: f64 = per_line_error * 240.0;
    let drift_us: f64 = (drift_samples / fs) * 1e6;
    assert!((drift_samples - 51.2f64).abs() < 0.1);
    assert!((drift_us - 3.3333333333333335f64).abs() < 1e-6);
}

#[test]
fn nominal_sample_periods_across_standard_sdr_rates() {
    let rates = [
        15_360_000.0f64,
        20_000_000.0,
        25_000_000.0,
        30_720_000.0,
        61_440_000.0,
    ];
    let ntsc_line_hz: f64 = 15_750_000.0 / 1001.0;
    let pal_line_hz: f64 = 15_625.0;

    for &fs in &rates {
        let ntsc_period: f64 = fs / ntsc_line_hz;
        let pal_period: f64 = fs / pal_line_hz;

        // Verify periods are within standard physical bounds (63.5555 us for NTSC, 64.0 us for PAL)
        let ntsc_time_us: f64 = (ntsc_period / fs) * 1e6;
        let pal_time_us: f64 = (pal_period / fs) * 1e6;
        assert!((ntsc_time_us - 63.55555555555556f64).abs() < 1e-9);
        assert!((pal_time_us - 64.0f64).abs() < 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Extended Synthetic Fixture & Ground Truth
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn extended_fixture_records_correct_ground_truth_structure() {
    let base = SyntheticVideoConfig {
        sample_rate: 15_360_000,
        is_pal: false,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Flat(50.0),
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };
    let cfg = ExtendedSyntheticConfig::new(base);
    let fixture = generate_extended_fixture(&cfg, 3);

    assert_eq!(fixture.total_fields, 3);
    assert_eq!(fixture.ground_truth_fields.len(), 3);

    // Parities alternate
    assert_eq!(fixture.ground_truth_fields[0].parity, FieldParity::First);
    assert_eq!(fixture.ground_truth_fields[1].parity, FieldParity::Second);
    assert_eq!(fixture.ground_truth_fields[2].parity, FieldParity::First);

    // Verify pulse counts in field 0: 6 eq + 6 broad + 6 post-eq = 18 VBI pulses
    let field0_pulses: Vec<_> = fixture
        .ground_truth_pulses
        .iter()
        .filter(|p| p.field_index == 0)
        .collect();
    let eq_count = field0_pulses
        .iter()
        .filter(|p| p.kind == PulseKind::Equalizing)
        .count();
    let broad_count = field0_pulses
        .iter()
        .filter(|p| p.kind == PulseKind::Broad)
        .count();
    assert_eq!(eq_count, 12); // 6 pre-eq + 6 post-eq
    assert_eq!(broad_count, 6);

    // Pulse centers are strictly monotonically increasing
    for w in fixture.ground_truth_pulses.windows(2) {
        assert!(w[1].center_sample > w[0].center_sample);
    }
}

#[test]
fn clock_mismatch_accurately_scales_sample_interval() {
    let base = SyntheticVideoConfig {
        sample_rate: 15_360_000,
        is_pal: false,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Flat(50.0),
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };

    let cfg_nominal = ExtendedSyntheticConfig::new(base);
    let fix_nominal = generate_extended_fixture(&cfg_nominal, 2);

    let cfg_fast = ExtendedSyntheticConfig::new(base).with_clock_error_ppm(100.0); // +100 ppm
    let fix_fast = generate_extended_fixture(&cfg_fast, 2);

    // When camera is 100 ppm fast, duration of 1 line in receiver samples should be smaller by ~100 ppm
    let p0_nom = fix_nominal.ground_truth_pulses[30].center_sample;
    let p1_nom = fix_nominal.ground_truth_pulses[130].center_sample;
    let dt_nom = p1_nom - p0_nom;

    let p0_fast = fix_fast.ground_truth_pulses[30].center_sample;
    let p1_fast = fix_fast.ground_truth_pulses[130].center_sample;
    let dt_fast = p1_fast - p0_fast;

    let ratio = dt_nom / dt_fast;
    let expected_ratio = 1.0 + 100.0 * 1e-6;
    assert!(
        (ratio - expected_ratio).abs() < 1e-4,
        "ratio: {}, expected: {}",
        ratio,
        expected_ratio
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Baseline Reconstructor Performance & Failure Characterization
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn baseline_clean_ntsc_and_pal_decode() {
    for is_pal in [false, true] {
        let sample_rate = 15_360_000;
        let base = SyntheticVideoConfig {
            sample_rate,
            is_pal,
            deviation_hz: 3_000_000.0,
            pattern: TestPattern::Flat(50.0),
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        };
        let fixture = generate_extended_fixture(&ExtendedSyntheticConfig::new(base), 3);

        let mut recon = FrameReconstructor::new(sample_rate, is_pal, 3_000_000.0, false);
        let mut frame = vec![0u32; recon.width * recon.height];

        let consumed = recon.reconstruct_frame_into(&fixture.demod, &mut frame);
        assert!(consumed.is_some(), "clean signal must decode");
        assert!(
            recon.latest_sync_quality() >= 0.90,
            "clean sync quality must be >= 0.90, got {}",
            recon.latest_sync_quality()
        );
    }
}

#[test]
fn baseline_fails_when_vbi_is_erased() {
    // Characterizes existing behavior:
    // In the baseline reconstructor, when the VBI serration group is missing and
    // no density spike triggers fallback, `reconstruct_frame_into` returns `None`.
    let sample_rate = 15_360_000;
    let base = SyntheticVideoConfig {
        sample_rate,
        is_pal: false,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Flat(50.0),
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };
    let impairments = ImpairmentSchedule {
        erase_vbi_fields: vec![0], // Field 0 has no VBI
        ..Default::default()
    };
    let cfg = ExtendedSyntheticConfig::new(base).with_impairments(impairments);
    let fixture = generate_extended_fixture(&cfg, 2);

    let mut recon = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    // Only present field 0's samples
    let field0_samples = (sample_rate as f64 / 15734.2657 * 263.0) as usize;
    let res = recon.reconstruct_frame_into(&fixture.demod[..field0_samples], &mut frame);

    // Baseline characterization: field 0 without VBI cannot be decoded by baseline reconstructor!
    assert!(
        res.is_none(),
        "Baseline characterization: missing VBI returns None"
    );
}

#[test]
fn baseline_pure_noise_rejection() {
    let sample_rate = 15_360_000;
    let mut recon = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    let mut rng = 0x1234_5678_9abc_def0u64;
    let noise: Vec<f32> = (0..sample_rate as usize / 10)
        .map(|_| orecchiette_fpv_drone_analog_rs::synthetic::gaussian_noise(&mut rng) * 0.5)
        .collect();

    let res = recon.reconstruct_frame_into(&noise, &mut frame);
    assert!(
        res.is_none(),
        "Pure noise must never achieve field lock in baseline"
    );
}
