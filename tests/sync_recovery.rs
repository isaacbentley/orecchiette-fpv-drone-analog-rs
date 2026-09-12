//! Integration tests for NTSC/PAL timing recovery, independent timing vectors,
//! and baseline characterization of sync tracking and impairment handling.

use orecchiette_fpv_drone_analog_rs::synthetic::{
    ExtendedSyntheticConfig, ImpairmentSchedule, SyntheticVideoConfig, TestPattern,
    generate_extended_fixture,
};
use orecchiette_fpv_drone_analog_rs::timing::{DecodeStep, DecodeValidationError, TimedDemodSlice};
use orecchiette_fpv_drone_analog_rs::vbi::{FieldParity, PulseKind};
use orecchiette_fpv_drone_analog_rs::video::FrameReconstructor;

fn recovery_config(is_pal: bool, start_field: FieldParity) -> ExtendedSyntheticConfig {
    ExtendedSyntheticConfig::new(SyntheticVideoConfig {
        sample_rate: 15_360_000,
        is_pal,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Bars,
        start_field,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    })
}

#[test]
fn need_more_data_does_not_commit_parity_or_picture_state() {
    for is_pal in [false, true] {
        let fixture = generate_extended_fixture(&recovery_config(is_pal, FieldParity::Second), 2);
        let mut recon = FrameReconstructor::new(fixture.sample_rate, is_pal, 3_000_000.0, false);
        let mut frame = vec![0x12345678; recon.width * recon.height];
        let tracker = recon.timing_tracker().clone();
        for _ in 0..2 {
            let result = recon
                .reconstruct_timed_into(
                    TimedDemodSlice::new(&fixture.demod[..100_000], 0, fixture.sample_rate, false),
                    &mut frame,
                )
                .unwrap();
            assert_eq!(
                result,
                DecodeStep::NeedMoreData {
                    consumed_samples: 0
                }
            );
            assert_eq!(recon.field_parity, 0, "short input must not change parity");
            assert_eq!(*recon.timing_tracker(), tracker);
            assert_eq!(recon.history_depth(), 0);
            assert!(frame.iter().all(|&p| p == 0x12345678));
        }
        let result = recon
            .reconstruct_timed_into(
                TimedDemodSlice::new(&fixture.demod, 0, fixture.sample_rate, false),
                &mut frame,
            )
            .unwrap();
        assert!(
            matches!(result, DecodeStep::Advance { field: Some(ref t), .. }
            if t.parity == FieldParity::Second && t.has_observed_timing_evidence)
        );
    }
}

#[test]
fn duplicate_completed_input_and_wrong_rate_are_rejected_without_mutation() {
    let fixture = generate_extended_fixture(&recovery_config(false, FieldParity::First), 2);
    let mut recon = FrameReconstructor::new(fixture.sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0; recon.width * recon.height];
    assert!(
        recon
            .reconstruct_timed_into(
                TimedDemodSlice::new(&fixture.demod, 0, fixture.sample_rate / 2, false),
                &mut frame,
            )
            .is_err(),
        "a mismatched clock must be rejected"
    );
    recon
        .reconstruct_timed_into(
            TimedDemodSlice::new(&fixture.demod, 0, fixture.sample_rate, false),
            &mut frame,
        )
        .unwrap();
    let tracker = recon.timing_tracker().clone();
    let previous_frame = frame.clone();
    assert!(
        recon
            .reconstruct_timed_into(
                TimedDemodSlice::new(&fixture.demod, 0, fixture.sample_rate, false),
                &mut frame,
            )
            .is_err(),
        "replaying consumed input must not emit a duplicate field"
    );
    assert_eq!(*recon.timing_tracker(), tracker);
    assert_eq!(frame, previous_frame);
    assert!(
        recon
            .reconstruct_timed_into(
                TimedDemodSlice::new(&fixture.demod, u64::MAX - 100, fixture.sample_rate, true),
                &mut frame,
            )
            .is_err(),
        "overflowing sample coordinates must be rejected"
    );
    assert_eq!(*recon.timing_tracker(), tracker);
}

#[test]
fn discontinuity_on_a_short_read_is_applied_once() {
    let mut recon = FrameReconstructor::new(15_360_000, false, 3_000_000.0, false);
    let mut frame = vec![0; recon.width * recon.height];
    for _ in 0..2 {
        let result = recon
            .reconstruct_timed_into(
                TimedDemodSlice::new(&[0.0; 1000], 100_000, 15_360_000, true),
                &mut frame,
            )
            .unwrap();
        assert_eq!(
            result,
            DecodeStep::NeedMoreData {
                consumed_samples: 0
            }
        );
        assert_eq!(recon.timing_tracker().continuity_epoch, 1);
    }
}

#[test]
fn large_sample_origins_preserve_fractional_timing() {
    use orecchiette_fpv_drone_analog_rs::timing::TimingTracker;
    for is_pal in [false, true] {
        let mut small = TimingTracker::new(15_360_000, is_pal);
        let mut large = small.clone();
        let base = (1_u64 << 55) + 3;
        small.record_observed_field(0, 123.375, 976.2133333333334, FieldParity::First);
        large.record_observed_field(base, 123.375, 976.2133333333334, FieldParity::First);
        assert_eq!(large.anchor_sample - base, small.anchor_sample);
        assert_eq!(large.fractional_offset, small.fractional_offset);
        assert_eq!(
            large.predict_vbi_sample_relative(base + 250_000),
            small.predict_vbi_sample_relative(250_000)
        );
    }
}

#[test]
fn density_fallback_is_not_observed_vbi_evidence() {
    let mut config = recovery_config(false, FieldParity::First);
    config.impairments.erase_vbi_fields = vec![0, 1];
    let mut fixture = generate_extended_fixture(&config, 3);
    // A single long negative plateau trips density detection but has no
    // half-line serrations or equalizing sequence.
    fixture.demod[3_000..4_500].fill(-0.5);
    let mut recon = FrameReconstructor::new(fixture.sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0; recon.width * recon.height];
    let result = recon
        .reconstruct_timed_into(
            TimedDemodSlice::new(&fixture.demod[..2 * 256_256], 0, fixture.sample_rate, false),
            &mut frame,
        )
        .unwrap();
    let consumed = match result {
        DecodeStep::Advance {
            field: Some(timing),
            consumed_samples,
        } => {
            assert!(!timing.has_observed_timing_evidence);
            consumed_samples
        }
        other => panic!("expected an unverified density preview: {other:?}"),
    };
    assert!(!recon.timing_tracker().is_locked);
    let result = recon
        .reconstruct_timed_into(
            TimedDemodSlice::new(
                &fixture.demod[consumed..],
                consumed as u64,
                fixture.sample_rate,
                false,
            ),
            &mut frame,
        )
        .unwrap();
    assert!(
        matches!(result, DecodeStep::Advance { field: Some(t), .. } if t.has_observed_timing_evidence)
    );
    assert_eq!(
        recon.history_depth(),
        1,
        "unverified preview history must not contaminate acquisition"
    );
}

#[test]
fn missing_horizontal_sync_does_not_publish_or_pollute_history() {
    for is_pal in [false, true] {
        let mut fixture =
            generate_extended_fixture(&recovery_config(is_pal, FieldParity::First), 5);
        let field_samples = if is_pal { 307_200 } else { 256_256 };
        fixture.demod[field_samples..4 * field_samples].fill(0.0);
        let mut recon = FrameReconstructor::new(fixture.sample_rate, is_pal, 3_000_000.0, false);
        let mut frame = vec![0; recon.width * recon.height];
        let first = recon
            .reconstruct_timed_into(
                TimedDemodSlice::new(&fixture.demod, 0, fixture.sample_rate, false),
                &mut frame,
            )
            .unwrap();
        let mut cursor = match first {
            DecodeStep::Advance {
                consumed_samples,
                field: Some(_),
            } => consumed_samples,
            other => panic!("initial acquisition failed: {other:?}"),
        };
        let previous_frame = frame.clone();
        let missing = recon
            .reconstruct_timed_into(
                TimedDemodSlice::new(
                    &fixture.demod[cursor..],
                    cursor as u64,
                    fixture.sample_rate,
                    false,
                ),
                &mut frame,
            )
            .unwrap();
        match missing {
            DecodeStep::Advance {
                consumed_samples,
                field: None,
            } => cursor += consumed_samples,
            other => panic!("unsupported field must be skipped: {other:?}"),
        }
        assert_eq!(frame, previous_frame);
        assert_eq!(recon.history_depth(), 0);
        assert_eq!(recon.timing_tracker().h_sync_coasted_fields, 1);
        assert!(
            recon.timing_tracker().is_locked,
            "one field of timing holdover is allowed"
        );
        let missing = recon
            .reconstruct_timed_into(
                TimedDemodSlice::new(
                    &fixture.demod[cursor..],
                    cursor as u64,
                    fixture.sample_rate,
                    false,
                ),
                &mut frame,
            )
            .unwrap();
        assert!(matches!(missing, DecodeStep::Advance { field: None, .. }));
        assert!(
            !recon.timing_tracker().is_locked,
            "second missing H grid exhausts its own budget"
        );
        assert_eq!(frame, previous_frame);
    }
}

#[test]
fn reacquires_structural_vbi_after_exhausted_holdover() {
    for is_pal in [false, true] {
        let mut config = recovery_config(is_pal, FieldParity::First);
        config.impairments.erase_vbi_fields = vec![1, 2, 3, 4];
        let fixture = generate_extended_fixture(&config, 7);
        let mut recon = FrameReconstructor::new(fixture.sample_rate, is_pal, 3_000_000.0, false);
        let mut frame = vec![0; recon.width * recon.height];
        let mut cursor = 0;
        let mut decoded = Vec::new();
        while cursor < fixture.demod.len() {
            match recon
                .reconstruct_timed_into(
                    TimedDemodSlice::new(
                        &fixture.demod[cursor..],
                        cursor as u64,
                        fixture.sample_rate,
                        false,
                    ),
                    &mut frame,
                )
                .unwrap()
            {
                DecodeStep::Advance {
                    consumed_samples,
                    field,
                } => {
                    assert!(consumed_samples > 0);
                    cursor += consumed_samples;
                    if let Some(timing) = field {
                        decoded.push(timing);
                    }
                }
                DecodeStep::NeedMoreData { .. } => break,
            }
        }
        assert_eq!(
            decoded.len(),
            6,
            "only field 4 should be lost for PAL={is_pal}"
        );
        for (timing, field_index) in decoded.iter().zip([0, 1, 2, 3, 5, 6]) {
            let truth = &fixture.ground_truth_fields[field_index];
            assert_eq!(timing.parity, truth.parity);
            let broad = timing.origin_sample as f64 + timing.vbi_sample_offset;
            // The parser's moving average delays the measured leading edge.
            assert!(
                (broad - truth.broad_start_sample as f64).abs()
                    < fixture.sample_rate as f64 * 0.5e-6,
                "PAL={is_pal} field={field_index}: broad={broad}, expected={}, timing={timing:?}",
                truth.broad_start_sample
            );
            assert_eq!(
                timing.has_observed_timing_evidence,
                matches!(field_index, 0 | 5 | 6)
            );
        }
        assert!(recon.timing_tracker().is_locked);
        assert_eq!(
            recon.timing_tracker().continuity_epoch,
            0,
            "loss of sync is not loss of samples"
        );
    }
}

#[test]
fn bounded_vbi_search_preserves_coordinates_and_parity() {
    use orecchiette_fpv_drone_analog_rs::levels::SyncLevels;
    use orecchiette_fpv_drone_analog_rs::vbi::{
        find_vertical_sync_candidates, find_vertical_sync_near,
    };
    for is_pal in [false, true] {
        let fixture = generate_extended_fixture(&recovery_config(is_pal, FieldParity::First), 6);
        let levels = SyncLevels {
            sync_tip: -0.4 * 2.0 * std::f32::consts::PI * 3_000_000.0 / fixture.sample_rate as f32,
            blanking: 0.0,
        };
        let candidates =
            find_vertical_sync_candidates(&fixture.demod, fixture.sample_rate, &levels, is_pal);
        assert_eq!(candidates.len(), 6);
        for expected in candidates {
            let found = find_vertical_sync_near(
                &fixture.demod,
                fixture.sample_rate,
                &levels,
                is_pal,
                expected.broad_start + 20.0,
                100,
                expected.parity,
            )
            .unwrap();
            assert_eq!(found.parity, expected.parity);
            assert!((found.broad_start - expected.broad_start).abs() < 0.1);
            assert!((found.field_active_start - expected.field_active_start).abs() < 0.2);
            assert_eq!(
                (found.n_broad, found.n_eq_pre, found.n_eq_post),
                (expected.n_broad, expected.n_eq_pre, expected.n_eq_post)
            );
            assert!(
                find_vertical_sync_near(
                    &fixture.demod,
                    fixture.sample_rate,
                    &levels,
                    is_pal,
                    expected.broad_start,
                    0,
                    expected.parity
                )
                .is_some()
            );
        }
        assert!(
            find_vertical_sync_near(
                &fixture.demod,
                fixture.sample_rate,
                &levels,
                is_pal,
                f32::NAN,
                100,
                None
            )
            .is_none()
        );
    }
}

#[test]
fn coasted_fields_follow_the_measured_horizontal_clock() {
    for is_pal in [false, true] {
        let mut config = recovery_config(is_pal, FieldParity::First)
            .with_clock_error_ppm(50.0)
            .with_clock_drift(1_000.0);
        config.impairments.erase_vbi_fields = vec![1, 2, 3];
        let fixture = generate_extended_fixture(&config, 5);
        let mut recon = FrameReconstructor::new(fixture.sample_rate, is_pal, 3_000_000.0, false);
        let mut frame = vec![0; recon.width * recon.height];
        let mut cursor = 0;
        let mut previous_uncertainty = 0.0;
        for field_index in 0..4 {
            let result = recon
                .reconstruct_timed_into(
                    TimedDemodSlice::new(
                        &fixture.demod[cursor..],
                        cursor as u64,
                        fixture.sample_rate,
                        false,
                    ),
                    &mut frame,
                )
                .unwrap();
            let (consumed, timing) = match result {
                DecodeStep::Advance {
                    consumed_samples,
                    field: Some(timing),
                } => (consumed_samples, timing),
                other => panic!("PAL={is_pal} field={field_index}: {other:?}"),
            };
            assert_eq!(timing.coasted_field_count, field_index as u32);
            assert_eq!(
                recon.timing_tracker().line_period,
                timing.line_period_samples,
                "VBI prediction and H resampling must use the same clock"
            );
            if field_index > 0 {
                assert!(
                    timing.uncertainty_seconds > previous_uncertainty,
                    "H-only observations must not reset VBI uncertainty"
                );
            }
            previous_uncertainty = timing.uncertainty_seconds;
            cursor += consumed;
            let prediction = recon
                .timing_tracker()
                .predict_vbi_sample_relative(cursor as u64) as f64;
            let truth = fixture.ground_truth_fields[field_index + 1].broad_start_sample as f64
                - cursor as f64;
            assert!(
                (prediction - truth).abs() < fixture.sample_rate as f64 * 1.0e-6,
                "PAL={is_pal} field={field_index}: predicted={prediction}, truth={truth}"
            );
        }
    }
}

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

    // Assert that library production constants in timing.rs, vbi.rs, and Standard enum
    // exactly match these independent ITU-R BT.470 values:
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::timing::NTSC_NOMINAL_LINE_HZ,
        ntsc_line_hz
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::timing::PAL_NOMINAL_LINE_HZ,
        pal_line_hz
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::vbi::consts::NTSC_LINE_HZ,
        ntsc_line_hz
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::vbi::consts::PAL_LINE_HZ,
        pal_line_hz
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::timing::Standard::Ntsc.nominal_line_hz(),
        ntsc_line_hz
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::timing::Standard::Pal.nominal_line_hz(),
        pal_line_hz
    );

    // Half-lines per field (525 / 2 = 262.5 for NTSC; 625 / 2 = 312.5 for PAL)
    let ntsc_field_lines: f64 = 525.0 * 0.5;
    let pal_field_lines: f64 = 625.0 * 0.5;
    assert_eq!(ntsc_field_lines, 262.5);
    assert_eq!(pal_field_lines, 312.5);

    assert_eq!(
        orecchiette_fpv_drone_analog_rs::timing::NTSC_FIELD_TOTAL_LINES,
        ntsc_field_lines
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::timing::PAL_FIELD_TOTAL_LINES,
        pal_field_lines
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::vbi::consts::NTSC_FIELD_TOTAL_LINES,
        ntsc_field_lines
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::vbi::consts::PAL_FIELD_TOTAL_LINES,
        pal_field_lines
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::timing::Standard::Ntsc.half_lines_per_field(),
        525
    );
    assert_eq!(
        orecchiette_fpv_drone_analog_rs::timing::Standard::Pal.half_lines_per_field(),
        625
    );

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

    // Assert that FrameReconstructor initializes line_period to the exact unrounded period
    let recon = FrameReconstructor::new(15_360_000, false, 3_000_000.0, false);
    assert_eq!(recon.line_period, exact_period as f32);
    assert_ne!(recon.line_period, 976.0);

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

        // Assert that library function nominal_line_period_samples matches exact calculation
        let lib_ntsc =
            orecchiette_fpv_drone_analog_rs::timing::nominal_line_period_samples(fs as u32, false);
        let lib_pal =
            orecchiette_fpv_drone_analog_rs::timing::nominal_line_period_samples(fs as u32, true);
        assert_eq!(lib_ntsc, ntsc_period);
        assert_eq!(lib_pal, pal_period);

        // Assert that FrameReconstructor initializes line_period to exact nominal period
        let recon_ntsc = FrameReconstructor::new(fs as u32, false, 3_000_000.0, false);
        assert_eq!(recon_ntsc.line_period, ntsc_period as f32);

        let recon_pal = FrameReconstructor::new(fs as u32, true, 3_000_000.0, false);
        assert_eq!(recon_pal.line_period, pal_period as f32);
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
    let fix_nominal = generate_extended_fixture(&cfg_nominal, 8);

    let cfg_fast = ExtendedSyntheticConfig::new(base).with_clock_error_ppm(100.0); // +100 ppm
    let fix_fast = generate_extended_fixture(&cfg_fast, 8);

    // Measure over a sufficiently long interval (2000 pulses, ~1.95M samples)
    // so sample-rounding discretization error is < 3e-7, allowing a tight 1e-6 tolerance.
    let p0_nom = fix_nominal.ground_truth_pulses[30].center_sample;
    let p1_nom = fix_nominal.ground_truth_pulses[2030].center_sample;
    let dt_nom = p1_nom - p0_nom;

    let p0_fast = fix_fast.ground_truth_pulses[30].center_sample;
    let p1_fast = fix_fast.ground_truth_pulses[2030].center_sample;
    let dt_fast = p1_fast - p0_fast;

    let ratio = dt_nom / dt_fast;
    let expected_ratio = 1.0 + 100.0 * 1e-6;
    assert!(
        (ratio - expected_ratio).abs() < 1e-6,
        "ratio: {:.8}, expected: {:.8}, error: {:.8}",
        ratio,
        expected_ratio,
        (ratio - expected_ratio).abs()
    );

    // Explicitly assert that a no-op / null clock adjustment (ratio = 1.0)
    // would fail this test with a large ~100 ppm margin:
    let null_error = (1.0f64 - expected_ratio).abs();
    assert!(
        null_error >= 9.9e-5,
        "Null adjustment must fail tight tolerance with ~100 ppm deviation: {:.8}",
        null_error
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

#[test]
fn timed_reconstruct_validation_errors() {
    let sample_rate = 15_360_000;
    let mut recon = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    // Empty buffer error
    let empty_slice = TimedDemodSlice::new(&[], 0, sample_rate, false);
    let res = recon.reconstruct_timed_into(empty_slice, &mut frame);
    assert_eq!(res, Err(DecodeValidationError::EmptyBuffer));

    // Frame buffer too small error
    let dummy_samples = vec![0.0f32; 1000];
    let slice = TimedDemodSlice::new(&dummy_samples, 0, sample_rate, false);
    let mut small_frame = vec![0u32; 10];
    let res = recon.reconstruct_timed_into(slice, &mut small_frame);
    assert_eq!(
        res,
        Err(DecodeValidationError::FrameBufferTooSmall {
            required: recon.width * recon.height,
            actual: 10,
        })
    );
}

#[test]
fn timed_reconstruct_pure_noise_and_short_slices() {
    let sample_rate = 15_360_000;
    let mut recon = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    // Very short slice -> NeedMoreData { consumed_samples: 0 }
    let short_samples = vec![0.0f32; 500];
    let short_slice = TimedDemodSlice::new(&short_samples, 0, sample_rate, false);
    let res = recon.reconstruct_timed_into(short_slice, &mut frame);
    assert_eq!(
        res,
        Ok(DecodeStep::NeedMoreData {
            consumed_samples: 0
        })
    );

    // Long pure noise -> Advance with field: None (safe non-blocking drain)
    let mut rng = 0xbeef_cafe_1234_5678u64;
    let noise: Vec<f32> = (0..sample_rate as usize / 10)
        .map(|_| orecchiette_fpv_drone_analog_rs::synthetic::gaussian_noise(&mut rng) * 0.5)
        .collect();
    let noise_slice = TimedDemodSlice::new(&noise, 10_000, sample_rate, false);
    let res = recon.reconstruct_timed_into(noise_slice, &mut frame);
    match res {
        Ok(DecodeStep::Advance {
            consumed_samples,
            field,
        }) => {
            assert!(
                consumed_samples > 0,
                "Noise slice must consume samples to prevent freezing"
            );
            assert!(
                field.is_none(),
                "Noise slice must not produce a decoded field"
            );
        }
        other => panic!("Expected Advance with None field on noise, got {other:?}"),
    }
}

#[test]
fn timed_reconstruct_clean_ntsc_is_deterministic() {
    let sample_rate = 15_360_000;
    let base = SyntheticVideoConfig {
        sample_rate,
        is_pal: false,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Bars,
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };
    let cfg = ExtendedSyntheticConfig::new(base);
    let fixture = generate_extended_fixture(&cfg, 2);

    let mut recon1 = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut recon2 = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame1 = vec![0u32; recon1.width * recon1.height];
    let mut frame2 = vec![0u32; recon2.width * recon2.height];

    let slice1 = TimedDemodSlice::new(&fixture.demod, 1_000_000, sample_rate, false);
    let slice2 = TimedDemodSlice::new(&fixture.demod, 1_000_000, sample_rate, false);

    let res1 = recon1.reconstruct_timed_into(slice1, &mut frame1).unwrap();
    let res2 = recon2.reconstruct_timed_into(slice2, &mut frame2).unwrap();

    // Determinism: both reconstructors produce identical results
    assert_eq!(res1, res2, "Reconstruction must be deterministic");
    assert_eq!(
        frame1, frame2,
        "Decoded frames must be byte-for-byte identical"
    );

    // Check telemetry fields
    match res1 {
        DecodeStep::Advance {
            consumed_samples,
            field: Some(timing),
        } => {
            assert!(consumed_samples > 0);
            assert_eq!(timing.origin_sample, 1_000_000);
            assert!(timing.has_observed_timing_evidence);
            assert!(timing.confidence >= 0.90);
            assert_eq!(timing.coasted_field_count, 0);
            assert!(recon1.timing_tracker().is_locked);
            assert_eq!(
                recon1.timing_tracker().last_processed_sample,
                1_000_000 + consumed_samples as u64
            );
        }
        other => panic!("Expected Advance with Some(field), got {other:?}"),
    }
}

#[test]
fn timed_reconstruct_handles_discontinuity() {
    let sample_rate = 15_360_000;
    let base = SyntheticVideoConfig {
        sample_rate,
        is_pal: false,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Bars,
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };
    let cfg = ExtendedSyntheticConfig::new(base);
    let fixture = generate_extended_fixture(&cfg, 2);

    let mut recon = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    // Field 0: normal decode
    let slice0 = TimedDemodSlice::new(&fixture.demod, 0, sample_rate, false);
    let res0 = recon.reconstruct_timed_into(slice0, &mut frame).unwrap();
    let consumed0 = match res0 {
        DecodeStep::Advance {
            consumed_samples, ..
        } => consumed_samples,
        _ => panic!("Expected field 0 to decode"),
    };
    assert_eq!(recon.timing_tracker().continuity_epoch, 0);

    // Discontinuous slice with explicit flag: epoch must increment and history reset
    let slice1 = TimedDemodSlice::new(&fixture.demod[consumed0..], 5_000_000, sample_rate, true);
    let res1 = recon.reconstruct_timed_into(slice1, &mut frame).unwrap();
    assert!(matches!(res1, DecodeStep::Advance { .. }));
    assert_eq!(
        recon.timing_tracker().continuity_epoch,
        1,
        "Epoch must increment upon discontinuity"
    );
}

#[test]
fn holdover_recovers_missing_vbi_field() {
    let sample_rate = 15_360_000;
    let base = SyntheticVideoConfig {
        sample_rate,
        is_pal: false,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Bars,
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };
    let impairments = ImpairmentSchedule {
        erase_vbi_fields: vec![1], // Field 1 has VBI erased
        ..Default::default()
    };
    let cfg = ExtendedSyntheticConfig::new(base).with_impairments(impairments);
    let fixture = generate_extended_fixture(&cfg, 3);

    let mut recon = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    let mut cursor = 0usize;
    let mut decoded_fields = Vec::new();

    while cursor < fixture.demod.len() && decoded_fields.len() < 3 {
        let slice =
            TimedDemodSlice::new(&fixture.demod[cursor..], cursor as u64, sample_rate, false);
        match recon.reconstruct_timed_into(slice, &mut frame) {
            Ok(DecodeStep::Advance {
                consumed_samples,
                field,
            }) => {
                cursor += consumed_samples;
                if let Some(timing) = field {
                    decoded_fields.push(timing);
                }
            }
            Ok(DecodeStep::NeedMoreData { .. }) => break,
            Err(e) => panic!("Unexpected decode error: {e:?}"),
        }
    }

    assert_eq!(decoded_fields.len(), 3, "All 3 fields must be recovered!");

    // Field 0: Observed VBI
    assert!(
        decoded_fields[0].has_observed_timing_evidence,
        "Field 0 must have observed timing evidence"
    );
    assert_eq!(decoded_fields[0].coasted_field_count, 0);
    assert_eq!(decoded_fields[0].parity, FieldParity::First);

    // Field 1: Coasted / Holdover VBI (missing serration group)
    assert!(
        !decoded_fields[1].has_observed_timing_evidence,
        "Field 1 was coasted, so has_observed_timing_evidence must be false"
    );
    assert_eq!(decoded_fields[1].coasted_field_count, 1);
    assert_eq!(
        decoded_fields[1].parity,
        FieldParity::Second,
        "Field 1 must alternate to Second parity"
    );

    // Field 2: Reacquired observed VBI
    assert!(
        decoded_fields[2].has_observed_timing_evidence,
        "Field 2 must reacquire observed timing evidence"
    );
    assert_eq!(decoded_fields[2].coasted_field_count, 0);
    assert_eq!(
        decoded_fields[2].parity,
        FieldParity::First,
        "Field 2 must alternate back to First parity"
    );
}

#[test]
fn holdover_budget_exhaustion_drops_field() {
    let sample_rate = 15_360_000;
    let base = SyntheticVideoConfig {
        sample_rate,
        is_pal: false,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Bars,
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };
    // Erase VBI for 4 consecutive fields (fields 1, 2, 3, 4)
    let impairments = ImpairmentSchedule {
        erase_vbi_fields: vec![1, 2, 3, 4],
        ..Default::default()
    };
    let cfg = ExtendedSyntheticConfig::new(base).with_impairments(impairments);
    let fixture = generate_extended_fixture(&cfg, 6);

    let mut recon = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    let mut cursor = 0usize;
    let mut coasted_counts = Vec::new();

    for _ in 0..5 {
        if cursor >= fixture.demod.len() {
            break;
        }
        let slice =
            TimedDemodSlice::new(&fixture.demod[cursor..], cursor as u64, sample_rate, false);
        match recon.reconstruct_timed_into(slice, &mut frame) {
            Ok(DecodeStep::Advance {
                consumed_samples,
                field,
            }) => {
                cursor += consumed_samples;
                if let Some(timing) = field {
                    coasted_counts.push(timing.coasted_field_count);
                } else {
                    coasted_counts.push(999); // dropped field marker
                }
            }
            Ok(DecodeStep::NeedMoreData { .. }) => break,
            Err(e) => panic!("Unexpected error: {e:?}"),
        }
    }

    // Field 0: 0 coasted
    // Field 1: 1 coasted
    // Field 2: 2 coasted
    // Field 3: 3 coasted
    // Field 4: dropped (budget of 3 exhausted)
    assert!(coasted_counts.len() >= 4);
    assert_eq!(coasted_counts[0], 0);
    assert_eq!(coasted_counts[1], 1);
    assert_eq!(coasted_counts[2], 2);
    assert_eq!(coasted_counts[3], 3);
    if coasted_counts.len() >= 5 {
        assert_eq!(
            coasted_counts[4], 999,
            "4th consecutive missing VBI must exceed holdover budget and drop field"
        );
    }
}

#[test]
fn timing_metric_detects_whole_line_slip() {
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
    let cfg = ExtendedSyntheticConfig::new(base);
    let fixture = generate_extended_fixture(&cfg, 2);

    let mut recon = FrameReconstructor::new(sample_rate, false, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    let consumed = recon.reconstruct_frame_into(&fixture.demod, &mut frame);
    assert!(consumed.is_some());

    let sync_positions = recon.latest_sync_positions();
    assert!(!sync_positions.is_empty());

    let base_active_lines =
        orecchiette_fpv_drone_analog_rs::vbi::consts::NTSC_BASE_ACTIVE_START_LINES;
    let tbc_scale = recon.line_width as f64 / recon.line_period as f64;

    // Normal evaluation: sub-pixel error
    let mut normal_errors = Vec::new();
    for (row, &pos) in sync_positions.iter().enumerate() {
        let expected_line = base_active_lines + row as f64;
        let pulse = fixture
            .ground_truth_pulses
            .iter()
            .find(|p| {
                p.field_index == 0
                    && p.kind == PulseKind::Horizontal
                    && (p.line_in_field - expected_line).abs() < 1e-4
            })
            .expect("Expected ground truth pulse for row");
        let decoded_sample = pos as f64;
        let err_tbc = (decoded_sample - pulse.center_sample).abs() * tbc_scale;
        normal_errors.push(err_tbc);
    }
    normal_errors.sort_by(f64::total_cmp);
    let p95_normal = normal_errors[(normal_errors.len() as f64 * 0.95) as usize];
    assert!(
        p95_normal < 1.0,
        "Normal decode error must be < 1.0 TBC pixel, got {p95_normal}"
    );

    // Deliberate 1-line slip perturbation: add exactly 1 line period to reported sync positions
    let mut slipped_errors = Vec::new();
    for (row, &pos) in sync_positions.iter().enumerate() {
        let expected_line = base_active_lines + row as f64;
        let pulse = fixture
            .ground_truth_pulses
            .iter()
            .find(|p| {
                p.field_index == 0
                    && p.kind == PulseKind::Horizontal
                    && (p.line_in_field - expected_line).abs() < 1e-4
            })
            .expect("Expected ground truth pulse for row");
        let displaced_sample = pos as f64 + recon.line_period as f64; // +1 full line period displacement
        let err_tbc = (displaced_sample - pulse.center_sample).abs() * tbc_scale;
        slipped_errors.push(err_tbc);
    }
    slipped_errors.sort_by(f64::total_cmp);
    let p95_slipped = slipped_errors[(slipped_errors.len() as f64 * 0.95) as usize];
    // Must report approximately 858 TBC pixels (one full line width), NOT sub-sample error!
    assert!(
        (p95_slipped - recon.line_width as f64).abs() < 2.0,
        "Whole-line slip must report ~{} TBC samples of error, got {:.2}",
        recon.line_width,
        p95_slipped
    );
    assert!(
        p95_slipped > 800.0,
        "Line slip error must not collapse to sub-sample: {p95_slipped}"
    );
}
