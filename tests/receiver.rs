//! Public receiver API contracts, exercised without a viewer or source driver.
use num_complex::Complex;
use orecchiette_fpv_drone_analog_rs::acquisition::{
    AcquisitionRecord, StandardEvidence, analyze_tuned_channel,
};
use orecchiette_fpv_drone_analog_rs::bands::{
    candidate_frequencies, channel_catalog, channel_name, lookup_channel_by_name,
};
use orecchiette_fpv_drone_analog_rs::decode::{
    DecodePlan, DecoderConfig, DemodulationMode, StreamingFpvDecoder, required_capture_rate,
};
use orecchiette_fpv_drone_analog_rs::scanner::{
    CandidatePolicy, FineTuneSelector, ScanProgress, ScanSelector, SweepBudget, SweepPolicy,
};
use orecchiette_fpv_drone_analog_rs::synthetic::{SyntheticVideoConfig, TestPattern, generate_iq};
use orecchiette_fpv_drone_analog_rs::timing::{DecodeValidationError, Standard};
use orecchiette_fpv_drone_analog_rs::types::{DetectionResult, SignalType};
use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;
use std::collections::HashSet;
use std::time::Duration;

fn config(standard: Standard) -> DecoderConfig {
    DecoderConfig {
        plan: DecodePlan::new(30_720_000, 3e6, 10e6, DemodulationMode::Auto).unwrap(),
        frequency_offset_hz: 0.0,
        standard,
        deemphasis_tau_s: 0.75e-6,
        temporal_window: 3,
        debug: false,
    }
}
fn fixture(standard: Standard) -> Vec<Complex<f32>> {
    generate_iq(
        &SyntheticVideoConfig {
            sample_rate: 30_720_000,
            is_pal: standard == Standard::Pal,
            deviation_hz: 3e6,
            pattern: TestPattern::Bars,
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        },
        4,
        0.0,
    )
}
#[test]
fn standalone_receiver_preserves_every_field_across_irregular_chunks() {
    for standard in [Standard::Ntsc, Standard::Pal] {
        let iq = fixture(standard);
        let decode = |chunk_size| {
            let mut decoder = StreamingFpvDecoder::new(config(standard)).unwrap();
            let mut frame = vec![0; decoder.reconstructor().width * decoder.reconstructor().height];
            let mut fields = Vec::new();
            for chunk in iq.chunks(chunk_size) {
                decoder.push_iq(chunk, false);
                decoder.push_iq(&[], false);
                while let Some(timing) = decoder.next_field_into(&mut frame).unwrap() {
                    fields.push((timing, frame.clone()));
                }
            }
            (
                fields,
                decoder.sample_coordinate(),
                decoder.buffered_samples(),
            )
        };
        let small = decode(16_381);
        let whole = decode(iq.len());
        assert!(small.0.len() >= 3);
        assert!(small.0.iter().all(|(t, _)| t.has_observed_timing_evidence));
        assert_eq!(small.0.len(), whole.0.len(), "field count {standard:?}");
        assert_eq!(
            (small.1, small.2),
            (whole.1, whole.2),
            "cursor {standard:?}"
        );
        for (index, ((a, pixels_a), (b, pixels_b))) in small.0.iter().zip(&whole.0).enumerate() {
            // The VBI quality score uses the available level-estimation
            // lookahead; timing and pixels must remain partition-independent.
            assert!((a.confidence - b.confidence).abs() < 1e-3);
            let mut normalized = a.clone();
            normalized.confidence = b.confidence;
            assert_eq!(&normalized, b, "timing {standard:?} field {index}");
            assert!(pixels_a == pixels_b, "pixels {standard:?} field {index}");
        }
    }
}
#[test]
fn invalid_output_does_not_consume_input_and_gaps_advance_one_epoch() {
    let mut decoder = StreamingFpvDecoder::new(config(Standard::Ntsc)).unwrap();
    decoder.push_iq(&fixture(Standard::Ntsc), false);
    let before = decoder.buffered_samples();
    assert!(matches!(
        decoder.next_field_into(&mut []),
        Err(DecodeValidationError::FrameBufferTooSmall { .. })
    ));
    assert_eq!(decoder.buffered_samples(), before);
    assert_eq!(decoder.sample_coordinate(), 0);
    let mut frame = vec![0; decoder.reconstructor().width * decoder.reconstructor().height];
    assert!(decoder.next_field_into(&mut frame).unwrap().is_some());
    assert!(decoder.reconstructor().timing_tracker().is_locked);
    let end = decoder.sample_coordinate() + decoder.buffered_samples() as u64;
    decoder.reset_discontinuity();
    assert_eq!(decoder.sample_coordinate(), end);
    assert_eq!(decoder.buffered_samples(), 0);
    assert_eq!(decoder.reconstructor().history_depth(), 0);
    assert_eq!(decoder.reconstructor().timing_tracker().continuity_epoch, 1);
    assert_eq!(decoder.cnr_db(), None);
}
#[test]
fn latency_discard_counts_known_samples_and_releases_history_once() {
    let mut decoder = StreamingFpvDecoder::new(config(Standard::Ntsc)).unwrap();
    decoder.push_iq(&fixture(Standard::Ntsc), false);
    let count = decoder.buffered_samples();
    assert_eq!(decoder.discard_pending_except(100), count - 100);
    assert_eq!(decoder.sample_coordinate(), (count - 100) as u64);
    assert_eq!(decoder.buffered_samples(), 100);
    assert_eq!(decoder.reconstructor().timing_tracker().continuity_epoch, 1);
    assert_eq!(decoder.discard_pending_except(100), 0);
    assert_eq!(decoder.reconstructor().timing_tracker().continuity_epoch, 1);
}
#[test]
fn invalid_decode_configuration_is_rejected_before_dsp_construction() {
    for (rate, dev, bandwidth) in [
        (0, 5e6, 14e6),
        (30_720_000, f32::NAN, 14e6),
        (30_720_000, 5e6, f32::INFINITY),
    ] {
        assert!(DecodePlan::new(rate, dev, bandwidth, DemodulationMode::Auto).is_err());
    }
    let mut cfg = config(Standard::Pal);
    cfg.frequency_offset_hz = f32::NAN;
    assert!(StreamingFpvDecoder::new(cfg).is_err());
    let mut cfg = config(Standard::Pal);
    cfg.plan = DecodePlan::new(1, 5e6, 14e6, DemodulationMode::Auto).unwrap();
    assert!(StreamingFpvDecoder::new(cfg).is_err());
    assert_eq!(required_capture_rate(5e6, 0.8), Some(17_500_000.0));
    assert_eq!(required_capture_rate(5e6, 0.0), None);
}
#[test]
fn acquisition_discards_a_gap_and_bounds_record_memory() {
    let mut record = AcquisitionRecord::new(1000);
    record.push(&[Complex::new(1.0, 0.0); 20], false);
    assert!(!record.is_ready());
    record.push(&[Complex::new(2.0, 0.0); 30], true);
    assert!(record.is_ready());
    assert_eq!(record.samples(), &[Complex::new(2.0, 0.0); 25]);
    record.push(&[Complex::new(3.0, 0.0); 30], false);
    assert_eq!(record.samples().len(), 25);
}
#[test]
fn acquisition_resolves_both_standards_from_the_contiguous_record() {
    for (standard, expected, fallback) in [
        (
            Standard::Ntsc,
            SignalType::AnalogVideoNtsc,
            SignalType::AnalogVideoPal,
        ),
        (
            Standard::Pal,
            SignalType::AnalogVideoPal,
            SignalType::AnalogVideoNtsc,
        ),
    ] {
        let mut record = AcquisitionRecord::new(30_720_000);
        for chunk in fixture(standard).chunks(65_536) {
            record.push(chunk, false);
        }
        let result =
            analyze_tuned_channel(record.samples(), 5865e6, 30_720_000, 3e6, fallback).unwrap();
        assert_eq!(result.standard.standard, expected);
        assert_ne!(result.standard.evidence, StandardEvidence::Fallback);
        if let Some(carrier) = result.measured_carrier_hz {
            assert!((carrier - 5865e6).abs() <= 5e6);
        }
    }
    assert!(analyze_tuned_channel(&[], 5865e6, 0, 3e6, SignalType::Unknown).is_err());
}
#[test]
fn every_catalog_entry_round_trips_and_is_a_snap_candidate() {
    for channel in channel_catalog() {
        assert_eq!(
            lookup_channel_by_name(&channel.name()),
            Some(channel.frequency_hz)
        );
        assert!(
            channel_name(channel.frequency_hz)
                .unwrap()
                .contains(&channel.display_name())
        );
        assert_eq!(
            candidate_frequencies(channel.frequency_hz as f64, 0.0),
            vec![channel.frequency_hz as f64]
        );
    }
    assert_eq!(lookup_channel_by_name("U4"), Some(5_373_000_000));
    assert_eq!(lookup_channel_by_name("T8"), Some(2_450_000_000));
    let aliases = candidate_frequencies(2460e6, 20e6);
    assert_eq!(
        aliases.len(),
        aliases
            .iter()
            .map(|f| *f as u64)
            .collect::<HashSet<_>>()
            .len()
    );
    assert!(candidate_frequencies(f64::NAN, 15e6).is_empty());
}
#[test]
fn sweep_requires_a_full_budget_for_every_hop_and_ignores_foreign_headers() {
    let mut progress = ScanProgress::new(&[1e9, 2e9], 2);
    assert!(progress.observe(3_000_000_000).is_none());
    assert!(!progress.observe(1_000_000_000).unwrap().sweep_complete);
    assert!(!progress.observe(2_000_000_000).unwrap().sweep_complete);
    assert!(!progress.observe(2_000_000_000).unwrap().sweep_complete);
    assert!(progress.observe(2_000_000_000).is_none());
    assert!(!progress.observe(1_000_000_000).unwrap().sweep_complete);
    assert!(progress.observe(1_000_000_000).unwrap().sweep_complete);
    progress.reset_sweep();
    assert!(progress.observe(1_000_000_000).is_none()); // drain final dwell
    assert!(progress.observe(2_000_000_000).unwrap().first_packet);
}
#[test]
fn a_single_tune_continues_after_each_sweep() {
    let mut progress = ScanProgress::new(&[5865e6], 2);
    for _ in 0..3 {
        assert!(!progress.observe(5_865_000_000).unwrap().sweep_complete);
        assert!(progress.observe(5_865_000_000).unwrap().sweep_complete);
        progress.reset_sweep();
    }
}
fn hit(frequency_hz: u64, rssi_dbm: f32) -> DetectionResult {
    DetectionResult {
        channel: None,
        frequency_hz,
        rssi_dbm,
        confidence: 0.8,
        bandwidth_hz: 14_000_000,
        signal_type: SignalType::AnalogVideoUnknown,
    }
}
#[test]
fn scan_selection_preserves_skips_and_off_grid_skip_tolerance() {
    let skipped = HashSet::from([5_865_000_000, 6_100_000_000]);
    let mut nearest = ScanSelector::new(&skipped, CandidatePolicy::NearestChannel);
    assert!(!nearest.consider(&hit(5_865_000_000, -10.0)));
    assert!(!nearest.consider(&hit(6_102_000_000, -10.0)));
    assert!(nearest.consider(&hit(5_800_000_000, -50.0)));
    assert!(!nearest.consider(&hit(5_845_000_000, -60.0)));
    assert!(!nearest.consider(&hit(5_845_000_000, f32::NAN)));
    assert_eq!(nearest.best().unwrap().0, 5800e6);
    let mut any = ScanSelector::new(&skipped, CandidatePolicy::AnyUnskippedChannel);
    assert!(any.consider(&hit(5_865_000_000, -10.0)));
    assert_eq!(any.fallback_frequency(5865e6), Some(5866e6)); // other nearby channels still allowed
}
#[test]
fn sensitive_cadence_is_configured_by_backend_costs() {
    let policy = SweepPolicy {
        fast: SweepBudget::flat(Duration::from_millis(10), 2),
        sensitive: SweepBudget::flat(Duration::from_millis(50), 6),
        sensitive_every: 4,
        narrow_capture_max_rate_hz: 30.72e6,
    };
    assert_eq!(
        (0..8)
            .map(|n| policy.budget(n, 61.44e6).packets_per_hop)
            .collect::<Vec<_>>(),
        [2, 2, 2, 6, 2, 2, 2, 6]
    );
    assert_eq!(policy.budget(3, 15.36e6).dwell, Duration::from_millis(10));
    let disabled = SweepPolicy {
        sensitive_every: 0,
        ..policy
    };
    assert_eq!(disabled.budget(3, 61.44e6).packets_per_hop, 2);
}
#[test]
fn fine_tuning_rejects_weak_sync_and_uses_rssi_only_near_equal_confidence() {
    let mut selector = FineTuneSelector::default();
    selector.consider(5865e6, 0.05, -40.0);
    assert!(selector.best().is_none());
    selector.consider(5866e6, 0.8, -60.0);
    selector.consider(5860e6, 0.6, -20.0);
    assert_eq!(selector.best().unwrap().0, 5866e6);
    selector.consider(5865e6, 0.79, -40.0);
    assert_eq!(selector.best().unwrap().0, 5865e6);
}
