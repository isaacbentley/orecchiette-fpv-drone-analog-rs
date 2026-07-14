//! End-to-end GPU-vs-CPU equivalence for the wideband sweep (GPU Phase
//! 2). The stage-level check (GPU's batched DDC vs `ddc_and_decimate`
//! directly) lives in `detector.rs`'s own test module
//! (`gpu_ddc_matches_cpu_ddc_and_decimate`) since it needs access to
//! private helpers; this file checks the thing a caller actually
//! observes — `FpvDetector::detect_from_iq` — is unaffected by turning
//! the `gpu` feature on.
//!
//! Both tests skip gracefully with no GPU adapter. Detection is
//! threshold-based (not a bit-exact recipe like the DJI crate's CRC
//! decode), so tolerance-based parity is the right bar here: matching
//! `SignalType` and a frequency within one 5 MHz probe step.

#![cfg(feature = "gpu")]

use num_complex::Complex;
use orecchiette_fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
use orecchiette_fpv_drone_analog_rs::gpu::GpuAnalog;
use orecchiette_fpv_drone_analog_rs::types::SignalType;
use std::f32::consts::PI;
use std::sync::Arc;

/// Mirrors `detection_test.rs`'s `make_fm_sync_iq` — kept as a local
/// copy since integration test binaries don't share code across files
/// without a `tests/common/mod.rs`, and this crate doesn't have one yet.
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

#[test]
fn gpu_and_cpu_detectors_agree_on_wideband_offset_signal() {
    let Some(gpu) = GpuAnalog::try_new() else {
        eprintln!(
            "No GPU adapter found; skipping gpu_and_cpu_detectors_agree_on_wideband_offset_signal"
        );
        return;
    };

    // Same fixture as detection_test.rs's
    // detect_from_iq_wideband_sliding_ddc_finds_offset_signal: 50 MSPS,
    // NTSC line rate, signal parked at -15 MHz.
    let sample_rate = 50_000_000u32;
    let n = 262_144;
    let offset_hz = -15_000_000.0f32;
    let iq = make_fm_sync_iq(sample_rate, n, 15734.0, 5_000_000.0, offset_hz);
    let center_freq = 5_800_000_000u64;

    let cpu_detector = AnalogFpvDetector::default();
    let gpu_detector = AnalogFpvDetector::with_gpu(Arc::new(gpu));

    let cpu_results = cpu_detector.detect_from_iq(&iq, center_freq, sample_rate);
    let gpu_results = gpu_detector.detect_from_iq(&iq, center_freq, sample_rate);

    assert!(
        !cpu_results.is_empty(),
        "CPU path missed the offset signal (fixture sanity check)"
    );
    assert!(
        !gpu_results.is_empty(),
        "GPU path missed the offset signal that the CPU path found"
    );

    let cpu_hit = &cpu_results[0];
    let gpu_hit = &gpu_results[0];

    assert_eq!(
        cpu_hit.signal_type, gpu_hit.signal_type,
        "GPU and CPU disagree on signal type: cpu={:?} gpu={:?}",
        cpu_hit.signal_type, gpu_hit.signal_type
    );
    assert_ne!(gpu_hit.signal_type, SignalType::Unknown);

    // Within one 5 MHz probe step of each other.
    let diff_mhz = (cpu_hit.frequency_hz as f64 - gpu_hit.frequency_hz as f64).abs() / 1e6;
    assert!(
        diff_mhz < 5.0,
        "GPU frequency {} too far from CPU frequency {} (diff {} MHz)",
        gpu_hit.frequency_hz,
        cpu_hit.frequency_hz,
        diff_mhz
    );
}

#[test]
fn gpu_and_cpu_detectors_agree_on_two_signal_capture() {
    let Some(gpu) = GpuAnalog::try_new() else {
        eprintln!(
            "No GPU adapter found; skipping gpu_and_cpu_detectors_agree_on_two_signal_capture"
        );
        return;
    };

    // Two independent PAL/NTSC signals far enough apart (40 MHz) that
    // clustering keeps them as separate detections, exercising the
    // multi-probe batching path with more than one accepted probe.
    let sample_rate = 80_000_000u32;
    let n = 524_288;
    let mut iq = make_fm_sync_iq(sample_rate, n, 15625.0, 4_000_000.0, -20_000_000.0);
    let iq2 = make_fm_sync_iq(sample_rate, n, 15734.0, 4_000_000.0, 20_000_000.0);
    for (a, b) in iq.iter_mut().zip(iq2.iter()) {
        *a += *b;
    }
    let center_freq = 5_800_000_000u64;

    let cpu_detector = AnalogFpvDetector::default();
    let gpu_detector = AnalogFpvDetector::with_gpu(Arc::new(gpu));

    let mut cpu_results = cpu_detector.detect_from_iq(&iq, center_freq, sample_rate);
    let mut gpu_results = gpu_detector.detect_from_iq(&iq, center_freq, sample_rate);
    cpu_results.sort_by_key(|r| r.frequency_hz);
    gpu_results.sort_by_key(|r| r.frequency_hz);

    assert_eq!(
        cpu_results.len(),
        gpu_results.len(),
        "GPU and CPU found a different number of signals: cpu={:?} gpu={:?}",
        cpu_results
            .iter()
            .map(|r| r.frequency_hz)
            .collect::<Vec<_>>(),
        gpu_results
            .iter()
            .map(|r| r.frequency_hz)
            .collect::<Vec<_>>()
    );
    for (c, g) in cpu_results.iter().zip(gpu_results.iter()) {
        assert_eq!(c.signal_type, g.signal_type);
        let diff_mhz = (c.frequency_hz as f64 - g.frequency_hz as f64).abs() / 1e6;
        assert!(
            diff_mhz < 5.0,
            "GPU/CPU frequency mismatch: cpu={} gpu={} diff={} MHz",
            c.frequency_hz,
            g.frequency_hz,
            diff_mhz
        );
    }
}
