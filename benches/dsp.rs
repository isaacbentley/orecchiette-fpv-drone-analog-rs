//! DSP hot-path benches, so "optimized" is an enforced number rather
//! than a review opinion. Run with `cargo bench`.
//!
//! Baselines (Apple silicon, 2026-08): `detect_from_iq` on a 65 k
//! chunk runs in single-digit ms at both rates; a reconstructed field
//! at 15.36 MSPS lands well under its own 16.7 ms real-time budget.

use criterion::{Criterion, criterion_group, criterion_main};
use orecchiette_fpv_drone_analog_rs::detector::{
    AnalogFpvDetector, FpvDetector, SpectralIntegrator,
};
use orecchiette_fpv_drone_analog_rs::synthetic::{
    SyntheticVideoConfig, TestPattern, generate_fields, generate_iq,
};
use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;
use orecchiette_fpv_drone_analog_rs::video::FrameReconstructor;
use std::hint::black_box;

fn cfg(sample_rate: u32) -> SyntheticVideoConfig {
    SyntheticVideoConfig {
        sample_rate,
        is_pal: false,
        deviation_hz: 5e6,
        pattern: TestPattern::Bars,
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    }
}

fn bench_detection(c: &mut Criterion) {
    // The live capture shape: one 65,536-sample chunk.
    for sample_rate in [25_000_000u32, 61_440_000] {
        let iq_full = generate_iq(&cfg(sample_rate), 2, 0.0);
        let chunk = &iq_full[..65_536];
        let label = format!("detect_from_iq_{}msps_65k", sample_rate / 1_000_000);
        c.bench_function(&label, |b| {
            let det = AnalogFpvDetector::default();
            b.iter(|| det.detect_from_iq(black_box(chunk), 5_800_000_000, sample_rate));
        });
        // Integrated variant — classifies every probe, so this bounds
        // the cost of the sensitivity feature.
        let label = format!(
            "detect_from_iq_integrated_{}msps_65k",
            sample_rate / 1_000_000
        );
        c.bench_function(&label, |b| {
            let det = AnalogFpvDetector::default();
            let mut integ = SpectralIntegrator::new(4);
            b.iter(|| {
                det.detect_from_iq_integrated(
                    black_box(chunk),
                    5_800_000_000,
                    sample_rate,
                    &mut integ,
                )
            });
        });
    }
}

fn bench_reconstruction(c: &mut Criterion) {
    let sample_rate = 15_360_000u32;
    let data = generate_fields(&cfg(sample_rate), 3);
    c.bench_function("reconstruct_field_ntsc_15msps", |b| {
        let mut recon = FrameReconstructor::new(sample_rate, false, 5e6, false);
        let mut frame = vec![0u32; recon.width * recon.height];
        b.iter(|| {
            let _ = recon.reconstruct_frame_into(black_box(&data), &mut frame);
        });
    });
}

criterion_group!(benches, bench_detection, bench_reconstruction);
criterion_main!(benches);
