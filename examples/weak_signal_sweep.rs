//! Progressive weak-signal characterization: walk a synthetic NTSC
//! signal from clean down to (and under) the noise floor and measure
//! every weak-signal mechanism at each step:
//!
//! - envelope CNR estimate (`levels::estimate_cnr_db`)
//! - single-shot detection (4 independent noise realizations)
//! - cross-batch integrated detection (`SpectralIntegrator`, 4 batches)
//! - discriminator vs PLL demod fidelity (MSE against the known
//!   baseband, and the PLL's threshold-extension gain in dB)
//! - field reconstruction sync quality from the discriminator demod
//!
//! Run with:
//!   cargo run --release --example weak_signal_sweep
//!
//! Deterministic (fixed xorshift seeds), so numbers are comparable
//! across runs and machines up to FP reassociation.

use num_complex::Complex;
use orecchiette_fpv_drone_analog_rs::demod::{PllFmDemod, fm_demod};
use orecchiette_fpv_drone_analog_rs::detector::{AnalogFpvDetector, SpectralIntegrator};
use orecchiette_fpv_drone_analog_rs::levels::estimate_cnr_db;
use orecchiette_fpv_drone_analog_rs::synthetic::{
    SyntheticVideoConfig, TestPattern, generate_fields, generate_iq,
};
use orecchiette_fpv_drone_analog_rs::types::SignalType;
use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;
use orecchiette_fpv_drone_analog_rs::video::FrameReconstructor;

fn gauss(s: &mut u64) -> f32 {
    let mut acc = 0.0f32;
    for _ in 0..12 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        acc += (*s >> 11) as f32 / (1u64 << 53) as f32;
    }
    acc - 6.0
}

fn add_awgn(iq: &mut [Complex<f32>], sigma: f32, seed: &mut u64) {
    if sigma <= 0.0 {
        return;
    }
    for z in iq.iter_mut() {
        z.re += sigma * gauss(seed);
        z.im += sigma * gauss(seed);
    }
}

fn mse(a: &[f32], truth: &[f32], skip: usize) -> f64 {
    let n = a.len().min(truth.len());
    let mut acc = 0.0f64;
    let mut count = 0u64;
    for i in skip..n {
        let d = (a[i] - truth[i]) as f64;
        acc += d * d;
        count += 1;
    }
    acc / count.max(1) as f64
}

fn main() {
    // Optional args: PLL loop bandwidth in Hz (default 2.5e6) and
    // sample rate in Hz (default 15.36e6), e.g.
    //   cargo run --release --example weak_signal_sweep -- 1.0e6 25e6
    // NOTE: the PLL clamps its normalised natural frequency to 0.5
    // rad/sample, so the effective loop bandwidth ceiling is
    // ~fs/(4π) ≈ fs · 0.08 — requests above that are all the same loop.
    let pll_loop_bw: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.5e6);
    let sample_rate: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v as u32)
        .unwrap_or(15_360_000);
    let deviation = 5.0e6f32;
    let cfg = SyntheticVideoConfig {
        sample_rate,
        is_pal: false,
        deviation_hz: deviation,
        pattern: TestPattern::Bars,
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };
    let truth = generate_fields(&cfg, 2);
    let clean = generate_iq(&cfg, 2, 0.0);
    let skip = 4_000; // demod/PLL settling

    println!(
        "NTSC bars, {} MSPS, {} MHz deviation, PLL loop bw {} MHz, 2 fields/batch, sigma per I/Q vs unit carrier",
        sample_rate as f64 / 1e6,
        deviation / 1e6,
        pll_loop_bw / 1e6
    );
    println!(
        "{:>5} | {:>8} | {:>12} | {:>16} | {:>10} {:>10} {:>7} | {:>6} {:>6}",
        "sigma",
        "CNRest",
        "single(4x)",
        "integrated(4)",
        "disc MSE",
        "pll MSE",
        "gain",
        "syncD",
        "syncP"
    );

    for &sigma in &[0.0f32, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2] {
        // ── CNR estimate on one realization ──
        let mut iq = clean.clone();
        let mut seed = 0xA5A5u64;
        add_awgn(&mut iq, sigma, &mut seed);
        let cnr = estimate_cnr_db(&iq)
            .map(|v| format!("{v:5.1} dB"))
            .unwrap_or_else(|| "  n/a".into());

        // ── single-shot detection over 4 independent realizations ──
        let det = AnalogFpvDetector::default();
        let mut hits = 0;
        let mut best = (SignalType::Unknown, 0.0f32);
        for b in 0..4u64 {
            let mut iq_b = clean.clone();
            let mut s = 0x1111u64.wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
            add_awgn(&mut iq_b, sigma, &mut s);
            let (st, conf) = det.detect_sync_pulses(&iq_b, sample_rate);
            if st.is_analog_video() {
                hits += 1;
                if conf > best.1 {
                    best = (st, conf);
                }
            }
        }
        let single = format!("{hits}/4 {:.2}", best.1);

        // ── integrated detection across the same 4 batches ──
        let mut integ = SpectralIntegrator::new(4);
        let mut last = (SignalType::Unknown, 0.0f32);
        for b in 0..4u64 {
            let mut iq_b = clean.clone();
            let mut s = 0x1111u64.wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
            add_awgn(&mut iq_b, sigma, &mut s);
            last = det.detect_sync_pulses_integrated(&iq_b, sample_rate, 5_800_000_000, &mut integ);
        }
        let tag = match last.0 {
            SignalType::AnalogVideoNtsc => "NTSC",
            SignalType::AnalogVideoPal => "PAL ",
            SignalType::AnalogVideoUnknown => "Vid?",
            _ => "none",
        };
        let integrated = format!("{tag} {:.2}", last.1);

        // ── demod fidelity: discriminator vs PLL ──
        // disc[j] estimates the same transition truth[j+1] carries;
        // the PLL's output aligns with truth[i] directly (see the PLL
        // equivalence test).
        let disc_out = fm_demod(&iq);
        let disc_mse = {
            let shifted: Vec<f32> = truth[1..].to_vec();
            mse(&disc_out, &shifted, skip)
        };
        let mut pll = PllFmDemod::new(sample_rate, pll_loop_bw, deviation * 1.2);
        let mut pll_out = Vec::new();
        pll.process_into(&iq, &mut pll_out);
        let pll_mse = mse(&pll_out, &truth, skip);
        let gain_db = 10.0 * (disc_mse / pll_mse.max(1e-18)).log10();

        // ── reconstruction from each demod: the system-level metric.
        // Raw MSE flatters the PLL (its loop inherently band-limits;
        // the discriminator's output carries full-bandwidth noise), so
        // whether sync survives deeper into the noise is the honest
        // comparison.
        let sync_q_of = |demod: &[f32]| -> String {
            let mut recon = FrameReconstructor::new(sample_rate, false, deviation, false)
                .with_temporal_window(1);
            let mut frame = vec![0u32; recon.width * recon.height];
            match recon.reconstruct_frame_into(demod, &mut frame) {
                Some(_) => format!("{:.2}", recon.latest_sync_quality()),
                None => "  — ".into(),
            }
        };
        let sync_disc = sync_q_of(&disc_out);
        let sync_pll = sync_q_of(&pll_out);

        println!(
            "{sigma:>5.1} | {cnr:>8} | {single:>12} | {integrated:>16} | {disc_mse:>10.2e} {pll_mse:>10.2e} {gain_db:>+6.1}d | {sync_disc:>6} {sync_pll:>6}"
        );
    }
}
