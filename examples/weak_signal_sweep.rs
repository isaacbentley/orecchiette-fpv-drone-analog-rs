//! Progressive weak-signal characterization: walk a synthetic NTSC
//! signal from clean down to (and under) the noise floor and measure
//! every weak-signal mechanism at each step:
//!
//! - envelope CNR estimate (`levels::estimate_cnr_db`)
//! - single-shot detection (4 independent noise realizations)
//! - cross-batch integrated detection (`SpectralIntegrator`, 4 batches)
//! - discriminator vs PLL demod fidelity
//! - field reconstruction sync quality and GCOR from each demod
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

#[derive(Clone, Copy, Debug)]
enum Impairment {
    Awgn,
    Multipath,
    BurstDropout,
    SlowFade,
    ImpulsiveNoise,
}

fn main() {
    let pll_loop_bw = 2.5e6f32;
    let sample_rate = 15_360_000u32;
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

    // Generate 4 fields so the PLL and phase 1 logic can settle on the first frame
    // and correctly output the second frame.
    let truth = generate_fields(&cfg, 4);
    let clean = generate_iq(&cfg, 4, 0.0);
    let skip = 4_000;

    #[allow(unused_mut)]
    let mut truth_recon = FrameReconstructor::new(sample_rate, false, deviation, true)
        .with_temporal_window(1)
        .with_matched_sync(true)
        .with_line_locked_clock(true);
    let mut truth_frame = vec![0u32; truth_recon.width * truth_recon.height];

    let mut offset = 1;
    for _ in 0..2 {
        if let Some(consumed) =
            truth_recon.reconstruct_frame_into(&truth[offset..], &mut truth_frame)
        {
            offset += consumed;
        } else {
            break;
        }
    }

    println!(
        "NTSC bars, {} MSPS, {} MHz deviation, PLL loop bw {} MHz, 4 fields/batch, sigma per I/Q vs unit carrier",
        sample_rate as f64 / 1e6,
        deviation / 1e6,
        pll_loop_bw / 1e6
    );

    for &impairment in &[
        Impairment::Awgn,
        Impairment::Multipath,
        Impairment::BurstDropout,
        Impairment::SlowFade,
        Impairment::ImpulsiveNoise,
    ] {
        println!("\n=== Profile: {:?} ===", impairment);
        println!(
            "{:>5} | {:>8} | {:>12} | {:>16} | {:>10} {:>10} | {:>5} {:>5} | {:>5} {:>5}",
            "sigma",
            "CNRest",
            "single(4x)",
            "integrated(4)",
            "disc MSE",
            "pll MSE",
            "syncD",
            "syncP",
            "gcorD",
            "gcorP"
        );

        let mut cliff_disc = None;
        let mut cliff_pll = None;

        for &sigma in &[0.0f32, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2] {
            let mut iq = clean.clone();
            let mut seed = 0xA5A5u64;

            match impairment {
                Impairment::Awgn => {}
                Impairment::Multipath => {
                    orecchiette_fpv_drone_analog_rs::impairments::add_multipath(
                        &mut iq,
                        &[(15, 0.5, 1.0, 50.0), (40, 0.2, 0.5, 200.0)],
                        sample_rate as f32,
                    );
                }
                Impairment::BurstDropout => {}
                Impairment::SlowFade => {
                    orecchiette_fpv_drone_analog_rs::impairments::add_slow_fade(
                        &mut iq,
                        sample_rate as f32,
                        60.0,
                        0.8,
                    );
                }
                Impairment::ImpulsiveNoise => {
                    orecchiette_fpv_drone_analog_rs::impairments::add_impulsive_noise(
                        &mut iq, 0.05, 5.0, &mut seed,
                    );
                }
            }
            add_awgn(&mut iq, sigma, &mut seed);
            if matches!(impairment, Impairment::BurstDropout) {
                orecchiette_fpv_drone_analog_rs::impairments::add_burst_dropouts(
                    &mut iq,
                    &[(10000, 2000), (50000, 1000)],
                    sigma,
                    &mut seed,
                );
            }

            let cnr = estimate_cnr_db(&iq)
                .map(|v| format!("{v:5.1} dB"))
                .unwrap_or_else(|| "  n/a".into());

            // single shot detection
            let det = AnalogFpvDetector::default();
            let mut hits = 0;
            let mut best = (SignalType::Unknown, 0.0f32);
            for b in 0..4u64 {
                let mut iq_b = clean.clone();
                let mut s = 0x1111u64.wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
                match impairment {
                    Impairment::Awgn => {}
                    Impairment::Multipath => {
                        orecchiette_fpv_drone_analog_rs::impairments::add_multipath(
                            &mut iq_b,
                            &[(15, 0.5, 1.0, 50.0), (40, 0.2, 0.5, 200.0)],
                            sample_rate as f32,
                        )
                    }
                    Impairment::BurstDropout => {}
                    Impairment::SlowFade => {
                        orecchiette_fpv_drone_analog_rs::impairments::add_slow_fade(
                            &mut iq_b,
                            sample_rate as f32,
                            60.0,
                            0.8,
                        )
                    }
                    Impairment::ImpulsiveNoise => {
                        orecchiette_fpv_drone_analog_rs::impairments::add_impulsive_noise(
                            &mut iq_b, 0.05, 5.0, &mut s,
                        )
                    }
                }
                add_awgn(&mut iq_b, sigma, &mut s);
                if matches!(impairment, Impairment::BurstDropout) {
                    orecchiette_fpv_drone_analog_rs::impairments::add_burst_dropouts(
                        &mut iq_b,
                        &[(10000, 2000), (50000, 1000)],
                        sigma,
                        &mut s,
                    );
                }

                let (st, conf) = det.detect_sync_pulses(&iq_b, sample_rate);
                if st.is_analog_video() {
                    hits += 1;
                    if conf > best.1 {
                        best = (st, conf);
                    }
                }
            }
            let single = format!("{hits}/4 {:.2}", best.1);

            // integrated detection
            let mut integ = SpectralIntegrator::new(4);
            let mut last = (SignalType::Unknown, 0.0f32);
            for b in 0..4u64 {
                let mut iq_b = clean.clone();
                let mut s = 0x1111u64.wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
                match impairment {
                    Impairment::Awgn => {}
                    Impairment::Multipath => {
                        orecchiette_fpv_drone_analog_rs::impairments::add_multipath(
                            &mut iq_b,
                            &[(15, 0.5, 1.0, 50.0), (40, 0.2, 0.5, 200.0)],
                            sample_rate as f32,
                        )
                    }
                    Impairment::BurstDropout => {}
                    Impairment::SlowFade => {
                        orecchiette_fpv_drone_analog_rs::impairments::add_slow_fade(
                            &mut iq_b,
                            sample_rate as f32,
                            60.0,
                            0.8,
                        )
                    }
                    Impairment::ImpulsiveNoise => {
                        orecchiette_fpv_drone_analog_rs::impairments::add_impulsive_noise(
                            &mut iq_b, 0.05, 5.0, &mut s,
                        )
                    }
                }
                add_awgn(&mut iq_b, sigma, &mut s);
                if matches!(impairment, Impairment::BurstDropout) {
                    orecchiette_fpv_drone_analog_rs::impairments::add_burst_dropouts(
                        &mut iq_b,
                        &[(10000, 2000), (50000, 1000)],
                        sigma,
                        &mut s,
                    );
                }
                last = det.detect_sync_pulses_integrated(
                    &iq_b,
                    sample_rate,
                    5_800_000_000,
                    &mut integ,
                );
            }
            let tag = match last.0 {
                SignalType::AnalogVideoNtsc => "NTSC",
                SignalType::AnalogVideoPal => "PAL ",
                SignalType::AnalogVideoUnknown => "Vid?",
                _ => "none",
            };
            let integrated = format!("{tag} {:.2}", last.1);

            // demod fidelity
            let disc_out = fm_demod(&iq);
            let disc_mse = {
                let shifted: Vec<f32> = truth[1..].to_vec();
                mse(&disc_out, &shifted, skip)
            };

            let mut pll = PllFmDemod::new(sample_rate, pll_loop_bw, deviation * 1.2);
            let mut pll_out = Vec::new();
            pll.process_into(&iq, &mut pll_out);
            let pll_mse = mse(&pll_out, &truth, skip);

            // reconstruction
            let eval_demod = |demod: &[f32], phase1: bool, name: &str| -> (String, f64) {
                #[allow(unused_mut)]
                let mut recon = FrameReconstructor::new(sample_rate, false, deviation, false)
                    .with_temporal_window(1)
                    .with_matched_sync(phase1)
                    .with_line_locked_clock(phase1)
                    .with_smart_doc(true, 0.5);
                #[cfg(feature = "neural-vsr")]
                {
                    recon =
                        recon.with_neural_restorer("models/temporal_quantized_trained.onnx", false);
                }
                let mut frame = vec![0u32; recon.width * recon.height];
                let mut last_res = None;

                let mut offset = 0;
                let mut processed_any = false;
                for _ in 0..2 {
                    if let Some(consumed) =
                        recon.reconstruct_frame_into(&demod[offset..], &mut frame)
                    {
                        offset += consumed;
                        processed_any = true;
                    } else {
                        break;
                    }
                }

                if processed_any {
                    // Dump frames for sigma=0.5
                    if (sigma - 0.5).abs() < 1e-3 {
                        let _ = std::fs::create_dir_all("target/frames");
                        use std::io::Write;
                        if let Ok(mut f) = std::fs::File::create(format!(
                            "target/frames/{:?}_{}_{}.ppm",
                            impairment, sigma, name
                        )) {
                            writeln!(f, "P3\n{} {}\n255", recon.width, recon.height).unwrap();
                            for &px in &frame {
                                writeln!(
                                    f,
                                    "{} {} {}",
                                    (px >> 16) & 0xFF,
                                    (px >> 8) & 0xFF,
                                    px & 0xFF
                                )
                                .unwrap();
                            }
                        }
                    }

                    let grad_corr =
                        orecchiette_fpv_drone_analog_rs::metrics::compute_gradient_correlation(
                            &frame,
                            &truth_frame,
                            recon.width,
                            recon.height,
                        );
                    last_res = Some((format!("{:.2}", recon.latest_sync_quality()), grad_corr));
                }

                last_res.unwrap_or_else(|| ("  — ".into(), 0.0))
            };

            let (sync_disc, gcor_disc) = eval_demod(&disc_out, true, "Disc");
            let (sync_pll, gcor_pll) = eval_demod(&pll_out, true, "PLL");

            // Also evaluate truth to generate the truth image
            let _ = eval_demod(&truth[1..], true, "Truth");

            if gcor_disc < 0.5 && cliff_disc.is_none() {
                cliff_disc = Some(sigma);
            }
            if gcor_pll < 0.5 && cliff_pll.is_none() {
                cliff_pll = Some(sigma);
            }

            println!(
                "{sigma:5.2} | {cnr} | {single:>12} | {integrated:>16} | {disc_mse:10.2e} {pll_mse:10.2e} | {sync_disc:>5} {sync_pll:>5} | {gcor_disc:5.2} {gcor_pll:5.2}"
            );
        }

        println!("---");
        println!(
            "Disc σ-cliff: {}",
            cliff_disc
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| ">2.2".into())
        );
        println!(
            "PLL  σ-cliff: {}",
            cliff_pll
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| ">2.2".into())
        );
    }
}
