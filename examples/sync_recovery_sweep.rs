//! Offline sweep runner evaluating baseline NTSC/PAL timing recovery,
//! line-position error, clock mismatch, and impairment resilience.
//!
//! Run with:
//!   cargo run --release --example sync_recovery_sweep

use std::time::Instant;

use orecchiette_fpv_drone_analog_rs::synthetic::{
    ExtendedSyntheticConfig, ImpairmentSchedule, SyntheticVideoConfig, TestPattern,
    generate_extended_fixture,
};
use orecchiette_fpv_drone_analog_rs::timing::{DecodeStep, TimedDemodSlice};
use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;
use orecchiette_fpv_drone_analog_rs::video::FrameReconstructor;

#[derive(Debug, Clone)]
struct Scenario {
    name: &'static str,
    is_pal: bool,
    sample_rate: u32,
    clock_error_ppm: f64,
    erase_vbi_fields: Vec<usize>,
    erase_hsync_pulses: Vec<usize>,
    noise_sigma: f32,
    n_fields: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct SweepReport {
    commit_analog: String,
    commit_viewer: Option<String>,
    target_os: String,
    target_arch: String,
    generator_configuration: GeneratorProvenance,
    results: Vec<ScenarioResult>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct GeneratorProvenance {
    test_pattern: &'static str,
    deviation_hz: f32,
    seed_derivation: &'static str,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ScenarioResult {
    name: &'static str,
    is_pal: bool,
    sample_rate: u32,
    clock_error_ppm: f64,
    fields_generated: usize,
    fields_decoded: usize,
    first_field_sample: Option<usize>,
    mean_sync_quality: f32,
    p95_h_error_tbc_samples: f64,
    signal_duration_s: f64,
    cpu_time_s: f64,
    cpu_ratio: f64, // cpu_time / signal_duration
}

fn run_scenario(sc: &Scenario) -> ScenarioResult {
    let base = SyntheticVideoConfig {
        sample_rate: sc.sample_rate,
        is_pal: sc.is_pal,
        deviation_hz: 3_000_000.0,
        pattern: TestPattern::Flat(50.0),
        start_field: FieldParity::First,
        noise_sigma: sc.noise_sigma,
        dc_offset: 0.0,
    };
    let impairments = ImpairmentSchedule {
        erase_vbi_fields: sc.erase_vbi_fields.clone(),
        erase_hsync_pulses: sc.erase_hsync_pulses.clone(),
        ..Default::default()
    };

    let cfg = ExtendedSyntheticConfig::new(base)
        .with_clock_error_ppm(sc.clock_error_ppm)
        .with_impairments(impairments);

    let fixture = generate_extended_fixture(&cfg, sc.n_fields);
    let signal_duration_s = fixture.demod.len() as f64 / sc.sample_rate as f64;

    let mut recon = FrameReconstructor::new(sc.sample_rate, sc.is_pal, 3_000_000.0, false);
    let mut frame = vec![0u32; recon.width * recon.height];

    // Gating threshold matching production worker in fpv-viewer-rs:
    // ~13 ms at 15.36 MSPS, requiring enough samples before asking reconstructor to commit.
    let min_samples_per_field = {
        let line_rate = if sc.is_pal {
            orecchiette_fpv_drone_analog_rs::timing::PAL_NOMINAL_LINE_HZ
        } else {
            orecchiette_fpv_drone_analog_rs::timing::NTSC_NOMINAL_LINE_HZ
        };
        let lines = (if sc.is_pal { 288 } else { 240 } + 22) as f64;
        ((sc.sample_rate as f64 / line_rate) * lines) as usize
    };

    // Realistic SDR / pipeline streaming chunk size (~16 ms) and production buffer bounds
    let chunk_size = (sc.sample_rate as usize / 60).max(16_384);
    let live_region_cap = sc.sample_rate as usize / 12; // ~83 ms live cap
    let keep_on_skip = sc.sample_rate as usize / 60; // ~16 ms kept on safety skip

    let start_time = Instant::now();
    let mut consumed_cursor = 0usize;
    let mut available_samples = chunk_size.min(fixture.demod.len());
    let mut fields_decoded = 0usize;
    let mut sync_qualities = Vec::new();
    let mut first_field_sample = None;
    let mut h_errors = Vec::new();
    let mut is_discontinuous = false;

    while consumed_cursor < fixture.demod.len() {
        // Attempt reconstruction on currently available streaming input
        while available_samples.saturating_sub(consumed_cursor) >= min_samples_per_field {
            let slice_data = &fixture.demod[consumed_cursor..available_samples];
            let timed_slice = TimedDemodSlice::new(
                slice_data,
                consumed_cursor as u64,
                sc.sample_rate,
                is_discontinuous,
            );
            is_discontinuous = false;

            match recon.reconstruct_timed_into(timed_slice, &mut frame) {
                Ok(DecodeStep::NeedMoreData { .. }) => break,
                Ok(DecodeStep::Advance {
                    consumed_samples,
                    field,
                }) => {
                    let field_consumed_cursor = consumed_cursor;
                    consumed_cursor += consumed_samples;

                    if let Some(_timing) = field {
                        if first_field_sample.is_none() {
                            first_field_sample = Some(field_consumed_cursor);
                        }
                        fields_decoded += 1;
                        sync_qualities.push(recon.latest_sync_quality());

                        // Evaluate horizontal timing error across all rendered lines,
                        // matching measurements strictly by ground-truth field and line identity.
                        let sync_positions = recon.latest_sync_positions();
                        if !sync_positions.is_empty() {
                            let field_active_start =
                                field_consumed_cursor as f64 + sync_positions[0] as f64;
                            if let Some(gt_field) =
                                fixture.ground_truth_fields.iter().min_by(|a, b| {
                                    (a.active_video_start_sample as f64 - field_active_start)
                                        .abs()
                                        .partial_cmp(
                                            &(b.active_video_start_sample as f64
                                                - field_active_start)
                                                .abs(),
                                        )
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                            {
                                let field_idx = gt_field.field_index;
                                let base_active_lines = if sc.is_pal {
                                    orecchiette_fpv_drone_analog_rs::vbi::consts::PAL_BASE_ACTIVE_START_LINES
                                } else {
                                    orecchiette_fpv_drone_analog_rs::vbi::consts::NTSC_BASE_ACTIVE_START_LINES
                                };
                                let active_start_lines = base_active_lines
                                    + if gt_field.parity == FieldParity::Second {
                                        0.5
                                    } else {
                                        0.0
                                    };

                                // Scale factor converting input-sample errors to TBC output-sample units (pixels)
                                let tbc_scale = recon.line_width as f64 / recon.line_period as f64;

                                for (row, &pos) in sync_positions.iter().enumerate() {
                                    let expected_line = active_start_lines + row as f64;
                                    if let Some(pulse) = fixture.ground_truth_pulses.iter().find(|p| {
                                        p.field_index == field_idx
                                            && p.kind == orecchiette_fpv_drone_analog_rs::vbi::PulseKind::Horizontal
                                            && (p.line_in_field - expected_line).abs() < 1e-4
                                    }) {
                                        let decoded_sample = field_consumed_cursor as f64 + pos as f64;
                                        let err_input = (decoded_sample - pulse.center_sample).abs();
                                        let err_tbc = err_input * tbc_scale;
                                        h_errors.push(err_tbc);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    let fallback = recon.samples_per_line;
                    consumed_cursor += fallback;
                    break;
                }
            }
        }

        if available_samples < fixture.demod.len() {
            // Simulate stream progression: next chunk arrives
            available_samples = (available_samples + chunk_size).min(fixture.demod.len());
        } else {
            // End of fixture stream reached; check if unconsumed region exceeds live capacity
            if available_samples.saturating_sub(consumed_cursor) > live_region_cap {
                consumed_cursor = available_samples.saturating_sub(keep_on_skip);
            } else {
                break;
            }
        }
    }
    let cpu_time_s = start_time.elapsed().as_secs_f64();

    h_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_h_error_tbc_samples = if !h_errors.is_empty() {
        let idx = ((h_errors.len() as f64 * 0.95) as usize).min(h_errors.len() - 1);
        h_errors[idx]
    } else {
        f64::NAN
    };

    let mean_sync_quality = if !sync_qualities.is_empty() {
        sync_qualities.iter().sum::<f32>() / sync_qualities.len() as f32
    } else {
        0.0
    };

    let cpu_ratio = if signal_duration_s > 0.0 {
        cpu_time_s / signal_duration_s
    } else {
        0.0
    };

    ScenarioResult {
        name: sc.name,
        is_pal: sc.is_pal,
        sample_rate: sc.sample_rate,
        clock_error_ppm: sc.clock_error_ppm,
        fields_generated: sc.n_fields,
        fields_decoded,
        first_field_sample,
        mean_sync_quality,
        p95_h_error_tbc_samples,
        signal_duration_s,
        cpu_time_s,
        cpu_ratio,
    }
}

fn main() {
    let commit_analog = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let commit_viewer = std::process::Command::new("git")
        .args(["-C", "../fpv-viewer-rs", "rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    println!("════════════════════════════════════════════════════════════════════════════════");
    println!(" NTSC/PAL Sync Recovery Baseline Sweep");
    println!(
        " Commit Analog: {} | Commit Viewer: {} | Seed Formula: 0x1234_5678_9abc_def0 ^ ...",
        commit_analog,
        commit_viewer.as_deref().unwrap_or("n/a")
    );
    println!("════════════════════════════════════════════════════════════════════════════════");

    let scenarios = vec![
        // Clean reference across rates
        Scenario {
            name: "clean_ntsc_15m36",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        Scenario {
            name: "clean_pal_15m36",
            is_pal: true,
            sample_rate: 15_360_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        Scenario {
            name: "clean_ntsc_20m",
            is_pal: false,
            sample_rate: 20_000_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        Scenario {
            name: "clean_ntsc_25m",
            is_pal: false,
            sample_rate: 25_000_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        Scenario {
            name: "clean_ntsc_30m72",
            is_pal: false,
            sample_rate: 30_720_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        // Clock mismatch
        Scenario {
            name: "clock_mismatch_plus_50ppm",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: 50.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        Scenario {
            name: "clock_mismatch_minus_50ppm",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: -50.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        Scenario {
            name: "clock_mismatch_plus_100ppm",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: 100.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        // Missing VBI (characterizes current baseline drop)
        Scenario {
            name: "missing_vbi_1field",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![2],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        Scenario {
            name: "missing_vbi_2fields",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![2, 3],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        // Missing H-sync pulses
        Scenario {
            name: "missing_hsync_4pulses",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![50, 51, 52, 53],
            noise_sigma: 0.0,
            n_fields: 6,
        },
        Scenario {
            name: "missing_hsync_16pulses",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: (50..66).collect(),
            noise_sigma: 0.0,
            n_fields: 6,
        },
        // AWGN noise
        Scenario {
            name: "noise_sigma_0_15",
            is_pal: false,
            sample_rate: 15_360_000,
            clock_error_ppm: 0.0,
            erase_vbi_fields: vec![],
            erase_hsync_pulses: vec![],
            noise_sigma: 0.15,
            n_fields: 6,
        },
    ];

    println!(
        "{:<30} | {:>6} | {:>6} | {:>8} | {:>9} | {:>8}",
        "Scenario", "Gen", "Dec", "SyncQ", "P95 TBCpx", "CPU s/s"
    );
    println!(
        "{:-<30}-+-{:-<6}-+-{:-<6}-+-{:-<8}-+-{:-<9}-+-{:-<8}",
        "", "", "", "", "", ""
    );

    let mut results = Vec::new();
    for sc in &scenarios {
        let res = run_scenario(sc);
        println!(
            "{:<30} | {:>6} | {:>6} | {:>8.3} | {:>9.2} | {:>8.4}",
            res.name,
            res.fields_generated,
            res.fields_decoded,
            res.mean_sync_quality,
            res.p95_h_error_tbc_samples,
            res.cpu_ratio
        );
        results.push(res);
    }

    let report = SweepReport {
        commit_analog,
        commit_viewer,
        target_os: std::env::consts::OS.to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
        generator_configuration: GeneratorProvenance {
            test_pattern: "Flat(50.0)",
            deviation_hz: 3_000_000.0,
            seed_derivation: "0x1234_5678_9abc_def0 ^ (out_len * 0x9e3779b97f4a7c15)",
        },
        results,
    };

    println!("\n════════════════════════════════════════════════════════════════════════════════");
    println!(" Structured JSON Output");
    println!("════════════════════════════════════════════════════════════════════════════════");
    let json_output = serde_json::to_string_pretty(&report).expect("Valid JSON serialization");
    println!("{}", json_output);
}
