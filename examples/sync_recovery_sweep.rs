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

#[derive(Debug, Clone)]
struct ScenarioResult {
    name: &'static str,
    is_pal: bool,
    sample_rate: u32,
    clock_error_ppm: f64,
    fields_generated: usize,
    fields_decoded: usize,
    first_field_sample: Option<usize>,
    mean_sync_quality: f32,
    p95_h_error_samples: f64,
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

    // Decode in field-sized slices to simulate streaming buffer arrival
    let samples_per_field = (sc.sample_rate as f64 / if sc.is_pal { 15625.0 } else { 15734.2657 }
        * (if sc.is_pal { 312.5 } else { 262.5 })) as usize;
    let slice_window = samples_per_field + (samples_per_field / 10);

    let start_time = Instant::now();
    let mut cursor = 0usize;
    let mut fields_decoded = 0usize;
    let mut sync_qualities = Vec::new();
    let mut first_field_sample = None;
    let mut h_errors = Vec::new();

    while cursor < fixture.demod.len() {
        let end = (cursor + slice_window).min(fixture.demod.len());
        let slice = &fixture.demod[cursor..end];
        if slice.len() < samples_per_field {
            break;
        }

        if let Some(consumed) = recon.reconstruct_frame_into(slice, &mut frame) {
            if first_field_sample.is_none() {
                first_field_sample = Some(cursor);
            }
            fields_decoded += 1;
            sync_qualities.push(recon.latest_sync_quality());

            // Measure horizontal position error: compare reconstructor's sync phase against
            // the nearest ground truth H-sync pulse in this field
            let decoded_sync_phase = cursor as f64 + recon.sync_phase as f64;
            if let Some(nearest_pulse) = fixture
                .ground_truth_pulses
                .iter()
                .filter(|p| p.kind == orecchiette_fpv_drone_analog_rs::vbi::PulseKind::Horizontal)
                .min_by(|a, b| {
                    (a.center_sample - decoded_sync_phase)
                        .abs()
                        .partial_cmp(&(b.center_sample - decoded_sync_phase).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let err = (nearest_pulse.center_sample - decoded_sync_phase).abs();
                h_errors.push(err);
            }

            cursor += consumed.max(samples_per_field / 2);
        } else {
            // Advance by half a field if no frame decoded
            cursor += samples_per_field / 2;
        }
    }
    let cpu_time_s = start_time.elapsed().as_secs_f64();

    h_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_h_error_samples = if !h_errors.is_empty() {
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
        p95_h_error_samples,
        signal_duration_s,
        cpu_time_s,
        cpu_ratio,
    }
}

fn main() {
    println!("════════════════════════════════════════════════════════════════════════════════");
    println!(" NTSC/PAL Sync Recovery Baseline Sweep");
    println!(" Commit Analog: 5c9232b | Commit Viewer: bf93c8c | Seed: 42");
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
        "{:<30} | {:>6} | {:>6} | {:>8} | {:>8} | {:>8}",
        "Scenario", "Gen", "Dec", "SyncQ", "P95 H-Err", "CPU s/s"
    );
    println!(
        "{:-<30}-+-{:-<6}-+-{:-<6}-+-{:-<8}-+-{:-<8}-+-{:-<8}",
        "", "", "", "", "", ""
    );

    let mut results = Vec::new();
    for sc in &scenarios {
        let res = run_scenario(sc);
        println!(
            "{:<30} | {:>6} | {:>6} | {:>8.3} | {:>8.2} | {:>8.4}",
            res.name,
            res.fields_generated,
            res.fields_decoded,
            res.mean_sync_quality,
            res.p95_h_error_samples,
            res.cpu_ratio
        );
        results.push(res);
    }

    println!("\n════════════════════════════════════════════════════════════════════════════════");
    println!(" Structured JSON Output");
    println!("════════════════════════════════════════════════════════════════════════════════");

    println!("{{");
    println!("  \"commit_analog\": \"5c9232b\",");
    println!("  \"commit_viewer\": \"bf93c8c\",");
    println!("  \"seed\": 42,");
    println!("  \"target_os\": {:?},", std::env::consts::OS);
    println!("  \"target_arch\": {:?},", std::env::consts::ARCH);
    println!("  \"results\": [");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        println!("    {{");
        println!("      \"name\": {:?},", r.name);
        println!("      \"is_pal\": {},", r.is_pal);
        println!("      \"sample_rate\": {},", r.sample_rate);
        println!("      \"clock_error_ppm\": {},", r.clock_error_ppm);
        println!("      \"fields_generated\": {},", r.fields_generated);
        println!("      \"fields_decoded\": {},", r.fields_decoded);
        println!("      \"first_field_sample\": {:?},", r.first_field_sample);
        println!("      \"mean_sync_quality\": {:.4},", r.mean_sync_quality);
        println!(
            "      \"p95_h_error_samples\": {:.4},",
            if r.p95_h_error_samples.is_nan() {
                -1.0
            } else {
                r.p95_h_error_samples
            }
        );
        println!("      \"signal_duration_s\": {:.4},", r.signal_duration_s);
        println!("      \"cpu_time_s\": {:.4},", r.cpu_time_s);
        println!("      \"cpu_ratio\": {:.5}", r.cpu_ratio);
        println!("    }}{}", comma);
    }
    println!("  ]");
    println!("}}");
}
