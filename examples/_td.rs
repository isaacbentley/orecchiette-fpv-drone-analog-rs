use orecchiette_fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
use orecchiette_fpv_drone_analog_rs::synthetic::{SyntheticVideoConfig, TestPattern, generate_iq};
use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;
fn main() {
    let sr = 40_000_000u32;
    let cfg = SyntheticVideoConfig {
        sample_rate: sr,
        is_pal: true,
        deviation_hz: 5e6,
        pattern: TestPattern::Bars,
        start_field: FieldParity::First,
        noise_sigma: 0.0,
        dc_offset: 0.0,
    };
    let full = generate_iq(&cfg, 12, 0.0);
    let det = AnalogFpvDetector::default();
    let r = det.detect_from_iq(&full[..131_072], 5_800_000_000, sr);
    for d in &r {
        eprintln!(
            "RESULT {:?} @{:+.1} MHz conf {:.2}",
            d.signal_type,
            (d.frequency_hz as f64 - 5_800e6) / 1e6,
            d.confidence
        );
    }
    if r.is_empty() {
        eprintln!("RESULT none");
    }
}
