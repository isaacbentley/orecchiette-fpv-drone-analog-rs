# fpv-drone-analog-rs

[![CI](https://github.com/isaacbentley/orecchiette/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacbentley/orecchiette/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/fpv-drone-analog-rs.svg)](https://crates.io/crates/fpv-drone-analog-rs)
[![Docs.rs](https://docs.rs/fpv-drone-analog-rs/badge.svg)](https://docs.rs/fpv-drone-analog-rs)
[![Codecov](https://codecov.io/gh/isaacbentley/orecchiette/branch/main/graph/badge.svg)](https://codecov.io/gh/isaacbentley/orecchiette)
[![License](https://img.shields.io/crates/l/fpv-drone-analog-rs.svg)](https://crates.io/crates/fpv-drone-analog-rs)
[![MSRV](https://img.shields.io/badge/rustc-1.85+-ab6000.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)

A high-performance Rust crate for detecting analog FPV drone video signals using FM demodulation and spectral sync-pulse analysis.

## Features
- **Wideband Detection Strategy**: Utilizes a sliding Digital Down Converter (DDC) to scan capture bandwidths autonomously without relying on predefined channel tables, supporting rates from 1 MSPS to 100+ MSPS.
- **FM Demodulation**: Implements baseband video recovery from FM-modulated signals using polar phase differentiation (`arg(z[n] × conj(z[n-1]))`).
- **PAL/NTSC Classification**: Performs windowed FFT analysis to discriminate horizontal sync pulses at 15,625 Hz (PAL) and 15,734 Hz (NTSC).
- **Cepstral Analysis Validation**: Applies post-harmonic cepstrum validation (`IFFT(ln|FFT|²)`) to distinguish periodic pulse trains from multi-CW interference.
- **PAL Parity Tracking**: Utilizes a phase-delta discriminator to track and correct PAL V-switch line parity offsets.
- **Sync Extraction & Time Base Correction**: Employs median and MAD outlier rejection on raw sync tips, combined with Catmull-Rom cubic interpolation for sub-sample Time Base Correction (TBC).
- **Temporal Noise Reduction**: Features a configurable fixed-capacity ring buffer for multi-field temporal denoising, utilizing per-pixel median and motion-weighted blending to improve SNR on static regions.
- **Monochrome Rendering**: Outputs luma-only frames to ensure stability under high phase-noise conditions typical of analog FPV chroma bursts (see `DESIGN.md` for empirical details).
- **Signal Clustering**: Aggregates proximate DDC probe hits (within 25 MHz) to emit single, consolidated detection events.
- **Standardized Scoring**: Employs a 0.0–1.0 confidence scoring model consistent across workspace detection heuristics.
- **Hardware Agnostic**: Processes standard complex I/Q samples independent of the underlying SDR hardware.

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
fpv-drone-analog-rs = "0.1.0"
num-complex = "0.4"
```

## Usage

### Narrowband Detection (< 3 MSPS)
For isolated baseband signals where the FM carrier is already centered:

```rust
use fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
use num_complex::Complex;

let detector = AnalogFpvDetector::default();
let iq_data: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 65536]; // Replace with raw samples
let sample_rate = 1_000_000;
let center_freq = 5_800_000_000;

let results = detector.detect_from_iq(&iq_data, center_freq, sample_rate);
for res in &results {
    println!("Signal: {:?}, confidence: {:.2}", res.signal_type, res.confidence);
}
```

### Wideband Detection (≥ 3 MSPS)
For wideband captures containing multiple signals at arbitrary frequencies:

```rust
use fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
use num_complex::Complex;

let detector = AnalogFpvDetector::default();
let iq_data: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 262_144]; // 100 MSPS packet
let sample_rate = 100_000_000;
let center_freq = 5_800_000_000;

// Sliding DDC probe automatically sweeps the full 100 MHz bandwidth
let results = detector.detect_from_iq(&iq_data, center_freq, sample_rate);
for res in &results {
    println!("Found {:?} at {:.1} MHz, confidence {:.2}",
        res.signal_type,
        res.frequency_hz as f64 / 1e6,
        res.confidence);
}
```

## Detection Model
| Confidence | `SignalType` | Meaning |
| :--- | :--- | :--- |
| **0.6** | `AnalogVideoUnknown` | H-sync detected but FFT bin resolution too coarse to discriminate PAL (15625 Hz) from NTSC (15734 Hz); harmonic check passed |
| **0.6** | `AnalogVideoPal` / `AnalogVideoNtsc` | Classified via V-sync rate (50/60 Hz); only reachable with > 100 ms capture window |
| **0.8** | `AnalogVideoPal` / `AnalogVideoNtsc` | Distinct H-sync bin AND ≥ 2 harmonics above the −20 dB threshold (high-confidence pulse-train classification) |

The harmonic-consistency check is a *gate*: candidates with fewer than 2 harmonics above the −20 dB threshold are rejected as `Unknown` regardless of fundamental energy. This holds across both the bins-distinct and bin-collision paths.

Use `SignalType::is_analog_video()` to gate on "is this an analog FPV signal at all?" — returns `true` for all three video variants including `AnalogVideoUnknown`.

## Testing

```bash
cargo test -p fpv-drone-analog-rs
```

Tests generate FM-modulated synthetic IQ data programmatically — no large fixture files needed. Coverage includes narrowband PAL/NTSC, wideband sliding DDC, two-signal detection, noise rejection, CW rejection, clustering verification, the `StreamingDDC` mixer round-trip, the FM demodulator's near-±π precision, cepstrum gate verification (harmonic comb pass / flat spectrum reject / noise reject), and PAL parity self-correction under 1-line V-sync offset.

### End-to-end decode check

For a visual sanity check, use the workspace's `fpv_viewer` binary with `--debug`, which renders the full DDC → FM-demod → reconstruction pipeline live and dumps the first three frames to `/tmp/fpv_frame_*.png` before auto-exiting. See the top-level README for the full flag set.

See [DESIGN.md](./DESIGN.md) for the full architecture and math.

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later) - see the [LICENSE](../../LICENSE) file for details.
