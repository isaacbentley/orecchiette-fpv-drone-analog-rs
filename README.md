# orecchiette-fpv-drone-analog-rs

[![CI](https://github.com/isaacbentley/orecchiette-fpv-drone-analog-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacbentley/orecchiette-fpv-drone-analog-rs/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/isaacbentley/orecchiette-fpv-drone-analog-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/isaacbentley/orecchiette-fpv-drone-analog-rs)
[![License: GPL-3.0-or-later](https://img.shields.io/github/license/isaacbentley/orecchiette-fpv-drone-analog-rs.svg)](https://choosealicense.com/licenses/gpl-3.0/)

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
orecchiette-fpv-drone-analog-rs = "0.1.0"
num-complex = "0.4"
```

## Usage

### Narrowband Detection (< 3 MSPS)
For isolated baseband signals where the FM carrier is already centered:

```rust
use orecchiette_fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
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
use orecchiette_fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
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
cargo test -p orecchiette-fpv-drone-analog-rs
```

Tests generate FM-modulated synthetic IQ data programmatically — no large fixture files needed. Coverage includes narrowband PAL/NTSC, wideband sliding DDC, two-signal detection, noise rejection, CW rejection, clustering verification, the `StreamingDDC` mixer round-trip, the FM demodulator's near-±π precision, cepstrum gate verification (harmonic comb pass / flat spectrum reject / noise reject), and PAL parity self-correction under 1-line V-sync offset.

`video::FrameReconstructor` additionally has regression tests confirming `reconstruct_frame_into` degrades to `None` rather than panicking on a mismatched output-buffer size, empty input, and degenerate configuration (`sample_rate = 0`).

### End-to-end decode check

For a visual sanity check, use the [fpv-viewer-rs](https://github.com/isaacbentley/fpv-viewer-rs) binary with `--debug`, which renders the full DDC → FM-demod → reconstruction pipeline live and dumps the first three frames to `/tmp/fpv_frame_*.png` before auto-exiting. See the `fpv-viewer-rs` README for the full flag set.

See [DESIGN.md](./DESIGN.md) for the full architecture and math.

## MSRV & Semver Policy

- **MSRV:** This crate does not maintain an explicit Minimum Supported Rust Version (MSRV) policy and tracks the latest `stable` compiler.
- **Semver:** This crate follows semantic versioning. While in `0.x.y`, breaking API changes will result in a minor version bump (e.g. `0.1.x` to `0.2.0`).

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions on running the test suite and formatting your code before submitting a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later) - see the [LICENSE](LICENSE) file for details.
