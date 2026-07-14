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
- **Vertical-Sync (VBI) Parsing**: Classifies equalizing / serrated-broad / horizontal pulses by width, locates the true vertical-sync group, and determines field parity by direct hypothesis test against the standard's calibrated active-video timing — not a phase fit, which this signal's own structure can't actually carry (see `DESIGN.md` §7).
- **VBI-Confirmed Detection**: The detector cross-checks a harmonic-comb match against real, field-period-spaced vertical syncs, boosting confidence to 0.95 (or promoting a standard-ambiguous hit to 0.75) — essentially unfakeable by a non-video interferer.
- **FM Deviation Auto-Estimation**: Recovers a transmitter's true peak FM deviation directly from the demodulated waveform (`levels::estimate_fm_deviation`), with no sync lock required, so playback and detection thresholds don't depend on a fixed assumption that's wrong for a given VTX.
- **Optional Deemphasis**: A single-pole IIR deemphasis filter (`demod::Deemphasis`), off by default, approximates undoing a VTX's video pre-emphasis with unity DC gain (doesn't affect deviation estimation or sync detection either way).
- **Sync Extraction & Time Base Correction**: Employs median and MAD outlier rejection on raw sync tips, combined with Catmull-Rom cubic interpolation for sub-sample Time Base Correction (TBC).
- **Temporal Noise Reduction**: Features a configurable fixed-capacity ring buffer for multi-field temporal denoising, utilizing per-pixel median and motion-weighted blending to improve SNR on static regions.
- **Monochrome Rendering**: Outputs luma-only frames — analog FPV's color subcarrier carries comparatively little of what an operator needs, and low-SNR RF links look better in clean grayscale than in noisy decoded color (see `DESIGN.md` §9).
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
| **0.6** | `AnalogVideoPal` / `AnalogVideoNtsc` | Demoted from 0.8/0.95 by the opt-in `demote_unconfirmed_video` check (default off) — a harmonic-comb match with zero confirmed vertical-sync groups over ≥ 2.5 field periods |
| **0.75** | `AnalogVideoUnknown` | The 0.6 (bin-collision) case above, but real periodic vertical-sync structure was confirmed underneath |
| **0.8** | `AnalogVideoPal` / `AnalogVideoNtsc` | Distinct H-sync bin AND ≥ 2 harmonics above the −20 dB threshold (high-confidence pulse-train classification) |
| **0.95** | `AnalogVideoPal` / `AnalogVideoNtsc` | The 0.8 case above, additionally confirmed by real vertical-sync structure — essentially unfakeable by a non-video interferer |

The harmonic-consistency check is a *gate*: candidates with fewer than 2 harmonics above the −20 dB threshold are rejected as `Unknown` regardless of fundamental energy. This holds across both the bins-distinct and bin-collision paths.

Use `SignalType::is_analog_video()` to gate on "is this an analog FPV signal at all?" — returns `true` for all three video variants including `AnalogVideoUnknown`.

## Testing

```bash
cargo test -p orecchiette-fpv-drone-analog-rs
```

Tests generate FM-modulated synthetic IQ data programmatically — no large fixture files needed. `synthetic::generate_fields`/`generate_iq` build standards-shaped NTSC/PAL fields with real vertical-sync structure (equalizing/serrated-broad pulses, correct blanking, interlace parity), shared by every module's tests so generator and parser can't independently drift. Coverage includes narrowband PAL/NTSC, wideband sliding DDC, two-signal detection, noise rejection, CW rejection, clustering verification, the `StreamingDDC` mixer round-trip, the FM demodulator's near-±π precision, cepstrum gate verification (harmonic comb pass / flat spectrum reject / noise reject), the VBI parser (broad-group location, field-parity hypothesis test for both standards and parities, noise robustness), the confidence-tier policy (boost/promote/demote, as a pure function independent of any specific IQ signal), the FM-deviation estimator's accuracy across a range of deviation/sample-rate pairs, and the deemphasis filter's DC gain / −3 dB point / streaming continuity.

`video::FrameReconstructor` additionally has regression tests confirming `reconstruct_frame_into` degrades to `None` rather than panicking on a mismatched output-buffer size, empty input, and degenerate configuration (`sample_rate = 0`), plus geometry tests proving a test pattern lands on the correct output row for both NTSC and PAL (the standards-correct-blanking fix) and a dropped-field test proving the VBI parser's parity override — not a naive per-call toggle — recovers correct interlacing after a lost field.

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
