# orecchiette-fpv-drone-analog-rs

[![CI](https://github.com/isaacbentley/orecchiette-fpv-drone-analog-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacbentley/orecchiette-fpv-drone-analog-rs/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/isaacbentley/orecchiette-fpv-drone-analog-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/isaacbentley/orecchiette-fpv-drone-analog-rs)
[![License: GPL-3.0-or-later](https://img.shields.io/github/license/isaacbentley/orecchiette-fpv-drone-analog-rs.svg)](https://choosealicense.com/licenses/gpl-3.0/)

A Rust crate for detecting and decoding analog FPV drone video from raw
I/Q samples. It covers FM demodulation, sync detection, and frame
reconstruction, and is independent of SDR hardware.

Analog FPV video is an FM-modulated television signal. The crate searches
a capture for one, classifies it as PAL or NTSC, and reconstructs frames.

[DESIGN.md](./DESIGN.md) documents the architecture and the underlying
math. Section references below point into it.

## Features

### Detection

- **Wideband sweep.** A sliding down-converter scans the full capture
  bandwidth, from 1 MSPS to over 100 MSPS, without a predefined channel
  table.
- **PAL/NTSC classification.** An FFT measures the horizontal sync rate,
  15,625 Hz for PAL against 15,734 Hz for NTSC.
- **Interference rejection.** A candidate must present a consistent
  harmonic series, and a cepstral check (§3) distinguishes a genuine
  repeating pulse train from several continuous-wave carriers.
- **Vertical-sync confirmation.** Detections are cross-checked against
  real vertical sync groups at the correct field spacing, which a
  non-video signal cannot readily imitate (§7).
- **Cross-batch integration.** `SpectralIntegrator` averages magnitude
  spectra across batches for additional sensitivity. A calibrated test
  holds detection at a noise level where four independent single batches
  all fail (§11).
- **Clustering and scoring.** Detections within 25 MHz are merged into a
  single event, scored from 0.0 to 1.0.

### Decoding

- **FM demodulation.** The default is a quadrature discriminator,
  `arg(z[n] × conj(z[n-1]))`. `demod::PllFmDemod` is a phase-locked
  alternative for weak signals, measuring 6–17 dB better demodulated SNR
  and holding sync approximately one noise step longer at 25 MSPS and
  above (`examples/weak_signal_sweep.rs`). The discriminator remains
  preferable below that rate.
- **FM deviation estimation.** `levels::estimate_fm_deviation` recovers a
  transmitter's true peak deviation from the demodulated waveform without
  requiring sync lock, so downstream thresholds do not depend on a fixed
  assumption. `levels::estimate_cnr_db` provides a link-quality measure.
- **VBI parsing.** Pulses are classified by width into equalizing, broad,
  and horizontal; the parser locates the vertical sync group and resolves
  field parity by hypothesis test against the standard's active-video
  timing (§7).
- **Sync extraction and time-base correction.** Sync tips are filtered by
  median and MAD outlier rejection, then aligned to sub-sample accuracy
  with Catmull-Rom interpolation. Detection thresholds derive from the
  signal's own level distribution, so an upstream gain change such as
  deemphasis or AGC cannot push every sync tip out of range (§9.1).
- **Temporal noise reduction.** A fixed-capacity ring buffer of recent
  fields feeds a per-pixel median, blended according to local motion.
- **Deemphasis.** `demod::Deemphasis` is a unity-DC-gain single-pole IIR
  filter that inverts a transmitter's video pre-emphasis, suppressing the
  high-frequency noise that pre-emphasis would otherwise leave in the
  picture. It affects neither deviation estimation nor sync detection.
- **Burst-gated subcarrier notch.** The dot-crawl notch engages only when
  a Goertzel burst detector confirms a colour burst. Many FPV cameras are
  effectively monochrome, and on those the notch removed real luma
  detail.
- **Monochrome output.** Analog FPV's colour subcarrier carries little of
  the information an operator needs, and weak links resolve better in
  clean grayscale than in noisy decoded colour (§9).

### Optional

- **GPU acceleration** (`gpu` feature, disabled by default). Batches the
  sweep's per-probe down-conversion, filtering, and decimation into a
  single wgpu dispatch rather than running them sequentially on the CPU.
  This is the dominant cost on captures of 50 MSPS and above.
  Classification remains on the CPU (§10).

## Installation

```toml
[dependencies]
orecchiette-fpv-drone-analog-rs = "0.5.0"
num-complex = "0.4"
```

To enable the GPU sweep:

```toml
orecchiette-fpv-drone-analog-rs = { version = "0.5.0", features = ["gpu"] }
```

Construct one `GpuAnalog` and share it across detectors.
`AnalogFpvDetector` holds an `FftPlanner` and should be kept per-thread;
`GpuAnalog` is `Send + Sync`:

```rust
use orecchiette_fpv_drone_analog_rs::detector::AnalogFpvDetector;
use orecchiette_fpv_drone_analog_rs::gpu::GpuAnalog;
use std::sync::Arc;

// The CPU sweep is used automatically if try_new() returns None.
if let Some(gpu) = GpuAnalog::try_new() {
    let detector = AnalogFpvDetector::with_gpu(Arc::new(gpu));
    // ... detector.detect_from_iq(...) as usual
}
```

## Usage

`detect_from_iq` selects its strategy from the sample rate:

```rust
use orecchiette_fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
use num_complex::Complex;

let detector = AnalogFpvDetector::default();
let iq_data: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 262_144]; // raw samples
let results = detector.detect_from_iq(&iq_data, 5_800_000_000, 100_000_000);

for res in &results {
    println!("Found {:?} at {:.1} MHz, confidence {:.2}",
        res.signal_type, res.frequency_hz as f64 / 1e6, res.confidence);
}
```

Above roughly 25 MSPS the sliding down-converter sweeps the full
bandwidth and may return several signals at arbitrary frequencies. Below
that there are too few probe positions to form a grid, and the capture is
classified as a single slice at the tuned centre frequency, which suits
an already-centred baseband signal.

## Confidence levels

| Confidence | `SignalType` | Meaning |
| :--- | :--- | :--- |
| **0.6** | `AnalogVideoUnknown` | Horizontal sync detected, but FFT bin resolution is too coarse to separate PAL from NTSC. Harmonic check passed. |
| **0.6** | `AnalogVideoPal` / `AnalogVideoNtsc` | Demoted from 0.8 or 0.95 by the optional `demote_unconfirmed_video` check (disabled by default): harmonics matched, but no vertical sync was confirmed over 2.5 field periods. |
| **0.75** | `AnalogVideoUnknown` | The bin-collision case above, with real vertical sync structure confirmed underneath. |
| **0.8** | `AnalogVideoPal` / `AnalogVideoNtsc` | Distinct horizontal sync bin and at least 2 harmonics above −20 dB. |
| **0.95** | `AnalogVideoPal` / `AnalogVideoNtsc` | The 0.8 case, additionally confirmed by vertical sync structure. |

The harmonic-consistency check is a gate rather than a bonus: fewer than
2 harmonics above −20 dB is rejected as `Unknown` regardless of
fundamental energy.

`SignalType::is_analog_video()` reports whether a detection is analog FPV
video of any kind, including `AnalogVideoUnknown`.

## Testing

```bash
cargo test -p orecchiette-fpv-drone-analog-rs
```

Tests generate their own FM-modulated I/Q, so the repository carries no
large fixture files. `synthetic::generate_fields` and `generate_iq`
produce standards-shaped PAL and NTSC fields with complete vertical sync
structure, and every module's tests share them, so the generator and the
parser cannot drift apart independently.

Coverage spans detection (narrowband, wideband sweep, two-signal
captures, noise and carrier rejection, clustering, the cepstral gate, and
the confidence rules), DSP (down-converter round trip, demodulator
accuracy near ±π, deviation estimation, deemphasis response and
continuity across chunks), the VBI parser for both standards and both
field parities, and `FrameReconstructor` — geometry per standard,
interlace recovery after a dropped field, sync survival through an
upstream gain change, and returning `None` rather than panicking on
mismatched buffers or degenerate configuration.

`cargo bench` (criterion) times detection on realistic 65 k chunks at 25
and 61.44 MSPS and frame reconstruction at 15.36 MSPS, so real-time
performance is measured rather than assumed.

`--features gpu` adds a stage-level GPU-against-CPU comparison and two
end-to-end equivalence tests. All three skip cleanly when no GPU adapter
is present.

### Reference captures

`examples/make_reference_capture.rs` writes standards-conformant PAL and
NTSC captures in SigMF format (`.sigmf-data` and `.sigmf-meta`) using the
same generator as the test suite. Because they are correct by
construction, they can be used to establish whether a decoder is at
fault:

```bash
cargo run --release --example make_reference_capture -- --standard pal
cargo run --release --example make_reference_capture -- --standard ntsc
cargo run --release --example make_reference_capture -- --standard pal --noise-sigma 0.45 --out reference_pal_weak
```

A correct decoder reports the expected standard and geometry and holds
sync quality near 1.0 on every field of both parities. The clean PAL and
NTSC files decode at 0.99 and 1.00 respectively, with no interpolated
rows. If a real capture decodes poorly on alternate fields while these do
not, the fault lies in the capture.

The reference files carry no transmitter pre-emphasis, so
`--deemphasis-tau 0` reproduces the generated waveform exactly. The
viewer's default of 0.75 µs also decodes them correctly, though with
softer edges, as there is no pre-emphasis to invert. Both settings are
worth exercising: that gain change between demodulator and reconstructor
is the condition covered by `sync_survives_deemphasis_gain_change`
(§9.1).

### Visual verification

The [fpv-viewer-rs](https://github.com/isaacbentley/fpv-viewer-rs) binary
run with `--debug` renders the complete pipeline live and writes frames
1–3 and 30–32 to `fpv_frame_<freq>MHz_<n>.png`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for running the test suite and
formatting before submitting a pull request.

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later). See
[LICENSE](LICENSE).
