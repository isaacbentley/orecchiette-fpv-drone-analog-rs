//! Detection and decoding of analog FPV drone video from raw I/Q samples.
//!
//! Analog FPV video is an FM-modulated television signal. This crate
//! searches a capture for one, classifies it as PAL or NTSC, and
//! reconstructs frames. It is independent of SDR hardware: callers
//! supply `Complex<f32>` samples from any source.
//!
//! # Detection
//!
//! [`detector::AnalogFpvDetector`] implements [`detector::FpvDetector`].
//! Its strategy follows the sample rate: at or above 25 MSPS a sliding
//! down-converter sweeps the capture on a 5 MHz probe grid; below that
//! the grid yields too few probes, so the capture is classified as a
//! single baseband slice at the tuned centre.
//!
//! Detections carry a confidence from 0.6 to 0.95, rising as the
//! horizontal sync bin resolves PAL from NTSC and as vertical sync
//! structure is confirmed. [`detector::SpectralIntegrator`] accumulates
//! magnitude spectra across batches for additional sensitivity.
//!
//! # Decoding
//!
//! [`demod::fm_demod`] is a quadrature discriminator;
//! [`demod::PllFmDemod`] is a phase-locked alternative that measures
//! better at 25 MSPS and above. [`demod::Deemphasis`] inverts a
//! transmitter's video pre-emphasis. [`video::FrameReconstructor`]
//! extracts sync, corrects the time base, and assembles fields into
//! frames; [`vbi`] parses the vertical blanking interval and resolves
//! field parity.
//!
//! # Features
//!
//! - `gpu` — batches the sweep's per-probe down-conversion into a single
//!   wgpu dispatch. Classification stays on the CPU.
//! - `neural-vsr` — ONNX-based temporal restoration via `ort`.
//!
//! Both are disabled by default; the default build is CPU and rayon
//! only.
//!
//! # Example
//!
//! ```no_run
//! use orecchiette_fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
//! use num_complex::Complex;
//!
//! let detector = AnalogFpvDetector::default();
//! let iq: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 262_144];
//! for res in detector.detect_from_iq(&iq, 5_800_000_000, 100_000_000) {
//!     println!("{:?} at {} Hz, confidence {:.2}",
//!         res.signal_type, res.frequency_hz, res.confidence);
//! }
//! ```
//!
//! See `DESIGN.md` in the repository for the architecture and the
//! underlying math.

pub mod bands;
pub mod ddc;
pub mod demod;
pub mod detector;
pub mod frame_history;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod impairments;
pub mod levels;
pub mod metrics;
#[cfg(feature = "neural-vsr")]
pub mod neural;
pub mod scanner;
pub mod synthetic;
pub mod timing;
pub mod types;
pub mod vbi;
pub mod video;

pub use bands::*;
pub use ddc::*;
pub use demod::*;
pub use detector::*;
pub use frame_history::*;
pub use levels::*;
pub use scanner::*;
pub use timing::*;
pub use types::*;
pub use vbi::*;
pub use video::*;
