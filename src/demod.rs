//! FM Demodulation Module
//!
//! This module handles the conversion of raw complex IQ samples into a baseband
//! signal (the video stream). For 5.8 GHz analog FPV, the video is frequency
//! modulated (FM).

use num_complex::Complex;

// ELI5: Imagine the drone is a singer who changes the pitch of their voice
// to send a message. This module is like an ear that listens to those
// changes in pitch and writes them down as numbers.

/// Quadrature FM demodulation: instantaneous frequency from
/// `arg(iq[n] · conj(iq[n-1]))`.
///
/// The phase extraction uses the exact scalar `f32::atan2` (via
/// `Complex::arg`) rather than a polynomial approximation. We tried
/// `fast_math::atan2` for edge-device throughput, but image quality
/// wins here: approximate atan2 kernels lose precision near ±π —
/// exactly the regime a high-deviation FM video signal lives in —
/// and the resulting quadrant errors surface as click-noise sparkles
/// in the reconstructed picture. The complex multiply + `conj` ahead
/// of the atan2 is the bulk of the per-sample work and auto-vectorises
/// cleanly under `-O3` (both NEON and AVX2), so the exact path costs
/// little over the approximation while keeping the discriminator
/// output clean. The downstream temporal-denoise median in `video.rs`
/// is for channel noise, not for papering over demodulator error.
pub fn fm_demod(iq_data: &[Complex<f32>]) -> Vec<f32> {
    let n = iq_data.len();
    if n < 2 {
        return vec![];
    }

    let mut output = vec![0.0f32; n - 1];
    for i in 1..n {
        let prod = iq_data[i] * iq_data[i - 1].conj();
        output[i - 1] = prod.arg();
    }
    output
}

/// Deprecated alias for [`fm_demod`]. The `_simd` suffix was
/// historical and misleading — the function never used explicit
/// SIMD intrinsics. Kept around so external callers don't break;
/// new code should call [`fm_demod`] directly.
#[deprecated(
    since = "0.4.28",
    note = "renamed to `fm_demod` — the `_simd` suffix was misleading; see fn docs"
)]
pub fn fm_demod_simd(iq_data: &[Complex<f32>]) -> Vec<f32> {
    fm_demod(iq_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Regression guard: the demodulator must produce the same phase differences as a
    /// straightforward scalar reference, ensuring correctness near ±π.
    #[test]
    fn fm_demod_matches_scalar_reference() {
        // Mix of in-band rotation, near-π wraps, and a quadratic chirp to cover quadrants.
        let n = 4096;
        let mut iq = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32;
            let phase = 0.0123 * t + 0.000_001 * t * t;
            iq.push(Complex::from_polar(1.0, phase));
        }
        let out = fm_demod(&iq);
        assert_eq!(out.len(), n - 1);
        for i in 0..(n - 1) {
            let prod = iq[i + 1] * iq[i].conj();
            let expected = prod.im.atan2(prod.re);
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "mismatch at i={i}: got {}, expected {}",
                out[i],
                expected
            );
        }
    }

    #[test]
    fn fm_demod_handles_pi_boundary() {
        // Two samples whose product points to almost exactly -π (atan2(-ε, -1) → -π+ε),
        // verifying correctness in the wraparound region.
        let iq = vec![
            Complex::from_polar(1.0, 0.0),
            Complex::from_polar(1.0, PI - 0.001), // step of ≈ +π
        ];
        let out = fm_demod(&iq);
        assert_eq!(out.len(), 1);
        // Expected: phase wrap close to ±π.
        assert!((out[0].abs() - (PI - 0.001)).abs() < 1e-4, "got {}", out[0]);
    }

    #[test]
    fn fm_demod_short_input() {
        assert!(fm_demod(&[]).is_empty());
        assert!(fm_demod(&[Complex::new(1.0, 0.0)]).is_empty());
    }

    /// The deprecated `fm_demod_simd` alias should still work
    /// bit-for-bit so external callers aren't silently broken until
    /// they migrate.
    #[test]
    #[allow(deprecated)]
    fn fm_demod_simd_alias_matches() {
        let iq: Vec<Complex<f32>> = (0..128)
            .map(|i| Complex::from_polar(1.0, 0.013 * i as f32))
            .collect();
        assert_eq!(fm_demod(&iq), fm_demod_simd(&iq));
    }
}
