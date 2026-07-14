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

/// Default deemphasis time constant: 0.75 µs. Common in analog FPV
/// VTXs and camcorder-derived FM video links (a faster time constant
/// than broadcast CCIR 405-1's 50 µs audio deemphasis — video
/// deemphasis targets the much wider luma bandwidth, not audio).
/// Tune per-VTX via [`Deemphasis::new`]'s `tau_seconds` if a specific
/// transmitter's pre-emphasis curve is known.
pub const DEFAULT_DEEMPHASIS_TAU_S: f32 = 0.75e-6;

/// Single-pole IIR deemphasis filter (a digital approximation of the
/// analog RC low-pass a receiver would use to undo a VTX's pre-
/// emphasis), applied to [`fm_demod`]'s output.
///
/// Deliberately **not** a method on `FrameReconstructor`: the
/// reconstructor is called repeatedly on the *unconsumed tail* of a
/// persistent demod buffer (each call re-reads samples the previous
/// call already saw, advancing by `consumed`), so a stateful filter
/// living there would re-filter already-filtered samples every call.
/// The correct place is stream-side — once, right after [`fm_demod`],
/// before samples ever enter that persistent buffer.
///
/// `y[n] = α·x[n] + (1−α)·y[n−1]`, with
/// `α = 1 − exp(−1 / (sample_rate · τ))` — the impulse-invariant
/// discretization of a continuous first-order RC low-pass with time
/// constant `τ`. Unity DC gain by construction (steady state on a
/// constant input is `y = x`), so sync-tip/blanking levels — and
/// therefore [`crate::levels::estimate_fm_deviation`]'s swing
/// measurement and the reconstructor's AGC — are unaffected by
/// whether this filter is enabled.
pub struct Deemphasis {
    alpha: f32,
    y: f32,
    primed: bool,
}

impl Deemphasis {
    /// `tau_seconds` is the RC time constant; see
    /// [`DEFAULT_DEEMPHASIS_TAU_S`] for a reasonable default. Panics if
    /// `sample_rate` is 0 or `tau_seconds` isn't positive and finite —
    /// both make `alpha` undefined, and a filter silently doing
    /// nothing (or blowing up) is worse than a loud failure at
    /// construction, far from the hot path.
    pub fn new(sample_rate: u32, tau_seconds: f32) -> Self {
        assert!(sample_rate > 0, "Deemphasis: sample_rate must be > 0");
        assert!(
            tau_seconds.is_finite() && tau_seconds > 0.0,
            "Deemphasis: tau_seconds must be finite and > 0"
        );
        let alpha = 1.0 - (-1.0 / (sample_rate as f32 * tau_seconds)).exp();
        Self {
            alpha,
            y: 0.0,
            primed: false,
        }
    }

    /// Filter `data` in place, continuing the running state from any
    /// previous call (or from [`Self::reset`]/construction).
    pub fn process_in_place(&mut self, data: &mut [f32]) {
        for x in data.iter_mut() {
            if !self.primed {
                // Seed the state with the first sample rather than 0.0
                // so a call starting on a non-zero DC level doesn't
                // spend the first ~few/α samples settling from zero —
                // audible/visible as a brief fade-in on every rebuild.
                self.y = *x;
                self.primed = true;
            } else {
                self.y = self.alpha * *x + (1.0 - self.alpha) * self.y;
            }
            *x = self.y;
        }
    }

    /// Clear the running state (e.g. after a DDC/reconstructor rebuild
    /// on a target change, so the filter doesn't blend samples across
    /// an unrelated discontinuity).
    pub fn reset(&mut self) {
        self.y = 0.0;
        self.primed = false;
    }
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

    // ── Deemphasis ──────────────────────────────────────────────────

    #[test]
    fn deemphasis_has_unity_dc_gain() {
        let mut d = Deemphasis::new(10_000_000, DEFAULT_DEEMPHASIS_TAU_S);
        let mut data = vec![0.37f32; 1000];
        d.process_in_place(&mut data);
        // Steady state (well past the settling transient) must equal
        // the input exactly -- this is what keeps sync/blanking levels,
        // and therefore the deviation estimator's swing measurement,
        // unaffected by whether deemphasis is enabled.
        for &v in &data[500..] {
            assert!(
                (v - 0.37).abs() < 1e-6,
                "expected steady state 0.37, got {v}"
            );
        }
    }

    #[test]
    fn deemphasis_minus_3db_point_matches_raw_time_constant() {
        // Sweep a range of tones through the filter and find where the
        // output amplitude drops to 1/sqrt(2) of a very-low-frequency
        // reference -- that should land near f_c = 1 / (2*pi*tau).
        let sample_rate = 20_000_000u32;
        let tau = 0.75e-6f32;
        let f_c = 1.0 / (2.0 * PI * tau);

        let measure_gain = |freq: f32| -> f32 {
            let n = 20_000;
            let mut d = Deemphasis::new(sample_rate, tau);
            let mut data: Vec<f32> = (0..n)
                .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
                .collect();
            d.process_in_place(&mut data);
            // RMS over the settled tail as an amplitude proxy.
            let tail = &data[n / 2..];
            (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt()
        };

        let low_freq_gain = measure_gain(1_000.0); // near-DC reference, ~unity
        let at_fc_gain = measure_gain(f_c);
        let ratio = at_fc_gain / low_freq_gain;
        assert!(
            (ratio - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.05,
            "gain ratio at f_c={f_c:.0} Hz was {ratio}, expected ~{:.4} (-3 dB)",
            std::f32::consts::FRAC_1_SQRT_2
        );
    }

    #[test]
    fn deemphasis_streaming_matches_one_shot() {
        let sample_rate = 15_360_000u32;
        let signal: Vec<f32> = (0..2000).map(|i| (0.01 * i as f32).sin() * 0.5).collect();

        let mut one_shot = signal.clone();
        Deemphasis::new(sample_rate, DEFAULT_DEEMPHASIS_TAU_S).process_in_place(&mut one_shot);

        let mut streamed = signal.clone();
        let mut d = Deemphasis::new(sample_rate, DEFAULT_DEEMPHASIS_TAU_S);
        let (a, b) = streamed.split_at_mut(700);
        d.process_in_place(a);
        d.process_in_place(b);

        for i in 0..signal.len() {
            assert!(
                (one_shot[i] - streamed[i]).abs() < 1e-6,
                "mismatch at i={i}: one_shot={} streamed={}",
                one_shot[i],
                streamed[i]
            );
        }
    }

    #[test]
    fn deemphasis_reset_clears_state_like_a_fresh_instance() {
        let sample_rate = 15_360_000u32;
        let mut d = Deemphasis::new(sample_rate, DEFAULT_DEEMPHASIS_TAU_S);
        let mut warm_up = vec![0.9f32; 200];
        d.process_in_place(&mut warm_up);

        d.reset();
        let mut after_reset = vec![0.2f32; 50];
        d.process_in_place(&mut after_reset);

        let mut fresh = Deemphasis::new(sample_rate, DEFAULT_DEEMPHASIS_TAU_S);
        let mut fresh_data = vec![0.2f32; 50];
        fresh.process_in_place(&mut fresh_data);

        assert_eq!(
            after_reset, fresh_data,
            "reset should behave like a fresh instance"
        );
    }

    #[test]
    #[should_panic(expected = "sample_rate")]
    fn deemphasis_rejects_zero_sample_rate() {
        Deemphasis::new(0, DEFAULT_DEEMPHASIS_TAU_S);
    }

    #[test]
    #[should_panic(expected = "tau_seconds")]
    fn deemphasis_rejects_non_positive_tau() {
        Deemphasis::new(15_360_000, 0.0);
    }
}
