//! Streaming Digital Down-Converter (DDC) with a real anti-alias FIR.
//!
//! The windowed-sinc FIR design loop and the per-sample convolution
//! both index by sample position (FIR tap index and circular-buffer
//! offset arithmetic respectively); iterator chains would obscure the
//! convolution structure that's the whole point of the function.
#![allow(clippy::needless_range_loop)]
//!
//! This module provides [`StreamingDDC`], a mixer + Blackman-windowed-
//! sinc FIR low-pass (default 63 taps, > 50 dB stopband). It is used
//! both by the live `fpv_viewer` binary on its channel-decode hot path
//! and — since v0.4.29 — by the detector's sliding-DDC sweep via the
//! private `ddc_and_decimate` helper in `detector.rs` (which
//! constructs a `StreamingDDC` per probe and calls
//! `process_decimated`). The earlier length-N boxcar (`sum/N`) it
//! replaced had a sinc magnitude response with poor stopband
//! attenuation — adjacent-band energy leaked back into the
//! discriminator passband, costing CNR margin in the FM
//! threshold-effect region and synthesising spurious harmonic content
//! from strong out-of-band tones. The windowed-sinc FIR closes that
//! gap (> 50 dB stopband at `target_rate/3` cutoff).
//!
//! ## Phase tracking
//!
//! The mixer LO uses phasor recursion: each step is a single complex
//! multiply by `exp(j·phase_adv)` instead of a `sincos` call.
//! Magnitude is renormalised every sample with a first-order Newton
//! step (`0.5·(3 − |φ|²)`) so f32 round-off doesn't drift |phasor|
//! away from 1.

use num_complex::Complex;
use std::f32::consts::PI;

/// Default tap count for the anti-alias FIR. 63 taps with a
/// Blackman window gives > 50 dB stopband attenuation, which is
/// roughly 40 dB better than the boxcar of equivalent length.
pub const DEFAULT_FIR_TAPS: usize = 63;

/// Streaming mixer + low-pass FIR for use in real-time and offline
/// pipelines. Preserves filter delay-line state across `process()`
/// calls, so a long capture can be processed in chunks without
/// boundary artefacts.
///
/// ## Convolution layout
///
/// The delay line is stored at **double length** (`2 · num_taps`).
/// Each new sample is written to both `delay_line[idx]` and
/// `delay_line[idx + num_taps]`. The convolution then reads
/// `num_taps` contiguous slots starting at `idx + 1`, never crossing
/// the buffer wrap. This removes the modulo from the FIR inner loop
/// — the modulo creates a data dependency that defeats LLVM's
/// auto-vectoriser, so dropping it lets the inner loop compile down
/// to a tight FMA chain on NEON / AVX2. Cost is one extra store and
/// `num_taps` extra `Complex<f32>` of memory; gain is ~3-4× on the
/// inner kernel.
///
/// Taps are stored **pre-reversed** in `taps_for_conv` so the
/// convolution iterates both `delay_line` and `taps_for_conv` in
/// the same forward direction. The original `taps` are also kept
/// for callers that want to introspect the impulse response.
pub struct StreamingDDC {
    /// LO phasor — kept on the unit circle by per-sample Newton
    /// renormalisation. `phasor.re` is `cos(current_phase)`,
    /// `phasor.im` is `sin(current_phase)`.
    phasor: Complex<f32>,
    /// Per-sample phasor advance, `exp(j·-2π·freq_offset/fs)`.
    step_phasor: Complex<f32>,
    /// Real FIR impulse response (Blackman-windowed sinc), in
    /// natural time order. Kept for caller introspection / future
    /// design changes.
    taps: Vec<f32>,
    /// Convolution coefficients: `taps_for_conv[k]` multiplies the
    /// k-th-oldest sample in the contiguous `[idx+1, idx+1+N)`
    /// window. For symmetric (linear-phase) FIRs this equals
    /// `taps`; we still pre-reverse here so a non-symmetric FIR
    /// (e.g. a future Kaiser-windowed asymmetric design) drops in
    /// without an indexing rewrite.
    taps_for_conv: Vec<f32>,
    /// Doubled-length delay line: `delay_line[i] == delay_line[i +
    /// num_taps]` is maintained as an invariant by the dual-write
    /// pattern in `process_into`.
    delay_line: Vec<Complex<f32>>,
    /// Current write index into the lower half of `delay_line`,
    /// in `[0, num_taps)`.
    idx: usize,
    /// Counter for polyphase decimation.
    decimation_counter: usize,
}

impl StreamingDDC {
    /// Construct a new DDC with the default 63-tap Blackman-windowed
    /// sinc filter. `cutoff_hz` is the one-sided cutoff frequency
    /// (passband edge); typically set this to
    /// `fm_deviation + headroom` where headroom covers chroma peaks
    /// (about 7 MHz for PAL).
    pub fn new(freq_offset_hz: f32, sample_rate: u32, cutoff_hz: f32) -> Self {
        Self::with_taps(freq_offset_hz, sample_rate, cutoff_hz, DEFAULT_FIR_TAPS)
    }

    /// Construct with a caller-specified tap count. Use this when
    /// the default 63 taps don't fit a particular SNR / latency
    /// budget. Odd tap counts give a true linear-phase response.
    ///
    /// `num_taps` is clamped to a minimum of 3: fewer taps make the
    /// Blackman window collapse to all-zeros (its endpoints are ~0 by
    /// construction), which would normalise the impulse response to
    /// `NaN`, and `num_taps == 0` would underflow the tap-design
    /// arithmetic. A sub-3-tap anti-alias FIR is meaningless anyway.
    pub fn with_taps(
        freq_offset_hz: f32,
        sample_rate: u32,
        cutoff_hz: f32,
        num_taps: usize,
    ) -> Self {
        let num_taps = num_taps.max(3);
        // `sample_rate` is a divisor for both the LO phase advance and the
        // normalised cutoff; a 0 rate would make `phase_adv` / `cutoff_norm`
        // ±Inf → NaN taps and a NaN LO. Clamp to 1 Hz (degenerate but finite).
        let sample_rate = sample_rate.max(1);
        let phase_adv = -2.0 * PI * freq_offset_hz / sample_rate as f32;
        let (step_im, step_re) = phase_adv.sin_cos();
        let step_phasor = Complex::new(step_re, step_im);

        let cutoff_norm = cutoff_hz / sample_rate as f32;
        let m = (num_taps - 1) as f32 / 2.0;
        let mut taps = vec![0.0f32; num_taps];
        for i in 0..num_taps {
            let n = i as f32 - m;
            // Windowed sinc — exact value at n=0 is `2·cutoff_norm`
            // (the impulse-response peak), avoiding the 0/0 form.
            let sinc = if n.abs() < 1e-6 {
                2.0 * cutoff_norm
            } else {
                (2.0 * PI * cutoff_norm * n).sin() / (PI * n)
            };
            // Blackman window — > 50 dB stopband.
            let window = 0.42 - 0.5 * (2.0 * PI * i as f32 / (num_taps - 1) as f32).cos()
                + 0.08 * (4.0 * PI * i as f32 / (num_taps - 1) as f32).cos();
            taps[i] = sinc * window;
        }
        // Normalise to unity gain at DC. A degenerate design (e.g.
        // `cutoff_hz == 0`, which zeros every sinc term → `sum == 0`) would
        // otherwise divide by zero and yield NaN taps; fall back to a centred
        // unit impulse (unity-gain passthrough) so the filter stays finite.
        let sum: f32 = taps.iter().sum();
        if sum.is_finite() && sum.abs() > 1e-20 {
            for t in &mut taps {
                *t /= sum;
            }
        } else {
            for t in taps.iter_mut() {
                *t = 0.0;
            }
            taps[num_taps / 2] = 1.0;
        }

        // Pre-reversed taps for the doubled-buffer convolution.
        // `taps_for_conv[k]` will multiply the k-th-oldest sample in
        // the contiguous read window starting at `idx+1`.
        let taps_for_conv: Vec<f32> = taps.iter().rev().copied().collect();

        Self {
            phasor: Complex::new(1.0, 0.0),
            step_phasor,
            taps,
            taps_for_conv,
            // Doubled-length delay line: the upper half mirrors the
            // lower half via the dual-write in `process_into`.
            delay_line: vec![Complex::new(0.0, 0.0); 2 * num_taps],
            idx: 0,
            decimation_counter: 0,
        }
    }

    /// Number of FIR taps; used by callers that need to allocate a
    /// matching scratch buffer.
    pub fn num_taps(&self) -> usize {
        self.taps.len()
    }

    /// Mix + filter a chunk of complex samples. Output length equals
    /// input length (this is a non-decimating filter — pair with a
    /// downstream decimator if you need rate reduction).
    pub fn process(&mut self, iq: &[Complex<f32>]) -> Vec<Complex<f32>> {
        let mut output = Vec::with_capacity(iq.len());
        self.process_into(iq, &mut output);
        output
    }

    /// Same as [`Self::process`] but appends into a caller-supplied
    /// `Vec`, letting the caller reuse the allocation across chunks.
    pub fn process_into(&mut self, iq: &[Complex<f32>], output: &mut Vec<Complex<f32>>) {
        self.process_into_decimated(iq, output, 1);
    }

    /// Mix + filter + decimate a chunk of complex samples. Output length is
    /// approximately `input.len() / decimation_factor`.
    pub fn process_decimated(
        &mut self,
        iq: &[Complex<f32>],
        decimation_factor: usize,
    ) -> Vec<Complex<f32>> {
        // `decimation_factor` of 0 is nonsensical (and would divide by
        // zero in the capacity estimate); treat it as 1 (no decimation),
        // matching the clamp in `process_into_decimated`.
        let decimation_factor = decimation_factor.max(1);
        let mut output = Vec::with_capacity(iq.len() / decimation_factor + 1);
        self.process_into_decimated(iq, &mut output, decimation_factor);
        output
    }

    /// Same as [`Self::process_decimated`] but appends into a caller-supplied `Vec`.
    pub fn process_into_decimated(
        &mut self,
        iq: &[Complex<f32>],
        output: &mut Vec<Complex<f32>>,
        decimation_factor: usize,
    ) {
        // Guard the stride: a factor of 0 would make `decimation_counter
        // >= decimation_factor` always true (resetting to 0 every sample,
        // i.e. no decimation) but is a caller error — normalise to 1.
        let decimation_factor = decimation_factor.max(1);
        let num_taps = self.taps.len();
        for &sample in iq {
            // Mix: sample × phasor.
            let mixed = Complex::new(
                sample.re * self.phasor.re - sample.im * self.phasor.im,
                sample.re * self.phasor.im + sample.im * self.phasor.re,
            );

            // Dual-write: keep `delay_line[idx]` and
            // `delay_line[idx + num_taps]` in sync so the read
            // window below is always contiguous.
            self.delay_line[self.idx] = mixed;
            self.delay_line[self.idx + num_taps] = mixed;

            // Only run the FIR convolution if this sample aligns with the decimation stride.
            if self.decimation_counter == 0 {
                let mut sum_re = 0.0f32;
                let mut sum_im = 0.0f32;
                let win = &self.delay_line[self.idx + 1..self.idx + 1 + num_taps];
                let taps = &self.taps_for_conv[..];
                for k in 0..num_taps {
                    sum_re += win[k].re * taps[k];
                    sum_im += win[k].im * taps[k];
                }
                output.push(Complex::new(sum_re, sum_im));
            }

            self.idx = if self.idx + 1 == num_taps {
                0
            } else {
                self.idx + 1
            };

            self.decimation_counter += 1;
            if self.decimation_counter >= decimation_factor {
                self.decimation_counter = 0;
            }

            // Advance LO phasor and renormalise. `0.5·(3 − |φ|²)`
            // is a single-step Newton iteration for `1/sqrt(x)` near
            // x=1 — one MAC, no transcendental.
            self.phasor *= self.step_phasor;
            let mag_sq = self.phasor.re * self.phasor.re + self.phasor.im * self.phasor.im;
            let inv = 0.5 * (3.0 - mag_sq);
            self.phasor.re *= inv;
            self.phasor.im *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure tone at DC should pass through (LO at 0 Hz, FIR is unity-
    /// gain at DC).
    #[test]
    fn ddc_passes_dc() {
        let mut ddc = StreamingDDC::new(0.0, 1_000_000, 100_000.0);
        let iq = vec![Complex::new(1.0, 0.0); 1024];
        let out = ddc.process(&iq);
        // After filter settling (~num_taps samples), output should
        // approach the input.
        let tail = &out[200..];
        let mean: f32 = tail.iter().map(|c| c.re).sum::<f32>() / tail.len() as f32;
        assert!((mean - 1.0).abs() < 0.01, "DC gain drifted: {}", mean);
    }

    /// A tone at the LO offset frequency should be down-converted to
    /// DC. Feed a +200 kHz tone, set LO = +200 kHz, expect DC-ish
    /// output magnitude ≈ 1.
    #[test]
    fn ddc_mixes_tone_to_dc() {
        let fs = 1_000_000u32;
        let f_tone = 200_000.0f32;
        let n = 4096;
        let iq: Vec<Complex<f32>> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                Complex::from_polar(1.0, 2.0 * PI * f_tone * t)
            })
            .collect();

        let mut ddc = StreamingDDC::new(f_tone, fs, 50_000.0);
        let out = ddc.process(&iq);
        // After settling, the magnitude should be close to 1 and the
        // signal close to DC. Average over the tail.
        let tail = &out[200..];
        let mean_mag: f32 = tail.iter().map(|c| c.norm()).sum::<f32>() / tail.len() as f32;
        assert!(
            (mean_mag - 1.0).abs() < 0.05,
            "tone-to-DC magnitude drift: {}",
            mean_mag,
        );
    }

    /// Regression test for the doubled-delay-line indexing. Replaces
    /// the FIR taps with an **asymmetric** impulse response (a unit
    /// impulse at tap 0, zeros elsewhere) and verifies the filter
    /// implements a 1-sample delay rather than some shifted version
    /// of it. Using a symmetric FIR like the default Blackman-sinc
    /// would mask any "convolution is reading in the wrong direction"
    /// bug because the same tap sits at both ends of the impulse.
    #[test]
    fn convolution_handles_asymmetric_taps() {
        // LO at 0 Hz so the mixer is identity — we want to isolate
        // the FIR's behaviour.
        let mut ddc = StreamingDDC::new(0.0, 1_000_000, 100_000.0);
        // Overwrite the designed FIR with a unit impulse at tap 0.
        // taps_for_conv keeps its reversed layout so we have to
        // rebuild it from the new taps; the doubled delay line
        // doesn't need any reset because it's already zero-filled.
        let n = ddc.taps.len();
        ddc.taps = vec![0.0; n];
        ddc.taps[0] = 1.0; // impulse: y[n] = x[n - (N-1)]
        ddc.taps_for_conv = ddc.taps.iter().rev().copied().collect();

        // Feed a known sequence and verify the output is a delayed
        // copy. `taps[0] = 1.0, taps[N-1] = 0.0` means tap-0
        // multiplies the *newest* sample in the natural convolution
        // sense; equivalently `taps_for_conv[N-1] = 1.0` so the FIR
        // returns the newest input. A 1-tap impulse at position 0
        // should pass the input through unchanged with zero delay.
        let n_samples = 200;
        let iq: Vec<Complex<f32>> = (0..n_samples)
            .map(|i| Complex::new(i as f32, 0.0))
            .collect();
        let out = ddc.process(&iq);
        for (i, c) in out.iter().enumerate() {
            assert!(
                (c.re - i as f32).abs() < 1e-3 && c.im.abs() < 1e-3,
                "impulse-at-tap-0 should pass input through unchanged: out[{}] = {:?}, expected ({}, 0)",
                i,
                c,
                i,
            );
        }

        // Now try the opposite asymmetric case: impulse at the last
        // tap. This should produce a delay equal to (N-1) samples —
        // the FIR sees the oldest sample in its window.
        let mut ddc = StreamingDDC::new(0.0, 1_000_000, 100_000.0);
        ddc.taps = vec![0.0; n];
        ddc.taps[n - 1] = 1.0;
        ddc.taps_for_conv = ddc.taps.iter().rev().copied().collect();
        let out = ddc.process(&iq);
        // For i < n-1 the delay line still has the initial zeros at
        // the oldest position; output should be ~0.
        for (i, c) in out.iter().enumerate().take(n - 1) {
            assert!(
                c.re.abs() < 1e-3 && c.im.abs() < 1e-3,
                "during warmup the oldest-sample tap should see zero-filled delay line: out[{}] = {:?}",
                i,
                c,
            );
        }
        // After warmup, output should be input delayed by (n-1).
        for (i, c) in out.iter().enumerate().skip(n - 1) {
            let expected = (i - (n - 1)) as f32;
            assert!(
                (c.re - expected).abs() < 1e-3,
                "impulse-at-tap-{} should delay by {}: out[{}] = {:?}, expected ({}, 0)",
                n - 1,
                n - 1,
                i,
                c,
                expected,
            );
        }
    }

    /// A `decimation_factor` of 0 is a caller error but must not panic
    /// (it divides the capacity estimate). It is normalised to 1, so the
    /// output length equals the input length.
    #[test]
    fn decimation_factor_zero_does_not_panic() {
        let mut ddc = StreamingDDC::new(0.0, 1_000_000, 100_000.0);
        let iq = vec![Complex::new(1.0, 0.0); 256];
        let out = ddc.process_decimated(&iq, 0);
        assert_eq!(out.len(), iq.len(), "factor 0 should behave like factor 1");
    }

    /// Degenerate tap counts must not produce a `NaN` filter. The
    /// Blackman window is ~0 at its endpoints, so a 1- or 2-tap design
    /// would normalise to `NaN`; `with_taps` clamps to 3.
    #[test]
    fn degenerate_tap_counts_are_clamped_not_nan() {
        for n in [0usize, 1, 2, 3] {
            let ddc = StreamingDDC::with_taps(0.0, 1_000_000, 100_000.0, n);
            assert!(ddc.num_taps() >= 3, "tap count {n} not clamped");
            let dc_gain: f32 = ddc.taps.iter().sum();
            assert!(
                dc_gain.is_finite() && (dc_gain - 1.0).abs() < 1e-4,
                "taps for n={n} not unity-gain/finite: sum={dc_gain}"
            );
        }
    }

    /// Degenerate design parameters (`sample_rate == 0`, `cutoff_hz == 0`)
    /// must yield a finite, unity-DC-gain filter and a finite LO step,
    /// never NaN taps or a NaN phasor.
    #[test]
    fn degenerate_design_params_dont_nan() {
        for (fs, cutoff) in [(0u32, 100_000.0f32), (1_000_000, 0.0), (0, 0.0)] {
            let mut ddc = StreamingDDC::with_taps(50_000.0, fs, cutoff, 63);
            let dc_gain: f32 = ddc.taps.iter().sum();
            assert!(
                dc_gain.is_finite() && (dc_gain - 1.0).abs() < 1e-4,
                "fs={fs} cutoff={cutoff}: taps not finite unity-gain (sum={dc_gain})"
            );
            assert!(
                ddc.step_phasor.re.is_finite() && ddc.step_phasor.im.is_finite(),
                "fs={fs} cutoff={cutoff}: LO step phasor is not finite"
            );
            // Filtering must stay finite too.
            let out = ddc.process(&[Complex::new(1.0, 0.0); 128]);
            assert!(
                out.iter().all(|c| c.re.is_finite() && c.im.is_finite()),
                "fs={fs} cutoff={cutoff}: DDC output contains non-finite samples"
            );
        }
    }

    /// Phasor magnitude should stay on the unit circle across long
    /// runs — verifies the Newton renormalisation keeps round-off
    /// drift bounded.
    #[test]
    fn phasor_magnitude_remains_unity() {
        let mut ddc = StreamingDDC::new(123_456.7, 1_000_000, 100_000.0);
        // Process enough samples to amplify any drift.
        let iq = vec![Complex::new(0.0, 0.0); 200_000];
        let _ = ddc.process(&iq);
        let mag = ddc.phasor.norm();
        assert!(
            (mag - 1.0).abs() < 1e-3,
            "phasor magnitude drifted: {}",
            mag,
        );
    }
}
