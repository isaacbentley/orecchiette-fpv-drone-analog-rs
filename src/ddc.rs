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
//! both by the live fpv-viewer binary on its channel-decode hot path
//! and by the detector's sliding-DDC sweep via the private
//! `ddc_and_decimate` helper in `detector.rs` (which constructs a
//! `StreamingDDC` per probe and calls `process_decimated`). The
//! earlier length-N boxcar (`sum/N`) the sweep used had a sinc
//! magnitude response with poor stopband
//! attenuation — adjacent-band energy leaked back into the
//! discriminator passband, costing CNR margin in the FM
//! threshold-effect region and synthesising spurious harmonic content
//! from strong out-of-band tones. The windowed-sinc FIR closes that
//! gap (> 50 dB stopband at `target_rate/3` cutoff).
//!
//! ## Phase tracking
//!
//! For frequency translation, the mixer LO uses phasor recursion: each step is a single complex
//! multiply by `exp(j·phase_adv)` instead of a `sincos` call.
//! Magnitude is renormalised every sample with a first-order Newton
//! step (`0.5·(3 − |φ|²)`) so f32 round-off doesn't drift |phasor|
//! away from 1. A centered channel with an identity LO skips mixing and
//! oscillator updates while retaining the full anti-alias filter.

use num_complex::Complex;
use std::f32::consts::PI;
use wide::f32x8;

/// Default tap count for the anti-alias FIR. 63 taps with a
/// Blackman window gives > 50 dB stopband attenuation, which is
/// roughly 40 dB better than the boxcar of equivalent length.
pub const DEFAULT_FIR_TAPS: usize = 63;

/// Design a Blackman-windowed-sinc low-pass FIR, in natural (non-reversed)
/// impulse-response order, normalised to unity DC gain. Factored out of
/// [`StreamingDDC::with_taps`] so the `gpu` feature's batched DDC compute
/// shader can upload the *exact* same filter design the CPU path uses —
/// the sliding-DDC sweep's PAL/NTSC classification depends on the FIR's
/// passband/stopband shape, so GPU and CPU probes must share one design
/// function rather than risk two implementations drifting apart.
///
/// `num_taps` is clamped to a minimum of 3 (see [`StreamingDDC::with_taps`]
/// for why) and `sample_rate` to a minimum of 1 Hz.
pub(crate) fn design_fir_taps(cutoff_hz: f32, sample_rate: u32, num_taps: usize) -> Vec<f32> {
    let num_taps = num_taps.max(3);
    let sample_rate = sample_rate.max(1);
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
    taps
}

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
/// (the modulo creates a data dependency and an unpredictable branch)
/// and — more importantly now — makes the read window contiguous,
/// which is what lets the kernel reinterpret it as interleaved
/// `[re,im]` floats and go explicitly SIMD. Cost is one extra store
/// and `num_taps` extra `Complex<f32>` of memory.
///
/// Note the inner loop is *not* auto-vectorised and never was: it is a
/// float dot-product reduction, and LLVM may not reassociate FP adds,
/// so the scalar `sum += …` form only ever became a sequential FMA
/// chain regardless of `target-cpu`. The kernel therefore writes the
/// lanes itself, with four independent accumulators to overlap multiply-add
/// latency (see `process_kernel`). This changes summation order, not taps.
///
/// Taps are stored **pre-reversed** (and pair-duplicated for the SIMD
/// kernel) in `taps_dup`, so the convolution iterates both `delay_line`
/// and the taps in the same forward direction. The original `taps` are
/// also kept for callers that want to introspect the impulse response.
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
    /// Convolution coefficients, **pre-reversed** (so the kernel walks
    /// `delay_line` and the taps in the same forward direction) and
    /// **pair-duplicated** — `[h0,h0,h1,h1,…]` — so the SIMD kernel in
    /// `process_into_decimated` can multiply them straight against the
    /// interleaved `[re,im]` float view of the delay-line window: even
    /// lanes accumulate the real products, odd lanes the imaginary
    /// ones. `taps_dup[2k]` recovers plain reversed tap `k`. Derived
    /// from `taps` only via [`Self::derive_conv_taps`].
    taps_dup: Vec<f32>,
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
        // `sample_rate` is a divisor for the normalised cutoff (and the
        // LO phase advance below); clamp to 1 Hz (degenerate but finite).
        let sample_rate = sample_rate.max(1);
        let taps = design_fir_taps(cutoff_hz, sample_rate, num_taps);
        Self::from_designed_taps(freq_offset_hz, sample_rate, taps)
    }

    /// Construct from an already-designed impulse response (e.g. one
    /// cached by the detector's sweep, which builds a `StreamingDDC`
    /// per probe with identical `cutoff/sample_rate` — re-running the
    /// tap-design `sin_cos` loop per probe was pure waste). `taps` must
    /// come from [`design_fir_taps`] (or be an equivalent finite,
    /// unity-DC-gain design of length ≥ 3).
    pub(crate) fn from_designed_taps(
        freq_offset_hz: f32,
        sample_rate: u32,
        taps: Vec<f32>,
    ) -> Self {
        debug_assert!(taps.len() >= 3, "FIR needs at least 3 taps");
        let num_taps = taps.len();
        // A 0 rate would make `phase_adv` ±Inf → a NaN LO. Clamp to 1 Hz.
        let sample_rate = sample_rate.max(1);
        let phase_adv = -2.0 * PI * freq_offset_hz / sample_rate as f32;
        let (step_im, step_re) = phase_adv.sin_cos();
        let step_phasor = Complex::new(step_re, step_im);

        // Pre-reversed taps for the doubled-buffer convolution.
        // `taps_for_conv[k]` will multiply the k-th-oldest sample in
        // the contiguous read window starting at `idx+1`.
        let taps_dup = Self::derive_conv_taps(&taps);

        Self {
            phasor: Complex::new(1.0, 0.0),
            step_phasor,
            taps,
            taps_dup,
            // Doubled-length delay line: the upper half mirrors the
            // lower half via the dual-write in `process_into`.
            delay_line: vec![Complex::new(0.0, 0.0); 2 * num_taps],
            idx: 0,
            decimation_counter: 0,
        }
    }

    /// Builds the kernel's `taps_dup` layout from the natural-order
    /// impulse response. `taps` is the single source of truth and
    /// `taps_dup` is a cache of it, so it is only ever produced here —
    /// anything that reassigns `taps` must rebuild through this or the
    /// filter silently keeps convolving with the old response.
    fn derive_conv_taps(taps: &[f32]) -> Vec<f32> {
        let mut taps_dup = Vec::with_capacity(taps.len() * 2);
        for &t in taps.iter().rev() {
            taps_dup.push(t);
            taps_dup.push(t);
        }
        taps_dup
    }

    /// Number of FIR taps; used by callers that need to allocate a
    /// matching scratch buffer.
    /// Clear the filter state after a break in the signal.
    ///
    /// The delay line holds the last `num_taps` samples and convolves
    /// them with whatever arrives next. Across a dropped chunk those
    /// samples are no longer adjacent to the new ones, so the filter
    /// would blend two moments that never touched — a transient the
    /// discriminator turns into a frequency spike, which reads
    /// downstream as a sync edge that was never transmitted.
    ///
    /// The phasor is deliberately *not* reset: it tracks the mixer's
    /// own phase, which is a function of elapsed samples rather than of
    /// signal continuity, and restarting it would put a step in the
    /// down-conversion where there was none.
    pub fn reset(&mut self) {
        self.delay_line
            .iter_mut()
            .for_each(|s| *s = Complex::new(0.0, 0.0));
        self.idx = 0;
        self.decimation_counter = 0;
    }

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
        // A centered channel needs the FIR but no frequency translation.
        // Dispatch once per chunk, so the identity case has neither an
        // oscillator recurrence nor a branch in its per-sample loop.
        let identity = Complex::new(1.0, 0.0);
        if self.step_phasor == identity && self.phasor == identity {
            self.process_kernel::<false>(iq, output, decimation_factor);
        } else {
            self.process_kernel::<true>(iq, output, decimation_factor);
        }
    }

    fn process_kernel<const MIX: bool>(
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
            let mixed = if MIX {
                Complex::new(
                    sample.re * self.phasor.re - sample.im * self.phasor.im,
                    sample.re * self.phasor.im + sample.im * self.phasor.re,
                )
            } else {
                sample
            };

            // Dual-write: keep `delay_line[idx]` and
            // `delay_line[idx + num_taps]` in sync so the read
            // window below is always contiguous.
            self.delay_line[self.idx] = mixed;
            self.delay_line[self.idx + num_taps] = mixed;

            // Only run the FIR convolution if this sample aligns with the decimation stride.
            if self.decimation_counter == 0 {
                // The dual-write above guarantees this window is
                // contiguous, so view it as interleaved [re,im] floats
                // and run the dot product in f32x8 lanes against the
                // pair-duplicated taps (even lanes → re, odd → im).
                //
                // Explicit SIMD is required: this is a float reduction,
                // and LLVM may not reassociate FP adds, so the scalar
                // `sum += …` form compiles to a sequential FMA chain no
                // matter the target-cpu. Splitting into lanes does
                // reorder the summation, which for a FIR is benign (and
                // if anything more accurate than one long chain).
                let win = &self.delay_line[self.idx + 1..self.idx + 1 + num_taps];
                let flat: &[f32] = bytemuck::cast_slice(win);
                // Independent accumulators let the CPU overlap multiply-adds
                // instead of waiting on one dependency chain for all taps.
                let mut acc0 = f32x8::ZERO;
                let mut acc1 = f32x8::ZERO;
                let mut acc2 = f32x8::ZERO;
                let mut acc3 = f32x8::ZERO;
                let (fv, ftail) = flat.as_chunks::<32>();
                let (hv, htail) = self.taps_dup.as_chunks::<32>();
                for (v, t) in fv.iter().zip(hv) {
                    let (v, _) = v.as_chunks::<8>();
                    let (t, _) = t.as_chunks::<8>();
                    acc0 = f32x8::from(v[0]).mul_add(f32x8::from(t[0]), acc0);
                    acc1 = f32x8::from(v[1]).mul_add(f32x8::from(t[1]), acc1);
                    acc2 = f32x8::from(v[2]).mul_add(f32x8::from(t[2]), acc2);
                    acc3 = f32x8::from(v[3]).mul_add(f32x8::from(t[3]), acc3);
                }
                let (fv, ftail) = ftail.as_chunks::<8>();
                let (hv, htail) = htail.as_chunks::<8>();
                for (&v, &t) in fv.iter().zip(hv) {
                    acc0 = f32x8::from(v).mul_add(f32x8::from(t), acc0);
                }
                let a = ((acc0 + acc1) + (acc2 + acc3)).to_array();
                let mut sum_re = a[0] + a[2] + a[4] + a[6];
                let mut sum_im = a[1] + a[3] + a[5] + a[7];
                // `flat` is 2·num_taps long, so the tail is an even
                // number of floats — i.e. whole [re,im] pairs.
                let (fv, _) = ftail.as_chunks::<2>();
                let (hv, _) = htail.as_chunks::<2>();
                for (vp, tp) in fv.iter().zip(hv) {
                    sum_re += vp[0] * tp[0];
                    sum_im += vp[1] * tp[1];
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
            if MIX {
                self.phasor *= self.step_phasor;
                let mag_sq = self.phasor.re * self.phasor.re + self.phasor.im * self.phasor.im;
                let inv = 0.5 * (3.0 - mag_sq);
                self.phasor.re *= inv;
                self.phasor.im *= inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent direct convolution with f64 accumulation checks the SIMD
    /// grouping, short/tail tap counts, mixer state and reset semantics.
    #[test]
    fn streaming_kernel_matches_direct_convolution_across_gaps() {
        let iq: Vec<_> = (0..701)
            .map(|i| Complex::new((i as f32 * 0.731).sin(), (i as f32 * 0.317).cos()))
            .collect();
        for taps in [3, 4, 7, 16, 31, 63, 127, 128] {
            for offset in [0.0, 0.1, -850_000.0] {
                for factor in [0usize, 1, 3, 4] {
                    let mut ddc = StreamingDDC::with_taps(offset, 15_360_000, 3e6, taps);
                    let mut phase = Complex::new(1.0f32, 0.0);
                    for segment in [&iq[..173], &iq[173..]] {
                        ddc.reset();
                        let mixed: Vec<_> = segment
                            .iter()
                            .map(|&sample| {
                                let result = sample * phase;
                                phase *= ddc.step_phasor;
                                let inv = 0.5 * (3.0 - phase.norm_sqr());
                                phase.re *= inv;
                                phase.im *= inv;
                                result
                            })
                            .collect();
                        let expected: Vec<_> = (0..mixed.len())
                            .step_by(factor.max(1))
                            .map(|i| {
                                let mut sum = Complex::new(0.0f64, 0.0);
                                for k in 0..taps.min(i + 1) {
                                    sum.re += f64::from(mixed[i - k].re) * f64::from(ddc.taps[k]);
                                    sum.im += f64::from(mixed[i - k].im) * f64::from(ddc.taps[k]);
                                }
                                Complex::new(sum.re as f32, sum.im as f32)
                            })
                            .collect();
                        let sentinel = Complex::new(42.0, -42.0);
                        let mut actual = vec![sentinel];
                        for part in segment.chunks(17) {
                            ddc.process_into_decimated(&[], &mut actual, factor);
                            ddc.process_into_decimated(part, &mut actual, factor);
                        }
                        assert_eq!(actual[0], sentinel, "output must append");
                        assert_eq!(actual.len() - 1, expected.len());
                        for (i, (a, b)) in actual[1..].iter().zip(expected).enumerate() {
                            assert!(
                                (*a - b).norm() < 2e-6,
                                "taps={taps} offset={offset} factor={factor} sample={i}: {a:?} != {b:?}"
                            );
                        }
                    }
                }
            }
        }
    }

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
        ddc.taps_dup = StreamingDDC::derive_conv_taps(&ddc.taps);

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
        ddc.taps_dup = StreamingDDC::derive_conv_taps(&ddc.taps);
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

    /// Decimation phase must survive chunk boundaries.
    ///
    /// `decimation_counter` is struct state, so a stream fed in chunks
    /// has to produce exactly what the same stream fed in one call
    /// produces. If the counter reset per call, every chunk would
    /// restart the stride at phase 0 and the output would carry a
    /// timing discontinuity at each boundary — inaudible in a length
    /// check but fatal to a decoder downstream, which is exactly how
    /// the viewer consumes this (65,536-sample packets, one call each).
    ///
    /// Chunk sizes here are deliberately NOT multiples of the factor,
    /// so the phase genuinely carries a non-zero remainder across most
    /// boundaries.
    #[test]
    fn decimation_phase_survives_chunk_boundaries() {
        let sample_rate = 4_000_000;
        // A tone off DC, so both the mixer phase and the FIR history
        // have to line up for the outputs to match. Chunk sizes below
        // are deliberately not multiples of the factors.
        let iq: Vec<Complex<f32>> = (0..8_192)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let ph = 2.0 * PI * 250_000.0 * t;
                Complex::new(ph.cos(), ph.sin())
            })
            .collect();

        // A non-zero LO, so the mixer phasor has to carry across calls
        // too — with the LO at DC the mixer is the identity and this
        // would only exercise the FIR history and the stride.
        let lo = 137_000.0;
        for factor in [2usize, 3, 4, 5] {
            let mut whole = StreamingDDC::new(lo, sample_rate, 400_000.0);
            let mut one_shot = Vec::new();
            whole.process_into_decimated(&iq, &mut one_shot, factor);

            for chunk in [100usize, 333, 1024, 4096] {
                let mut streamed_ddc = StreamingDDC::new(lo, sample_rate, 400_000.0);
                let mut streamed = Vec::new();
                for part in iq.chunks(chunk) {
                    streamed_ddc.process_into_decimated(part, &mut streamed, factor);
                }
                assert_eq!(
                    streamed.len(),
                    one_shot.len(),
                    "factor {factor}, chunk {chunk}: sample count diverged"
                );
                for (i, (a, b)) in one_shot.iter().zip(&streamed).enumerate() {
                    assert!(
                        (a - b).norm() < 1e-6,
                        "factor {factor}, chunk {chunk}: sample {i} diverged, \
                         {a:?} one-shot vs {b:?} streamed"
                    );
                }
            }
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

    /// After a gap the delay line holds samples that are no longer
    /// adjacent to what arrives next; convolving the two blends moments
    /// that never touched. Reset must leave nothing of the old signal.
    #[test]
    fn reset_leaves_no_tail_of_the_previous_signal() {
        let mut ddc = StreamingDDC::new(0.0, 1_000_000, 200_000.0);
        let loud: Vec<Complex<f32>> = (0..512).map(|_| Complex::new(10.0, -10.0)).collect();
        let mut out = Vec::new();
        ddc.process_into(&loud, &mut out);

        // Silence straight after the burst still rings: that is the
        // filter's tail, which is correct when the signal is continuous.
        let mut ringing = Vec::new();
        ddc.process_into(&vec![Complex::new(0.0, 0.0); 128], &mut ringing);
        let rung: f32 = ringing.iter().map(|c| c.norm()).sum();
        assert!(rung > 0.0, "a continuous stream should carry its tail");

        // After a reset it must not.
        ddc.reset();
        let mut clean = Vec::new();
        ddc.process_into(&vec![Complex::new(0.0, 0.0); 128], &mut clean);
        let leaked: f32 = clean.iter().map(|c| c.norm()).sum();
        assert!(
            leaked < 1e-6,
            "reset still leaked {leaked} of the previous signal"
        );
    }
}
