//! Standards-shaped synthetic NTSC/PAL field generator, for tests (both
//! in this crate and downstream) and for regenerating orecchiette's
//! `synthetic_fpv` fixture.
//!
//! Unlike the crate's existing ad hoc test helpers (`make_pal_pulse_train`
//! in `detector.rs`, `make_synthetic_demod` in `detector.rs`), this
//! generator produces a *complete* field: pre-equalizing pulses,
//! serrated broad (vertical sync) pulses, post-equalizing pulses,
//! standards-correct blanking, and an active-video test pattern — the
//! full structure [`crate::vbi`]'s parser (added alongside the
//! reconstructor integration) needs to exercise, not just a bare
//! line-rate comb.
//!
//! Output is in the same "demod-domain" convention as
//! [`crate::demod::fm_demod`]'s return value (instantaneous frequency,
//! radians/sample) — [`generate_fields`] can be fed directly to
//! [`crate::video::FrameReconstructor::reconstruct_frame_into`] or to
//! [`crate::levels::estimate_fm_deviation`] with no intermediate
//! modulation step. [`generate_iq`] additionally FM-modulates that
//! baseband onto a complex carrier, for tests that exercise the
//! detector's `detect_from_iq` entry point.

use crate::vbi::{FieldParity, consts};
use num_complex::Complex;
use std::f32::consts::PI;

/// Output frame width, matching [`crate::video::FrameReconstructor::width`]
/// — [`TestPattern::WhiteSquare`] columns are specified in this
/// coordinate space so a geometry test's assertion lines up directly
/// with pixels the reconstructor actually produces.
pub const OUTPUT_WIDTH: usize = 720;

/// Back-porch duration used for regular (non-vertical-sync) lines,
/// between the H-sync pulse and the start of active-video content.
/// Not standards-critical for this generator's purposes — it just
/// needs to be a few microseconds so [`crate::levels::estimate_fm_deviation`]'s
/// porch-sampling window lands on blanking level rather than active
/// content (see the accuracy discussion in `levels.rs`).
const BACK_PORCH_S: f64 = 5.8e-6;

/// A test pattern painted into a field's active-video region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestPattern {
    /// Constant luma level, in IRE units (0 = black/blanking level,
    /// 100 = peak white).
    Flat(f32),
    /// Linear ramp from 0 to 100 IRE across each active line's width.
    GrayRamp,
    /// Eight vertical luma steps (0..100 IRE) across the line width.
    Bars,
    /// A bright (100 IRE) rectangle on an otherwise black (0 IRE)
    /// field, for reconstruction-geometry tests. `row0`/`h` are in
    /// *field-local active-line* coordinates (`0..field_lines`, i.e.
    /// what [`crate::video::FrameReconstructor::reconstruct_frame_into`]
    /// renders on a single call, before interlace merging) — not the
    /// final interlaced frame's row range. `col0`/`w` are in
    /// [`OUTPUT_WIDTH`]-relative columns.
    WhiteSquare {
        row0: usize,
        col0: usize,
        h: usize,
        w: usize,
    },
}

/// Configuration for [`generate_fields`] / [`generate_iq`].
#[derive(Debug, Clone, Copy)]
pub struct SyntheticVideoConfig {
    pub sample_rate: u32,
    pub is_pal: bool,
    /// True peak FM deviation this signal should carry, in Hz — what
    /// [`crate::levels::estimate_fm_deviation`] is expected to recover.
    pub deviation_hz: f32,
    pub pattern: TestPattern,
    /// Which field the generated sequence starts on. Only matters for
    /// tests that care about absolute parity (e.g. asserting the
    /// parser reports `FieldParity::Second` for a sequence that starts
    /// there); for multi-field sequences, parity always alternates
    /// field-to-field regardless of the starting value.
    pub start_field: FieldParity,
    /// Standard deviation of additive Gaussian-ish noise, in the same
    /// radians/sample units as the output.
    pub noise_sigma: f32,
    /// Constant offset added to every sample, simulating a DDC tuning
    /// error or DC bias — the estimator and parser are both meant to
    /// be invariant to this.
    pub dc_offset: f32,
}

struct StandardParams {
    line_hz: f64,
    field_total_lines: f64,
    active_lines: usize,
    eq_pulses: usize,
    broad_pulses: usize,
    posteq_pulses: usize,
    eq_width_s: f64,
    broad_low_s: f64,
    base_active_start_lines: f64,
}

fn params(is_pal: bool) -> StandardParams {
    if is_pal {
        StandardParams {
            line_hz: consts::PAL_LINE_HZ,
            field_total_lines: consts::PAL_FIELD_TOTAL_LINES,
            active_lines: consts::PAL_ACTIVE_LINES,
            eq_pulses: consts::PAL_EQ_PULSES,
            broad_pulses: consts::PAL_BROAD_PULSES,
            posteq_pulses: consts::PAL_POSTEQ_PULSES,
            eq_width_s: consts::PAL_EQ_WIDTH_S,
            broad_low_s: consts::PAL_BROAD_LOW_S,
            base_active_start_lines: consts::PAL_BASE_ACTIVE_START_LINES,
        }
    } else {
        StandardParams {
            line_hz: consts::NTSC_LINE_HZ,
            field_total_lines: consts::NTSC_FIELD_TOTAL_LINES,
            active_lines: consts::NTSC_ACTIVE_LINES,
            eq_pulses: consts::NTSC_EQ_PULSES,
            broad_pulses: consts::NTSC_BROAD_PULSES,
            posteq_pulses: consts::NTSC_POSTEQ_PULSES,
            eq_width_s: consts::NTSC_EQ_WIDTH_S,
            broad_low_s: consts::NTSC_BROAD_LOW_S,
            base_active_start_lines: consts::NTSC_BASE_ACTIVE_START_LINES,
        }
    }
}

/// Emit a constant-level segment of `dur_s` seconds, using a running
/// `(elapsed_seconds, emitted_samples)` pair so segment durations that
/// aren't a whole number of samples don't accumulate rounding drift
/// over a long multi-field sequence (each segment rounds to the
/// nearest sample boundary in *absolute* elapsed time, not relative to
/// the previous segment).
#[inline]
fn advance_flat(
    out: &mut Vec<f32>,
    emitted: &mut usize,
    t: &mut f64,
    fs: f64,
    dur_s: f64,
    level: f32,
) {
    *t += dur_s;
    let target = (*t * fs).round() as usize;
    let n = target.saturating_sub(*emitted);
    out.resize(out.len() + n, level);
    *emitted = target;
}

/// Same accounting as [`advance_flat`], but each sample's level comes
/// from `level_at(frac)` where `frac` is that sample's fractional
/// position (0.0..1.0) within the segment.
#[inline]
fn advance_fn(
    out: &mut Vec<f32>,
    emitted: &mut usize,
    t: &mut f64,
    fs: f64,
    dur_s: f64,
    mut level_at: impl FnMut(f32) -> f32,
) {
    *t += dur_s;
    let target = (*t * fs).round() as usize;
    let n = target.saturating_sub(*emitted);
    for k in 0..n {
        let frac = if n > 1 {
            k as f32 / (n - 1) as f32
        } else {
            0.0
        };
        out.push(level_at(frac));
    }
    *emitted = target;
}

fn pattern_ire(pattern: &TestPattern, active_line_idx: usize, col: usize) -> f32 {
    match *pattern {
        TestPattern::Flat(ire) => ire,
        TestPattern::GrayRamp => (col as f32 / OUTPUT_WIDTH as f32) * 100.0,
        TestPattern::Bars => {
            const N_BARS: usize = 8;
            let bar = (col * N_BARS / OUTPUT_WIDTH).min(N_BARS - 1);
            bar as f32 * (100.0 / (N_BARS - 1) as f32)
        }
        TestPattern::WhiteSquare { row0, col0, h, w } => {
            if active_line_idx >= row0
                && active_line_idx < row0 + h
                && col >= col0
                && col < col0 + w
            {
                100.0
            } else {
                0.0
            }
        }
    }
}

#[inline]
fn xorshift01(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32 / u64::MAX as f32).clamp(0.0, 1.0)
}

/// Irwin-Hall approximation of a unit Gaussian: sum of 12 U(0,1)
/// samples has mean 6, variance 1, so subtracting 6 gives an
/// approximately N(0,1) value without pulling in `rand_distr`.
#[inline]
fn gaussian_noise(state: &mut u64) -> f32 {
    let mut sum = 0.0f32;
    for _ in 0..12 {
        sum += xorshift01(state);
    }
    sum - 6.0
}

/// Generate `n_fields` consecutive fields of demod-domain baseband
/// (radians/sample), alternating parity starting from
/// `cfg.start_field`. See the module doc for the output convention and
/// [`TestPattern`] for the coordinate spaces used by
/// `WhiteSquare`.
pub fn generate_fields(cfg: &SyntheticVideoConfig, n_fields: usize) -> Vec<f32> {
    let p = params(cfg.is_pal);
    let fs = cfg.sample_rate as f64;
    let radians_per_volt = 2.0 * PI * cfg.deviation_hz / cfg.sample_rate as f32;
    // 40 IRE == SYNC_TO_BLANK_FRACTION (0.4) of full deviation swing,
    // so 1 IRE == 0.01 * radians_per_volt. See levels.rs for why this
    // exact convention matters.
    let rad_per_ire = radians_per_volt * 0.01;
    let sync_tip = -40.0 * rad_per_ire;
    let blank = 0.0f32;
    let half_line_s = 0.5 / p.line_hz;
    let line_s = 1.0 / p.line_hz;

    let mut out = Vec::new();
    let mut emitted = 0usize;
    let mut t = 0.0f64;
    let mut parity = cfg.start_field;

    for _ in 0..n_fields {
        let active_start_lines = p.base_active_start_lines
            + if parity == FieldParity::Second {
                0.5
            } else {
                0.0
            };

        let mut lines_in_field = 0.0f64;

        for _ in 0..p.eq_pulses {
            advance_flat(&mut out, &mut emitted, &mut t, fs, p.eq_width_s, sync_tip);
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                half_line_s - p.eq_width_s,
                blank,
            );
        }
        lines_in_field += p.eq_pulses as f64 * 0.5;

        for _ in 0..p.broad_pulses {
            advance_flat(&mut out, &mut emitted, &mut t, fs, p.broad_low_s, sync_tip);
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                half_line_s - p.broad_low_s,
                blank,
            );
        }
        lines_in_field += p.broad_pulses as f64 * 0.5;

        for _ in 0..p.posteq_pulses {
            advance_flat(&mut out, &mut emitted, &mut t, fs, p.eq_width_s, sync_tip);
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                half_line_s - p.eq_width_s,
                blank,
            );
        }
        lines_in_field += p.posteq_pulses as f64 * 0.5;

        while lines_in_field + 1.0 <= active_start_lines {
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                consts::H_SYNC_WIDTH_S,
                sync_tip,
            );
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                line_s - consts::H_SYNC_WIDTH_S,
                blank,
            );
            lines_in_field += 1.0;
        }
        let leftover = active_start_lines - lines_in_field;
        if leftover > 1e-9 {
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                leftover / p.line_hz,
                blank,
            );
            lines_in_field = active_start_lines;
        }

        for active_line_idx in 0..p.active_lines {
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                consts::H_SYNC_WIDTH_S,
                sync_tip,
            );
            advance_flat(&mut out, &mut emitted, &mut t, fs, BACK_PORCH_S, blank);
            let active_content_s = line_s - consts::H_SYNC_WIDTH_S - BACK_PORCH_S;
            advance_fn(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                active_content_s,
                |frac| {
                    let col = ((frac * OUTPUT_WIDTH as f32) as usize).min(OUTPUT_WIDTH - 1);
                    pattern_ire(&cfg.pattern, active_line_idx, col) * rad_per_ire
                },
            );
            lines_in_field += 1.0;
        }

        while lines_in_field + 1.0 <= p.field_total_lines {
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                consts::H_SYNC_WIDTH_S,
                sync_tip,
            );
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                line_s - consts::H_SYNC_WIDTH_S,
                blank,
            );
            lines_in_field += 1.0;
        }
        let trailing = p.field_total_lines - lines_in_field;
        if trailing > 1e-9 {
            advance_flat(
                &mut out,
                &mut emitted,
                &mut t,
                fs,
                trailing / p.line_hz,
                blank,
            );
        }

        parity = match parity {
            FieldParity::First => FieldParity::Second,
            FieldParity::Second => FieldParity::First,
        };
    }

    if cfg.dc_offset != 0.0 || cfg.noise_sigma > 0.0 {
        let mut rng =
            0x1234_5678_9abc_def0u64 ^ (out.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for v in out.iter_mut() {
            *v += cfg.dc_offset;
            if cfg.noise_sigma > 0.0 {
                *v += gaussian_noise(&mut rng) * cfg.noise_sigma;
            }
        }
    }

    out
}

#[inline]
fn wrap_phase(mut p: f32) -> f32 {
    const TWO_PI: f32 = 2.0 * PI;
    while p > PI {
        p -= TWO_PI;
    }
    while p < -PI {
        p += TWO_PI;
    }
    p
}

/// FM-modulate [`generate_fields`]'s baseband onto a complex carrier,
/// with an optional frequency offset (matching how the detector's
/// sliding-DDC sweep sees a signal that isn't exactly centred).
///
/// Callers must keep `deviation_hz` well under `sample_rate / 4` (with
/// margin) — this is a physical Nyquist constraint on FM modulation
/// itself, not an implementation limit: a peak instantaneous frequency
/// excursion approaching `sample_rate / 2` aliases regardless of how
/// carefully it's demodulated. [`generate_fields`]'s baseband has no
/// such constraint (it's not modulated), so estimator/parser tests that
/// don't need real IQ should prefer it directly.
pub fn generate_iq(
    cfg: &SyntheticVideoConfig,
    n_fields: usize,
    carrier_offset_hz: f32,
) -> Vec<Complex<f32>> {
    let baseband = generate_fields(cfg, n_fields);
    let carrier_rad = 2.0 * PI * carrier_offset_hz / cfg.sample_rate as f32;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(baseband.len());
    for &b in &baseband {
        phase = wrap_phase(phase + b + carrier_rad);
        out.push(Complex::new(phase.cos(), phase.sin()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demod::fm_demod;

    fn base_config(is_pal: bool, deviation_hz: f32, sample_rate: u32) -> SyntheticVideoConfig {
        SyntheticVideoConfig {
            sample_rate,
            is_pal,
            deviation_hz,
            pattern: TestPattern::Flat(0.0),
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        }
    }

    /// Count below-threshold runs of a given width range directly on
    /// the raw (unsmoothed) baseband — a cheap structural check
    /// independent of `levels.rs`'s own pulse scan, so this test can't
    /// pass merely because the two implementations share a bug.
    fn count_pulses(data: &[f32], fs: f64, min_us: f64, max_us: f64, threshold: f32) -> usize {
        let min_w = (fs * min_us * 1e-6) as usize;
        let max_w = (fs * max_us * 1e-6) as usize;
        let mut count = 0;
        let mut i = 0;
        while i < data.len() {
            if data[i] < threshold {
                let start = i;
                while i < data.len() && data[i] < threshold {
                    i += 1;
                }
                let w = i - start;
                if w >= min_w.max(1) && w <= max_w {
                    count += 1;
                }
            } else {
                i += 1;
            }
        }
        count
    }

    #[test]
    fn ntsc_field_has_expected_pulse_counts() {
        let cfg = base_config(false, 5e6, 15_360_000);
        let data = generate_fields(&cfg, 1);
        let fs = cfg.sample_rate as f64;
        let rad_per_ire = (2.0 * PI * cfg.deviation_hz / cfg.sample_rate as f32) * 0.01;
        let threshold = -20.0 * rad_per_ire; // halfway between sync tip (-40 IRE) and blank (0)

        let eq_or_h = count_pulses(&data, fs, 1.5, 5.0, threshold);
        let broad = count_pulses(&data, fs, 20.0, 32.0, threshold);
        // 6 pre-eq + 6 post-eq = 12 narrow pulses, plus one H-sync per
        // remaining-blanking/active line thereafter. First parity's
        // active start (21.0 lines) sits exactly 12.0 lines past the
        // 9.0-line trio, so there are 12 plain-blanking H-sync pulses
        // before the 240 active-line H-sync pulses, plus 1 more
        // trailing-blanking pulse after active video (lines_in_field
        // 261.0 -> 262.0, still short of the 262.5-line field total).
        assert_eq!(eq_or_h, 12 + 12 + 240 + 1, "narrow (eq/H-sync) pulse count");
        assert_eq!(broad, 6, "broad (vsync) pulse count");
    }

    #[test]
    fn pal_field_has_expected_pulse_counts() {
        let cfg = base_config(true, 5e6, 15_360_000);
        let data = generate_fields(&cfg, 1);
        let fs = cfg.sample_rate as f64;
        let rad_per_ire = (2.0 * PI * cfg.deviation_hz / cfg.sample_rate as f32) * 0.01;
        let threshold = -20.0 * rad_per_ire;

        let eq_or_h = count_pulses(&data, fs, 1.5, 5.0, threshold);
        let broad = count_pulses(&data, fs, 20.0, 32.0, threshold);
        // 5 pre-eq + 5 post-eq = 10 narrow pulses. First parity's
        // active start (23.5 lines) sits exactly 16.0 lines past the
        // 7.5-line trio, so there are 16 plain-blanking H-sync pulses
        // before the 288 active-line H-sync pulses, plus 1 more
        // trailing-blanking pulse after active video (lines_in_field
        // 311.5 -> 312.5, landing exactly on the 312.5-line field
        // total).
        assert_eq!(eq_or_h, 10 + 16 + 288 + 1, "narrow (eq/H-sync) pulse count");
        assert_eq!(broad, 5, "broad (vsync) pulse count");
    }

    #[test]
    fn broad_pulse_width_matches_standard() {
        for (is_pal, expected_us) in [(false, 27.1f64), (true, 27.3f64)] {
            let cfg = base_config(is_pal, 5e6, 15_360_000);
            let data = generate_fields(&cfg, 1);
            let rad_per_ire = (2.0 * PI * cfg.deviation_hz / cfg.sample_rate as f32) * 0.01;
            let threshold = -20.0 * rad_per_ire;
            // Find the first broad-width run and measure it directly.
            let fs = cfg.sample_rate as f64;
            let min_w = (fs * 20e-6) as usize;
            let mut i = 0;
            let mut found = None;
            while i < data.len() {
                if data[i] < threshold {
                    let start = i;
                    while i < data.len() && data[i] < threshold {
                        i += 1;
                    }
                    if i - start >= min_w {
                        found = Some(i - start);
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            let width_samples = found.expect("expected a broad pulse");
            let width_us = width_samples as f64 / fs * 1e6;
            assert!(
                (width_us - expected_us).abs() < 1.0,
                "is_pal={is_pal}: broad width {width_us} vs expected {expected_us}"
            );
        }
    }

    #[test]
    fn consecutive_fields_alternate_parity_via_half_line_offset() {
        // Two consecutive fields' broad-pulse groups must be offset
        // from each other by half a line relative to the H grid — the
        // structural signature of interlace. Detect it indirectly: the
        // total sample count of field 1 (start -> second field's first
        // pre-eq pulse) must correspond to a half-integer number of
        // lines (262.5 for NTSC), not a whole number.
        let cfg = base_config(false, 5e6, 15_360_000);
        let one_field = generate_fields(&cfg, 1).len();
        let two_fields = generate_fields(&cfg, 2).len();
        let field2_len = two_fields - one_field;
        let fs = cfg.sample_rate as f64;
        let line_period_samples = fs / consts::NTSC_LINE_HZ;
        let field1_lines = one_field as f64 / line_period_samples;
        let field2_lines = field2_len as f64 / line_period_samples;
        assert!(
            (field1_lines - 262.5).abs() < 0.6,
            "field 1 length {field1_lines} lines, expected ~262.5"
        );
        assert!(
            (field2_lines - 262.5).abs() < 0.6,
            "field 2 length {field2_lines} lines, expected ~262.5"
        );
    }

    #[test]
    fn field_duration_matches_standard_line_count() {
        for (is_pal, expected_lines, line_hz) in [
            (false, 262.5, consts::NTSC_LINE_HZ),
            (true, 312.5, consts::PAL_LINE_HZ),
        ] {
            let cfg = base_config(is_pal, 5e6, 15_360_000);
            let n = generate_fields(&cfg, 4).len();
            let fs = cfg.sample_rate as f64;
            let avg_field_lines = (n as f64 / 4.0) / (fs / line_hz);
            assert!(
                (avg_field_lines - expected_lines).abs() < 0.05,
                "is_pal={is_pal}: avg field length {avg_field_lines} lines, expected {expected_lines}"
            );
        }
    }

    #[test]
    fn generate_iq_round_trips_through_fm_demod() {
        // Moderate, Nyquist-safe deviation/rate pair (see generate_iq's
        // doc for why extreme combinations aren't valid FM signals in
        // the first place, independent of this crate's code).
        let cfg = base_config(false, 2e6, 20_000_000);
        let baseband = generate_fields(&cfg, 1);
        let iq = generate_iq(&cfg, 1, 0.0);
        let recovered = fm_demod(&iq);
        assert_eq!(recovered.len(), baseband.len() - 1);
        // fm_demod's first output corresponds to baseband[1..], with a
        // small phase-wrap tolerance instead of exact equality (both
        // are legitimate representations of the same instantaneous
        // frequency near ±π).
        let mut max_err = 0.0f32;
        for i in 0..recovered.len() {
            let err = (recovered[i] - baseband[i + 1]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-2, "max round-trip error {max_err}");
    }
}
