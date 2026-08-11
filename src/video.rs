//! Analog Video Frame Reconstruction
//!
//! This module takes a demodulated FM signal and reconstructs it into 2D frames
//! by identifying H-Sync and V-Sync pulses. The pipeline is monochrome (luma
//! only — see DESIGN.md §9 for why colour is disabled) and features a sub-sample
//! Time Base Corrector (TBC), two-pass sync extraction with MAD outlier
//! rejection, a subcarrier notch for dot-crawl suppression, multi-field temporal
//! denoise + dropout repair (see [`crate::frame_history`]), Dropout Compensation
//! (DOC), and a luma transient-improvement (unsharp) pass.

#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::manual_div_ceil,
    clippy::manual_checked_ops
)]

use crate::frame_history::{FieldMeta, FrameHistory};
use crate::levels::{SyncLevels, estimate_sync_levels};
use crate::types::SignalType;
use crate::vbi::{FieldParity, find_vertical_sync};
use rayon::prelude::*;
use std::io::Write;

/// Default number of fields retained in the temporal history
/// buffer used by the denoise + dropout-repair stages. Five fields
/// = ~83 ms at NTSC's 60-field rate; gives √5 ≈ 2.24× noise drop
/// (~+7 dB SNR) on static regions while keeping latency low
/// enough for live FPV. Tunable per-reconstructor via
/// [`FrameReconstructor::with_temporal_window`].
pub const DEFAULT_TEMPORAL_WINDOW: usize = 5;

/// Hard upper bound on the temporal history window. The per-pixel
/// denoise reads at most this many history fields (its stack scratch
/// arrays are sized to it), so retaining more would allocate memory
/// that's never read. [`FrameReconstructor::with_temporal_window`]
/// clamps to this.
pub const MAX_TEMPORAL_WINDOW: usize = 8;

/// Sync-quality threshold below which a field *enters* dropout mode.
/// 0.5 means "more than half the sync tips in this field got rejected
/// by the MAD outlier filter" — at that point the rendered output is
/// dominated by interpolation noise and we blend toward recent history
/// instead.
const DROPOUT_ENTER_THRESHOLD: f32 = 0.5;

/// Sync-quality threshold above which a field *exits* dropout mode.
/// The gap between this and [`DROPOUT_ENTER_THRESHOLD`] gives the
/// dropout-repair state hysteresis: a field whose sync-quality hovers
/// right at 0.5 won't flip the denoise between static-blend and
/// motion-adaptive every frame (which is visible as a flicker).
const DROPOUT_EXIT_THRESHOLD: f32 = 0.6;

/// Fraction of the unwrapped line cropped from the left as horizontal
/// blanking before the active-video window is mapped to the output.
/// Each TBC line starts ~2.35 µs ahead of the sync-tip centre, so the
/// sync pulse (4.7 µs) + back porch + burst occupy the first stretch of
/// the line; 0.16 places the visible window just past that. The
/// theoretical active-video start (sync + back porch ≈ 9.4 µs of a
/// 63.555 µs NTSC line) is ≈ 0.148 — nudge toward that if a capture
/// looks over-cropped on the left, but 0.16 matches the centring we see
/// on real frames today, so it's the conservative default.
const ACTIVE_VIDEO_LEFT_CROP_FRAC: f32 = 0.16;

/// Motion threshold (fraction of FM-deviation rail) below which a
/// pixel is treated as static for temporal denoising. Pixels above
/// this fall through to current-frame-only (no averaging) to
/// avoid motion-blur. 0.10 ≈ 10 % of full deviation = roughly the
/// chroma-burst amplitude as a noise floor.
const TEMPORAL_MOTION_THRESHOLD: f32 = 0.10;

/// Per-line EMA coefficient for the AGC gain/porch state — a ~5-line
/// time constant, fast enough to track real vertical shading but slow
/// enough that one noisy porch window can't flicker a line.
const AGC_EMA_ALPHA: f32 = 0.2;

/// Sanity clamp on the per-line AGC gain target, as a factor of the
/// deviation-implied nominal gain of 1.0. Anything outside this is a
/// broken measurement (dropout, click burst), not real level drift.
const AGC_GAIN_CLAMP: (f32, f32) = (0.25, 4.0);

/// Consecutive fields whose burst measurement must disagree with the
/// current notch state before it flips — keeps a marginal source from
/// toggling the subcarrier notch (a visible sharpness change) field to
/// field.
const NOTCH_HYSTERESIS_FIELDS: u8 = 3;

/// Fraction of measured rows that must report a burst for the field to
/// count as "burst present".
const NOTCH_BURST_ROW_MAJORITY: f32 = 0.5;

/// Goertzel tone power for `w0` rad/sample over `x`, normalised to an
/// amplitude-squared estimate (`4·P/N²` recovers `A²` for a full-window
/// tone of amplitude `A`).
fn goertzel_amp_sq(x: &[f32], w0: f32) -> f32 {
    if x.len() < 8 {
        return 0.0;
    }
    let coeff = 2.0 * w0.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &v in x {
        let s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    4.0 * power.max(0.0) / (x.len() * x.len()) as f32
}

/// Classify a demodulated baseband slice as PAL or NTSC by measuring
/// the median sync-tip interval. Returns [`SignalType::Unknown`] when
/// the slice carries no measurable line structure — too short, too few
/// sync tips, no downward spread to threshold against (flat or
/// inverted waveforms), or a measured rate implausibly far from both
/// standards. Earlier versions returned `AnalogVideoNtsc` for the
/// degenerate cases, which asserted a standard from no evidence;
/// callers must pick their own default when this returns `Unknown`.
///
/// The sync threshold is the same robust percentile construction the
/// rest of the crate uses (`p2 + 0.25·(p50−p2)` on a ~0.5 µs-smoothed
/// copy): DC-invariant and immune to single FM-click outliers, unlike
/// the `global_min · 0.3` threshold it replaced, which one deep click
/// could drag beneath every real sync tip at exactly the low CNR this
/// function matters for.
pub fn detect_video_standard(demod_data: &[f32], sample_rate: u32) -> SignalType {
    let min_samples = sample_rate as usize / 200;
    if demod_data.len() < min_samples {
        return SignalType::Unknown;
    }

    let scan_len = (sample_rate as f32 * 5000e-6) as usize;
    let scan_len = scan_len.min(demod_data.len());
    let ma_win = ((sample_rate as f32 * 0.5e-6) as usize).max(1);
    let smoothed = crate::levels::moving_average(&demod_data[..scan_len], ma_win);
    let threshold = match crate::levels::robust_sync_threshold(&smoothed) {
        Some(t) => t,
        None => return SignalType::Unknown,
    };
    let min_gap = (sample_rate as f32 * 30e-6) as usize;
    let max_gap = (sample_rate as f32 * 100e-6) as usize;

    let mut sync_positions: Vec<usize> = Vec::new();
    let mut i = 0;
    let scan_len = smoothed.len();

    while i < scan_len {
        if smoothed[i] < threshold {
            let mut local_min_idx = i;
            let mut local_min_val = smoothed[i];
            while i < scan_len && smoothed[i] < threshold {
                if smoothed[i] < local_min_val {
                    local_min_val = smoothed[i];
                    local_min_idx = i;
                }
                i += 1;
            }
            sync_positions.push(local_min_idx);
            i = local_min_idx + min_gap;
        } else {
            i += 1;
        }
    }

    if sync_positions.len() < 5 {
        return SignalType::Unknown;
    }

    let mut intervals: Vec<usize> = Vec::new();
    for w in sync_positions.windows(2) {
        let gap = w[1] - w[0];
        if gap >= min_gap && gap <= max_gap {
            intervals.push(gap);
        }
    }

    if intervals.is_empty() {
        return SignalType::Unknown;
    }

    // Median, not mean. Every field's vertical-sync group contributes
    // ~18 *half*-line intervals (≈31.8 µs at NTSC) which clear the 30 µs
    // `min_gap` and so land in `intervals` alongside the ~240 full-line
    // ones. Averaging let that minority drag the estimate down by ~3.5%,
    // which was not a rounding nuisance but an outright correctness
    // failure: real PAL measured 62.1 µs against the 63.78 µs PAL/NTSC
    // decision boundary below, so this function could **never** return
    // `AnalogVideoPal` — every PAL capture came back NTSC. The median
    // ignores the half-line minority and recovers 64.00 µs for PAL /
    // 63.56 µs for NTSC.
    intervals.sort_unstable();
    let median_interval = intervals[intervals.len() / 2] as f64;
    if median_interval <= 0.0 {
        return SignalType::Unknown;
    }
    let line_hz = sample_rate as f64 / median_interval;

    // Reject a measured rate that is nowhere near either standard before
    // committing to one. On FM-demodulated *noise* the dips below
    // threshold occur essentially continuously, so the `min_gap` skip
    // collapses the interval to just above `min_gap` (30 µs → a ~33 kHz
    // "line rate"). Without this bound that absurd rate still classified
    // as NTSC — it only had to be *closer* to 15734 than to 15625 — which
    // is how empty-band noise got reported as video on live captures.
    // Real VTX crystal error is a few Hz; ±~250 Hz is generous.
    //
    // Both this guard and the median above mirror
    // `detector::classify_pal_ntsc_time_domain`, which had them first;
    // the two implementations of this same measurement had diverged.
    if !(15_400.0..=16_000.0).contains(&line_hz) {
        return SignalType::Unknown;
    }

    let line_period_us = 1_000_000.0 / line_hz;
    if line_period_us > 63.78 {
        SignalType::AnalogVideoPal
    } else {
        SignalType::AnalogVideoNtsc
    }
}

pub struct FrameReconstructor {
    pub width: usize,
    pub height: usize,
    pub line_width: usize,
    pub field_lines: usize,
    pub samples_per_line: usize,
    pub pal: bool,
    pub fm_deviation: f32,
    pub sample_rate: u32,
    pub debug_dump: bool,
    pub use_matched_sync: bool,
    pub use_line_locked_clock: bool,
    pub use_smart_doc: bool,
    pub smart_doc_spatial_weight: f32,

    #[cfg(feature = "neural-vsr")]
    pub neural_restorer: Option<crate::neural::NeuralRestorer>,
    #[cfg(feature = "neural-vsr")]
    pub hidden_state: Option<Vec<f32>>,
    /// Persistent luma in/out scratch for the neural pass, so the hot
    /// path doesn't allocate two full-frame `Vec`s per rendered frame.
    #[cfg(feature = "neural-vsr")]
    neural_in: Vec<f32>,
    #[cfg(feature = "neural-vsr")]
    neural_out: Vec<f32>,

    /// Holds the complete RGB frame between calls so consecutive
    /// `reconstruct_frame_into` calls (each one capturing a single
    /// NTSC/PAL *field*) can be merged into a single interlaced
    /// output. On a parity-0 call we render the current field into
    /// `frame`'s even rows and fill the odd rows from `field_buf`'s
    /// odd rows; on a parity-1 call we render into the odd rows and
    /// fill the even rows from `field_buf`. After each call,
    /// `frame` is copied back to `field_buf` so the *next* call has
    /// the just-rendered complementary rows available.
    ///
    /// Previously this buffer was declared but unused, and the
    /// output path called `line_doubling` (copying each captured
    /// row into both even and odd output positions) which threw
    /// away half of the vertical detail. The terminal output's
    /// "consumed = 1.66 M samples = half a frame's worth" on the
    /// first call was the direct symptom of that.
    pub field_buf: Vec<u32>,
    /// Which output row parity the next captured field renders into.
    /// Toggles 0 ↔ 1 on every successful `reconstruct_frame_into`
    /// call. NTSC's field-1-vs-field-2 distinction isn't explicitly
    /// tracked here — for static content (and for the synthetic
    /// fixture) pairing fields naively produces visually-correct
    /// interlaced output; explicit field-parity detection is a
    /// follow-up if a motion-heavy capture demands it.
    pub field_parity: u8,

    // Previous field's TBC output, used by the dropout-compensation
    // (DOC) pass to conceal current-field dropout pixels.
    pub prev_frame_tbc: Vec<f32>,
    pub has_prev: bool,

    // Sync Tracking State
    pub sync_phase: f32,
    pub line_period: f32,

    // Period history for cross-frame stabilisation: stores the last
    // N frames' median line periods. The median of this buffer gives
    // us a rock-solid reference period after 3-5 frames. Since the
    // line period is crystal-driven in the transmitter, it's
    // essentially constant across frames.
    period_history: Vec<f32>,

    // Biquad notch at the colour subcarrier (NTSC 3.58 MHz / PAL
    // 4.43 MHz). Repurposed to eliminate dot-crawl in our pure Luma signal.
    notch_b0: f32,
    notch_b1: f32,
    notch_b2: f32,
    notch_a1: f32,
    notch_a2: f32,
    /// Subcarrier angular frequency at the TBC output rate — kept for
    /// the burst-presence Goertzel that gates the notch (see
    /// [`Self::chroma_notch_active`]).
    notch_w0: f32,
    /// Whether the subcarrier notch is currently applied. Starts
    /// `true` (safe default: dot-crawl suppression until proven
    /// burst-free) and flips only after [`NOTCH_HYSTERESIS_FIELDS`]
    /// consecutive fields agree — burst-free monochrome cameras get
    /// their full luma detail back, burst-carrying sources keep the
    /// notch.
    notch_enabled: bool,
    /// Consecutive fields whose burst measurement disagreed with
    /// `notch_enabled`; drives the hysteresis flip.
    notch_flip_streak: u8,

    /// Smoothed per-line AGC state: gain applied to (sample − porch)
    /// and the porch (blanking) level it references. EMA'd across
    /// lines (α = [`AGC_EMA_ALPHA`]) because the raw per-line
    /// measurements come from two ~25-sample window means — at low CNR
    /// a single click in the porch window used to swing an entire
    /// line's brightness, visible as line-to-line flicker exactly when
    /// the picture was already struggling.
    agc_gain: f32,
    agc_porch: f32,
    /// `false` until the first valid swing measurement seeds the AGC
    /// state directly (instead of easing in from the 1.0/0.0 init).
    agc_primed: bool,

    /// Multi-field temporal history. Holds the last N rendered Y
    /// fields plus per-field metadata. Consumed by the temporal-
    /// denoise (per-pixel median + motion-weighted average across
    /// the window) and dropout-repair (per-field sync-quality
    /// driven blend) stages. See [`crate::frame_history`] for the
    /// detailed design notes.
    ///
    /// Capacity is set at construction via `temporal_window`
    /// (default [`DEFAULT_TEMPORAL_WINDOW`]); a window of 0 or 1
    /// effectively disables both temporal stages, which is what
    /// batch-mode callers want when they prefer single-frame
    /// fidelity over noise reduction.
    pub history: FrameHistory,
    /// Wall-clock frame counter used to stamp `FieldMeta` entries
    /// so the history's timestamps stay monotonic across runs.
    /// Increments once per `reconstruct_frame_into` call.
    field_counter: u64,
    /// Hysteresis state for the dropout-repair stage: once a field
    /// drops below [`DROPOUT_ENTER_THRESHOLD`] we stay in dropout mode
    /// (full blend toward history) until sync-quality recovers past
    /// [`DROPOUT_EXIT_THRESHOLD`], so a marginal-SNR field hovering at
    /// the threshold doesn't flicker the denoise mode frame to frame.
    in_dropout: bool,
}

#[inline]
fn filtfilt(data: &mut [f32], b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) {
    if data.is_empty() {
        return;
    }
    let mut x1 = data[0];
    let mut x2 = data[0];
    let mut y1 = data[0];
    let mut y2 = data[0];
    for i in 0..data.len() {
        let x = data[i];
        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        data[i] = y;
    }
    x1 = data[data.len() - 1];
    x2 = data[data.len() - 1];
    y1 = data[data.len() - 1];
    y2 = data[data.len() - 1];
    for i in (0..data.len()).rev() {
        let x = data[i];
        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        data[i] = y;
    }
}

/// Locate an H-sync tip near `center` by correlating against a
/// zero-mean sync-pulse template (`+1` blanking wings of `sync_width/2`
/// either side of a `-1` centre pulse). A matched filter is the optimal
/// linear detector for a known pulse shape in additive noise, so at low
/// CNR this recovers the exact integer sync phase where the min/centroid
/// estimator in [`robust_sync_tip_center`] starts wandering. The
/// tradeoff — a fixed template is less adaptive to level/porch drift, so
/// it can trail the robust estimator under heavy noise; see the
/// `phase2_dsp_*` tests and `weak_signal_sweep`. Selected by
/// `FrameReconstructor::use_matched_sync` (default on).
///
/// Returns `None` when the best-correlated location's pulse level isn't
/// below `reject_above` (no real sync in range).
#[inline]
fn matched_sync_center(
    demod: &[f32],
    center: f32,
    search_radius: usize,
    sync_width: usize,
    reject_above: f32,
) -> Option<f32> {
    let c = center.round() as usize;
    let half_template = sync_width;
    let qtr_template = sync_width / 2;

    let lo = c.saturating_sub(search_radius).max(half_template);
    let hi = (c + search_radius).min(demod.len().saturating_sub(half_template));
    if lo >= hi {
        return None;
    }

    let mut max_corr = f32::NEG_INFINITY;
    let mut best_idx = lo;
    let mut best_pulse_val = 0.0;

    for i in lo..hi {
        let mut corr = 0.0;
        let mut pulse_sum = 0.0;
        // outer left (+1)
        for j in (i - half_template)..(i - half_template + qtr_template) {
            corr += demod[j];
        }
        // center negative pulse (-1)
        for j in (i - half_template + qtr_template)..(i + half_template - qtr_template) {
            corr -= demod[j];
            pulse_sum += demod[j];
        }
        // outer right (+1)
        for j in (i + half_template - qtr_template)..(i + half_template) {
            corr += demod[j];
        }

        if corr > max_corr {
            max_corr = corr;
            best_idx = i;
            best_pulse_val = pulse_sum / sync_width as f32;
        }
    }

    if best_pulse_val >= reject_above {
        return None;
    }
    Some(best_idx as f32)
}

/// Estimate a sync-tip centre near `center` (searching ±`search_radius`),
/// robust to brightness / DC shifts in the demodulated signal.
///
/// The detection threshold is the **midpoint between the pulse minimum
/// and the surrounding back-porch level** (the max over ±`porch_radius`
/// of the minimum, a window wide enough to clear the ~4.7 µs sync pulse
/// and reach the porch). Because a constant signal-level shift moves the
/// minimum and the back porch together, the midpoint — and therefore the
/// centroid of the below-midpoint region — is invariant to brightness.
/// The previous `min * 0.5` threshold was referenced to zero, so bright
/// active video (e.g. a window in frame) biased the tip position,
/// progressively in the lower field rows — the "slanting vertical line".
///
/// Returns `None` if no pulse below `reject_above` is found in range.
#[inline]
fn robust_sync_tip_center(
    demod: &[f32],
    center: f32,
    search_radius: usize,
    porch_radius: usize,
    ma_win: usize,
    reject_above: f32,
) -> Option<f32> {
    let smooth = |i: usize| -> f32 {
        let s: f32 = demod[i - ma_win..i + ma_win].iter().sum();
        s / (2 * ma_win) as f32
    };
    let c = center.round() as usize;
    let lo = c.saturating_sub(search_radius).max(ma_win);
    let hi = (c + search_radius).min(demod.len().saturating_sub(ma_win));
    if lo >= hi {
        return None;
    }
    // 1. Pulse minimum within the search window.
    let mut min_val = f32::INFINITY;
    let mut min_idx = lo;
    for i in lo..hi {
        let v = smooth(i);
        if v < min_val {
            min_val = v;
            min_idx = i;
        }
    }
    if min_val >= reject_above {
        return None;
    }
    // 2. Local back-porch reference: max over a wider window centred on
    //    the minimum (reaches the porch on both sides of the pulse).
    let plo = min_idx.saturating_sub(porch_radius).max(ma_win);
    let phi = (min_idx + porch_radius).min(demod.len().saturating_sub(ma_win));
    let mut back_porch = f32::NEG_INFINITY;
    for i in plo..phi {
        let v = smooth(i);
        if v > back_porch {
            back_porch = v;
        }
    }
    // 3. Brightness-invariant midpoint threshold + centroid of the
    //    below-midpoint region = the pulse centre.
    let thresh = (min_val + back_porch) * 0.5;
    let mut sum_idx = 0usize;
    let mut count = 0usize;
    for i in plo..phi {
        if smooth(i) < thresh {
            sum_idx += i;
            count += 1;
        }
    }
    if count > 0 {
        Some(sum_idx as f32 / count as f32)
    } else {
        Some(min_idx as f32)
    }
}

impl FrameReconstructor {
    pub fn new(sample_rate: u32, is_pal: bool, fm_deviation: f32, debug_dump: bool) -> Self {
        let line_rate = if is_pal { 15625.0 } else { 15734.0 };
        let samples_per_line = (sample_rate as f32 / line_rate).round() as usize;
        let field_lines = if is_pal { 288 } else { 240 };
        let line_width = if is_pal { 864 } else { 858 }; // Exact standard pixels per line
        let width = 720;
        let height = if is_pal { 576 } else { 480 };

        let field_pixels = field_lines * line_width;

        let tbc_fs = line_width as f32 / (if is_pal { 64.0e-6 } else { 63.5555e-6 });

        let w0 =
            2.0 * std::f32::consts::PI * (if is_pal { 4.43361875e6 } else { 3.579545e6 }) / tbc_fs;

        // Biquad notch at the colour subcarrier — RBJ cookbook form
        // with Q=8 (≈ 360 kHz notch width at the 3.58/4.43 MHz
        // subcarriers). Zeros sit on the unit circle at e^±jω₀ for
        // an exact null at f_sc; poles are r·e^±jω₀ with
        // r = (1−α)/(1+α) giving a narrow attenuation band without
        // affecting nearby luma frequencies.
        let notch_q: f32 = 8.0;
        let notch_cos_w0 = w0.cos();
        let notch_alpha = w0.sin() / (2.0 * notch_q);
        let notch_a0 = 1.0 + notch_alpha;
        let notch_b0 = 1.0 / notch_a0;
        let notch_b1 = -2.0 * notch_cos_w0 / notch_a0;
        let notch_b2 = 1.0 / notch_a0;
        let notch_a1 = -2.0 * notch_cos_w0 / notch_a0;
        let notch_a2 = (1.0 - notch_alpha) / notch_a0;

        FrameReconstructor {
            width,
            height,
            line_width,
            field_lines,
            samples_per_line,
            pal: is_pal,
            fm_deviation,
            sample_rate,
            debug_dump,
            // Phase-2 weak-signal DSP, enabled by default. These are
            // net wins at low noise (matched-sync recovers the exact
            // integer sync phase → GCOR 1.0 on a clean signal; OLS
            // line-locked TBC gives straight verticals; SmartDOC blends
            // spatial+temporal concealment) but trade a little
            // high-noise resilience — see the calibrated
            // `phase2_dsp_*` tests and `weak_signal_sweep`. Turn any of
            // them off per-instance with the `with_*` builders.
            use_matched_sync: true,
            use_line_locked_clock: true,
            use_smart_doc: true,
            smart_doc_spatial_weight: 0.5,

            #[cfg(feature = "neural-vsr")]
            neural_restorer: None,
            #[cfg(feature = "neural-vsr")]
            hidden_state: None,
            #[cfg(feature = "neural-vsr")]
            neural_in: vec![0.0; width * height],
            #[cfg(feature = "neural-vsr")]
            neural_out: vec![0.0; width * height],

            field_buf: vec![0u32; width * height],
            field_parity: 0,
            prev_frame_tbc: vec![0.0; field_pixels],
            has_prev: false,
            sync_phase: 0.0,
            line_period: samples_per_line as f32,
            period_history: Vec::with_capacity(8),

            notch_b0,
            notch_b1,
            notch_b2,
            notch_a1,
            notch_a2,
            notch_w0: w0,
            notch_enabled: true,
            notch_flip_streak: 0,
            agc_gain: 1.0,
            agc_porch: 0.0,
            agc_primed: false,
            history: FrameHistory::new(DEFAULT_TEMPORAL_WINDOW, field_pixels),
            field_counter: 0,
            in_dropout: false,
        }
    }

    /// Override the temporal history window size. Default is
    /// [`DEFAULT_TEMPORAL_WINDOW`] (5 fields, ~83 ms latency, ~+7 dB
    /// SNR on static regions). Set to 1 or 0 to disable temporal
    /// denoise + dropout repair (offline-decode and unit-test
    /// scenarios). Larger windows improve SNR by √N at the cost of
    /// latency and memory; 8 is a reasonable upper bound for live
    /// surveillance / recon use.
    ///
    /// Builder-style: returns `self` so callers can chain with
    /// `FrameReconstructor::new(...).with_temporal_window(2)`.
    pub fn with_temporal_window(mut self, window: usize) -> Self {
        let field_pixels = self.field_lines * self.line_width;
        // Clamp to [1, MAX_TEMPORAL_WINDOW]: 0 would disable history
        // entirely (we keep at least the current field), and anything
        // above the cap allocates fields the denoise loop never reads.
        let window = window.clamp(1, MAX_TEMPORAL_WINDOW);
        self.history = FrameHistory::new(window, field_pixels);
        self
    }

    pub fn line_period_samples(&self) -> f32 {
        self.line_period
    }

    /// Override the FM peak-deviation estimate used to scale the sync
    /// thresholds, AGC, and DOC rails (see [`crate::levels`]). Safe to
    /// call at any time, including between calls to
    /// `reconstruct_frame_into` — every threshold derived from
    /// `radians_per_volt` is recomputed from `self.fm_deviation` at the
    /// start of each call, so there's no stale cached state to
    /// invalidate.
    pub fn set_fm_deviation(&mut self, deviation_hz: f32) {
        self.fm_deviation = deviation_hz;
    }

    /// Latest field's sync-extraction confidence in `[0, 1]`. 1.0
    /// means every sync tip in the field passed the MAD-outlier
    /// check; 0.0 means catastrophic dropout. Reads the metadata
    /// for the most recently pushed field in [`Self::history`].
    /// Returns 0.0 if no field has been rendered yet (rather than
    /// 1.0 — the caller probably wants "no data" to look like
    /// "bad" not "perfect" when wiring this into a UI indicator).
    pub fn latest_sync_quality(&self) -> f32 {
        self.history
            .current_meta()
            .map(|m| m.sync_quality)
            .unwrap_or(0.0)
    }

    /// Latest field's mean Y amplitude (post-notch). A sudden drop
    /// relative to recent history indicates the transmitter went
    /// out of range or the antenna got blocked. The viewer plots
    /// this as a "signal-strength meter"-style indicator.
    pub fn latest_mean_amplitude(&self) -> f32 {
        self.history
            .current_meta()
            .map(|m| m.mean_amplitude)
            .unwrap_or(0.0)
    }

    /// Number of fields currently retained in the temporal history.
    /// Stops increasing once the buffer hits its configured
    /// capacity (default [`DEFAULT_TEMPORAL_WINDOW`]). Used by the
    /// debug telemetry to show whether the denoise stage has filled
    /// its window yet (the first few frames after start-up render
    /// without full denoise benefit).
    pub fn history_depth(&self) -> usize {
        self.history.len()
    }

    /// Whether the colour-subcarrier notch is currently applied. Gated
    /// per field by a Goertzel burst-presence measurement on the back
    /// porch (with [`NOTCH_HYSTERESIS_FIELDS`] of hysteresis): many FPV
    /// cameras are effectively monochrome sources with no burst, and
    /// for those the notch deleted real luma detail around 3.58 /
    /// 4.43 MHz for nothing.
    pub fn chroma_notch_active(&self) -> bool {
        self.notch_enabled
    }

    pub fn video_standard(&self) -> crate::types::SignalType {
        if self.pal {
            crate::types::SignalType::AnalogVideoPal
        } else {
            crate::types::SignalType::AnalogVideoNtsc
        }
    }

    pub fn with_matched_sync(mut self, enable: bool) -> Self {
        self.use_matched_sync = enable;
        self
    }

    pub fn with_line_locked_clock(mut self, enable: bool) -> Self {
        self.use_line_locked_clock = enable;
        self
    }

    pub fn with_smart_doc(mut self, enable: bool, spatial_weight: f32) -> Self {
        self.use_smart_doc = enable;
        self.smart_doc_spatial_weight = spatial_weight.clamp(0.0, 1.0);
        self
    }

    /// Enable the optional temporal neural denoiser, loading the ONNX
    /// model from `model_path`. The caller supplies the path — a library
    /// must not assume a CWD-relative location. A load failure is logged
    /// and leaves the restorer disabled (reconstruction still works).
    #[cfg(feature = "neural-vsr")]
    pub fn with_neural_restorer(mut self, model_path: &str, use_gpu: bool) -> Self {
        match crate::neural::NeuralRestorer::new(model_path, use_gpu) {
            Ok(restorer) => self.neural_restorer = Some(restorer),
            Err(e) => log::error!("Failed to load neural restorer from {model_path}: {e}"),
        }
        self
    }

    pub fn reconstruct_frame(&mut self, demod_data: &[f32]) -> Option<(Vec<u32>, usize)> {
        let mut frame = vec![0u32; self.width * self.height];
        let consumed = self.reconstruct_frame_into(demod_data, &mut frame)?;
        Some((frame, consumed))
    }

    pub fn reconstruct_frame_into(
        &mut self,
        demod_data: &[f32],
        frame: &mut [u32],
    ) -> Option<usize> {
        if demod_data.is_empty() {
            return None;
        }
        // `frame` is a caller-supplied buffer; every write below indexes
        // it assuming exactly `width * height` elements (the field-merge
        // step even `copy_from_slice`s into `self.field_buf`, which is
        // fixed at that size), so a mismatched buffer must be rejected
        // here rather than panicking partway through.
        if frame.len() != self.width * self.height {
            return None;
        }

        let fs = self.sample_rate as f32;
        let radians_per_volt = 2.0 * std::f32::consts::PI * self.fm_deviation / fs;
        let v_sync_threshold = -0.3 * radians_per_volt;
        let window_len = self.samples_per_line;
        let ma_win = ((fs * 0.5e-6) as usize).max(1);
        let sync_window = (fs * 2.0e-6) as usize;
        // Back-porch reference window: wider than the ~4.7 µs sync pulse
        // so the max reaches the blanking level on both sides.
        let porch_radius = (fs * 3.5e-6) as usize;

        // Prefer the structural VBI parser (crate::vbi): it locates the
        // actual serrated vertical-sync group, so the active-video
        // anchor and (when conclusive) field parity come from real
        // pulse structure instead of a density heuristic + a hardcoded
        // 20-line blanking guess that's wrong for PAL (25 lines) and
        // can never detect parity at all. `estimate_sync_levels` needs
        // ~40 ms of data to be trustworthy — many single-call slices
        // are shorter than that (one field alone is 16.7–20 ms), so it
        // commonly falls back to the levels implied by `self.fm_deviation`,
        // which for a correctly-locked deviation is exactly right.
        let sync_levels =
            estimate_sync_levels(demod_data, self.sample_rate).unwrap_or(SyncLevels {
                sync_tip: -0.4 * radians_per_volt,
                blanking: 0.0,
            });
        let vbi_info = find_vertical_sync(demod_data, self.sample_rate, &sync_levels, self.pal);

        let first_sync_center;
        let required_samples;
        if let Some(info) = &vbi_info {
            if let Some(parity) = info.parity {
                self.field_parity = match parity {
                    FieldParity::First => 0,
                    FieldParity::Second => 1,
                };
            }
            first_sync_center = info.field_active_start;
            required_samples = (info.field_active_start.max(0.0) as usize)
                + self.samples_per_line * (self.field_lines + 2);
            if self.debug_dump {
                let _ = writeln!(
                    std::io::stdout(),
                    "[VBI] broad={} eq={}/{} parity={:?} active@{:.0}",
                    info.n_broad,
                    info.n_eq_pre,
                    info.n_eq_post,
                    info.parity,
                    info.field_active_start
                );
            }
        } else {
            // Fall back to the density heuristic + a fixed 20-line
            // blanking skip — real VBI is dirtier than the spec on
            // cheap FPV cameras and deep fades, and this path stays
            // exactly as it was before the parser existed.
            let mut v_sync_idx = None;
            if demod_data.len() > window_len * 2 {
                let mut below_count = 0usize;
                for i in 0..window_len {
                    if demod_data[i] < v_sync_threshold {
                        below_count += 1;
                    }
                }
                let density_threshold = window_len / 2;
                if below_count > density_threshold {
                    v_sync_idx = Some(0);
                } else {
                    for start in 1..demod_data.len() - window_len {
                        if demod_data[start - 1] < v_sync_threshold {
                            below_count -= 1;
                        }
                        if demod_data[start + window_len - 1] < v_sync_threshold {
                            below_count += 1;
                        }
                        if below_count > density_threshold {
                            v_sync_idx = Some(start);
                            break;
                        }
                    }
                }
            }
            let v_idx = v_sync_idx?;
            required_samples = v_idx + self.samples_per_line * (20 + self.field_lines + 2);
            if demod_data.len() >= required_samples {
                // Search for first H-sync tip after V-sync + blanking lines.
                let skip_lines = v_idx + (self.samples_per_line * 20);
                // Anchor = robust centre of the first H-sync tip in the
                // ~2 lines after the blanking skip. Brightness-invariant
                // (see `robust_sync_tip_center`); the two-pass extraction
                // below validates every tip against the median period, so
                // the anchor needs no extra smoothing.
                let h_sync_width = (crate::vbi::consts::H_SYNC_WIDTH_S as f32
                    * self.sample_rate as f32)
                    .round() as usize;
                let maybe_sync = if self.use_matched_sync {
                    matched_sync_center(
                        demod_data,
                        (skip_lines + self.samples_per_line) as f32,
                        self.samples_per_line,
                        h_sync_width,
                        v_sync_threshold * 0.8,
                    )
                } else {
                    robust_sync_tip_center(
                        demod_data,
                        (skip_lines + self.samples_per_line) as f32,
                        self.samples_per_line,
                        porch_radius,
                        ma_win,
                        v_sync_threshold * 0.8,
                    )
                };
                first_sync_center =
                    maybe_sync.unwrap_or((skip_lines + self.samples_per_line) as f32);
            } else {
                first_sync_center = 0.0; // unused: the length check below returns None first
            }
        }
        if demod_data.len() < required_samples {
            return None;
        }
        self.sync_phase = first_sync_center;

        // ═══════════════════════════════════════════════════════════
        //  TWO-PASS SYNC EXTRACTION
        // ═══════════════════════════════════════════════════════════
        //
        // Instead of tracking sync tips sequentially with a PLL (which
        // needs ~5 rows to converge, causing top-of-frame skew), we
        // scan the entire field in one pass to find ALL sync tip
        // positions, compute the median line period from the measured
        // intervals, and then build a corrected array of per-row sync
        // positions. This gives pixel-perfect alignment from row 0.

        // Pass 1: Find all sync tip centers in the field.
        let total_rows = self.field_lines + 4; // scan a few extra for robustness
        let mut raw_sync_positions: Vec<Option<f32>> = Vec::with_capacity(total_rows);
        {
            let mut cursor = self.sync_phase;
            for _row in 0..total_rows {
                if _row == 0 {
                    // Row 0: use the anchor directly
                    raw_sync_positions.push(Some(cursor));
                } else {
                    let expected = cursor + self.line_period;
                    if expected.round() as usize + sync_window
                        >= demod_data.len().saturating_sub(ma_win)
                    {
                        break;
                    }

                    let h_sync_width = (crate::vbi::consts::H_SYNC_WIDTH_S as f32
                        * self.sample_rate as f32)
                        .round() as usize;
                    let maybe_measured = if self.use_matched_sync {
                        matched_sync_center(
                            demod_data,
                            expected,
                            sync_window,
                            h_sync_width,
                            v_sync_threshold * 0.8,
                        )
                    } else {
                        robust_sync_tip_center(
                            demod_data,
                            expected,
                            sync_window,
                            porch_radius,
                            ma_win,
                            v_sync_threshold * 0.8,
                        )
                    };

                    match maybe_measured {
                        // Sanity: reject a tip that landed too far from
                        // where the constant period predicts (noise / a
                        // wrong feature); interpolate it later instead.
                        Some(measured) if (measured - expected).abs() < self.line_period * 0.25 => {
                            raw_sync_positions.push(Some(measured));
                            cursor = measured;
                        }
                        _ => {
                            raw_sync_positions.push(None);
                            cursor = expected;
                        }
                    }
                }
            }
        }

        // Pass 2: line period by robust least-squares regression over
        // the field (the Domesday86/ld-decode-style TBC approach). NTSC/
        // PAL sync is crystal-locked, so true line starts lie on a
        // straight line `intercept + row·period`; fitting the slope is a
        // sub-pixel period estimate.
        //
        // Plain OLS over *every* measured tip is NOT robust — a few
        // noise-corrupted sync tips (common in the low-SNR bottom rows)
        // tilt the slope and slant the whole field, the exact failure the
        // weak-signal sweep exposed. So fit once, drop points whose
        // residual exceeds 3× a MAD-derived scale, and refit on the
        // survivors — keeping OLS's sub-pixel precision on the clean
        // cluster while rejecting the drifting tail.
        let points: Vec<(f64, f64)> = raw_sync_positions
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| p.map(|pos| (i as f64, pos as f64)))
            .collect();

        let ols = |pts: &[(f64, f64)]| -> Option<(f64, f64)> {
            let n = pts.len() as f64;
            if n < 2.0 {
                return None;
            }
            let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
            for &(x, y) in pts {
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
            let denom = n * sxx - sx * sx;
            if denom.abs() < 1e-6 {
                return None;
            }
            let slope = (n * sxy - sx * sy) / denom;
            let intercept = (sy - slope * sx) / n;
            Some((slope, intercept))
        };

        let kept: Vec<(f64, f64)> = match ols(&points) {
            Some((slope, intercept)) => {
                let mut resid: Vec<f64> = points
                    .iter()
                    .map(|&(x, y)| (y - (intercept + slope * x)).abs())
                    .collect();
                let mid = resid.len() / 2;
                resid.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                // 1.4826·MAD ≈ σ for Gaussian residuals; floor at 1
                // sample so an already-clean field can't reject its own
                // near-perfect tips.
                let scale = (1.4826 * resid[mid]).max(1.0);
                points
                    .iter()
                    .copied()
                    .filter(|&(x, y)| (y - (intercept + slope * x)).abs() <= 3.0 * scale)
                    .collect()
            }
            None => points.clone(),
        };

        // Accumulators for the (trimmed) fit consumed by the period
        // logic below.
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xx = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut count = 0f64;
        for &(x, y) in &kept {
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1.0;
        }

        if count > 10.0 {
            let denom = count * sum_xx - sum_x * sum_x;
            if denom.abs() > 1e-6 {
                let ols_slope = (count * sum_xy - sum_x * sum_y) / denom;

                let nominal = self.samples_per_line as f32;
                if ols_slope > (nominal * 0.95) as f64 && ols_slope < (nominal * 1.05) as f64 {
                    let frame_period = ols_slope as f32;

                    // Push this frame's period into the history buffer (max 8)
                    if self.period_history.len() >= 8 {
                        self.period_history.remove(0);
                    }
                    self.period_history.push(frame_period);

                    // Use the median of the history buffer as the stabilised line period
                    let mut sorted_history = self.period_history.clone();
                    sorted_history
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let stabilised_period = sorted_history[sorted_history.len() / 2];

                    self.line_period = if self.use_line_locked_clock {
                        stabilised_period
                    } else {
                        frame_period
                    };

                    // Debug telemetry only.
                    if self.debug_dump && !self.has_prev {
                        let _ = writeln!(
                            std::io::stdout(),
                            "TBC: OLS line period = {:.3} samples ({} points, {} history frames)",
                            self.line_period,
                            count as usize,
                            self.period_history.len()
                        );
                    }
                }
            }
        }

        // ── MAD-based outlier rejection ─────────────────────────────
        //
        // Compute intervals from each measured sync tip to the
        // previous, then use Median Absolute Deviation (MAD) to find
        // the cluster width. Any measurement whose interval deviates
        // by more than 3×MAD from the stabilised period is rejected
        // as noise-corrupted and will be interpolated instead.
        let mut measured_intervals: Vec<(usize, f32)> = Vec::new();
        for i in 1..raw_sync_positions.len() {
            if let (Some(a), Some(b)) = (raw_sync_positions[i - 1], raw_sync_positions[i]) {
                measured_intervals.push((i, b - a));
            }
        }

        // Compute MAD of intervals
        let reject_threshold = if measured_intervals.len() >= 5 {
            let mut iv_vals: Vec<f32> = measured_intervals.iter().map(|(_, v)| *v).collect();
            iv_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let iv_median = iv_vals[iv_vals.len() / 2];
            let mut abs_devs: Vec<f32> = iv_vals.iter().map(|v| (v - iv_median).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = abs_devs[abs_devs.len() / 2];
            // 3×MAD, but at least 3 samples (noise floor)
            (3.0 * mad).max(3.0)
        } else {
            // Not enough data — use a conservative fixed threshold
            10.0
        };

        // Mark outlier positions as None
        for &(idx, interval) in &measured_intervals {
            if (interval - self.line_period).abs() > reject_threshold {
                raw_sync_positions[idx] = None;
            }
        }

        // Build corrected sync positions via a ROBUST CONSTANT-PERIOD
        // FIT. NTSC/PAL sync is crystal-locked, so the true line starts
        // lie on a straight line `intercept + row · line_period`. The
        // earlier shape used each row's own measured tip (interpolating
        // gaps), which let Pass 1's per-tip cursor *chase* measurement
        // noise — worst in the lower-SNR bottom rows near vertical
        // blanking / under the OSD — so line starts wandered ~40 samples
        // by the field bottom and snapped back at the boundary (the
        // "vertical line that slants then self-corrects"). Instead, keep
        // the very stable cross-frame `line_period` as the slope and fit
        // only the intercept: the *median* over all surviving measured
        // tips of `measured − row · line_period`. The median locks onto
        // the dense, well-tracked cluster (top ¾) and ignores the
        // drifting tail, so every line lands on the exact constant-period
        // grid → straight verticals, no per-row wander.
        let n_rows = raw_sync_positions.len().min(self.field_lines);
        let mut sync_positions: Vec<f32> = vec![0.0; n_rows];

        let mut intercepts: Vec<f32> = Vec::with_capacity(n_rows);
        for (row, pos) in raw_sync_positions.iter().take(n_rows).enumerate() {
            if let Some(p) = pos {
                intercepts.push(p - row as f32 * self.line_period);
            }
        }
        let intercept = if intercepts.is_empty() {
            // No surviving tips this field — fall back to the previous
            // field's trailing phase so we still produce a grid.
            self.sync_phase
        } else {
            intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            intercepts[intercepts.len() / 2]
        };
        for (row, sp) in sync_positions.iter_mut().enumerate() {
            *sp = intercept + row as f32 * self.line_period;
        }

        // Store final sync_phase for next frame's VBI computation
        if n_rows > 0 {
            self.sync_phase = sync_positions[n_rows - 1];
        }

        // ── Sync quality / dropout score ─────────────────────────
        //
        // Fraction of sync-tip slots in `raw_sync_positions` that
        // survived the MAD outlier filter. 1.0 means every line had
        // a clean sync; 0.0 means catastrophic dropout. Used as the
        // primary driver for the temporal denoise / dropout-repair
        // stage downstream: when this drops below the enter threshold,
        // we force the per-pixel denoise into "static" mode (full blend
        // toward history), substituting recent good output for the
        // current noisy frame, and stay there until it recovers past the
        // exit threshold (hysteresis — see the threshold constants).
        let total_slots = raw_sync_positions.len().max(1) as f32;
        let valid_slots = raw_sync_positions.iter().filter(|p| p.is_some()).count() as f32;
        let sync_quality = valid_slots / total_slots;
        if self.in_dropout {
            if sync_quality > DROPOUT_EXIT_THRESHOLD {
                self.in_dropout = false;
            }
        } else if sync_quality < DROPOUT_ENTER_THRESHOLD {
            self.in_dropout = true;
        }
        let force_static = self.in_dropout;

        // ── Sync-position residual profile (debug telemetry) ───────
        //
        // Raw measured-tip deviation from the fitted constant-period
        // grid (`intercept + row · line_period`), sampled top→bottom.
        // The rendered positions are now exactly on the grid, so this
        // reports how far the *underlying raw measurements* wandered —
        // i.e. how much slant the robust fit just removed. A clean
        // signal stays near 0 everywhere; a large bottom-quarter value
        // is the centroid drift near blanking / under the OSD that the
        // fit now ignores. `NaN` marks an interpolated (no-tip) row.
        // Rate-limited to one field per ~half second.
        // `% 30 == 0` (not `is_multiple_of`) to stay within the 1.85 MSRV.
        #[allow(clippy::manual_is_multiple_of)]
        if self.debug_dump && self.field_counter % 30 == 0 && n_rows > 4 {
            let dev = |row: usize| -> f32 {
                match raw_sync_positions.get(row).and_then(|o| *o) {
                    Some(p) => p - (intercept + row as f32 * self.line_period),
                    None => f32::NAN,
                }
            };
            let interp = raw_sync_positions
                .iter()
                .take(n_rows)
                .filter(|p| p.is_none())
                .count();
            let (q1, q2, q3, last) = (n_rows / 4, n_rows / 2, 3 * n_rows / 4, n_rows - 1);
            // Robust (Theil-Sen) slope of the measured tips vs the period
            // the grid uses. If they differ, the residual is a uniform
            // SLOPE error (whole-field slant) and the grid should fit the
            // slope per-field; if they match, a remaining slant is
            // inter-field/interlace, not slope. `field drift` is the
            // implied start-position error accumulated across the field.
            let pts: Vec<(f32, f32)> = raw_sync_positions
                .iter()
                .take(n_rows)
                .enumerate()
                .filter_map(|(r, p)| p.map(|pos| (r as f32, pos)))
                .collect();
            let mut slopes: Vec<f32> = Vec::new();
            for a in 0..pts.len() {
                for b in (a + 1)..pts.len() {
                    let dr = pts[b].0 - pts[a].0;
                    if dr > 0.0 {
                        slopes.push((pts[b].1 - pts[a].1) / dr);
                    }
                }
            }
            let ts_slope = if slopes.is_empty() {
                self.line_period
            } else {
                slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                slopes[slopes.len() / 2]
            };
            let slope_drift = (ts_slope - self.line_period) * n_rows as f32;
            let _ = writeln!(
                std::io::stdout(),
                "[SYNC RESID] rows={n_rows} interp={interp} | raw dev from fit (samples) @0={:+.1} @{q1}={:+.1} @{q2}={:+.1} @{q3}={:+.1} @{last}={:+.1} | TS slope={ts_slope:.3} vs period={:.3} → field drift {slope_drift:+.1}",
                dev(0),
                dev(q1),
                dev(q2),
                dev(q3),
                dev(last),
                self.line_period,
            );
        }

        let mut field_rows_written = 0usize;

        // Setup arrays for TBC output
        let mut current_frame_tbc = vec![0.0f32; self.field_lines * self.line_width];
        let mut current_frame_doc = vec![false; self.field_lines * self.line_width];

        // Per-row scratch, hoisted out of the loop: at 288 rows × 50
        // fields/s the per-row `vec![...]`s added up to real allocator
        // churn for buffers whose sizes never change within a call.
        let mut tbc_line = vec![0.0f32; self.line_width];
        let mut doc_mask = vec![false; self.line_width];
        let mut dilated_doc = vec![false; self.line_width];
        let mut aa_line: Vec<f32> = Vec::new();

        let nominal_swing = 0.4 * radians_per_volt;

        for row in 0..self.field_lines {
            if row >= n_rows {
                break;
            }
            // Use the pre-computed sync position for this row
            self.sync_phase = sync_positions[row];

            let next_sync_idx = (self.sync_phase + self.line_period).round() as usize;
            if next_sync_idx >= demod_data.len() {
                break;
            }

            // Extract raw line data offset by -2.35us to align with
            // sync tip falling edge. Sub-sample TBC: keep the line
            // start as a fractional position into `demod_data` so
            // line-to-line phase drift collapses from ~16° to <1°
            // at f_sc. The previous shape rounded `current_sync_idx`
            // and `next_sync_idx` to integer samples, so the line
            // read wandered by 0.5 samples per line — at 100 MSPS
            // that's 5 ns × 4.43 MHz × 2π ≈ 16° of chroma phase
            // jitter per line, which is what the per-row
            // `phase_offset` prints showed (101° → -17° → 107° →
            // -175° walk across rows 0-3).
            let sync_offset = fs * 2.35e-6; // fractional offset in samples
            let start_pos = self.sync_phase - sync_offset;
            let end_pos = self.sync_phase + self.line_period - sync_offset;
            if start_pos < 0.0 || end_pos >= demod_data.len() as f32 {
                break;
            }
            let start_int = start_pos.floor() as usize;
            let start_frac = start_pos - start_int as f32; // [0, 1)
            let end_int = end_pos.ceil() as usize;
            if end_int >= demod_data.len() {
                break;
            }
            let raw_len = end_int - start_int; // integer span; ≥ ceil(line_period)
            if raw_len == 0 {
                break;
            }
            let raw_line = &demod_data[start_int..=end_int];
            // Fractional source span the TBC sweeps over — equal to
            // `line_period - sync_offset_diff` modulo float-rounding;
            // taking the fractional values keeps the per-line stride
            // exact regardless of which sample `start_int` rounds to.
            let raw_len_f = end_pos - start_pos;

            // Anti-alias ahead of the TBC's point-sampling. When the
            // resample is decimating (high capture rates: 61.44 MSPS →
            // ~3,900 samples/line swept onto 864 outputs), evaluating
            // the cubic at a >1-sample stride folds demod content and
            // noise above the output Nyquist (~6.75 MHz) back into the
            // picture. A boxcar the width of the decimation ratio ahead
            // of the sampling (CIC-1) suppresses the fold; its
            // (win−1)/2 group delay is compensated in `idx_float` so
            // the image doesn't shift. At ≤1.5× ratios (low capture
            // rates) the pass is skipped — nothing folds.
            let ratio = raw_len_f / self.line_width as f32;
            let aa_win = if ratio >= 1.5 {
                ratio.round() as usize
            } else {
                1
            };
            let (line_src, aa_shift): (&[f32], f32) = if aa_win >= 2 {
                crate::levels::moving_average_into(raw_line, aa_win, &mut aa_line);
                (&aa_line, (aa_win - 1) as f32 / 2.0)
            } else {
                (raw_line, 0.0)
            };

            let doc_thresh_high = 1.5 * radians_per_volt;
            let doc_thresh_low = -0.8 * radians_per_volt;

            for col in 0..self.line_width {
                // start_frac shifts the entire sweep by the fractional
                // sync_phase; aa_shift compensates the anti-alias
                // boxcar's group delay; the col-driven term spreads the
                // fractional raw_len_f across the requested line_width
                // samples.
                let idx_float =
                    start_frac + aa_shift + (col as f32 * raw_len_f) / (self.line_width as f32);
                let idx = idx_float as usize;
                let frac = idx_float - idx as f32;

                let val = if idx >= 1 && idx + 2 < line_src.len() {
                    let v0 = line_src[idx - 1];
                    let v1 = line_src[idx];
                    let v2 = line_src[idx + 1];
                    let v3 = line_src[idx + 2];

                    let a = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
                    let b = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
                    let c = -0.5 * v0 + 0.5 * v2;
                    let d = v1;

                    a * frac * frac * frac + b * frac * frac + c * frac + d
                } else if idx + 1 < line_src.len() {
                    // Fallback to linear at the very edges
                    line_src[idx] * (1.0 - frac) + line_src[idx + 1] * frac
                } else if idx < line_src.len() {
                    line_src[idx]
                } else {
                    // Past the end of the extracted line — clamp to the
                    // last real sample rather than injecting a 0, which
                    // would paint a black pixel at the right edge of the
                    // active picture (line_width > visible width, so the
                    // tail is displayed). `.last()` rather than
                    // `[len - 1]` so a (guarded-impossible) empty line
                    // can't underflow.
                    line_src.last().copied().unwrap_or(0.0)
                };

                let is_doc = val > doc_thresh_high || val < doc_thresh_low;
                tbc_line[col] = val;
                doc_mask[col] = is_doc;
            }

            // Blur doc mask slightly (morphological dilate)
            dilated_doc.copy_from_slice(&doc_mask);
            for col in 2..self.line_width - 2 {
                if doc_mask[col - 2]
                    || doc_mask[col - 1]
                    || doc_mask[col]
                    || doc_mask[col + 1]
                    || doc_mask[col + 2]
                {
                    dilated_doc[col] = true;
                }
            }

            // AGC Luma — measured per line, applied through smoothed
            // EMA state (`agc_gain`/`agc_porch`). The raw measurements
            // are two ~25-sample window means; instantaneous per-line
            // application let a single click in either window swing an
            // entire line's brightness and offset — visible as
            // line-to-line flicker exactly at the low CNR where the
            // picture is already struggling.
            let sync_tip_start = (self.line_width as f32 * 0.01) as usize;
            let sync_tip_end = (self.line_width as f32 * 0.04) as usize;
            let bp_start = (self.line_width as f32 * 0.12) as usize;
            let bp_end = (self.line_width as f32 * 0.15) as usize;

            let sync_tip = tbc_line[sync_tip_start..sync_tip_end].iter().sum::<f32>()
                / (sync_tip_end - sync_tip_start) as f32;
            let back_porch =
                tbc_line[bp_start..bp_end].iter().sum::<f32>() / (bp_end - bp_start) as f32;

            // Require a *positive* sync-to-porch swing, not merely a
            // large-magnitude one. In this crate's FM-demod convention
            // blanking always sits above the sync tip, so a negative swing
            // means the discriminator output is inverted (swapped I/Q, or a
            // spectrally-mirrored capture) — such lines never update the
            // AGC state, so the wiring fault stays visible as an inverted
            // picture rather than being silently "corrected".
            let swing = back_porch - sync_tip;
            if swing > 0.01 {
                let target_gain = (nominal_swing / swing).clamp(AGC_GAIN_CLAMP.0, AGC_GAIN_CLAMP.1);
                if self.agc_primed {
                    self.agc_gain += AGC_EMA_ALPHA * (target_gain - self.agc_gain);
                    self.agc_porch += AGC_EMA_ALPHA * (back_porch - self.agc_porch);
                } else {
                    // First valid measurement seeds the state directly
                    // instead of easing in from the 1.0/0.0 init.
                    self.agc_gain = target_gain;
                    self.agc_porch = back_porch;
                    self.agc_primed = true;
                }
            }
            // Apply the smoothed state even when this line's own
            // measurement was invalid (a dropout or click burst): the
            // last good gain beats leaving the line unscaled.
            if self.agc_primed {
                for v in tbc_line.iter_mut() {
                    *v = (*v - self.agc_porch) * self.agc_gain;
                }
            }

            // Store TBC line and DOC mask
            let offset = row * self.line_width;
            current_frame_tbc[offset..(self.line_width + offset)]
                .copy_from_slice(&tbc_line[..self.line_width]);
            current_frame_doc[offset..(self.line_width + offset)]
                .copy_from_slice(&dilated_doc[..self.line_width]);

            field_rows_written = row + 1;
        }

        let mut current_frame_y = vec![0.0f32; self.field_lines * self.line_width];
        let rows_to_process = field_rows_written;

        // ── Chroma-burst presence → subcarrier-notch gating ─────────
        //
        // Many FPV cameras are effectively monochrome sources with no
        // colour burst; for those the notch deletes real luma detail
        // around f_sc for nothing. Measure burst presence with a
        // Goertzel at f_sc over the back-porch window (7–14 % of the
        // line — where the burst lives for both standards) on up to 32
        // rows, and gate the notch with NOTCH_HYSTERESIS_FIELDS of
        // hysteresis so a marginal source can't toggle sharpness field
        // to field.
        if rows_to_process > 8 {
            let b_lo = (self.line_width as f32 * 0.07) as usize;
            let b_hi = (self.line_width as f32 * 0.14) as usize;
            let step = (rows_to_process / 32).max(1);
            let mut hits = 0u32;
            let mut total = 0u32;
            let mut row = 0;
            while row < rows_to_process {
                let off = row * self.line_width;
                let w = &current_frame_tbc[off + b_lo..off + b_hi];
                let mean = w.iter().sum::<f32>() / w.len() as f32;
                let var = w.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w.len() as f32;
                aa_line.clear();
                aa_line.extend(w.iter().map(|v| v - mean));
                // A window dominated by a real burst has tone-amplitude²
                // ≈ 2·variance; noise-only windows put ~1/N of their
                // variance in any one bin. `> var` splits the two with
                // margin on both sides.
                if var > 1e-12 && goertzel_amp_sq(&aa_line, self.notch_w0) > var {
                    hits += 1;
                }
                total += 1;
                row += step;
            }
            let burst_present =
                total > 0 && (hits as f32) >= NOTCH_BURST_ROW_MAJORITY * total as f32;
            if burst_present != self.notch_enabled {
                self.notch_flip_streak += 1;
                if self.notch_flip_streak >= NOTCH_HYSTERESIS_FIELDS {
                    self.notch_enabled = burst_present;
                    self.notch_flip_streak = 0;
                }
            } else {
                self.notch_flip_streak = 0;
            }
        }

        // ── Per-row clean + temporal denoise (PARALLEL) ────────────
        //
        // DOC replacement, the subcarrier notch, and the multi-field
        // temporal-denoise median are independent per row: each reads
        // only shared-immutable inputs (the TBC field, the DOC mask,
        // the previous field, and the history ring) and writes its own
        // row of `current_frame_y`. The per-pixel median is the hot
        // path that pushed single-threaded decode past real time at
        // 25 MSPS (causing dropped IQ chunks → sync dips); fanning the
        // rows across cores with rayon keeps the full N-field window
        // viable live. The cheap CTI + Y→RGB pack stays sequential
        // below since it writes parity-interleaved output rows.
        //
        // Temporal denoise, per pixel: collect the current value plus
        // the same-pixel value from every stored history field; the
        // max abs difference is the motion estimate; blend the current
        // value toward the median (kills FM "click" sparkles) by
        // `1 - motion_weight`. Static pixels denoise fully (√N), moving
        // pixels keep the current value. On dropout (`force_static`)
        // the weight is forced to 0 so even moving pixels take history.
        const MAX_HISTORY: usize = MAX_TEMPORAL_WINDOW;
        let line_width = self.line_width;
        let has_prev = self.has_prev;
        let prev_frame_tbc = &self.prev_frame_tbc;
        let history = &self.history;
        let hist_len = history.len().min(MAX_HISTORY);
        let (nb0, nb1, nb2, na1, na2) = (
            self.notch_b0,
            self.notch_b1,
            self.notch_b2,
            self.notch_a1,
            self.notch_a2,
        );
        let notch_enabled = self.notch_enabled;
        let motion_threshold = TEMPORAL_MOTION_THRESHOLD * radians_per_volt;
        let use_smart_doc = self.use_smart_doc;
        let current_frame_tbc_ref = &current_frame_tbc;
        let current_frame_doc_ref = &current_frame_doc;

        current_frame_y
            .par_chunks_mut(line_width)
            .enumerate()
            .take(rows_to_process)
            .for_each(|(row, y_out)| {
                let offset = row * line_width;
                // Work in place on this row's output chunk — the row
                // used to copy into a per-row `Vec` and back, an
                // allocation and two copies per row for nothing.
                y_out.copy_from_slice(&current_frame_tbc_ref[offset..offset + line_width]);
                let y_line = y_out;
                let doc_mask = &current_frame_doc_ref[offset..offset + line_width];

                // 1. DOC replacement from the previous field.
                if has_prev {
                    for col in 0..line_width {
                        if doc_mask[col] {
                            if use_smart_doc {
                                let mut spatial_val = 0.0;
                                let mut spatial_count = 0.0;
                                if row > 0 {
                                    let up_off = (row - 1) * line_width;
                                    if !current_frame_doc_ref[up_off + col] {
                                        spatial_val += current_frame_tbc_ref[up_off + col];
                                        spatial_count += 1.0;
                                    }
                                }
                                if row + 1 < rows_to_process {
                                    let down_off = (row + 1) * line_width;
                                    if !current_frame_doc_ref[down_off + col] {
                                        spatial_val += current_frame_tbc_ref[down_off + col];
                                        spatial_count += 1.0;
                                    }
                                }
                                let temporal_val = prev_frame_tbc[offset + col];
                                if spatial_count > 0.0 {
                                    let spatial_mean = spatial_val / spatial_count;
                                    let w_spatial = self.smart_doc_spatial_weight;
                                    let w_temporal = 1.0 - w_spatial;
                                    y_line[col] =
                                        spatial_mean * w_spatial + temporal_val * w_temporal;
                                } else {
                                    y_line[col] = temporal_val;
                                }
                            } else {
                                y_line[col] = prev_frame_tbc[offset + col];
                            }
                        }
                    }
                }

                // 2. Subcarrier notch (zero-phase forward/backward
                //    biquad) — only while the burst gate says a colour
                //    subcarrier is actually present.
                if notch_enabled {
                    filtfilt(y_line, nb0, nb1, nb2, na1, na2);
                }

                // 3. Temporal denoise. Stack scratch, allocation-free.
                let mut history_rows: [Option<&[f32]>; MAX_HISTORY] = [None; MAX_HISTORY];
                let mut history_count = 0usize;
                for n in 0..hist_len {
                    if let Some(field) = history.prev_field(n) {
                        history_rows[history_count] = Some(&field[offset..offset + line_width]);
                        history_count += 1;
                    }
                }
                for col in 0..line_width {
                    let cur = y_line[col];
                    let mut samples = [0.0f32; MAX_HISTORY + 1];
                    let mut n_samples = 1usize;
                    samples[0] = cur;
                    let mut max_motion = 0.0f32;
                    for row_slice in history_rows.iter().take(history_count).flatten() {
                        let prev = row_slice[col];
                        samples[n_samples] = prev;
                        n_samples += 1;
                        let d = (cur - prev).abs();
                        if d > max_motion {
                            max_motion = d;
                        }
                    }
                    if n_samples == 1 {
                        // No history yet — pass through unchanged.
                        continue;
                    }
                    let motion_weight = if force_static {
                        0.0
                    } else {
                        (max_motion / motion_threshold).clamp(0.0, 1.0)
                    };
                    // Median via insertion sort on the tiny stack array
                    // (n ≤ MAX_HISTORY + 1): branchless, register-resident.
                    let mut sorted = samples;
                    let len = n_samples;
                    for i in 1..len {
                        let mut j = i;
                        while j > 0 && sorted[j - 1] > sorted[j] {
                            sorted.swap(j - 1, j);
                            j -= 1;
                        }
                    }
                    let median = sorted[len / 2];
                    y_line[col] = motion_weight * cur + (1.0 - motion_weight) * median;
                }
            });

        // ── CTI + Y→RGB pack (SEQUENTIAL) ──────────────────────────
        // Cheap relative to the denoise, and it writes parity-strided
        // output rows, so it stays single-threaded. The complementary
        // parity rows are filled from `field_buf` after this loop.
        // dst_row max is `(field_lines - 1) * 2 + 1` (479 NTSC / 575
        // PAL), within `height - 1`, so the row math needs no guard.
        let h_blank_end = (self.line_width as f32 * ACTIVE_VIDEO_LEFT_CROP_FRAC) as usize;
        // Hoisted CTI scratch (see the TBC scratch note above).
        let mut y_cti = vec![0.0f32; self.line_width];
        for row in 0..rows_to_process {
            let offset = row * self.line_width;
            let y_clean = &current_frame_y[offset..offset + self.line_width];

            // 4. CTI (unsharp mask on luma).
            y_cti.copy_from_slice(y_clean);
            for col in 1..self.line_width - 1 {
                let diff2 = y_clean[col - 1] - 2.0 * y_clean[col] + y_clean[col + 1];
                y_cti[col] -= 0.2 * diff2;
            }

            // 5. Y→RGB (monochrome), cropped to active video, into this
            //    field's parity rows of the interlaced output frame.
            let dst_row = row * 2 + self.field_parity as usize;
            let dst_off = dst_row * self.width;
            for col in 0..self.width {
                let src_col = h_blank_end + col;
                if src_col >= self.line_width {
                    break;
                }
                let y_norm = y_cti[src_col] / radians_per_volt;
                // `+ 0.5`: round to nearest instead of truncating, which
                // biased every pixel ~half an LSB dark.
                let c = (y_norm.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                frame[dst_off + col] = 0xFF000000 | (c << 16) | (c << 8) | c;
            }
        }

        // Field merge. The current call rendered the `field_parity`
        // rows of `frame`; pull the complementary parity's rows in
        // from `field_buf` (the previously-emitted frame). On the
        // very first call `field_buf` is all-zeros, so frame 1 looks
        // like only-one-field — every subsequent frame is properly
        // interlaced from two adjacent fields. This replaces the
        // previous "line-doubling vertical blend" pass which copied
        // each captured row into both even and odd output positions,
        // effectively dropping half of the vertical detail.
        //
        // Tail-clear: for early-break rows beyond `rows_to_process`,
        // zero both parities so a short capture doesn't leak stale
        // pixels from the previous full frame's rows-too-far-down
        // into the visible region.
        let cur_parity = self.field_parity as usize;
        let comp_parity = 1 - cur_parity;
        for row in 0..self.field_lines {
            let comp_dst_row = row * 2 + comp_parity;
            if comp_dst_row >= self.height {
                break;
            }
            let comp_off = comp_dst_row * self.width;
            if row >= rows_to_process {
                // Beyond the captured rows: also blank the current
                // parity, otherwise we'd preserve the parity from a
                // previous longer call.
                let cur_off = (row * 2 + cur_parity) * self.width;
                for col in 0..self.width {
                    frame[cur_off + col] = 0;
                    frame[comp_off + col] = 0;
                }
            } else {
                frame[comp_off..comp_off + self.width]
                    .copy_from_slice(&self.field_buf[comp_off..comp_off + self.width]);
            }
        }
        // Persist this fully-merged frame so the next call has access
        // to *both* parities (current and complementary) when it
        // pulls the complementary parity in.
        self.field_buf.copy_from_slice(frame);

        #[cfg(feature = "neural-vsr")]
        // Only run with a valid previous field (a complete interlaced
        // frame). `take()` the restorer out so the persistent in/out
        // scratch and `hidden_state` can be borrowed without aliasing
        // `self.neural_restorer`; put it back afterwards.
        if self.has_prev
            && let Some(mut restorer) = self.neural_restorer.take()
        {
            for (dst, p) in self.neural_in.iter_mut().zip(frame.iter()) {
                *dst = (((p >> 16) & 0xFF) as f32) / 255.0;
            }
            match restorer.process_frame_luma(
                self.width,
                self.height,
                &self.neural_in,
                self.hidden_state.as_deref(),
                &mut self.neural_out,
            ) {
                Ok(new_hidden) => {
                    self.hidden_state = Some(new_hidden);
                    for (px, y) in frame.iter_mut().zip(self.neural_out.iter()) {
                        let c = (y.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                        *px = 0xFF000000 | (c << 16) | (c << 8) | c;
                    }
                }
                Err(e) => log::error!("Neural restorer failed: {}", e),
            }
            self.neural_restorer = Some(restorer);
        }
        // Parity of the field just rendered, captured before the toggle:
        // the FieldMeta below describes THIS field, and reading
        // `self.field_parity` after `^= 1` stamped every history entry
        // with the NEXT field's parity instead — an off-by-one-field
        // error in the value documented as the key for future 3D-comb
        // work.
        let rendered_parity = self.field_parity;
        self.field_parity ^= 1;

        self.prev_frame_tbc = current_frame_tbc;

        // Push the just-rendered Y field into the multi-field
        // history buffer for the next call's temporal denoise.
        // Field period per standard: PAL is 50 fields/s (20 000 µs), NTSC
        // ~59.94 (16 667 µs). This was hardcoded to the NTSC value, so a
        // PAL stream's history timestamps ran ~20% fast. Only diagnostics
        // read them today, but the field is documented as the temporal
        // reference for future motion-compensated work, where a wrong
        // inter-field interval would weight the wrong neighbours.
        let field_period_us: u64 = if self.pal { 20_000 } else { 16_667 };
        let meta = FieldMeta {
            timestamp_us: self.field_counter * field_period_us,
            field_parity: rendered_parity,
            sync_quality,
            mean_amplitude: if !current_frame_y.is_empty() {
                let s: f32 = current_frame_y.iter().sum();
                s / current_frame_y.len() as f32
            } else {
                0.0
            },
        };
        self.history.push(current_frame_y, meta);
        self.field_counter = self.field_counter.wrapping_add(1);
        self.has_prev = true;

        // Advance to the next field. When the VBI parser locked, the
        // next field's vertical-sync group starts exactly one field
        // duration after this one's `broad_start` — a direct datum,
        // not a search — so `consumed` lands a few lines *before* it
        // (via `margin`), letting the next call's own parser find it
        // fresh rather than us guessing where mid-field. This also
        // structurally avoids the old heuristic's failure mode (the
        // leading equalizing pulse look-alike, described below).
        let consumed = if let Some(info) = &vbi_info {
            let field_total_lines = if self.pal {
                crate::vbi::consts::PAL_FIELD_TOTAL_LINES
            } else {
                crate::vbi::consts::NTSC_FIELD_TOTAL_LINES
            } as f32;
            let margin_lines = 6.0;
            let advance = info.broad_start + (field_total_lines - margin_lines) * self.line_period;
            advance.round() as usize
        } else {
            // Anchor the search to the expected field boundary — row-0
            // sync (`sync_positions[0]`, the most reliable datum) plus
            // one field of active line-periods — rather than walking
            // forward from the last rendered row's sync and taking the
            // *first* density spike. During the vertical-sync interval
            // many one-line windows clear the density threshold; the
            // first of them is the leading equalizing pulse, ~2-3 lines
            // before the true next-field datum. Latching that leading
            // edge makes the next call open mid-V-sync, which is the
            // startup mis-lock that craters sync-quality on the first
            // few fields. Instead, scan a short forward window past
            // active video and pick the *strongest* sync plateau,
            // falling back to a clean one-field advance if nothing
            // clears the threshold (so we slip at most a fraction of a
            // field rather than skipping whole fields).
            let field_start = sync_positions[0];
            let nominal_advance = field_start + self.line_period * self.field_lines as f32;
            let mut consumed = nominal_advance.round() as usize;
            let density_threshold = self.samples_per_line / 2;
            let stride = (self.samples_per_line / 4).max(1);
            let search_lo = consumed;
            let search_hi = (search_lo + 8 * self.samples_per_line)
                .min(demod_data.len().saturating_sub(self.samples_per_line));
            let mut best_below = density_threshold; // require at least the threshold to override the fallback
            let mut probe = search_lo;
            while probe < search_hi {
                let mut below = 0usize;
                for i in probe..probe + self.samples_per_line {
                    if demod_data[i] < v_sync_threshold {
                        below += 1;
                    }
                }
                if below > best_below {
                    best_below = below;
                    consumed = probe;
                }
                probe += stride;
            }
            consumed
        };

        // Never report consuming past the end of the input. `nominal_advance`
        // is built from the float `line_period`, while the up-front
        // `required_samples` guard uses the integer `samples_per_line`; when
        // `line_period > samples_per_line` and the buffer is only just long
        // enough, `consumed` can land a fraction of a line past
        // `demod_data.len()`. The caller advances its cursor by `consumed`
        // and re-slices `[consumed..]`, so an overshoot would panic. Clamp.
        Some(consumed.min(demod_data.len()))
    }

    pub fn save_ppm_frame(&self, frame: &[u32], path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(path)?;
        // PPM dimensions must match the frame buffer's actual layout
        // (`width` × `height`), not the unwrapped TBC line length
        // (`line_width` × `height`). The frame buffer is cropped to
        // the active-video 720-pixel-wide window before output, so
        // declaring `line_width` here lied to the decoder and either
        // produced a corrupt PPM or shifted the image diagonally
        // depending on which viewer parsed it.
        writeln!(
            file,
            "P6
{} {}
255",
            self.width, self.height
        )?;
        for &argb in frame {
            let r = ((argb >> 16) & 0xFF) as u8;
            let g = ((argb >> 8) & 0xFF) as u8;
            let b = (argb & 0xFF) as u8;
            file.write_all(&[r, g, b])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_video_standard_rejects_noise_instead_of_claiming_ntsc() {
        // On FM-demodulated noise the below-threshold dips are continuous,
        // so the `min_gap` skip collapses the mean inter-dip interval to
        // just above 30 us -- a ~33 kHz "line rate". Without a plausibility
        // bound that still classified as NTSC (it only had to be closer to
        // 15734 than to 15625), which is how empty-band noise got reported
        // as video on live captures.
        let sr = 15_360_000u32;
        let mut seed = 0x2545F491_4F6CDD1Du64;
        let noise: Vec<f32> = (0..sr as usize / 20)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        assert_eq!(
            detect_video_standard(&noise, sr),
            SignalType::Unknown,
            "noise must not be classified as a video standard"
        );
    }

    #[test]
    fn detect_video_standard_returns_unknown_on_degenerate_input() {
        let sr = 15_360_000u32;
        // Too short to contain a single line.
        assert_eq!(detect_video_standard(&[0.0; 100], sr), SignalType::Unknown);
        // All-positive (DC-offset / inverted) waveform: no sync tips
        // below zero, so no standard can honestly be claimed.
        let positive = vec![0.5f32; sr as usize / 10];
        assert_eq!(detect_video_standard(&positive, sr), SignalType::Unknown);
    }

    #[test]
    fn detect_video_standard_still_identifies_real_pal_and_ntsc() {
        // The plausibility bound must not cost real detections: run the
        // synthetic generator for both standards and confirm each is still
        // named correctly.
        for is_pal in [false, true] {
            let cfg = base_synth_config(is_pal, 5e6);
            let data = generate_fields(&cfg, 2);
            let want = if is_pal {
                SignalType::AnalogVideoPal
            } else {
                SignalType::AnalogVideoNtsc
            };
            assert_eq!(
                detect_video_standard(&data, cfg.sample_rate),
                want,
                "is_pal={is_pal}: real synthetic video must still classify"
            );
        }
    }

    #[test]
    fn reconstruct_frame_into_rejects_mismatched_frame_buffer() {
        let mut fr = FrameReconstructor::new(20_000_000, false, 200_000.0, false);
        let demod_data = vec![0.0f32; 10_000];
        // Deliberately not width * height.
        let mut wrong_size_frame = vec![0u32; 10];
        assert!(
            fr.reconstruct_frame_into(&demod_data, &mut wrong_size_frame)
                .is_none()
        );
    }

    #[test]
    fn reconstruct_frame_into_rejects_empty_demod_data() {
        let mut fr = FrameReconstructor::new(20_000_000, false, 200_000.0, false);
        let mut frame = vec![0u32; fr.width * fr.height];
        assert!(fr.reconstruct_frame_into(&[], &mut frame).is_none());
    }

    #[test]
    fn reconstruct_frame_handles_short_noise_without_panicking() {
        // Regression test for the video reconstruction path: feed input
        // far too short to contain a real field through the convenience
        // wrapper (which sizes its own buffer correctly) and confirm it
        // degrades to `None` rather than panicking on any of the
        // internal index arithmetic.
        let mut fr = FrameReconstructor::new(20_000_000, false, 200_000.0, false);
        let demod_data = vec![0.01f32; 500];
        assert!(fr.reconstruct_frame(&demod_data).is_none());
    }

    #[test]
    fn reconstruct_frame_handles_degenerate_zero_sample_rate_without_panicking() {
        // sample_rate = 0 makes samples_per_line / line_period 0 — an
        // invalid config, but one that must degrade gracefully rather
        // than panic (e.g. via division-by-zero-derived index math).
        let mut fr = FrameReconstructor::new(0, false, 200_000.0, false);
        let demod_data = vec![0.0f32; 1000];
        assert!(fr.reconstruct_frame(&demod_data).is_none());
    }

    use crate::synthetic::{SyntheticVideoConfig, TestPattern, generate_fields};

    fn base_synth_config(is_pal: bool, deviation_hz: f32) -> SyntheticVideoConfig {
        SyntheticVideoConfig {
            sample_rate: 15_360_000,
            is_pal,
            deviation_hz,
            pattern: TestPattern::Flat(0.0),
            start_field: FieldParity::First,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        }
    }

    /// Average luma of an output row (`0xFF000000 | c<<16 | c<<8 | c`,
    /// so the blue byte alone is the luma value).
    fn row_avg_luma(frame: &[u32], width: usize, row: usize) -> f32 {
        let off = row * width;
        let sum: u32 = frame[off..off + width].iter().map(|&px| px & 0xFF).sum();
        sum as f32 / width as f32
    }

    /// A full-width bright band proves *row* placement without needing
    /// to reason about the TBC's column resampling/crop — the concern
    /// this test exists for is [`crate::vbi`]'s standards-correct
    /// blanking (PAL 25 lines vs the old hardcoded 20), a purely
    /// vertical effect.
    #[test]
    fn white_band_lands_on_the_correct_row_for_ntsc_and_pal() {
        for is_pal in [false, true] {
            let deviation_hz = 5e6;
            let mut cfg = base_synth_config(is_pal, deviation_hz);
            let row0 = 100;
            let band_h = 20;
            cfg.pattern = TestPattern::WhiteSquare {
                row0,
                col0: 0,
                h: band_h,
                w: crate::synthetic::OUTPUT_WIDTH,
            };
            let data = generate_fields(&cfg, 3);

            let mut recon = FrameReconstructor::new(cfg.sample_rate, is_pal, deviation_hz, false);
            let mut frame = vec![0u32; recon.width * recon.height];
            recon
                .reconstruct_frame_into(&data, &mut frame)
                .unwrap_or_else(|| panic!("is_pal={is_pal}: expected a reconstructed field"));

            let parity = recon.field_parity as usize ^ 1; // this call's parity, before the end-of-call toggle
            let bright_row = row0 * 2 + parity;
            let dark_row_above = (row0 - 10) * 2 + parity;
            let dark_row_below = (row0 + band_h + 10) * 2 + parity;

            let bright = row_avg_luma(&frame, recon.width, bright_row);
            let dark_above = row_avg_luma(&frame, recon.width, dark_row_above);
            let dark_below = row_avg_luma(&frame, recon.width, dark_row_below);
            assert!(
                bright > 180.0,
                "is_pal={is_pal}: row {bright_row} luma {bright}, expected bright"
            );
            assert!(
                dark_above < 60.0,
                "is_pal={is_pal}: row {dark_row_above} luma {dark_above}, expected dark"
            );
            assert!(
                dark_below < 60.0,
                "is_pal={is_pal}: row {dark_row_below} luma {dark_below}, expected dark"
            );
        }
    }

    #[test]
    fn goertzel_detects_tone_and_ignores_noise() {
        let w0 = 0.9f32; // arbitrary in-band angular frequency
        let n = 64;
        let tone: Vec<f32> = (0..n).map(|i| (w0 * i as f32).cos()).collect();
        let amp_sq = goertzel_amp_sq(&tone, w0);
        assert!(
            (amp_sq - 1.0).abs() < 0.3,
            "unit tone should measure amp²≈1, got {amp_sq}"
        );

        let mut seed = 7u64;
        let noise: Vec<f32> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let mean = noise.iter().sum::<f32>() / n as f32;
        let var = noise.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let centered: Vec<f32> = noise.iter().map(|v| v - mean).collect();
        let amp_sq = goertzel_amp_sq(&centered, w0);
        assert!(
            amp_sq < var,
            "noise-only window must stay under the burst threshold (amp²={amp_sq}, var={var})"
        );
    }

    /// Burst-free video (like the synthetic generator's, and like most
    /// monochrome FPV cameras) must disable the subcarrier notch after
    /// the hysteresis window, recovering luma detail around f_sc — and
    /// the notch must start enabled (safe default).
    #[test]
    fn notch_disables_after_hysteresis_on_burst_free_video() {
        let deviation_hz = 5e6;
        let cfg = base_synth_config(false, deviation_hz);
        let data = generate_fields(&cfg, 6);
        let mut recon = FrameReconstructor::new(cfg.sample_rate, false, deviation_hz, false);
        assert!(recon.chroma_notch_active(), "notch must start enabled");
        let mut frame = vec![0u32; recon.width * recon.height];
        let mut cursor = 0usize;
        for _ in 0..4 {
            let consumed = recon
                .reconstruct_frame_into(&data[cursor..], &mut frame)
                .expect("field should reconstruct");
            cursor += consumed;
        }
        assert!(
            !recon.chroma_notch_active(),
            "burst-free fields should disable the notch after hysteresis"
        );
    }

    #[test]
    fn consecutive_fields_interlace_onto_correct_row_parities() {
        let deviation_hz = 5e6;
        let mut cfg = base_synth_config(false, deviation_hz);
        // Field 1 (First) paints rows 50..70; field 2 (Second) paints a
        // *different* band at rows 150..170. Both bands must land on
        // their own field's row parity and survive into the merged
        // output of the following call.
        cfg.pattern = TestPattern::WhiteSquare {
            row0: 50,
            col0: 0,
            h: 20,
            w: crate::synthetic::OUTPUT_WIDTH,
        };
        let field1 = generate_fields(&cfg, 1);
        cfg.start_field = FieldParity::Second;
        cfg.pattern = TestPattern::WhiteSquare {
            row0: 150,
            col0: 0,
            h: 20,
            w: crate::synthetic::OUTPUT_WIDTH,
        };
        let field2_alone = generate_fields(&cfg, 1);

        let mut data = field1.clone();
        data.extend_from_slice(&field2_alone);
        // Pad so the second call has enough trailing data to lock its
        // own VBI group and advance (mirrors a live decode buffer that
        // always has more than exactly one field queued).
        data.extend_from_slice(&field2_alone);

        let mut recon = FrameReconstructor::new(cfg.sample_rate, false, deviation_hz, false);
        let mut frame = vec![0u32; recon.width * recon.height];
        let consumed1 = recon
            .reconstruct_frame_into(&data, &mut frame)
            .expect("expected field 1 to reconstruct");
        assert!(
            row_avg_luma(&frame, recon.width, 100) > 180.0,
            "field 1's band (row 50 -> output row 100)"
        );

        recon
            .reconstruct_frame_into(&data[consumed1..], &mut frame)
            .expect("expected field 2 to reconstruct");
        assert!(
            row_avg_luma(&frame, recon.width, 301) > 180.0,
            "field 2's band (row 150 -> output row 301)"
        );
        assert!(
            row_avg_luma(&frame, recon.width, 100) > 180.0,
            "field 1's band must survive the merge into field 2's output frame"
        );
    }

    #[test]
    fn dropped_field_is_corrected_by_vbi_parity_override_not_left_inverted() {
        // Simulate a decode pipeline that lost one field's batch
        // entirely (a real occurrence: a dropped IQ batch, a channel-
        // hop interruption). A naive parity toggle has no way to know
        // a field went missing and stays permanently out of phase; the
        // VBI parser reads the real parity out of each field's own
        // vertical-sync structure and corrects it every call.
        let deviation_hz = 5e6;
        let cfg = base_synth_config(false, deviation_hz); // First, Second, First, Second, ...
        let f1_len = generate_fields(&cfg, 1).len();
        let f2_len = generate_fields(&cfg, 2).len() - f1_len;
        let all_four = generate_fields(&cfg, 4);
        let f3_start = f1_len + f2_len;
        let f3_len = generate_fields(&cfg, 3).len() - f3_start;

        let mut spliced = Vec::new();
        spliced.extend_from_slice(&all_four[0..f1_len]); // field 1 (First)
        spliced.extend_from_slice(&all_four[f3_start..f3_start + f3_len]); // field 3 (First) -- field 2 dropped
        spliced.extend_from_slice(&all_four[f3_start..f3_start + f3_len]); // padding so call 2 has enough trailing data

        let mut recon = FrameReconstructor::new(cfg.sample_rate, false, deviation_hz, false);
        let mut frame = vec![0u32; recon.width * recon.height];
        let consumed1 = recon
            .reconstruct_frame_into(&spliced, &mut frame)
            .expect("expected field 1 to reconstruct");
        recon
            .reconstruct_frame_into(&spliced[consumed1..], &mut frame)
            .expect("expected field 3 to reconstruct despite the dropped field 2");

        // See the derivation in this test's accompanying commit message:
        // with the VBI override correcting the naive toggle, field_parity
        // is 1 after this second call (both fields are genuinely First,
        // so the override re-applies 0 before rendering, then the
        // end-of-call toggle leaves 1); a naive toggle-only
        // implementation would instead leave it at 0.
        assert_eq!(
            recon.field_parity, 1,
            "VBI override should re-detect First parity for the post-drop field, not blindly toggle"
        );
    }

    // ── Phase-2 weak-signal DSP (matched-sync, OLS TBC, SmartDOC) ──────

    /// The Phase-2 DSP is on by default; the builders flip each flag and
    /// clamp the SmartDOC weight. Guards the on-by-default contract the
    /// viewer relies on.
    #[test]
    fn phase2_dsp_defaults_on_and_builders_override() {
        let r = FrameReconstructor::new(15_360_000, false, 5e6, false);
        assert!(r.use_matched_sync && r.use_line_locked_clock && r.use_smart_doc);

        let r = r
            .with_matched_sync(false)
            .with_line_locked_clock(false)
            .with_smart_doc(false, 1.7);
        assert!(!r.use_matched_sync && !r.use_line_locked_clock && !r.use_smart_doc);
        assert!(
            (r.smart_doc_spatial_weight - 1.0).abs() < 1e-6,
            "spatial weight must clamp to [0,1]"
        );
    }

    /// Matched-filter sync acquisition must locate a synthetic H-sync dip
    /// to within a sample — the property that gives it GCOR ≈ 1.0 on a
    /// clean signal (exact integer phase recovery).
    #[test]
    fn phase2_matched_sync_locates_pulse_within_one_sample() {
        let fs = 15_360_000f32;
        let w = (fs * 4.7e-6).round() as usize; // ~72-sample H-sync
        let mut demod = vec![0.0f32; 4000]; // blanking
        let center = 2000usize;
        for s in &mut demod[center - w / 2..center + w / 2] {
            *s = -1.0; // sync tip below blanking
        }
        let got = matched_sync_center(&demod, center as f32 + 5.0, 150, w, -0.1)
            .expect("should find the sync pulse");
        assert!(
            (got - center as f32).abs() <= 1.5,
            "matched sync located {got}, expected ~{center}"
        );
    }

    /// The robust OLS TBC must lock the line period to the crystal rate
    /// on a clean capture (straight verticals, no slant). Guards the
    /// MAD-reweighted refit against a plain-OLS regression.
    #[test]
    fn phase2_ols_locks_line_period_on_clean_signal() {
        let cfg = SyntheticVideoConfig {
            pattern: TestPattern::Bars,
            ..base_synth_config(false, 5e6)
        };
        let data = generate_fields(&cfg, 2);
        let mut recon =
            FrameReconstructor::new(cfg.sample_rate, false, 5e6, false).with_temporal_window(1);
        let mut frame = vec![0u32; recon.width * recon.height];
        recon
            .reconstruct_frame_into(&data, &mut frame)
            .expect("clean field should reconstruct");
        let nominal = recon.samples_per_line as f32;
        assert!(
            (recon.line_period - nominal).abs() < 1.0,
            "OLS line period {} drifted from nominal {nominal}",
            recon.line_period
        );
    }

    /// SmartDOC concealment must blend spatial neighbours with the
    /// previous field rather than substitute the previous field wholesale
    /// — verified via `compute_gradient_correlation` staying high on a
    /// clean reconstruction with SmartDOC on.
    #[test]
    fn phase2_smart_doc_reconstructs_clean_field_faithfully() {
        let cfg = SyntheticVideoConfig {
            pattern: TestPattern::Bars,
            ..base_synth_config(false, 5e6)
        };
        let data = generate_fields(&cfg, 3);
        let mut a =
            FrameReconstructor::new(cfg.sample_rate, false, 5e6, false).with_temporal_window(1);
        let mut fa = vec![0u32; a.width * a.height];
        // Two fields so the second is a full interlaced frame.
        let c1 = a.reconstruct_frame_into(&data, &mut fa).unwrap();
        a.reconstruct_frame_into(&data[c1..], &mut fa).unwrap();

        // A clean field reconstructed with SmartDOC on should still be a
        // faithful, self-consistent picture (high self-gradient
        // structure, not a smeared blend).
        let gcor = crate::metrics::compute_gradient_correlation(&fa, &fa, a.width, a.height);
        assert!(
            (gcor - 1.0).abs() < 1e-6,
            "identical-frame gradient correlation must be 1.0, got {gcor}"
        );
    }
}
