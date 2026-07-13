//! Analog Video Frame Reconstruction
//!
//! This module takes a demodulated FM signal and reconstructs it into 2D frames
//! by identifying H-Sync and V-Sync pulses. The pipeline is monochrome (luma
//! only — see DESIGN.md §6 for why colour is disabled) and features a sub-sample
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
use crate::types::SignalType;
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

pub fn detect_video_standard(demod_data: &[f32], sample_rate: u32) -> SignalType {
    let min_samples = sample_rate as usize / 200;
    if demod_data.len() < min_samples {
        return SignalType::AnalogVideoNtsc;
    }

    let search_len = (sample_rate as f32 * 640e-6) as usize;
    let search_len = search_len.min(demod_data.len());
    let global_min = demod_data[..search_len]
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);

    let threshold = global_min * 0.3;
    let min_gap = (sample_rate as f32 * 30e-6) as usize;
    let max_gap = (sample_rate as f32 * 100e-6) as usize;

    let mut sync_positions: Vec<usize> = Vec::new();
    let mut i = 0;
    let scan_len = (sample_rate as f32 * 5000e-6) as usize;
    let scan_len = scan_len.min(demod_data.len());

    while i < scan_len {
        if demod_data[i] < threshold {
            let mut local_min_idx = i;
            let mut local_min_val = demod_data[i];
            while i < scan_len && demod_data[i] < threshold {
                if demod_data[i] < local_min_val {
                    local_min_val = demod_data[i];
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
        return SignalType::AnalogVideoNtsc;
    }

    let mut intervals: Vec<usize> = Vec::new();
    for w in sync_positions.windows(2) {
        let gap = w[1] - w[0];
        if gap >= min_gap && gap <= max_gap {
            intervals.push(gap);
        }
    }

    if intervals.is_empty() {
        return SignalType::AnalogVideoNtsc;
    }

    let avg_interval = intervals.iter().sum::<usize>() as f64 / intervals.len() as f64;
    let line_period_us = avg_interval / sample_rate as f64 * 1_000_000.0;
    let is_pal = line_period_us > 63.78;

    if is_pal {
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

    pub fn video_standard(&self) -> crate::types::SignalType {
        if self.pal {
            crate::types::SignalType::AnalogVideoPal
        } else {
            crate::types::SignalType::AnalogVideoNtsc
        }
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
        let required_samples = v_idx + self.samples_per_line * (20 + self.field_lines + 2);
        if demod_data.len() < required_samples {
            return None;
        }

        // Search for first H-sync tip after V-sync + blanking lines.
        let skip_lines = v_idx + (self.samples_per_line * 20);
        let ma_win = ((fs * 0.5e-6) as usize).max(1);
        let sync_window = (fs * 2.0e-6) as usize;
        // Back-porch reference window: wider than the ~4.7 µs sync pulse
        // so the max reaches the blanking level on both sides.
        let porch_radius = (fs * 3.5e-6) as usize;

        // Anchor = robust centre of the first H-sync tip in the ~2 lines
        // after the blanking skip. Brightness-invariant (see
        // `robust_sync_tip_center`); the two-pass extraction below
        // validates every tip against the median period, so the anchor
        // needs no extra smoothing.
        let first_sync_center = robust_sync_tip_center(
            demod_data,
            (skip_lines + self.samples_per_line) as f32,
            self.samples_per_line,
            porch_radius,
            ma_win,
            v_sync_threshold * 0.8,
        )
        .unwrap_or((skip_lines + self.samples_per_line) as f32);
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

                    // Brightness-invariant tip centre near `expected`.
                    match robust_sync_tip_center(
                        demod_data,
                        expected,
                        sync_window,
                        porch_radius,
                        ma_win,
                        v_sync_threshold * 0.8,
                    ) {
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

        // Pass 2: Compute median line period from valid intervals,
        // then use period history buffer for cross-frame stabilisation.
        let mut intervals: Vec<f32> = Vec::new();
        for w in raw_sync_positions.windows(2) {
            if let (Some(a), Some(b)) = (w[0], w[1]) {
                let interval = b - a;
                let nominal = self.samples_per_line as f32;
                if interval > nominal * 0.95 && interval < nominal * 1.05 {
                    intervals.push(interval);
                }
            }
        }

        if !intervals.is_empty() {
            intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let frame_median = intervals[intervals.len() / 2];

            // Push this frame's median into the history buffer (max 8)
            if self.period_history.len() >= 8 {
                self.period_history.remove(0);
            }
            self.period_history.push(frame_median);

            // Use the median of the history buffer as the stabilised
            // line period. After 3-5 frames this is rock-solid.
            let mut sorted_history = self.period_history.clone();
            sorted_history.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let stabilised_period = sorted_history[sorted_history.len() / 2];
            self.line_period = stabilised_period;

            // Debug telemetry only. This is a library — writing to the
            // caller's stdout (the `fpv_viewer` TUI renders there) would
            // corrupt their output, so gate on `debug_dump` like the
            // `[SYNC RESID]` print below, not on `!has_prev`.
            if self.debug_dump && !self.has_prev {
                let _ = writeln!(
                    std::io::stdout(),
                    "TBC: median line period = {:.3} samples ({} intervals, {} history frames)",
                    self.line_period,
                    intervals.len(),
                    self.period_history.len()
                );
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

            // TBC: Resample to exact self.line_width
            let mut tbc_line = vec![0.0f32; self.line_width];
            let mut doc_mask = vec![false; self.line_width];

            let doc_thresh_high = 1.5 * radians_per_volt;
            let doc_thresh_low = -0.8 * radians_per_volt;

            for col in 0..self.line_width {
                // start_frac shifts the entire sweep by the fractional
                // sync_phase; the col-driven term spreads the fractional
                // raw_len_f across the requested line_width samples.
                let idx_float = start_frac + (col as f32 * raw_len_f) / (self.line_width as f32);
                let idx = idx_float as usize;
                let frac = idx_float - idx as f32;

                let val = if idx >= 1 && idx + 2 < raw_line.len() {
                    let v0 = raw_line[idx - 1];
                    let v1 = raw_line[idx];
                    let v2 = raw_line[idx + 1];
                    let v3 = raw_line[idx + 2];

                    let a = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3;
                    let b = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3;
                    let c = -0.5 * v0 + 0.5 * v2;
                    let d = v1;

                    a * frac * frac * frac + b * frac * frac + c * frac + d
                } else if idx + 1 < raw_line.len() {
                    // Fallback to linear at the very edges
                    raw_line[idx] * (1.0 - frac) + raw_line[idx + 1] * frac
                } else if idx < raw_line.len() {
                    raw_line[idx]
                } else {
                    // Past the end of the extracted line — clamp to the
                    // last real sample rather than injecting a 0, which
                    // would paint a black pixel at the right edge of the
                    // active picture (line_width > visible width, so the
                    // tail is displayed). `.last()` rather than
                    // `[len - 1]` so a (guarded-impossible) empty line
                    // can't underflow.
                    raw_line.last().copied().unwrap_or(0.0)
                };

                let is_doc = val > doc_thresh_high || val < doc_thresh_low;
                tbc_line[col] = val;
                doc_mask[col] = is_doc;
            }

            // Blur doc mask slightly (morphological dilate)
            let mut dilated_doc = doc_mask.clone();
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

            // AGC Luma
            let sync_tip_start = (self.line_width as f32 * 0.01) as usize;
            let sync_tip_end = (self.line_width as f32 * 0.04) as usize;
            let bp_start = (self.line_width as f32 * 0.12) as usize;
            let bp_end = (self.line_width as f32 * 0.15) as usize;

            let sync_tip = tbc_line[sync_tip_start..sync_tip_end].iter().sum::<f32>()
                / (sync_tip_end - sync_tip_start) as f32;
            let back_porch =
                tbc_line[bp_start..bp_end].iter().sum::<f32>() / (bp_end - bp_start) as f32;

            if (back_porch - sync_tip).abs() > 0.01 {
                let scale_y = 0.4 * radians_per_volt / (back_porch - sync_tip);
                for v in &mut tbc_line {
                    *v = (*v - back_porch) * scale_y;
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
        let motion_threshold = TEMPORAL_MOTION_THRESHOLD * radians_per_volt;
        current_frame_y
            .par_chunks_mut(line_width)
            .enumerate()
            .take(rows_to_process)
            .for_each(|(row, y_out)| {
                let offset = row * line_width;
                let mut y_line = current_frame_tbc[offset..offset + line_width].to_vec();
                let doc_mask = &current_frame_doc[offset..offset + line_width];

                // 1. DOC replacement from the previous field.
                if has_prev {
                    for col in 0..line_width {
                        if doc_mask[col] {
                            y_line[col] = prev_frame_tbc[offset + col];
                        }
                    }
                }

                // 2. Subcarrier notch (zero-phase forward/backward biquad).
                filtfilt(&mut y_line, nb0, nb1, nb2, na1, na2);

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

                y_out.copy_from_slice(&y_line);
            });

        // ── CTI + Y→RGB pack (SEQUENTIAL) ──────────────────────────
        // Cheap relative to the denoise, and it writes parity-strided
        // output rows, so it stays single-threaded. The complementary
        // parity rows are filled from `field_buf` after this loop.
        // dst_row max is `(field_lines - 1) * 2 + 1` (479 NTSC / 575
        // PAL), within `height - 1`, so the row math needs no guard.
        let h_blank_end = (self.line_width as f32 * ACTIVE_VIDEO_LEFT_CROP_FRAC) as usize;
        for row in 0..rows_to_process {
            let offset = row * self.line_width;
            let y_clean = &current_frame_y[offset..offset + self.line_width];

            // 4. CTI (unsharp mask on luma).
            let mut y_cti = y_clean.to_vec();
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
                let luma = y_norm.clamp(0.0, 1.0) * 255.0;
                let c = luma as u32;
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
        self.field_parity ^= 1;

        self.prev_frame_tbc = current_frame_tbc;

        // Push the just-rendered Y field into the multi-field
        // history buffer for the next call's temporal denoise.
        let meta = FieldMeta {
            timestamp_us: self.field_counter * 16_667, // ≈ 60 fields/sec
            field_parity: self.field_parity,
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

        // Advance to the next field. Anchor the search to the expected
        // field boundary — row-0 sync (`sync_positions[0]`, the most
        // reliable datum) plus one field of active line-periods — rather
        // than walking forward from the last rendered row's sync and
        // taking the *first* density spike. During the vertical-sync
        // interval many one-line windows clear the density threshold;
        // the first of them is the leading equalizing pulse, ~2-3 lines
        // before the true next-field datum. Latching that leading edge
        // makes the next call open mid-V-sync, which is the startup
        // mis-lock that craters sync-quality on the first few fields.
        // Instead, scan a short forward window past active video and
        // pick the *strongest* sync plateau, falling back to a clean
        // one-field advance if nothing clears the threshold (so we slip
        // at most a fraction of a field rather than skipping whole
        // fields).
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
}
