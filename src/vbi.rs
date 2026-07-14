//! Vertical Blanking Interval (VBI) timing conventions and parsing.
//!
//! [`consts`] is the single source of truth for NTSC/PAL vertical-sync
//! pulse timing, shared by [`crate::synthetic`] (which generates
//! standards-shaped fields for testing) and this module's own pulse
//! parser (added alongside the reconstructor integration). Keeping both
//! sides of "what a field's vertical sync looks like" in one place is
//! deliberate — a generator and parser that each hand-derive the same
//! numbers independently drift apart silently.
//!
//! ## A note on "standards-correct"
//!
//! The pulse *shapes and counts* below (pre-equalizing / serrated-broad /
//! post-equalizing, 6/6/6 for NTSC and 5/5/5 for PAL) follow the
//! broadcast standards. The *exact line at which active video begins*
//! is a genuinely fuzzy, convention-dependent number in real broadcast
//! practice (sources cite anywhere from line 20 to line 22 for NTSC).
//! Rather than pick one and risk a half-line mismatch against whatever
//! the parser later derives independently, [`consts::BASE_ACTIVE_START_LINES`]
//! is this crate's *own* self-consistent convention: the generator lays
//! fields out against it, and the parser's active-video datum is
//! calibrated against the generator (see the geometry tests in
//! `video.rs`), not re-derived from a spec table. Internal consistency
//! is what both the reconstructor and the detector's VBI-confirm stage
//! actually depend on.

/// Field parity — which of the two interlaced fields a slice belongs
/// to. NTSC/PAL fields differ by exactly half a line's worth of phase
/// in their vertical-sync timing relative to the horizontal grid: field
/// one's vertical-sync sequence starts in phase with the H grid,
/// field two's starts half a line later. That half-line offset is what
/// makes the two fields interlace into a full frame instead of
/// overwriting each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldParity {
    /// Vertical sync starts in phase with the horizontal grid (whole-line offset).
    First,
    /// Vertical sync starts half a line out of phase with the horizontal grid.
    Second,
}

pub mod consts {
    //! Timing constants, in seconds and line-counts, for NTSC and PAL
    //! vertical sync. See the module-level doc for how these were
    //! chosen and why they're shared between the generator and parser.

    /// Horizontal line rate (Hz). Matches [`crate::video::FrameReconstructor::new`]'s
    /// `line_rate` exactly — the generator and reconstructor must agree
    /// on this or their line-period assumptions diverge.
    pub const NTSC_LINE_HZ: f64 = 15734.0;
    pub const PAL_LINE_HZ: f64 = 15625.0;

    /// Total field duration, in lines. Standard values (262.5 / 312.5)
    /// — always a half-integer, which is the whole mechanism behind
    /// interlace: two fields of a half-integer line count tile into a
    /// whole-integer frame.
    pub const NTSC_FIELD_TOTAL_LINES: f64 = 262.5;
    pub const PAL_FIELD_TOTAL_LINES: f64 = 312.5;

    /// Active (visible) picture lines per field. Matches
    /// [`crate::video::FrameReconstructor::new`]'s `field_lines`.
    pub const NTSC_ACTIVE_LINES: usize = 240;
    pub const PAL_ACTIVE_LINES: usize = 288;

    /// Pulse counts in each of the three vertical-sync groups
    /// (pre-equalizing, serrated-broad, post-equalizing). Each pulse
    /// occupies one half-line slot, so a group of N pulses spans N/2
    /// lines.
    pub const NTSC_EQ_PULSES: usize = 6;
    pub const NTSC_BROAD_PULSES: usize = 6;
    pub const NTSC_POSTEQ_PULSES: usize = 6;
    pub const PAL_EQ_PULSES: usize = 5;
    pub const PAL_BROAD_PULSES: usize = 5;
    pub const PAL_POSTEQ_PULSES: usize = 5;

    /// Equalizing-pulse width (seconds) — brief, at twice line rate.
    pub const NTSC_EQ_WIDTH_S: f64 = 2.3e-6;
    pub const PAL_EQ_WIDTH_S: f64 = 2.35e-6;

    /// Serrated broad-pulse "low" duration (seconds) — the pulse is at
    /// sync-tip level for this long, then briefly returns to blanking
    /// (the serration) for the rest of its half-line slot, which is
    /// what keeps the horizontal oscillator in lock during vertical
    /// sync.
    pub const NTSC_BROAD_LOW_S: f64 = 27.1e-6;
    pub const PAL_BROAD_LOW_S: f64 = 27.3e-6;

    /// Ordinary horizontal-sync pulse width (seconds), used both for
    /// the plain-blanking lines between the vertical-sync groups and
    /// active video, and for active-video lines themselves.
    pub const H_SYNC_WIDTH_S: f64 = 4.7e-6;

    /// This crate's own convention (see module doc) for where First-
    /// parity active video starts, in lines measured from the start of
    /// the field's vertical-sync sequence (i.e. from the first
    /// pre-equalizing pulse). Second parity starts exactly half a line
    /// later — that's the interlace offset. Chosen so the gap between
    /// the vertical-sync group and active video is a whole number of
    /// lines for First parity (12 for NTSC, 16 for PAL).
    pub const NTSC_BASE_ACTIVE_START_LINES: f64 = 21.0;
    pub const PAL_BASE_ACTIVE_START_LINES: f64 = 23.5;

    /// Lines from the *pre-equalizing* group start to the *broad*
    /// group start (i.e. the pre-eq group's own duration).
    pub const NTSC_EQ_GROUP_LINES: f64 = NTSC_EQ_PULSES as f64 / 2.0;
    pub const PAL_EQ_GROUP_LINES: f64 = PAL_EQ_PULSES as f64 / 2.0;

    /// Lines from the *broad* group's first pulse to active video —
    /// i.e. [`NTSC_BASE_ACTIVE_START_LINES`] minus the pre-eq group's
    /// own span. This is the number [`crate::vbi::find_vertical_sync`]
    /// (added with the reconstructor integration) actually uses as its
    /// datum, and what the Phase-1 geometry tests calibrate against.
    pub const NTSC_BROAD_TO_ACTIVE_LINES: f64 = NTSC_BASE_ACTIVE_START_LINES - NTSC_EQ_GROUP_LINES;
    pub const PAL_BROAD_TO_ACTIVE_LINES: f64 = PAL_BASE_ACTIVE_START_LINES - PAL_EQ_GROUP_LINES;
}

use crate::levels::{SyncLevels, median, moving_average};

/// Which vertical-sync pulse family a below-threshold run's width
/// matches. Classification is by width alone — a full field's pulse
/// train has each family's width well separated from the others (2.3
/// vs 4.7 vs ~27 µs, all >2× apart), so a single FM click can't flip a
/// pulse from one family to another the way an edge-triggered decision
/// could.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulseKind {
    /// Pre/post-equalizing pulse (~2.3–2.35 µs).
    Equalizing,
    /// Ordinary H-sync pulse, in either a plain-blanking or an
    /// active-video line (~4.7 µs).
    Horizontal,
    /// Serrated vertical-sync ("broad") pulse (~27.1–27.3 µs low).
    Broad,
}

/// A single below-threshold run, classified by width.
#[derive(Debug, Clone, Copy)]
pub struct SyncPulse {
    pub start: usize,
    pub end: usize,
    /// Midpoint sample index, for callers that want it. The spacing
    /// and grid-fit math in this module deliberately uses `start`
    /// instead (see [`count_adjacent_run`]'s doc): different pulse
    /// families have different widths, so comparing centers across a
    /// family boundary (e.g. an equalizing pulse next to a broad
    /// pulse) introduces a spurious offset of roughly half that width
    /// difference.
    pub center: f32,
    pub width_us: f32,
    pub kind: PulseKind,
}

const EQ_WIDTH_RANGE_US: (f32, f32) = (1.5, 3.5);
const H_WIDTH_RANGE_US: (f32, f32) = (3.5, 8.0);
const BROAD_WIDTH_RANGE_US: (f32, f32) = (20.0, 32.0);

/// Minimum number of consecutive half-line-spaced broad pulses
/// required before a run is trusted as the vertical-sync group. The
/// standards specify 6 (NTSC) / 5 (PAL); requiring only 4 tolerates
/// one or two corrupted pulses without losing lock entirely.
const MIN_BROAD_RUN: usize = 4;

/// Tolerance (as a fraction of the nominal half-line period) for
/// accepting a broad-pulse-to-broad-pulse gap as "half-line spaced".
const BROAD_SPACING_TOLERANCE: f32 = 0.15;

/// Looser tolerance used only for the informational pre/post
/// equalizing-pulse run count — these aren't load-bearing for parity
/// or the active-video datum, so a wider net is fine.
const EQ_RUN_TOLERANCE: f32 = 0.25;

/// How far (in nominal line periods) to search for horizontal-sync
/// pulses to fit a local line-period/phase grid against.
const H_GRID_SEARCH_LINES: f32 = 20.0;

/// Slice below `levels.sync_tip + 0.5 · levels.swing()` (the midpoint
/// between sync-tip and blanking) into runs, classify each run's width
/// against [`PulseKind`], and discard anything that doesn't match one
/// of the three known pulse families (clicks, dropouts, active-video
/// noise). Data is smoothed with the same ~0.5 µs moving average
/// [`crate::levels::estimate_fm_deviation`] uses, for the same reason
/// (suppress FM click noise without blurring pulse edges at this
/// timescale).
pub fn extract_pulses(demod: &[f32], sample_rate: u32, levels: &SyncLevels) -> Vec<SyncPulse> {
    if sample_rate == 0 || demod.is_empty() {
        return Vec::new();
    }
    let fs = sample_rate as f32;
    let ma_win = ((fs * 0.5e-6) as usize).max(1);
    let smoothed = moving_average(demod, ma_win);
    let threshold = levels.sync_tip + 0.5 * levels.swing();

    let mut pulses = Vec::new();
    let n = smoothed.len();
    let mut i = 0usize;
    while i < n {
        if smoothed[i] < threshold {
            let start = i;
            while i < n && smoothed[i] < threshold {
                i += 1;
            }
            let end = i;
            let width_us = (end - start) as f32 / fs * 1e6;
            let kind = if width_us >= EQ_WIDTH_RANGE_US.0 && width_us < EQ_WIDTH_RANGE_US.1 {
                Some(PulseKind::Equalizing)
            } else if width_us >= H_WIDTH_RANGE_US.0 && width_us <= H_WIDTH_RANGE_US.1 {
                Some(PulseKind::Horizontal)
            } else if width_us >= BROAD_WIDTH_RANGE_US.0 && width_us <= BROAD_WIDTH_RANGE_US.1 {
                Some(PulseKind::Broad)
            } else {
                None
            };
            if let Some(kind) = kind {
                let center = (start + end) as f32 / 2.0;
                pulses.push(SyncPulse {
                    start,
                    end,
                    center,
                    width_us,
                    kind,
                });
            }
        } else {
            i += 1;
        }
    }
    pulses
}

/// Result of a successful [`find_vertical_sync`] parse.
#[derive(Debug, Clone, Copy)]
pub struct VerticalSyncInfo {
    /// Leading edge (sample index, as `f32` for the sub-sample line
    /// arithmetic below) of the vertical-sync group's first broad
    /// pulse — the datum [`field_active_start`] is measured from.
    pub broad_start: f32,
    /// Sample index at which this field's active video begins,
    /// derived from `broad_start` plus the standard's calibrated
    /// broad-to-active line count (see [`consts`]), adjusted by half a
    /// line when [`parity`] is conclusively `Second`.
    pub field_active_start: f32,
    /// `None` when the local horizontal grid couldn't be fit (too few
    /// nearby H-sync pulses) or the broad group's phase against that
    /// grid was ambiguous — callers should keep whatever parity
    /// prediction they already had rather than treat this as an error.
    pub parity: Option<FieldParity>,
    pub n_broad: usize,
    pub n_eq_pre: usize,
    pub n_eq_post: usize,
    /// Locally-fit line period (samples), when a grid fit succeeded.
    pub line_period: Option<f32>,
}

/// Fit `(period, intercept)` to the horizontal-sync grid from
/// `Horizontal`-kind pulses within `H_GRID_SEARCH_LINES` nominal line
/// periods of `near`, using the same median-period / median-intercept
/// technique as the reconstructor's own cross-field line-period
/// stabilisation. Returns `None` if fewer than 4 usable pulses survive
/// (too little data for a trustworthy fit).
fn fit_h_grid(pulses: &[SyncPulse], nominal_period: f32, near: f32) -> Option<f32> {
    let window = nominal_period * H_GRID_SEARCH_LINES;
    let mut candidates: Vec<f32> = pulses
        .iter()
        .filter(|p| p.kind == PulseKind::Horizontal && (p.start as f32 - near).abs() <= window)
        .map(|p| p.start as f32)
        .collect();
    if candidates.len() < 4 {
        return None;
    }
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut intervals: Vec<f32> = candidates.windows(2).map(|w| w[1] - w[0]).collect();
    intervals.retain(|&d| (d - nominal_period).abs() < nominal_period * 0.1);
    if intervals.len() < 3 {
        return None;
    }
    Some(median(&mut intervals))
}

/// Count a contiguous run of `kind`-classified pulses spaced
/// `half_period` apart, starting from `anchor_start` and walking in
/// `direction` (`-1` = backward, `+1` = forward). Compares pulse
/// *start* (leading edge), not `center` — every VBI pulse family leads
/// on the same half-line grid regardless of its own width, but their
/// widths differ enough (2.3 µs eq vs 27 µs broad) that comparing
/// centers across a kind boundary introduces a spurious offset of
/// roughly half that width difference, large enough to miss the very
/// next pulse. Purely informational (feeds `n_eq_pre`/`n_eq_post`), so
/// uses a looser tolerance than the broad-group detection itself.
fn count_adjacent_run(
    pulses: &[SyncPulse],
    anchor_start: f32,
    half_period: f32,
    direction: f32,
    kind: PulseKind,
) -> usize {
    let mut count = 0usize;
    let mut expected = anchor_start;
    loop {
        expected += direction * half_period;
        let found = pulses
            .iter()
            .filter(|p| {
                p.kind == kind && (p.start as f32 - expected).abs() < half_period * EQ_RUN_TOLERANCE
            })
            .min_by(|a, b| {
                (a.start as f32 - expected)
                    .abs()
                    .partial_cmp(&(b.start as f32 - expected).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        match found {
            Some(p) => {
                count += 1;
                expected = p.start as f32;
            }
            None => break,
        }
    }
    count
}

/// Parse a demod slice's vertical-sync structure: locate the serrated
/// broad-pulse group, determine field parity, and derive where active
/// video starts.
///
/// Parity is determined by direct hypothesis test, not by fitting a
/// phase against an independently-indexed local grid: this field's own
/// plain-blanking H-sync pulses start immediately after the
/// vertical-sync group at a fixed cadence *regardless of parity* (the
/// generator in [`crate::synthetic`] — and real broadcast video —
/// only encodes the half-line difference in *where active video
/// starts* relative to that cadence, not in the cadence's phase
/// against `broad_start`). So rather than fit a grid and read off a
/// phase, this computes both standards' predicted active-video start
/// (`broad_to_active` lines, or that plus half a line) and checks
/// which one an actual H-sync pulse confirms.
///
/// Returns `None` when no run of at least [`MIN_BROAD_RUN`] half-line-
/// spaced broad pulses is found at all — callers should fall back to
/// whatever less-structural vsync detection they already have (real
/// VBI is dirtier than the spec on cheap FPV cameras; only the broad
/// group is treated as load-bearing here). A `Some` result's `parity`
/// field can still independently be `None` (see its doc).
pub fn find_vertical_sync(
    demod: &[f32],
    sample_rate: u32,
    levels: &SyncLevels,
    is_pal: bool,
) -> Option<VerticalSyncInfo> {
    if sample_rate == 0 {
        return None;
    }
    let pulses = extract_pulses(demod, sample_rate, levels);
    let nominal_line_hz = if is_pal {
        consts::PAL_LINE_HZ
    } else {
        consts::NTSC_LINE_HZ
    };
    let nominal_period = sample_rate as f32 / nominal_line_hz as f32;
    let half_period = nominal_period * 0.5;

    let broads: Vec<&SyncPulse> = pulses
        .iter()
        .filter(|p| p.kind == PulseKind::Broad)
        .collect();
    let mut group: Option<(usize, usize)> = None; // (start_idx, end_idx_inclusive) into `broads`
    let mut i = 0usize;
    while i < broads.len() {
        let mut j = i;
        while j + 1 < broads.len() {
            let gap = broads[j + 1].start as f32 - broads[j].start as f32;
            if (gap - half_period).abs() <= half_period * BROAD_SPACING_TOLERANCE {
                j += 1;
            } else {
                break;
            }
        }
        if j - i + 1 >= MIN_BROAD_RUN {
            group = Some((i, j));
            break;
        }
        i = j + 1;
    }
    let (gi, gj) = group?;
    let n_broad = gj - gi + 1;
    let broad_start = broads[gi].start as f32;
    let last_start = broads[gj].start as f32;

    let n_eq_pre = count_adjacent_run(
        &pulses,
        broad_start,
        half_period,
        -1.0,
        PulseKind::Equalizing,
    );
    let n_eq_post =
        count_adjacent_run(&pulses, last_start, half_period, 1.0, PulseKind::Equalizing);

    // Refine the line period from this field's own plain H-sync pulses
    // (prefer after the group — guaranteed present in any single-
    // field-plus slice — falling back to before it). Only the period
    // is used from this fit; see the doc comment on why an intercept-
    // based phase can't carry parity here.
    let fitted_period = fit_h_grid(&pulses, nominal_period, last_start + nominal_period * 3.0)
        .or_else(|| fit_h_grid(&pulses, nominal_period, broad_start - nominal_period * 3.0));
    let period_for_datum = fitted_period.unwrap_or(nominal_period);

    let broad_to_active = if is_pal {
        consts::PAL_BROAD_TO_ACTIVE_LINES
    } else {
        consts::NTSC_BROAD_TO_ACTIVE_LINES
    } as f32;
    let candidate_first = broad_start + broad_to_active * period_for_datum;
    let candidate_second = broad_start + (broad_to_active + 0.5) * period_for_datum;
    let tol = period_for_datum * 0.2;
    let has_first = pulses
        .iter()
        .any(|p| p.kind == PulseKind::Horizontal && (p.start as f32 - candidate_first).abs() < tol);
    let has_second = pulses.iter().any(|p| {
        p.kind == PulseKind::Horizontal && (p.start as f32 - candidate_second).abs() < tol
    });
    let parity = match (has_first, has_second) {
        (true, false) => Some(FieldParity::First),
        (false, true) => Some(FieldParity::Second),
        _ => None,
    };

    let field_active_start = match parity {
        Some(FieldParity::Second) => candidate_second,
        _ => candidate_first,
    };

    Some(VerticalSyncInfo {
        broad_start,
        field_active_start,
        parity,
        n_broad,
        n_eq_pre,
        n_eq_post,
        line_period: fitted_period,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{SyntheticVideoConfig, TestPattern, generate_fields};

    fn synth_config(is_pal: bool, start_field: FieldParity) -> SyntheticVideoConfig {
        SyntheticVideoConfig {
            sample_rate: 15_360_000,
            is_pal,
            deviation_hz: 5e6,
            pattern: TestPattern::Bars,
            start_field,
            noise_sigma: 0.0,
            dc_offset: 0.0,
        }
    }

    fn levels_for(cfg: &SyntheticVideoConfig) -> SyncLevels {
        let radians_per_volt =
            2.0 * std::f32::consts::PI * cfg.deviation_hz / cfg.sample_rate as f32;
        let rad_per_ire = radians_per_volt * 0.01;
        SyncLevels {
            sync_tip: -40.0 * rad_per_ire,
            blanking: 0.0,
        }
    }

    #[test]
    fn extract_pulses_classifies_ntsc_field_correctly() {
        let cfg = synth_config(false, FieldParity::First);
        let data = generate_fields(&cfg, 1);
        let levels = levels_for(&cfg);
        let pulses = extract_pulses(&data, cfg.sample_rate, &levels);
        let broad = pulses.iter().filter(|p| p.kind == PulseKind::Broad).count();
        let eq = pulses
            .iter()
            .filter(|p| p.kind == PulseKind::Equalizing)
            .count();
        assert_eq!(broad, 6);
        assert_eq!(eq, 12); // 6 pre + 6 post
    }

    #[test]
    fn find_vertical_sync_locates_ntsc_group_and_first_parity() {
        let cfg = synth_config(false, FieldParity::First);
        let data = generate_fields(&cfg, 2); // trailing field gives the parser H pulses to fit against
        let levels = levels_for(&cfg);
        let info =
            find_vertical_sync(&data, cfg.sample_rate, &levels, false).expect("expected a parse");
        assert_eq!(info.n_broad, 6);
        assert_eq!(info.n_eq_pre, 6);
        assert_eq!(info.n_eq_post, 6);
        assert_eq!(info.parity, Some(FieldParity::First));
        // broad_start should sit near the very beginning of the field
        // (after the 6-pulse pre-eq group, ~3 lines in).
        let nominal_period = cfg.sample_rate as f32 / consts::NTSC_LINE_HZ as f32;
        assert!((info.broad_start - 3.0 * nominal_period).abs() < nominal_period * 0.5);
    }

    #[test]
    fn find_vertical_sync_detects_second_parity() {
        let cfg = synth_config(false, FieldParity::Second);
        let data = generate_fields(&cfg, 2);
        let levels = levels_for(&cfg);
        let info =
            find_vertical_sync(&data, cfg.sample_rate, &levels, false).expect("expected a parse");
        assert_eq!(info.parity, Some(FieldParity::Second));
    }

    #[test]
    fn find_vertical_sync_works_for_pal_both_parities() {
        for start in [FieldParity::First, FieldParity::Second] {
            let cfg = synth_config(true, start);
            let data = generate_fields(&cfg, 2);
            let levels = levels_for(&cfg);
            let info = find_vertical_sync(&data, cfg.sample_rate, &levels, true)
                .expect("expected a parse");
            assert_eq!(info.n_broad, 5);
            assert_eq!(info.parity, Some(start));
        }
    }

    #[test]
    fn find_vertical_sync_returns_none_on_pure_noise() {
        let sr = 15_360_000u32;
        let n = 400_000;
        let mut seed = 999u64;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let levels = SyncLevels {
            sync_tip: -0.5,
            blanking: 0.0,
        };
        assert!(find_vertical_sync(&data, sr, &levels, false).is_none());
    }

    #[test]
    fn find_vertical_sync_recovers_parity_under_realistic_noise() {
        let mut cfg = synth_config(false, FieldParity::Second);
        let radians_per_volt =
            2.0 * std::f32::consts::PI * cfg.deviation_hz / cfg.sample_rate as f32;
        cfg.noise_sigma = 0.1 * crate::levels::SYNC_TO_BLANK_FRACTION * radians_per_volt;
        let data = generate_fields(&cfg, 2);
        let levels = levels_for(&cfg);
        let info = find_vertical_sync(&data, cfg.sample_rate, &levels, false)
            .expect("expected a parse under noise");
        assert_eq!(info.parity, Some(FieldParity::Second));
    }
}
