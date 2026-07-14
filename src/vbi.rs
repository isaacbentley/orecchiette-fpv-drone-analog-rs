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
