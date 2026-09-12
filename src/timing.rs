//! Centralized standard television timing descriptors and coordinate conventions.
//!
//! Specifications per ITU-R BT.470 (Tables 1 and 2):
//! - NTSC 525/59.94:
//!   - Nominal line rate: 15_750_000 / 1001 Hz ≈ 15,734.2657 Hz
//!   - Half-lines per field: 525 (262.5 lines / field)
//!   - Subcarrier cycles per line: 455 / 2 = 227.5
//!   - Colour sequence length: 4 fields
//! - PAL 625/50:
//!   - Nominal line rate: 15,625 Hz
//!   - Half-lines per field: 625 (312.5 lines / field)
//!   - Subcarrier cycles per line: 1135 / 4 + 1 / 625 ≈ 283.7516
//!   - Colour sequence length: 8 fields

/// Exact rational line rate for NTSC 525/59.94 in Hz (ITU-R BT.470 Table 1).
pub const NTSC_NOMINAL_LINE_HZ: f64 = 15_750_000.0 / 1001.0;

/// Exact rational line rate for PAL 625/50 in Hz (ITU-R BT.470 Table 1).
pub const PAL_NOMINAL_LINE_HZ: f64 = 15_625.0;

/// Half-lines per field for NTSC (525).
pub const NTSC_HALF_LINES_PER_FIELD: usize = 525;

/// Half-lines per field for PAL (625).
pub const PAL_HALF_LINES_PER_FIELD: usize = 625;

/// Subcarrier cycles per line for NTSC (455 / 2 = 227.5).
pub const NTSC_SUBCARRIER_CYCLES_PER_LINE: f64 = 455.0 / 2.0;

/// Subcarrier cycles per line for PAL (1135 / 4 + 1 / 625).
pub const PAL_SUBCARRIER_CYCLES_PER_LINE: f64 = 1135.0 / 4.0 + 1.0 / 625.0;

/// Colour frame sequence length in fields for NTSC (4 fields).
pub const NTSC_COLOUR_SEQUENCE_FIELDS: usize = 4;

/// Colour frame sequence length in fields for PAL (8 fields).
pub const PAL_COLOUR_SEQUENCE_FIELDS: usize = 8;

/// Total field duration in lines for NTSC (262.5).
pub const NTSC_FIELD_TOTAL_LINES: f64 = 262.5;

/// Total field duration in lines for PAL (312.5).
pub const PAL_FIELD_TOTAL_LINES: f64 = 312.5;

/// Active visible picture lines per field for NTSC (240).
pub const NTSC_ACTIVE_LINES: usize = 240;

/// Active visible picture lines per field for PAL (288).
pub const PAL_ACTIVE_LINES: usize = 288;

/// Standard television descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standard {
    Ntsc,
    Pal,
}

impl Standard {
    #[inline]
    pub const fn from_is_pal(is_pal: bool) -> Self {
        if is_pal { Self::Pal } else { Self::Ntsc }
    }

    #[inline]
    pub const fn is_pal(&self) -> bool {
        matches!(self, Self::Pal)
    }

    #[inline]
    pub const fn nominal_line_hz(&self) -> f64 {
        match self {
            Self::Ntsc => NTSC_NOMINAL_LINE_HZ,
            Self::Pal => PAL_NOMINAL_LINE_HZ,
        }
    }

    #[inline]
    pub const fn half_lines_per_field(&self) -> usize {
        match self {
            Self::Ntsc => NTSC_HALF_LINES_PER_FIELD,
            Self::Pal => PAL_HALF_LINES_PER_FIELD,
        }
    }

    #[inline]
    pub const fn field_total_lines(&self) -> f64 {
        match self {
            Self::Ntsc => NTSC_FIELD_TOTAL_LINES,
            Self::Pal => PAL_FIELD_TOTAL_LINES,
        }
    }

    #[inline]
    pub const fn active_lines(&self) -> usize {
        match self {
            Self::Ntsc => NTSC_ACTIVE_LINES,
            Self::Pal => PAL_ACTIVE_LINES,
        }
    }

    #[inline]
    pub const fn subcarrier_cycles_per_line(&self) -> f64 {
        match self {
            Self::Ntsc => NTSC_SUBCARRIER_CYCLES_PER_LINE,
            Self::Pal => PAL_SUBCARRIER_CYCLES_PER_LINE,
        }
    }

    #[inline]
    pub const fn colour_sequence_fields(&self) -> usize {
        match self {
            Self::Ntsc => NTSC_COLOUR_SEQUENCE_FIELDS,
            Self::Pal => PAL_COLOUR_SEQUENCE_FIELDS,
        }
    }
}

/// Derive nominal line period in samples as an exact `f64`.
#[inline]
pub fn nominal_line_period_samples(sample_rate: u32, is_pal: bool) -> f64 {
    sample_rate as f64 / Standard::from_is_pal(is_pal).nominal_line_hz()
}

/// Derive nominal half-line period in samples as an exact `f64`.
#[inline]
pub fn nominal_half_line_period_samples(sample_rate: u32, is_pal: bool) -> f64 {
    nominal_line_period_samples(sample_rate, is_pal) * 0.5
}

/// Derive nominal field period in samples as an exact `f64`.
#[inline]
pub fn nominal_field_period_samples(sample_rate: u32, is_pal: bool) -> f64 {
    let std = Standard::from_is_pal(is_pal);
    std.field_total_lines() * nominal_line_period_samples(sample_rate, is_pal)
}

/// Calibrated timing conventions and coordinate conversions between sync pulse events.
///
/// H-sync pulse center, broad-pulse leading edge, and active-video start are distinct
/// physical events. Filtering delay and standard back-porch timing must be calibrated
/// explicitly rather than derived naively from quartz frequencies alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibratedCoordinateMap {
    pub is_pal: bool,
    pub sample_rate: u32,
    /// Calibrated delay from broad pulse leading edge to first active video line anchor, in lines.
    pub broad_to_active_lines: f64,
    /// Standard H-sync pulse duration in seconds (nominal ~4.7 µs).
    pub h_sync_width_s: f64,
    /// Standard back-porch duration in seconds (nominal ~5.8 µs).
    pub back_porch_s: f64,
}

impl CalibratedCoordinateMap {
    pub fn new(sample_rate: u32, is_pal: bool) -> Self {
        Self {
            is_pal,
            sample_rate,
            broad_to_active_lines: if is_pal {
                crate::vbi::consts::PAL_BROAD_TO_ACTIVE_LINES
            } else {
                crate::vbi::consts::NTSC_BROAD_TO_ACTIVE_LINES
            },
            h_sync_width_s: crate::vbi::consts::H_SYNC_WIDTH_S,
            back_porch_s: 5.8e-6,
        }
    }

    /// Offset from broad pulse leading edge to active video start in samples.
    #[inline]
    pub fn broad_to_active_start_samples(
        &self,
        line_period_samples: f64,
        is_second_parity: bool,
    ) -> f64 {
        let lines = self.broad_to_active_lines + if is_second_parity { 0.5 } else { 0.0 };
        lines * line_period_samples
    }

    /// Conversion offset from H-sync pulse center to leading edge in samples.
    #[inline]
    pub fn h_sync_center_to_leading_edge(&self) -> f64 {
        -(self.h_sync_width_s * self.sample_rate as f64) * 0.5
    }

    /// Conversion offset from H-sync leading edge to pulse center in samples.
    #[inline]
    pub fn h_sync_leading_edge_to_center(&self) -> f64 {
        (self.h_sync_width_s * self.sample_rate as f64) * 0.5
    }

    /// H-sync pulse width in integer samples.
    #[inline]
    pub fn h_sync_width_samples(&self) -> usize {
        (self.h_sync_width_s * self.sample_rate as f64).round() as usize
    }

    /// Back-porch duration in integer samples.
    #[inline]
    pub fn back_porch_samples(&self) -> usize {
        (self.back_porch_s * self.sample_rate as f64).round() as usize
    }
}
