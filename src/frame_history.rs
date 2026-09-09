//! Multi-field temporal history for cross-frame noise reduction,
//! sparkle suppression, and dropout repair.
//!
//! Holds the last N rendered field outputs (Y values) plus
//! per-field quality metadata. Consumers in `video.rs`:
//!
//! - **Temporal denoise** combines the per-pixel value across the
//!   N most recent fields with motion-adaptive weighting — static
//!   pixels get √N noise reduction, moving pixels fall back to the
//!   current value.
//! - **Sparkle suppression** uses a 3-field median (handled inside
//!   the denoise step) to kill impulse noise (FM "click" bursts at
//!   low CNR) without smearing motion edges.
//! - **Dropout repair** detects per-field sync degradation via
//!   [`FieldMeta::sync_quality`] and blends the current field's
//!   rendered output toward the average of the two prior fields
//!   when the sync extraction reported abnormally high outlier-
//!   rejection rates.
//!
//! The buffer is a fixed-capacity FIFO. On `push`, if the buffer
//! is full the oldest field is evicted. `current_field()` returns
//! the most recently pushed field; `prev_field(n)` indexes back N
//! steps (n=1 is the field before the current one).
//!
//! ## Memory cost
//!
//! Each field is `field_lines × line_width × sizeof(f32)` bytes. For
//! NTSC at 858×240 that's 824 KB. The default capacity of 5 fields
//! is ~4.1 MB per channel — well within budget even on embedded SDR
//! hosts. Capacity is configurable so callers can dial in their own
//! latency-vs-quality trade-off (low-latency racing FPV might want
//! N=2; recon / surveillance can push to N=8 for maximum SNR
//! recovery).

use std::collections::VecDeque;

/// Per-field metadata captured during rendering. Used downstream
/// by the temporal-denoise and dropout-repair stages to weigh
/// each historical field's contribution.
#[derive(Debug, Clone, Copy)]
pub struct FieldMeta {
    /// Capture timestamp in microseconds since reconstructor start.
    /// Used only for diagnostics today; reserved for future
    /// motion-compensated comb work where temporal proximity
    /// matters.
    pub timestamp_us: u64,
    /// Field parity (0 or 1) — even or odd output rows. NTSC's
    /// chroma phase inverts between same-parity fields two frames
    /// apart, so a future colour-recovery pipeline would key the
    /// 3D comb decision off this field.
    pub field_parity: u8,
    /// Sync-extraction confidence: 1.0 means every sync tip in the
    /// field passed the MAD-outlier check; 0.0 means every tip was
    /// rejected (catastrophic dropout). The dropout-repair stage
    /// fires when this drops below `DROPOUT_THRESHOLD`.
    pub sync_quality: f32,
    /// Mean amplitude of the Y signal in the field (post-notch).
    /// A sudden drop relative to recent history indicates the
    /// transmitter went out of range or the antenna got blocked —
    /// useful as a secondary dropout signal independent of sync.
    pub mean_amplitude: f32,
}

impl Default for FieldMeta {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            field_parity: 0,
            sync_quality: 1.0,
            mean_amplitude: 0.0,
        }
    }
}

/// Ring buffer of recent rendered field Y data + metadata.
///
/// Both `y_fields` and `meta` are kept the same length and aligned —
/// `y_fields[i]` corresponds to `meta[i]` for all `i`.
pub struct FrameHistory {
    /// Per-field Y values. Each entry is `field_lines × line_width`
    /// `f32` samples, laid out row-major
    /// (`y[row * line_width + col]`).
    y_fields: VecDeque<Vec<f32>>,
    /// Parallel queue of per-field metadata.
    meta: VecDeque<FieldMeta>,
    /// Maximum number of fields retained. Older fields are evicted
    /// FIFO when `push` would exceed this.
    capacity: usize,
    /// Pre-known per-field size (`field_lines × line_width`). Used
    /// to validate `push` payloads — mismatched sizes would corrupt
    /// the indexed accessors downstream.
    field_size: usize,
}

impl FrameHistory {
    /// Construct a new buffer that holds up to `capacity` fields,
    /// each of size `field_size` `f32`s. Allocates the backing
    /// queues lazily — empty until the first `push`.
    pub fn new(capacity: usize, field_size: usize) -> Self {
        Self {
            y_fields: VecDeque::with_capacity(capacity),
            meta: VecDeque::with_capacity(capacity),
            capacity,
            field_size,
        }
    }

    /// Add a new field's Y data and metadata. If the buffer is at
    /// capacity, the oldest field is evicted first. Panics in debug
    /// builds if `y_field.len() != field_size` — that would
    /// silently corrupt subsequent indexed accesses.
    pub fn push(&mut self, y_field: Vec<f32>, meta: FieldMeta) {
        debug_assert_eq!(
            y_field.len(),
            self.field_size,
            "FrameHistory::push: field size mismatch ({} vs expected {})",
            y_field.len(),
            self.field_size,
        );
        if self.y_fields.len() >= self.capacity {
            self.y_fields.pop_front();
            self.meta.pop_front();
        }
        self.y_fields.push_back(y_field);
        self.meta.push_back(meta);
    }

    /// Number of fields currently held. Ranges from 0 (just
    /// constructed) up to `capacity()`.
    pub fn len(&self) -> usize {
        self.y_fields.len()
    }

    /// True when no fields have been pushed yet. The temporal
    /// denoise and dropout-repair stages short-circuit on this.
    pub fn is_empty(&self) -> bool {
        self.y_fields.is_empty()
    }

    /// Configured maximum capacity. Use this in callers that want
    /// to size their own per-pixel scratch arrays.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The most recently pushed field's Y data, or `None` if the
    /// buffer is empty. `current_field()` and the various
    /// `prev_field(n)` accessors share the same indexing convention:
    /// `current_field() == prev_field(0)`.
    pub fn current_field(&self) -> Option<&[f32]> {
        self.y_fields.back().map(|v| v.as_slice())
    }

    /// The Nth-most-recent field. `prev_field(0)` is identical to
    /// `current_field()`; `prev_field(1)` is the field immediately
    /// before the current one; etc. Returns `None` if `n` is past
    /// the end of the buffer (i.e. fewer than `n + 1` pushes have
    /// happened).
    pub fn prev_field(&self, n: usize) -> Option<&[f32]> {
        if n >= self.y_fields.len() {
            return None;
        }
        let idx = self.y_fields.len() - 1 - n;
        Some(self.y_fields[idx].as_slice())
    }

    /// Per-field metadata accessors, parallel to `current_field` /
    /// `prev_field`. Used by the dropout-repair stage to consult
    /// `sync_quality` and `mean_amplitude`.
    pub fn current_meta(&self) -> Option<&FieldMeta> {
        self.meta.back()
    }

    pub fn prev_meta(&self, n: usize) -> Option<&FieldMeta> {
        if n >= self.meta.len() {
            return None;
        }
        let idx = self.meta.len() - 1 - n;
        Some(&self.meta[idx])
    }

    /// Iterator over all retained fields, oldest first. Used by the
    /// temporal-denoise step which sweeps the full window per pixel.
    pub fn iter_fields(&self) -> impl Iterator<Item = &[f32]> {
        self.y_fields.iter().map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_for(stamp: u64) -> FieldMeta {
        FieldMeta {
            timestamp_us: stamp,
            field_parity: (stamp & 1) as u8,
            sync_quality: 1.0,
            mean_amplitude: 0.5,
        }
    }

    #[test]
    fn push_and_index_roundtrip() {
        let mut h = FrameHistory::new(3, 4);
        h.push(vec![1.0, 2.0, 3.0, 4.0], meta_for(10));
        h.push(vec![5.0, 6.0, 7.0, 8.0], meta_for(20));
        assert_eq!(h.len(), 2);
        assert_eq!(h.current_field().unwrap(), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(h.prev_field(1).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(h.prev_field(2).is_none());
        assert_eq!(h.current_meta().unwrap().timestamp_us, 20);
        assert_eq!(h.prev_meta(1).unwrap().timestamp_us, 10);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut h = FrameHistory::new(2, 1);
        h.push(vec![1.0], meta_for(10));
        h.push(vec![2.0], meta_for(20));
        h.push(vec![3.0], meta_for(30)); // evicts the 10-stamped entry
        assert_eq!(h.len(), 2);
        assert_eq!(h.current_field().unwrap(), &[3.0]);
        assert_eq!(h.prev_field(1).unwrap(), &[2.0]);
        assert_eq!(h.current_meta().unwrap().timestamp_us, 30);
        assert_eq!(h.prev_meta(1).unwrap().timestamp_us, 20);
    }

    #[test]
    fn iter_fields_yields_oldest_first() {
        let mut h = FrameHistory::new(4, 1);
        h.push(vec![1.0], meta_for(1));
        h.push(vec![2.0], meta_for(2));
        h.push(vec![3.0], meta_for(3));
        let collected: Vec<f32> = h.iter_fields().map(|f| f[0]).collect();
        assert_eq!(collected, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn empty_history_short_circuits_cleanly() {
        let h = FrameHistory::new(3, 5);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert!(h.current_field().is_none());
        assert!(h.prev_field(0).is_none());
        assert!(h.current_meta().is_none());
    }
}
