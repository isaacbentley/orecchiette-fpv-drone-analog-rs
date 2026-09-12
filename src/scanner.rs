use crate::bands::{FpvChannel, get_all_channels};
use crate::detector::FpvDetector;
use crate::types::DetectionResult;
use num_complex::Complex;

/// Channel-table-driven scanner: tune to each channel in turn and run
/// the detector on one capture per channel.
///
/// Note that the 5.8 GHz band plans overlap heavily (A/B/E/F/R share
/// the 5645–5945 MHz range, and L/D interleave below it), so one
/// physical transmitter typically produces detections on several
/// adjacent channel entries; results are tagged per-channel and NOT
/// deduplicated across channels — callers wanting one hit per emitter
/// should merge results within ~one video bandwidth (25 MHz) of each
/// other, keeping the strongest.
pub struct FpvScanner<D: FpvDetector> {
    detector: D,
    channels: Vec<FpvChannel>,
}

impl<D: FpvDetector> FpvScanner<D> {
    pub fn new(detector: D) -> Self {
        Self {
            detector,
            channels: get_all_channels(),
        }
    }

    pub fn with_channels(detector: D, channels: Vec<FpvChannel>) -> Self {
        Self { detector, channels }
    }

    /// Scan all channels. The callback should tune the hardware and return IQ data.
    pub fn scan<F>(&self, mut get_iq: F, sample_rate: u32) -> Vec<DetectionResult>
    where
        F: FnMut(u64) -> Vec<Complex<f32>>,
    {
        let mut all_results = Vec::new();

        for channel in &self.channels {
            let iq_data = get_iq(channel.frequency_hz);
            if iq_data.is_empty() {
                continue;
            }

            let results = self
                .detector
                .detect_from_iq(&iq_data, channel.frequency_hz, sample_rate);
            for mut res in results {
                res.channel = Some(channel.clone());
                all_results.push(res);
            }
        }

        all_results
    }
}

use crate::bands::{
    CHANNEL_SNAP_TOLERANCE_MHZ, get_candidate_fpv_channels, snap_to_nearest_fpv_channel,
};
use std::collections::HashSet;
use std::time::Duration;
/// How many capture packets per hop get fed to the detector before the
/// rest of the dwell is drained.
///
/// `detect_from_iq_integrated` accumulates magnitude spectra per
/// frequency, and a single 65,536-sample packet is below the length at
/// which the sweep detects anything: with one packet per hop the first
/// sweep finds nothing and detection lands only on the second visit,
/// once the integrator holds two batches. Consecutive packets from one
/// hop show where it turns over (`--` = no detection, else confidence):
///
/// ```text
///   25.00 MSPS  packets 1..8:  --  0.80 0.80 0.80 0.80 0.80  --  0.80
///   61.44 MSPS  packets 1..8:  --  0.80 0.80 0.80 0.80 0.80 0.80 0.80
/// ```
///
/// Two gives first-visit detection, and on a *clean* signal beyond two
/// measures no better: confidence plateaus at 0.80 and never promotes.
/// The table above is exactly that case, which is why it reads as a
/// plateau.
///
/// It does not hold under noise. Resetting the integrator between trials
/// and varying capture position, synthetic centred NTSC gives:
///
/// ```text
///                                  within 2 packets   within 8
///   25 MSPS,    component σ 1.0          7/8             8/8
///   25 MSPS,    component σ 1.5          2/8             8/8
///   61.44 MSPS, component σ 1.0          5/8             8/8
///   61.44 MSPS, component σ 1.5          4/8             4/8
/// ```
///
/// So two packets is the right *fast pass* and the wrong *verdict*: at
/// σ 1.5 it finds a quarter of what eight finds. Hops that show energy
/// So two packets is the right fast pass at a wide capture rate and the
/// wrong verdict at a narrow one, where a packet is four times the
/// signal for a third of the cost — see [`detect_packets_per_hop`].
///
/// The cost is CPU, not time on air: two packets is ~5.2 ms of signal at
/// 25 MSPS but ~18 ms of detection work, so a hop becomes CPU-bound
/// rather than dwell-bound.
pub const DETECT_PACKETS_PER_HOP: usize = 2;
/// Packets fed to the detector on a hop at a *narrow* capture rate.
///
/// A packet is a fixed 65,536 samples, so what it is worth depends
/// entirely on the rate: 1.07 ms of signal at 61.44 MSPS against 4.27 ms
/// at 15.36, and it costs 6.41 ms of detector against 2.49. Narrow
/// captures are the case where more packets are both more useful and
/// cheaper, and six of them is 14.9 ms of CPU inside a 25 ms dwell —
/// where six at 61.44 MSPS would be 38.5 ms and miss the hop entirely.
///
/// Six because integration needs the room. Measured against a real A1
/// transmitter attenuated toward its cliff, a 15.36 MSPS capture that
/// never confirms in two packets first detects on the fourth; the
/// evidence is not present earlier, it is *built* by accumulating
/// spectra. Two packets cannot see it however cleverly they are gated.
pub const DETECT_PACKETS_NARROW: usize = 6;

/// Above this rate a packet is too little signal and too much detector
/// to spend [`DETECT_PACKETS_NARROW`] of them on.
pub const NARROW_CAPTURE_MAX_RATE_HZ: f64 = 30_720_000.0;

/// What one sweep may spend per hop.
///
/// Bundled because they are one decision: packets cost detector time,
/// and the dwell has to be long enough to pay for them.
#[derive(Clone, Copy)]
pub struct SweepBudget {
    pub dwell: Duration,
    pub packets_per_hop: usize,
}

/// How many packets one hop feeds the detector at `sample_rate`.
pub fn detect_packets_per_hop(sample_rate_hz: f64) -> usize {
    if sample_rate_hz > 0.0 && sample_rate_hz <= NARROW_CAPTURE_MAX_RATE_HZ {
        DETECT_PACKETS_NARROW
    } else {
        DETECT_PACKETS_PER_HOP
    }
}

/// Plan the fewest tune centres that put every channel in `channels_hz`
/// inside some capture window of width `span_hz`.
///
/// Greedy interval cover, the same shape as `orecchiette`'s
/// `plan_tune_centers`: walk the sorted channels, anchor on the leftmost
/// one not yet covered, absorb every channel within `span_hz` of it, and
/// emit the **midpoint of the group actually absorbed** rather than the
/// anchor, which holds the absorbed channels as far from the window
/// edges — where localisation degrades — as the group allows. Anchoring
/// left and taking as much as fits is optimal in the number of windows
/// for covering points on a line with a fixed width.
///
/// The coverage test is on channel *centres*, not whole channel widths.
/// `orecchiette` subtracts a 20 MHz allowance from the reach because it
/// needs a whole DJI channel inside one window to demodulate it. Here the
/// window only has to be good enough to *detect and localise* a carrier,
/// which the measurements behind the caller's usable-span fraction show needs the
/// carrier centre inside the span and nothing more. Charging every narrow
/// analog channel a 20 MHz guard would roughly halve the reach at
/// 25 MSPS for no gain.
pub fn plan_tune_centers(channels_hz: &[f64], span_hz: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = channels_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup();

    let reach = span_hz.max(0.0);
    let mut centers = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let first = sorted[i];
        let mut last = first;
        let mut j = i;
        while j < sorted.len() && sorted[j] - first <= reach {
            last = sorted[j];
            j += 1;
        }
        // `j > i` always holds: the anchor satisfies the window test
        // (distance 0), so the inner loop advances at least once and the
        // outer loop cannot spin. This guards only a NaN-poisoned
        // comparison ordering escaping the filter above.
        if j == i {
            j = i + 1;
        }
        centers.push((first + last) * 0.5);
        i = j;
    }
    centers
}

impl SweepBudget {
    pub fn flat(dwell: Duration, packets_per_hop: usize) -> Self {
        Self {
            dwell,
            packets_per_hop: packets_per_hop.max(1),
        }
    }
}
/// Cadence with caller-supplied tuning/processing costs. Sweep numbers are zero-based.
pub struct SweepPolicy {
    pub fast: SweepBudget,
    pub sensitive: SweepBudget,
    /// Zero disables periodic sensitive sweeps.
    pub sensitive_every: u32,
    pub narrow_capture_max_rate_hz: f64,
}
impl SweepPolicy {
    pub fn budget(&self, n: u32, sample_rate_hz: f64) -> SweepBudget {
        if sample_rate_hz > 0.0 && sample_rate_hz <= self.narrow_capture_max_rate_hz {
            SweepBudget::flat(self.fast.dwell, self.sensitive.packets_per_hop)
        } else if self.sensitive_every > 0 && n % self.sensitive_every == self.sensitive_every - 1 {
            self.sensitive
        } else {
            self.fast
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HopObservation {
    pub first_packet: bool,
    pub sweep_complete: bool,
}
/// Packet budget and completion tracking for a tune plan. Foreign/late centre
/// headers do not count toward coverage. Every hop must receive its full budget.
pub struct ScanProgress {
    expected: HashSet<u64>,
    completed: HashSet<u64>,
    last: Option<u64>,
    packets: usize,
    budget: usize,
}
impl ScanProgress {
    pub fn new(centers_hz: &[f64], packets_per_hop: usize) -> Self {
        Self {
            expected: centers_hz
                .iter()
                .filter(|f| f.is_finite() && **f >= 0.0)
                .map(|f| *f as u64)
                .collect(),
            completed: HashSet::new(),
            last: None,
            packets: 0,
            budget: packets_per_hop.max(1),
        }
    }
    pub fn observe(&mut self, center_hz: u64) -> Option<HopObservation> {
        if !self.expected.contains(&center_hz) {
            return None;
        }
        if self.last != Some(center_hz) {
            self.last = Some(center_hz);
            self.packets = 0;
        }
        if self.packets >= self.budget {
            return None;
        }
        self.packets += 1;
        if self.packets == self.budget {
            self.completed.insert(center_hz);
        }
        Some(HopObservation {
            first_packet: self.packets == 1,
            sweep_complete: self.completed.len() == self.expected.len(),
        })
    }
    /// Keep draining the final dwell until a retune. A one-tune plan has no
    /// retune, so immediately allow its next batch instead.
    pub fn reset_sweep(&mut self) {
        self.completed.clear();
        if self.expected.len() == 1 {
            self.last = None;
            self.packets = 0;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CandidatePolicy {
    NearestChannel,
    AnyUnskippedChannel,
}
/// Strongest selectable coarse hit. Skip keys are tuned frequencies in Hz.
pub struct ScanSelector {
    skipped: HashSet<u64>,
    policy: CandidatePolicy,
    best: Option<(f64, f32, crate::types::SignalType)>,
}
impl ScanSelector {
    pub fn new(skipped: &HashSet<u64>, policy: CandidatePolicy) -> Self {
        Self {
            skipped: skipped.clone(),
            policy,
            best: None,
        }
    }
    /// True when this detection becomes the new best hit (useful for UI logging).
    pub fn consider(&mut self, hit: &DetectionResult) -> bool {
        if !hit.rssi_dbm.is_finite() {
            return false;
        }
        let frequency = hit.frequency_hz as f64;
        let selectable = self.fallback_frequency(frequency).is_some();
        if selectable && self.best.is_none_or(|(_, rssi, _)| hit.rssi_dbm > rssi) {
            self.best = Some((frequency, hit.rssi_dbm, hit.signal_type));
            true
        } else {
            false
        }
    }
    /// Known channels worth testing, ordered by distance and excluding skips.
    pub fn refinement_candidates(&self, frequency_hz: f64) -> Vec<f64> {
        get_candidate_fpv_channels(frequency_hz)
            .into_iter()
            .filter(|c| !self.skipped.contains(&(c.round() as u64)))
            .collect()
    }
    /// Safe coarse fallback, including when a fine-tune probe is inconclusive.
    /// Never return a skipped nearest channel merely because refinement failed.
    pub fn fallback_frequency(&self, frequency_hz: f64) -> Option<f64> {
        if !frequency_hz.is_finite() {
            return None;
        }
        if get_candidate_fpv_channels(frequency_hz).is_empty() {
            return (!self
                .skipped
                .iter()
                .any(|&s| (s as f64 - frequency_hz).abs() <= CHANNEL_SNAP_TOLERANCE_MHZ * 1e6))
            .then_some(frequency_hz);
        }
        match self.policy {
            CandidatePolicy::NearestChannel => {
                let snapped = snap_to_nearest_fpv_channel(frequency_hz);
                (!self.skipped.contains(&(snapped.round() as u64))).then_some(snapped)
            }
            CandidatePolicy::AnyUnskippedChannel => {
                self.refinement_candidates(frequency_hz).first().copied()
            }
        }
    }
    pub fn best(&self) -> Option<(f64, f32, crate::types::SignalType)> {
        self.best
    }
}

/// Rank fine-tune evidence using confidence, then RSSI within the confidence
/// tolerance. Unknown/weak sync remains a caller-visible failed refinement.
#[derive(Default)]
pub struct FineTuneSelector {
    best: Option<(f64, f32, f32)>,
}
impl FineTuneSelector {
    pub fn consider(&mut self, frequency_hz: f64, confidence: f32, rssi_dbm: f32) {
        if !frequency_hz.is_finite() || !confidence.is_finite() || !rssi_dbm.is_finite() {
            return;
        }
        if self.best.is_none_or(|(_, c, r)| {
            confidence > c + 0.05 || (confidence > c - 0.05 && rssi_dbm > r)
        }) {
            self.best = Some((frequency_hz, confidence, rssi_dbm));
        }
    }
    pub fn best(&self) -> Option<(f64, f32, f32)> {
        self.best.filter(|(_, confidence, _)| *confidence > 0.1)
    }
}

/// Receiver acquisition accepts unresolved analog-video evidence (0.6). Standard
/// classification is performed later on a contiguous record by `acquisition`.
pub const RECEIVER_MIN_CONFIDENCE: f32 = 0.55;
pub fn receiver_detector() -> crate::detector::AnalogFpvDetector {
    let mut detector = crate::detector::AnalogFpvDetector::default();
    detector.min_confidence = RECEIVER_MIN_CONFIDENCE;
    detector
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A span too narrow to group anything must degrade to one tune per
    /// channel rather than silently dropping channels — and must not
    /// spin forever doing it.
    #[test]
    fn tune_planner_degrades_to_one_tune_per_channel_when_span_is_zero() {
        let channels = vec![5_658e6, 5_695e6, 5_732e6];
        let hops = plan_tune_centers(&channels, 0.0);
        assert_eq!(hops, channels);
    }

    #[test]
    fn tune_planner_handles_degenerate_input() {
        assert!(plan_tune_centers(&[], 50e6).is_empty());
        // Duplicates collapse rather than each claiming a tune.
        assert_eq!(
            plan_tune_centers(&[5_800e6, 5_800e6, 5_800e6], 0.0).len(),
            1
        );
        // A non-finite entry must not poison the sort into an infinite loop.
        let hops = plan_tune_centers(&[5_800e6, f64::NAN, 5_810e6], 50e6);
        assert_eq!(hops.len(), 1);
    }

    /// A packet is a fixed 65,536 samples, so its worth depends on the
    /// rate: 4.27 ms of signal at 15.36 MSPS against 1.07 at 61.44, for
    /// 2.49 ms of detector against 6.41. Narrow captures get the
    /// integration budget because there it is both more useful and
    /// affordable.
    #[test]
    fn a_narrow_capture_gets_the_integration_budget() {
        assert_eq!(detect_packets_per_hop(15_360_000.0), DETECT_PACKETS_NARROW);
        assert_eq!(detect_packets_per_hop(30_720_000.0), DETECT_PACKETS_NARROW);
    }

    /// Six passes at 61.44 MSPS is 38.5 ms of detector against a 25 ms
    /// dwell — the hop would end before the budget was spent.
    #[test]
    fn a_wide_capture_keeps_the_fast_pass() {
        assert_eq!(detect_packets_per_hop(61_440_000.0), DETECT_PACKETS_PER_HOP);
        assert!(
            detect_packets_per_hop(61_440_000.0) < detect_packets_per_hop(15_360_000.0),
            "a wide capture must spend fewer passes than a narrow one"
        );
    }

    /// A nonsense rate must not silently pick the expensive branch.
    #[test]
    fn a_degenerate_rate_keeps_the_fast_pass() {
        assert_eq!(detect_packets_per_hop(0.0), DETECT_PACKETS_PER_HOP);
        assert_eq!(detect_packets_per_hop(-1.0), DETECT_PACKETS_PER_HOP);
        assert_eq!(detect_packets_per_hop(f64::NAN), DETECT_PACKETS_PER_HOP);
    }
}
