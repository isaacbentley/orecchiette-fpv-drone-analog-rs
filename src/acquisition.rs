//! Channel acquisition, standard resolution and receiver lock policy.
//! Inputs are IQ and evidence; source deadlines, hardware tuning and UI belong
//! to the caller. A measured carrier is for identification, not automatic DDC correction.
use crate::ddc::StreamingDDC;
use crate::decode::LUMA_HEADROOM_HZ;
use crate::demod::fm_demod;
use crate::detector::{AnalogFpvDetector, FpvDetector};
use crate::types::{DetectionResult, SignalType};
use crate::video::detect_video_standard;
use num_complex::Complex;
use std::time::Duration;

/// A 25 ms contiguous record resolves the PAL/NTSC line-rate difference.
pub const STANDARD_DETECT_RECORD: Duration = Duration::from_millis(25);

/// Bounded contiguous acquisition input. A source gap discards the old record.
/// Construct a new record when the source sample rate or tuned centre changes.
pub struct AcquisitionRecord {
    samples: Vec<Complex<f32>>,
    target: usize,
}
impl AcquisitionRecord {
    pub fn new(sample_rate: u32) -> Self {
        let target = ((sample_rate as f64 * STANDARD_DETECT_RECORD.as_secs_f64()) as usize).max(1);
        Self {
            samples: Vec::with_capacity(target),
            target,
        }
    }
    pub fn push(&mut self, iq: &[Complex<f32>], discontinuous: bool) {
        if discontinuous {
            self.samples.clear();
        }
        let take = iq.len().min(self.target - self.samples.len());
        self.samples.extend_from_slice(&iq[..take]);
    }
    pub fn is_ready(&self) -> bool {
        self.samples.len() >= self.target
    }
    pub fn samples(&self) -> &[Complex<f32>] {
        &self.samples
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardEvidence {
    Classifier,
    LineTiming,
    Fallback,
}
#[derive(Debug, Clone)]
pub struct StandardResolution {
    pub standard: SignalType,
    pub evidence: StandardEvidence,
}

/// Resolve an ambiguous classification using demodulated line spacing. The
/// caller supplies its fallback (for example, a scan hint or operator default).
pub fn resolve_standard(
    tagged: SignalType,
    baseband_iq: &[Complex<f32>],
    sample_rate: u32,
    fallback: SignalType,
) -> StandardResolution {
    if matches!(
        tagged,
        SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc
    ) {
        return StandardResolution {
            standard: tagged,
            evidence: StandardEvidence::Classifier,
        };
    }
    let measured = detect_video_standard(&fm_demod(baseband_iq), sample_rate);
    if matches!(
        measured,
        SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc
    ) {
        StandardResolution {
            standard: measured,
            evidence: StandardEvidence::LineTiming,
        }
    } else {
        StandardResolution {
            standard: fallback,
            evidence: StandardEvidence::Fallback,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcquisitionResult {
    pub standard: StandardResolution,
    pub classifier_confidence: f32,
    /// Biased localization estimate. Never apply directly as a DDC correction.
    pub measured_carrier_hz: Option<f64>,
}
/// Classify a channel already tuned to DC and refine its identity on the same
/// record. Localization is bounded to one 65,536-sample probe and the decode
/// passband; an unrelated stronger carrier cannot steal the picture's label.
pub fn analyze_tuned_channel(
    iq: &[Complex<f32>],
    center_hz: f64,
    sample_rate: u32,
    fm_deviation_hz: f32,
    fallback: SignalType,
) -> Result<AcquisitionResult, crate::decode::DecodeConfigError> {
    validate_probe(sample_rate, fm_deviation_hz, center_hz)?;
    let detector = AnalogFpvDetector::default();
    let passband = fm_deviation_hz + LUMA_HEADROOM_HZ;
    let measured_carrier_hz = carrier_for_decoded_channel(
        detector
            .detect_from_iq(&iq[..iq.len().min(65_536)], center_hz as u64, sample_rate)
            .into_iter()
            .map(|h| h.frequency_hz as f64),
        center_hz,
        passband as f64,
    );
    let (tagged, classifier_confidence) = detector.detect_sync_pulses(iq, sample_rate);
    let standard = match tagged {
        SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc => StandardResolution {
            standard: tagged,
            evidence: StandardEvidence::Classifier,
        },
        SignalType::Unknown => StandardResolution {
            standard: fallback,
            evidence: StandardEvidence::Fallback,
        },
        other => {
            let mut ddc = StreamingDDC::new(0.0, sample_rate, passband);
            resolve_standard(other, &ddc.process(iq), sample_rate, fallback)
        }
    };
    Ok(AcquisitionResult {
        standard,
        classifier_confidence,
        measured_carrier_hz,
    })
}

fn validate_probe(
    sample_rate: u32,
    deviation: f32,
    frequency: f64,
) -> Result<(), crate::decode::DecodeConfigError> {
    if sample_rate == 0 || !deviation.is_finite() || deviation <= 0.0 || !frequency.is_finite() {
        return Err(crate::decode::DecodeConfigError(
            "probe requires a positive sample rate/deviation and finite frequency",
        ));
    }
    Ok(())
}

/// Filter an offset channel and classify its standard. Used for file playback;
/// there is no hardware-tuned centre or carrier refinement involved.
pub fn analyze_channel_standard(
    iq: &[Complex<f32>],
    offset_hz: f32,
    sample_rate: u32,
    deviation_hz: f32,
    fallback: SignalType,
) -> Result<(StandardResolution, f32), crate::decode::DecodeConfigError> {
    validate_probe(sample_rate, deviation_hz, offset_hz as f64)?;
    let mut ddc = StreamingDDC::new(offset_hz, sample_rate, deviation_hz + LUMA_HEADROOM_HZ);
    let baseband = ddc.process(iq);
    let (tagged, confidence) =
        AnalogFpvDetector::default().detect_sync_pulses(&baseband, sample_rate);
    let resolution = if tagged == SignalType::Unknown {
        StandardResolution {
            standard: fallback,
            evidence: StandardEvidence::Fallback,
        }
    } else {
        resolve_standard(tagged, &baseband, sample_rate, fallback)
    };
    Ok((resolution, confidence))
}

/// The lock check's running state, factored out so the rule is directly
/// unit-testable rather than buried in the reader closure.
///
/// Three outcomes per check, and the distinction that matters is
/// between the middle two:
///
/// - **Held**: our carrier is there and fields are coming out.
/// - **Provisional**: our carrier is there and no field has come out
///   since the last check. Acquisition looks like this; so does a
///   horizontal-sync train with no vertical sync, which scores 0.8
///   forever and reconstructs nothing. Bounded rather than trusted.
/// - **Lost**: nothing, or only hits that are not ours.
#[derive(Default)]
pub struct LockState {
    empty_checks: u32,
    provisional_checks: u32,
}

impl LockState {
    /// Fold one check in. `true` means release the channel.
    pub fn observe(&mut self, on_carrier: bool, decoding: bool) -> bool {
        match (on_carrier, decoding) {
            (true, true) => {
                self.empty_checks = 0;
                self.provisional_checks = 0;
                false
            }
            (true, false) => {
                // Our carrier is there. That answers the "is it still
                // ours" question regardless of whether a field came out,
                // so the miss counter resets — leaving it standing made
                // miss -> recovery -> miss release the channel on two
                // misses that were never consecutive, which is the one
                // thing `LOCK_EMPTY_CHECKS` is counting.
                self.empty_checks = 0;
                self.provisional_checks = self.provisional_checks.saturating_add(1);
                self.provisional_checks >= LOCK_PROVISIONAL_CHECKS
            }
            _ => {
                self.empty_checks = self.empty_checks.saturating_add(1);
                self.empty_checks >= LOCK_EMPTY_CHECKS
            }
        }
    }
}

/// Consecutive checks finding nothing of ours before the channel is
/// released.
///
/// Strictly consecutive: any check that sees our carrier clears this,
/// including one that saw no decoded field. That case is counted by
/// [`LOCK_PROVISIONAL_CHECKS`] instead — "the carrier is gone" and "the
/// carrier is here but undecodable" are different failures and are
/// timed separately.
///
/// Two, not four: the lock integrator holds ~4 checks of history, so an
/// empty result already means several consecutive looks found nothing.
/// Stacking a 4-check threshold on top made a dead channel linger for
/// up to 8 checks.
pub const LOCK_EMPTY_CHECKS: u32 = 2;

/// How far a continuing detection may sit from the channel we tuned to
/// and still count as *this* signal.
///
/// The lock check ran on whatever the capture held, so a transmitter
/// elsewhere in the 49 MHz span kept the tuned channel's lock alive
/// indefinitely. Sized like [`crate::bands::CHANNEL_SNAP_TOLERANCE_MHZ`]: past this,
/// a hit is more likely a different channel than an off-tune of ours.
pub const LOCK_CARRIER_TOLERANCE_HZ: f64 = 10e6;

/// Lock checks a carrier may go on being detected without a single
/// field coming out before we let it go.
///
/// A detection is not decodable video. A horizontal-sync pulse train
/// with no vertical sync scores 0.8 on the sweep detector and
/// reconstructs no frame at all, so "there is a hit" held the viewer on
/// a picture it could never draw. Fields reconstructing is the
/// confirmation; this is how long acquisition is allowed to take before
/// the absence of one is treated as an answer.
///
/// Checks run about twice a second, so eight is ~4 s — long against
/// standard classification plus a first field, short against staring at
/// an undecodable carrier.
pub const LOCK_PROVISIONAL_CHECKS: u32 = 8;

/// Which detection describes the carrier being decoded.
///
/// Not the strongest one. The decoder mixes the tuned centre down to DC
/// and keeps `passband_hz` either side, so a detection outside that is
/// filtered out before anything is drawn — it cannot be what is on
/// screen. Ranking by confidence instead picked a second transmitter
/// 20 MHz away that scored just as well, and labelled the picture after
/// the neighbour.
///
/// Among the candidates the decoder can actually see, the nearest to DC
/// is the one it is centred on.
pub fn carrier_for_decoded_channel(
    hits: impl IntoIterator<Item = f64>,
    tuned_hz: f64,
    passband_hz: f64,
) -> Option<f64> {
    hits.into_iter()
        .filter(|hz| (hz - tuned_hz).abs() <= passband_hz)
        .min_by(|a, b| (a - tuned_hz).abs().total_cmp(&(b - tuned_hz).abs()))
}

impl LockState {
    /// Only freshly observed timing counts as decoding; holdover pictures do not.
    pub fn observe_detections(
        &mut self,
        hits: &[DetectionResult],
        tuned_hz: f64,
        observed_timing: bool,
    ) -> bool {
        self.observe(
            hits.iter()
                .any(|h| (h.frequency_hz as f64 - tuned_hz).abs() <= LOCK_CARRIER_TOLERANCE_HZ),
            observed_timing,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bands::snap_to_nearest_fpv_channel;
    /// A healthy decode must never be released.
    #[test]
    fn a_decoding_carrier_holds_lock_indefinitely() {
        let mut lock = LockState::default();
        for i in 0..1000 {
            assert!(
                !lock.observe(true, true),
                "released a decoding carrier at check {i}"
            );
        }
    }

    /// The review's case: a horizontal-sync train with no vertical sync
    /// scores 0.8 on the sweep detector forever and reconstructs no
    /// frame. "There is a hit" held the viewer on a picture it could
    /// never draw.
    #[test]
    fn a_detected_but_undecodable_carrier_is_eventually_released() {
        let mut lock = LockState::default();
        let mut checks = 0;
        while !lock.observe(true, false) {
            checks += 1;
            assert!(checks < 100, "never released an undecodable carrier");
        }
        assert_eq!(checks + 1, LOCK_PROVISIONAL_CHECKS);
    }

    /// A hit that is not ours must not keep this channel alive — the
    /// lock check ran on the whole capture, so a transmitter elsewhere
    /// in a 49 MHz span did exactly that.
    #[test]
    fn a_hit_elsewhere_does_not_hold_our_channel() {
        let mut lock = LockState::default();
        // Someone else's strong, decodable signal.
        for _ in 0..LOCK_EMPTY_CHECKS - 1 {
            assert!(!lock.observe(false, true));
        }
        assert!(
            lock.observe(false, true),
            "another channel's detection kept ours locked"
        );
    }

    /// The review's sequence: a miss, then the carrier returns without
    /// a field, then another miss. Those two misses are not consecutive
    /// and must not release the channel.
    #[test]
    fn non_consecutive_misses_do_not_release_the_channel() {
        // miss -> carrier back but no field yet -> miss. Two misses,
        // not consecutive. `LOCK_EMPTY_CHECKS` counts consecutive ones,
        // so this must survive; before the fix the second miss released.
        let mut lock = LockState::default();
        assert!(!lock.observe(false, true), "a single miss released");
        assert!(!lock.observe(true, false), "a provisional check released");
        assert!(
            !lock.observe(false, true),
            "two non-consecutive misses released the channel"
        );

        // A decoded field clears both counters, so an isolated miss
        // between good checks can repeat indefinitely without drifting
        // toward release. (Good check first: the sequence above already
        // ended on a miss, and two in a row *should* release.)
        for _ in 0..50 {
            assert!(!lock.observe(true, true));
            assert!(!lock.observe(false, true), "an isolated miss released");
        }
    }

    /// Strictly consecutive misses still release, on the documented
    /// count — the fix must not make the channel un-releasable.
    #[test]
    fn consecutive_misses_still_release() {
        let mut lock = LockState::default();
        for _ in 0..LOCK_EMPTY_CHECKS - 1 {
            assert!(!lock.observe(false, false));
        }
        assert!(lock.observe(false, false));
    }

    /// And an undecodable carrier still runs out its own budget even
    /// though it keeps clearing the miss counter.
    #[test]
    fn a_provisional_carrier_still_times_out() {
        let mut lock = LockState::default();
        for _ in 0..LOCK_PROVISIONAL_CHECKS - 1 {
            assert!(!lock.observe(true, false));
        }
        assert!(lock.observe(true, false));
    }

    /// Acquisition stutter must not accumulate toward release: a check
    /// that sees both clears the provisional count.
    #[test]
    fn a_field_arriving_clears_the_provisional_count() {
        let mut lock = LockState::default();
        for _ in 0..LOCK_PROVISIONAL_CHECKS - 1 {
            assert!(!lock.observe(true, false));
        }
        assert!(!lock.observe(true, true), "a good check should recover");
        // Full budget available again.
        for _ in 0..LOCK_PROVISIONAL_CHECKS - 1 {
            assert!(!lock.observe(true, false));
        }
    }

    /// The review's reproduction: two transmitters, 5800 and 5820 MHz,
    /// captured at 5800. Both score 0.95. Ranking by confidence picked
    /// 5820 and labelled the picture after a transmitter the DDC never
    /// passes.
    #[test]
    fn refinement_labels_the_carrier_being_decoded_not_the_loudest() {
        let tuned = 5_800e6;
        let passband = 7e6; // 5 MHz deviation + luma headroom
        assert_eq!(
            carrier_for_decoded_channel(vec![5_800e6, 5_820e6], tuned, passband),
            Some(5_800e6)
        );
    }

    /// Order must not decide it either.
    #[test]
    fn refinement_is_independent_of_detection_order() {
        let tuned = 5_800e6;
        assert_eq!(
            carrier_for_decoded_channel(vec![5_820e6, 5_800e6], tuned, 7e6),
            carrier_for_decoded_channel(vec![5_800e6, 5_820e6], tuned, 7e6)
        );
    }

    /// A real offset inside the passband is still adopted — the whole
    /// point, since the sweep may have tuned a megahertz off.
    #[test]
    fn refinement_adopts_a_carrier_offset_inside_the_passband() {
        // Tuned to F7 (5860) with the carrier really near A1 (5865).
        let got = carrier_for_decoded_channel(vec![5_863.98e6], 5_860e6, 7e6);
        assert_eq!(got, Some(5_863.98e6));
        assert_eq!(snap_to_nearest_fpv_channel(got.unwrap()) / 1e6, 5865.0);
    }

    /// Nothing the decoder can see means no refinement, not a guess.
    #[test]
    fn refinement_declines_when_every_candidate_is_out_of_band() {
        assert_eq!(
            carrier_for_decoded_channel(vec![5_820e6, 5_780e6], 5_800e6, 7e6),
            None
        );
        assert_eq!(
            carrier_for_decoded_channel(Vec::<f64>::new(), 5_800e6, 7e6),
            None
        );
    }

    /// Holdover frames must not count as observed timing evidence for channel lock.
    /// If carrier is detected but only coasted/holdover frames are produced (no observed timing evidence),
    /// the scanner must still release the channel after the provisional timeout.
    #[test]
    fn holdover_fields_do_not_prevent_undecodable_carrier_timeout() {
        let mut lock = LockState::default();
        let mut checks = 0;
        let observed_timing_fields = 0u64;
        let mut frames_at_last_check = 0u64;

        // Simulate a carrier present where decoder produces holdover fields (worker_frames increments,
        // but observed_timing_fields stays unchanged).
        while checks < 100 {
            let decoding = observed_timing_fields > frames_at_last_check;
            frames_at_last_check = observed_timing_fields;
            if lock.observe(true, decoding) {
                break;
            }
            checks += 1;
        }
        assert_eq!(
            checks + 1,
            LOCK_PROVISIONAL_CHECKS,
            "holdover fields kept the channel locked"
        );
    }
}
