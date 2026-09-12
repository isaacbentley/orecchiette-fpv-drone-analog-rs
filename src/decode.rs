//! Hardware-independent configuration and streaming IQ-to-frame decoding.
//!
//! The caller owns scheduling and output buffers. `push_iq` preserves the DDC,
//! discriminator and filter state across chunks; `next_field_into` drains all
//! available fields using the same timed reconstruction path as the viewer.

//!
//! ```no_run
//! use orecchiette_fpv_drone_analog_rs::decode::{DecodePlan, DecoderConfig, DemodulationMode, StreamingFpvDecoder};
//! use orecchiette_fpv_drone_analog_rs::timing::Standard;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let plan = DecodePlan::new(30_720_000, 5e6, 14e6, DemodulationMode::Auto)?;
//! let mut decoder = StreamingFpvDecoder::new(DecoderConfig {
//!     plan, frequency_offset_hz: 0.0, standard: Standard::Ntsc,
//!     deemphasis_tau_s: 0.75e-6, temporal_window: 5, debug: false,
//! })?;
//! let mut frame = vec![0; decoder.reconstructor().width * decoder.reconstructor().height];
//! # let source_blocks: Vec<(Vec<num_complex::Complex<f32>>, bool)> = Vec::new();
//! for (iq, source_gap) in source_blocks {
//!     decoder.push_iq(&iq, source_gap);
//!     while let Some(timing) = decoder.next_field_into(&mut frame)? {
//!         // Display/record `frame`; only observed timing confirms receiver lock.
//!         println!("Observed timing: {}", timing.has_observed_timing_evidence);
//!     }
//! }
//! # Ok(()) }
//! ```

use crate::ddc::StreamingDDC;
use crate::demod::{Deemphasis, PllFmDemod, fm_demod_into};
use crate::timing::{DecodeStep, DecodeValidationError, FieldTiming, Standard, TimedDemodSlice};
use crate::video::FrameReconstructor;
use num_complex::Complex;
use std::time::{Duration, Instant};

pub const LUMA_HEADROOM_HZ: f32 = 2_000_000.0;
pub const DECODE_FIR_TAPS: usize = 127;
pub const PLL_LOOP_BW_HZ: f32 = 1.0e6;
pub const PLL_AUTO_MIN_SAMPLE_RATE_HZ: u32 = 25_000_000;
const FIR_TRANSITION_K: f32 = 2.33;

/// The requested demodulation mode, resolved after choosing the working rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DemodulationMode {
    #[default]
    Auto,
    Discriminator,
    Pll,
}

impl DemodulationMode {
    /// A PLL must be able to follow the FM deviation at its clamped bandwidth.
    pub fn use_pll(self, decode_rate_hz: u32, fm_deviation_hz: f32) -> bool {
        match self {
            Self::Discriminator => false,
            Self::Pll => true,
            Self::Auto => {
                let bandwidth =
                    (decode_rate_hz as f32 / (4.0 * std::f32::consts::PI)).min(PLL_LOOP_BW_HZ);
                decode_rate_hz >= PLL_AUTO_MIN_SAMPLE_RATE_HZ
                    && fm_deviation_hz > 0.0
                    && fm_deviation_hz <= bandwidth
            }
        }
    }
}

/// Maximum decimation whose output Nyquist band clears the FIR transition.
pub fn decode_decimation(sample_rate: u32, ddc_cutoff_hz: f32) -> usize {
    if sample_rate == 0 || !ddc_cutoff_hz.is_finite() || ddc_cutoff_hz <= 0.0 {
        return 1;
    }
    let transition = FIR_TRANSITION_K * sample_rate as f32 / DECODE_FIR_TAPS as f32;
    ((sample_rate as f32 / (2.0 * ddc_cutoff_hz + transition)).floor() as usize).max(1)
}

/// Minimum hardware capture rate that holds the widest planned channel inside
/// a device's usable span. Device-supported rates and tuning costs are external.
pub fn required_capture_rate(fm_deviation_hz: f32, usable_span_fraction: f64) -> Option<f64> {
    if !fm_deviation_hz.is_finite()
        || fm_deviation_hz <= 0.0
        || !usable_span_fraction.is_finite()
        || !(0.0..=1.0).contains(&usable_span_fraction)
        || usable_span_fraction == 0.0
    {
        return None;
    }
    Some(2.0 * (fm_deviation_hz as f64 + LUMA_HEADROOM_HZ as f64) / usable_span_fraction)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeConfigError(pub &'static str);
impl std::fmt::Display for DecodeConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for DecodeConfigError {}

/// A consistent filter/rate/demodulator selection. Derived fields are read-only.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodePlan {
    sample_rate: u32,
    fm_deviation: f32,
    ddc_cutoff: f32,
    decimation: usize,
    work_rate: u32,
    use_pll: bool,
}

impl DecodePlan {
    pub fn new(
        sample_rate: u32,
        fm_deviation: f32,
        bandwidth_hz: f32,
        mode: DemodulationMode,
    ) -> Result<Self, DecodeConfigError> {
        if sample_rate == 0 {
            return Err(DecodeConfigError("sample rate must be positive"));
        }
        if !fm_deviation.is_finite() || fm_deviation <= 0.0 {
            return Err(DecodeConfigError(
                "FM deviation must be finite and positive",
            ));
        }
        if !bandwidth_hz.is_finite() || bandwidth_hz <= 0.0 {
            return Err(DecodeConfigError(
                "channel bandwidth must be finite and positive",
            ));
        }
        let ddc_cutoff = (bandwidth_hz * 0.5)
            .min(fm_deviation + LUMA_HEADROOM_HZ)
            .max(fm_deviation);
        let mut decimation = decode_decimation(sample_rate, ddc_cutoff);
        if mode == DemodulationMode::Pll {
            decimation =
                decimation.min((sample_rate / PLL_AUTO_MIN_SAMPLE_RATE_HZ).max(1) as usize);
        }
        let work_rate = sample_rate / decimation as u32;
        Ok(Self {
            sample_rate,
            fm_deviation,
            ddc_cutoff,
            decimation,
            work_rate,
            use_pll: mode.use_pll(work_rate, fm_deviation),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn fm_deviation(&self) -> f32 {
        self.fm_deviation
    }
    pub fn ddc_cutoff_hz(&self) -> f32 {
        self.ddc_cutoff
    }
    pub fn decimation(&self) -> usize {
        self.decimation
    }
    pub fn work_rate(&self) -> u32 {
        self.work_rate
    }
    pub fn use_pll(&self) -> bool {
        self.use_pll
    }
}

#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub plan: DecodePlan,
    pub frequency_offset_hz: f32,
    pub standard: Standard,
    /// Zero disables deemphasis. Positive values are time constants in seconds.
    pub deemphasis_tau_s: f32,
    pub temporal_window: usize,
    pub debug: bool,
}

/// Optional elapsed wall-time instrumentation of the actual decoding stages.
#[derive(Debug, Clone, Default)]
pub struct DecodeStageTimings {
    pub ddc: Duration,
    pub demod: Duration,
    pub deemphasis: Duration,
    pub buffering: Duration,
    pub reconstruct: Duration,
}
impl DecodeStageTimings {
    pub fn total(&self) -> Duration {
        self.ddc + self.demod + self.deemphasis + self.buffering + self.reconstruct
    }
}

/// Stateful receiver core, independent of source drivers, threads and UI.
pub struct StreamingFpvDecoder {
    plan: DecodePlan,
    reconstructor: FrameReconstructor,
    ddc: StreamingDDC,
    shifted_iq: Vec<Complex<f32>>,
    iq_carry: Option<Complex<f32>>,
    demod_scratch: Vec<f32>,
    deemph: Option<Deemphasis>,
    pll: Option<PllFmDemod>,
    demod_buffer: Vec<f32>,
    demod_start: usize,
    current_sample_coordinate: u64,
    compact_after: usize,
    min_samples_per_field: usize,
    profiling: bool,
    timings: DecodeStageTimings,
    #[cfg(feature = "neural-vsr")]
    stashed_restorer: Option<crate::neural::NeuralRestorer>,
}

impl StreamingFpvDecoder {
    pub fn new(config: DecoderConfig) -> Result<Self, DecodeConfigError> {
        if !config.frequency_offset_hz.is_finite() {
            return Err(DecodeConfigError("frequency offset must be finite"));
        }
        if !config.deemphasis_tau_s.is_finite() || config.deemphasis_tau_s < 0.0 {
            return Err(DecodeConfigError(
                "deemphasis must be finite and nonnegative",
            ));
        }
        let plan = config.plan;
        if (plan.work_rate as f64) < config.standard.nominal_line_hz() {
            return Err(DecodeConfigError(
                "decode rate must provide at least one sample per video line",
            ));
        }
        let is_pal = config.standard == Standard::Pal;
        let reconstructor =
            FrameReconstructor::new(plan.work_rate, is_pal, plan.fm_deviation, config.debug)
                .with_temporal_window(config.temporal_window);
        let ddc = if plan.decimation > 1 {
            StreamingDDC::with_taps(
                config.frequency_offset_hz,
                plan.sample_rate,
                plan.ddc_cutoff,
                DECODE_FIR_TAPS,
            )
        } else {
            StreamingDDC::new(
                config.frequency_offset_hz,
                plan.sample_rate,
                plan.ddc_cutoff,
            )
        };
        let deemph = (config.deemphasis_tau_s > 0.0)
            .then(|| Deemphasis::new(plan.work_rate, config.deemphasis_tau_s));
        let pll = plan
            .use_pll
            .then(|| PllFmDemod::new(plan.work_rate, PLL_LOOP_BW_HZ, plan.fm_deviation * 1.2));
        let lines = if is_pal { 288 } else { 240 } + 22;
        let min_samples_per_field =
            (plan.work_rate as f64 / config.standard.nominal_line_hz() * lines as f64) as usize;
        let min_samples_per_field = min_samples_per_field.max(1);
        let compact_after = plan.work_rate as usize / 30;
        Ok(Self {
            plan,
            reconstructor,
            ddc,
            shifted_iq: Vec::new(),
            iq_carry: None,
            demod_scratch: Vec::new(),
            deemph,
            pll,
            demod_buffer: Vec::new(),
            demod_start: 0,
            current_sample_coordinate: 0,
            compact_after,
            min_samples_per_field,
            profiling: false,
            timings: DecodeStageTimings::default(),
            #[cfg(feature = "neural-vsr")]
            stashed_restorer: None,
        })
    }

    pub fn plan(&self) -> &DecodePlan {
        &self.plan
    }
    pub fn reconstructor(&self) -> &FrameReconstructor {
        &self.reconstructor
    }
    /// Coordinate at the beginning of the unconsumed demodulated region.
    pub fn sample_coordinate(&self) -> u64 {
        self.current_sample_coordinate
    }
    pub fn buffered_samples(&self) -> usize {
        self.demod_buffer.len() - self.demod_start
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
    }
    pub fn take_stage_timings(&mut self) -> DecodeStageTimings {
        std::mem::take(&mut self.timings)
    }
    pub fn cnr_db(&self) -> Option<f32> {
        crate::levels::estimate_cnr_db(&self.shifted_iq)
    }
    pub fn pll_phase_error_rms(&self) -> Option<f32> {
        self.pll.as_ref().map(PllFmDemod::phase_error_rms)
    }
    pub fn demodulated_chunk(&self) -> &[f32] {
        &self.demod_scratch
    }

    /// Reset all stream-dependent state once at a source gap. The origin counts
    /// known demodulated samples; unknown loss is represented by a new epoch.
    pub fn reset_discontinuity(&mut self) {
        self.iq_carry = None;
        self.shifted_iq.clear();
        self.demod_scratch.clear();
        self.ddc.reset();
        if let Some(p) = self.pll.as_mut() {
            p.reset();
        }
        if let Some(d) = self.deemph.as_mut() {
            d.reset();
        }
        self.current_sample_coordinate += self.buffered_samples() as u64;
        self.demod_buffer.clear();
        self.demod_start = 0;
        self.reconstructor.forget_history();
    }

    /// Append a continuous IQ chunk, or start a new epoch when `discontinuous`.
    /// Drain `next_field_into` before appending the next chunk to bound memory.
    pub fn push_iq(&mut self, samples: &[Complex<f32>], discontinuous: bool) {
        if discontinuous {
            self.reset_discontinuity();
        }
        let start = self.profiling.then(Instant::now);
        self.shifted_iq.clear();
        if self.pll.is_none()
            && let Some(prev) = self.iq_carry
        {
            self.shifted_iq.push(prev);
        }
        self.ddc
            .process_into_decimated(samples, &mut self.shifted_iq, self.plan.decimation);
        self.iq_carry = self.shifted_iq.last().copied();
        if let Some(t) = start {
            self.timings.ddc += t.elapsed();
        }
        let start = self.profiling.then(Instant::now);
        match self.pll.as_mut() {
            Some(p) => p.process_into(&self.shifted_iq, &mut self.demod_scratch),
            None => fm_demod_into(&self.shifted_iq, &mut self.demod_scratch),
        }
        if let Some(t) = start {
            self.timings.demod += t.elapsed();
        }
        let start = self.profiling.then(Instant::now);
        if let Some(d) = self.deemph.as_mut() {
            d.process_in_place(&mut self.demod_scratch);
        }
        if let Some(t) = start {
            self.timings.deemphasis += t.elapsed();
        }
        #[cfg(feature = "neural-vsr")]
        if self.reconstructor.neural_restorer.is_some()
            && let Some(cnr) = self.cnr_db()
        {
            self.reconstructor.set_neural_noise_level(cnr);
        }
        let start = self.profiling.then(Instant::now);
        if self.demod_start > self.compact_after {
            self.demod_buffer.drain(..self.demod_start);
            self.demod_start = 0;
        }
        self.demod_buffer.extend_from_slice(&self.demod_scratch);
        if let Some(t) = start {
            self.timings.buffering += t.elapsed();
        }
    }

    /// Render the next complete field, skipping unsupported fields internally.
    /// `None` means append more IQ. An invalid output buffer consumes no input.
    /// VBI confidence can vary slightly with the available level-estimation
    /// lookahead; it is not a chunk-independent checksum of a field.
    pub fn next_field_into(
        &mut self,
        frame: &mut [u32],
    ) -> Result<Option<FieldTiming>, DecodeValidationError> {
        let required = self.reconstructor.width * self.reconstructor.height;
        if frame.len() != required {
            return Err(DecodeValidationError::FrameBufferTooSmall {
                required,
                actual: frame.len(),
            });
        }
        while self.buffered_samples() >= self.min_samples_per_field {
            let slice = TimedDemodSlice::new(
                &self.demod_buffer[self.demod_start..],
                self.current_sample_coordinate,
                self.plan.work_rate,
                false,
            );
            let start = self.profiling.then(Instant::now);
            let result = self.reconstructor.reconstruct_timed_into(slice, frame);
            if let Some(t) = start {
                self.timings.reconstruct += t.elapsed();
            }
            match result? {
                DecodeStep::NeedMoreData { .. } => return Ok(None),
                DecodeStep::Advance {
                    consumed_samples,
                    field,
                } => {
                    self.demod_start += consumed_samples;
                    self.current_sample_coordinate += consumed_samples as u64;
                    if field.is_some() {
                        return Ok(field);
                    }
                }
            }
        }
        Ok(None)
    }

    /// Discard old pending demodulated samples when the caller's latency budget
    /// requires it. This invalidates picture/timing history but keeps the
    /// continuous upstream DDC and demodulator state.
    pub fn discard_pending_except(&mut self, keep_samples: usize) -> usize {
        let skipped = self.buffered_samples().saturating_sub(keep_samples);
        if skipped > 0 {
            self.demod_start += skipped;
            self.current_sample_coordinate += skipped as u64;
            self.reconstructor.forget_history();
        }
        skipped
    }

    #[cfg(feature = "neural-vsr")]
    pub fn load_neural_restorer(&mut self, path: &str, use_gpu: bool) -> ort::Result<()> {
        let restorer = crate::neural::NeuralRestorer::new(path, use_gpu)?;
        self.reconstructor.neural_restorer = Some(restorer);
        self.stashed_restorer = None;
        self.reconstructor.hidden_state = None;
        Ok(())
    }

    #[cfg(feature = "neural-vsr")]
    pub fn set_restoration_enabled(&mut self, enabled: bool) {
        if enabled && self.reconstructor.neural_restorer.is_none() {
            self.reconstructor.neural_restorer = self.stashed_restorer.take();
            self.reconstructor.hidden_state = None;
        } else if !enabled && self.reconstructor.neural_restorer.is_some() {
            self.stashed_restorer = self.reconstructor.neural_restorer.take();
            self.reconstructor.hidden_state = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn select_decode(rate: u32, deviation: f32, mode: DemodulationMode) -> (u32, bool) {
        let plan =
            DecodePlan::new(rate, deviation, 2.0 * (deviation + LUMA_HEADROOM_HZ), mode).unwrap();
        (plan.work_rate(), plan.use_pll())
    }
    /// The decode path may only shed bandwidth it has already filtered
    /// away. Energy at `work_rate - cutoff` folds onto the passband
    /// edge, so that frequency has to clear the filter's stopband edge.
    ///
    /// The rule has to be driven by the channel's own cutoff, not a
    /// fixed target rate: file playback defaults to a 17 MHz deviation,
    /// and "always decode at 15.36 MSPS" would fold a 19 MHz-wide
    /// channel onto itself.
    #[test]
    fn decode_decimation_never_folds_a_channel_onto_itself() {
        // A live channel (5 MHz deviation) and a file one (17 MHz).
        for cutoff in [5e6f32 + LUMA_HEADROOM_HZ, 17e6 + LUMA_HEADROOM_HZ] {
            for rate in [
                7_680_000u32,
                15_360_000,
                20_000_000,
                25_000_000,
                30_720_000,
                40_000_000,
                50_000_000,
                61_440_000,
                100_000_000,
            ] {
                let decim = decode_decimation(rate, cutoff);
                assert!(decim >= 1, "{rate} at {cutoff}: factor must be positive");
                if decim == 1 {
                    // Declining to decimate is always safe. Whether the
                    // capture was wide enough for the channel in the
                    // first place is not this function's business — at
                    // 7.68 MSPS a 7 MHz cutoff is already undersampled,
                    // and no stride fixes that.
                    continue;
                }
                let work_rate = (rate / decim as u32) as f32;
                // Nothing above the new Nyquist that could fold into
                // the passband is still inside the filter's transition.
                let stopband_edge =
                    cutoff + FIR_TRANSITION_K * rate as f32 / DECODE_FIR_TAPS as f32;
                assert!(
                    work_rate - cutoff >= stopband_edge,
                    "{:.2} MSPS at a {:.1} MHz cutoff decimated by {decim} to {:.2} MSPS: \
                     energy at {:.2} MHz folds onto the passband edge but the filter is \
                     still {:.2} MHz from its stopband",
                    rate as f64 / 1e6,
                    cutoff as f64 / 1e6,
                    work_rate as f64 / 1e6,
                    (work_rate - cutoff) as f64 / 1e6,
                    stopband_edge as f64 / 1e6
                );
            }
        }

        // The live defaults, spelled out: both wide Aaronia spans land
        // on the same working rate, and every rate a decoder already
        // kept up with is left alone.
        let live = 5e6f32 + LUMA_HEADROOM_HZ;
        assert_eq!(decode_decimation(61_440_000, live), 4);
        assert_eq!(decode_decimation(30_720_000, live), 2);
        for unchanged in [25_000_000u32, 20_000_000, 15_360_000, 7_680_000] {
            assert_eq!(
                decode_decimation(unchanged, live),
                1,
                "{unchanged} has nothing to shed at a 7 MHz cutoff"
            );
        }

        // A 19 MHz-wide file channel needs ~38 MSPS of working rate, so
        // it must not decimate below that.
        let file = 17e6f32 + LUMA_HEADROOM_HZ;
        assert_eq!(decode_decimation(61_440_000, file), 1);
        assert!(100_000_000 / decode_decimation(100_000_000, file) as u32 >= 38_000_000);

        // Degenerate input must not divide by zero or panic.
        assert_eq!(decode_decimation(0, live), 1);
        assert_eq!(decode_decimation(61_440_000, 0.0), 1);
        assert_eq!(decode_decimation(61_440_000, f32::NAN), 1);
    }

    /// The auto rule has to be about the deviation, not the rate. At
    /// FPV's 5 MHz the PLL measured 9-25 dB *worse* than the
    /// discriminator, because its loop cannot track that excursion —
    /// and it also costs 6.49 dB at 4.2 MHz luma.
    #[test]
    fn auto_does_not_pick_the_pll_at_an_fpv_deviation() {
        for rate in [25_000_000u32, 30_720_000, 61_440_000] {
            assert!(
                !DemodulationMode::Auto.use_pll(rate, 5_000_000.0),
                "auto chose the PLL at {rate} Hz for a 5 MHz deviation"
            );
        }
    }

    /// Through the *pipeline's* ordering — decimate, then choose —
    /// `Auto` resolves to the discriminator everywhere.
    ///
    /// This replaces a test that asked `use_pll(25_000_000, 500_000.0)`
    /// directly and concluded the PLL was reachable. The pipeline never
    /// poses that question: a 500 kHz deviation at 25 MSPS decimates by
    /// 4 first, so what `Auto` is actually asked about is 6.25 MSPS,
    /// where the loop is worth +1.8 dB rather than +9.9 and the rate
    /// gate correctly declines.
    #[test]
    fn auto_resolves_to_the_discriminator_through_the_real_ordering() {
        for capture in [15_360_000u32, 25_000_000, 30_720_000, 61_440_000] {
            for dev in [200_000.0f32, 500_000.0, 1_000_000.0, 5_000_000.0] {
                let (work, pll) = select_decode(capture, dev, DemodulationMode::Auto);
                assert!(
                    !pll,
                    "auto chose the PLL at capture {capture} dev {dev} \
                     (work rate {work})"
                );
            }
        }
    }

    /// `--demod pll` still gets the PLL, and gets it at a rate the loop
    /// can use rather than a decimated one.
    #[test]
    fn forcing_the_pll_holds_the_decode_rate_up() {
        let (work, pll) = select_decode(25_000_000, 500_000.0, DemodulationMode::Pll);
        assert!(pll, "--demod pll must force it");
        assert!(
            work >= PLL_AUTO_MIN_SAMPLE_RATE_HZ,
            "forced PLL decoded at {work}, below the rate its loop needs"
        );
    }

    /// Below the rate threshold the deviation is irrelevant.
    #[test]
    fn auto_keeps_the_discriminator_below_the_rate_threshold() {
        assert!(!DemodulationMode::Auto.use_pll(15_360_000, 500_000.0));
    }

    /// The clamp is real: at a low rate the loop cannot reach its
    /// nominal bandwidth, so a deviation it could otherwise track is
    /// still out of reach.
    #[test]
    fn the_loop_bandwidth_clamp_is_respected() {
        // fs / 4pi at 25 MSPS is ~1.99 MHz, so PLL_LOOP_BW_HZ binds.
        assert!(!DemodulationMode::Auto.use_pll(25_000_000, 1_500_000.0));
    }

    /// Explicit flags keep meaning what they say.
    #[test]
    fn explicit_demod_choices_ignore_the_heuristic() {
        assert!(DemodulationMode::Pll.use_pll(15_360_000, 5_000_000.0));
        assert!(!DemodulationMode::Discriminator.use_pll(61_440_000, 100_000.0));
    }

    /// A degenerate deviation must not select the PLL by accident.
    #[test]
    fn a_degenerate_deviation_keeps_the_discriminator() {
        assert!(!DemodulationMode::Auto.use_pll(25_000_000, 0.0));
        assert!(!DemodulationMode::Auto.use_pll(25_000_000, -1.0));
        assert!(!DemodulationMode::Auto.use_pll(25_000_000, f32::NAN));
    }
}
