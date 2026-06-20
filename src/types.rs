use crate::bands::FpvChannel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub channel: Option<FpvChannel>,
    pub frequency_hz: u64,
    pub confidence: f32, // 0.0 to 1.0
    pub rssi_dbm: f32,
    pub bandwidth_hz: u32,
    pub signal_type: SignalType,
}

/// The enum is `#[non_exhaustive]` so future variants (e.g. digital
/// FPV protocols, RC telemetry tags) can be added without breaking
/// downstream matches. Callers must include a wildcard arm or
/// exhaustively match the variants they care about.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignalType {
    /// No identifiable signal — silence, noise, or an unrecognised
    /// pattern. `detect_from_iq` filters these out before returning.
    Unknown,
    /// Confirmed analog FM video with NTSC line/field rates
    /// (15734 Hz / 60 Hz).
    AnalogVideoNtsc,
    /// Confirmed analog FM video with PAL line/field rates
    /// (15625 Hz / 50 Hz).
    AnalogVideoPal,
    /// Analog FM video detected (strong H-sync line-rate energy in
    /// the FM-demodulated spectrum) but the FFT bin resolution was
    /// too coarse to discriminate the 15625 Hz / 15734 Hz line
    /// spacing — which happens whenever per-bin resolution is
    /// ≥ ~109 Hz. The caller knows analog FPV is present but should
    /// not act on a PAL-vs-NTSC tag. Earlier code silently defaulted
    /// to NTSC here, which produced a false standards tag whenever
    /// the FFT couldn't separate the two line rates.
    AnalogVideoUnknown,
    /// Wideband digital signal, likely Wi-Fi / Bluetooth / LTE.
    WidebandDigital,
    /// Narrowband interferer (CW tone, single-carrier emitter).
    NarrowbandInterference,
}

impl SignalType {
    /// `true` for any of `AnalogVideoNtsc` / `AnalogVideoPal` /
    /// `AnalogVideoUnknown`. Callers should use this when they want
    /// to gate on "is this an analog FPV detection?" rather than
    /// `!= Unknown`, which would also accept `WidebandDigital` and
    /// `NarrowbandInterference`.
    pub fn is_analog_video(self) -> bool {
        matches!(
            self,
            SignalType::AnalogVideoNtsc
                | SignalType::AnalogVideoPal
                | SignalType::AnalogVideoUnknown,
        )
    }
}
