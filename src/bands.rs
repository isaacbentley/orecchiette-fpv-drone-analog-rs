use serde::{Deserialize, Serialize};

/// `#[non_exhaustive]` for the same reason as [`crate::types::SignalType`]:
/// band plans keep growing (BandD was added after 0.1.8), and downstream
/// exhaustive matches shouldn't break every time one lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FpvBand {
    BandA,
    BandB,
    BandE,
    Fatshark, // Often Band F
    Raceband, // Band R
    Lowband,  // Band L (5333-5613 MHz, 40 MHz grid)
    BandD,    // Boscam D / "5.3G" (5362-5621 MHz, 37 MHz grid)
    Band1200, // 1.2 GHz amateur / SM1370R 8-channel set (1240-1300 MHz)
    Band3300, // 3.3GHz - 4.875GHz
    /// The 9-channel "1.2G/1.3G" long-range VTX grid (1080-1360 MHz,
    /// 40 MHz spacing plus CH9 at 1258). See [`BAND_1200_WIDE_FREQS`].
    Band1200Wide,
    UltraLow,
    Band2400,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FpvChannel {
    pub band: FpvBand,
    pub channel: u8,
    pub frequency_hz: u64,
}

pub const RACEBAND_FREQS: [u64; 8] = [
    5_658_000_000,
    5_695_000_000,
    5_732_000_000,
    5_769_000_000,
    5_806_000_000,
    5_843_000_000,
    5_880_000_000,
    5_917_000_000,
];

pub const FATSHARK_FREQS: [u64; 8] = [
    5_740_000_000,
    5_760_000_000,
    5_780_000_000,
    5_800_000_000,
    5_820_000_000,
    5_840_000_000,
    5_860_000_000,
    5_880_000_000,
];

pub const BAND_A_FREQS: [u64; 8] = [
    5_865_000_000,
    5_845_000_000,
    5_825_000_000,
    5_805_000_000,
    5_785_000_000,
    5_765_000_000,
    5_745_000_000,
    5_725_000_000,
];

pub const BAND_B_FREQS: [u64; 8] = [
    5_733_000_000,
    5_752_000_000,
    5_771_000_000,
    5_790_000_000,
    5_809_000_000,
    5_828_000_000,
    5_847_000_000,
    5_866_000_000,
];

pub const BAND_E_FREQS: [u64; 8] = [
    5_705_000_000,
    5_685_000_000,
    5_665_000_000,
    5_645_000_000,
    5_885_000_000,
    5_905_000_000,
    5_925_000_000,
    5_945_000_000,
];

// Standard 48/56-channel VTX "Lowband" / "L" grid: 40 MHz spacing
// from 5333 MHz — the table real L-band hardware (Boscam/Aomway,
// Eachine, TBS 56CH units) transmits on.
//
// History: this array briefly held the 37 MHz-spaced 5362 grid to
// match the viewer's channel table, but that grid is the *Boscam D
// band* ([`BAND_D_FREQS`] below), not L — the reconciliation went in
// the wrong direction and made `--channel L1` tune 5362 MHz while a
// VTX set to L1 transmits at 5333 MHz, 29 MHz off. Both grids are now
// modelled under their correct names in this shared catalog.
pub const LOWBAND_FREQS: [u64; 8] = [
    5_333_000_000,
    5_373_000_000,
    5_413_000_000,
    5_453_000_000,
    5_493_000_000,
    5_533_000_000,
    5_573_000_000,
    5_613_000_000,
];

/// Boscam D band ("5.3G"): 37 MHz spacing from 5362 MHz, as found on
/// Eachine Pro58-class receivers and many multi-band VTXs' "D/5.3"
/// setting. Distinct from [`LOWBAND_FREQS`] — see the note there.
pub const BAND_D_FREQS: [u64; 8] = [
    5_362_000_000,
    5_399_000_000,
    5_436_000_000,
    5_473_000_000,
    5_510_000_000,
    5_547_000_000,
    5_584_000_000,
    5_621_000_000,
];

/// 1.2 GHz, the narrow amateur set: the 8 channels a SM1370R / VM1373R
/// receiver module steps through (1240–1300 MHz in 6 / 12 MHz steps —
/// the 23 cm ATV allocation). This is the table the cheap ESP32
/// "multi-band detector" builds use for their 1.2 GHz stage.
///
/// It is *not* the grid most long-range 1.2 GHz FPV VTXs transmit on —
/// that is [`BAND_1200_WIDE_FREQS`]. Both are scanned; they overlap at
/// 1240 and 1258 MHz and the orchestrator dedups by kHz.
pub const BAND_1200_FREQS: [u64; 8] = [
    1_240_000_000,
    1_246_000_000,
    1_258_000_000,
    1_264_000_000,
    1_276_000_000,
    1_282_000_000,
    1_294_000_000,
    1_300_000_000,
];

/// 1.2 GHz, the wide grid: the 9-channel "1.2G/1.3G" table the common
/// long-range analog VTX / RX modules (the 1.2G 9CH / 12CH units) use —
/// CH1–CH8 at 40 MHz spacing from 1080 MHz, plus CH9 at 1258 MHz.
/// Indexed in that CH order so `channel` matches the VTX's own menu;
/// the frequencies are therefore not monotonic.
pub const BAND_1200_WIDE_FREQS: [u64; 9] = [
    1_080_000_000,
    1_120_000_000,
    1_160_000_000,
    1_200_000_000,
    1_240_000_000,
    1_280_000_000,
    1_320_000_000,
    1_360_000_000,
    1_258_000_000,
];

/// 3.3 GHz band: 64 channels at 25 MHz spacing, 3300–4875 MHz.
pub fn get_3300_freqs() -> Vec<u64> {
    let mut freqs = Vec::new();
    for i in 0..64 {
        freqs.push(3_300_000_000 + (i as u64 * 25_000_000));
    }
    freqs
}

/// Ultra-low grid previously supported only by the viewer.
pub const ULTRA_LOW_FREQS: [u64; 8] = [
    5_300_000_000,
    5_325_000_000,
    5_348_000_000,
    5_373_000_000,
    5_398_000_000,
    5_423_000_000,
    5_448_000_000,
    5_473_000_000,
];
/// Nine-channel 2.4 GHz video grid. CH3 and CH8 share a carrier.
pub const BAND_2400_FREQS: [u64; 9] = [
    2_414_000_000,
    2_432_000_000,
    2_450_000_000,
    2_468_000_000,
    2_490_000_000,
    2_410_000_000,
    2_430_000_000,
    2_450_000_000,
    2_470_000_000,
];

impl FpvBand {
    /// Unambiguous short code used in channel identifiers.
    pub fn code(self) -> char {
        match self {
            Self::BandA => 'A',
            Self::BandB => 'B',
            Self::BandE => 'E',
            Self::Fatshark => 'F',
            Self::Raceband => 'R',
            Self::Lowband => 'L',
            Self::BandD => 'D',
            Self::UltraLow => 'U',
            Self::Band1200 => 'N',
            Self::Band1200Wide => 'W',
            Self::Band3300 => 'S',
            Self::Band2400 => 'T',
        }
    }
}
impl FpvChannel {
    pub fn name(&self) -> String {
        format!("{}{}", self.band.code(), self.channel)
    }
    pub fn display_name(&self) -> String {
        match self.band {
            FpvBand::Band1200Wide => format!(
                "1.{}G Ch{}",
                if self.channel <= 5 { 2 } else { 3 },
                self.channel
            ),
            FpvBand::Band2400 => format!("2.4G Ch{}", self.channel),
            FpvBand::Band3300 => format!("3.3G Ch{}", self.channel),
            _ => self.name(),
        }
    }
}

/// Shared catalog used by tuning, scanning, candidate matching and labels.
/// Aliases remain distinct entries; use candidate frequencies to deduplicate.
pub fn channel_catalog() -> &'static [FpvChannel] {
    static CATALOG: std::sync::OnceLock<Vec<FpvChannel>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut channels = Vec::new();
        let grid3300 = get_3300_freqs();
        for (band, frequencies) in [
            (FpvBand::BandA, BAND_A_FREQS.as_slice()),
            (FpvBand::BandB, &BAND_B_FREQS),
            (FpvBand::BandE, &BAND_E_FREQS),
            (FpvBand::Fatshark, &FATSHARK_FREQS),
            (FpvBand::Raceband, &RACEBAND_FREQS),
            (FpvBand::BandD, &BAND_D_FREQS),
            (FpvBand::Lowband, &LOWBAND_FREQS),
            (FpvBand::UltraLow, &ULTRA_LOW_FREQS),
            (FpvBand::Band1200Wide, &BAND_1200_WIDE_FREQS),
            (FpvBand::Band1200, &BAND_1200_FREQS),
            (FpvBand::Band2400, &BAND_2400_FREQS),
            (FpvBand::Band3300, grid3300.as_slice()),
        ] {
            channels.extend(
                frequencies
                    .iter()
                    .enumerate()
                    .map(|(i, &frequency_hz)| FpvChannel {
                        band,
                        channel: (i + 1) as u8,
                        frequency_hz,
                    }),
            );
        }
        channels
    })
}
pub fn get_all_channels() -> Vec<FpvChannel> {
    channel_catalog().to_vec()
}

/// Shared scan coverage selection, independent of CLI argument parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandSelection {
    Band58,
    All,
}
impl BandSelection {
    pub fn channels(self) -> impl Iterator<Item = &'static FpvChannel> {
        channel_catalog().iter().filter(move |c| {
            self == Self::All || (5_645_000_000..=5_945_000_000).contains(&c.frequency_hz)
        })
    }
}

/// Resolve case-insensitive A/B/E/F/R/L/D/U1–8, N1–8 (narrow 1.2 GHz),
/// W1–9 (wide 1.2 GHz), T1–9 (2.4 GHz), or S1–64 (3.3 GHz).
pub fn lookup_channel_by_name(name: &str) -> Option<u64> {
    let name = name.trim().to_ascii_uppercase();
    let mut chars = name.chars();
    let code = chars.next()?;
    let number = chars.as_str().parse::<u8>().ok()?;
    channel_catalog()
        .iter()
        .find(|c| c.band.code() == code && c.channel == number)
        .map(|c| c.frequency_hz)
}

/// Exact carrier label, including all aliases sharing that frequency.
pub fn channel_name(frequency_hz: u64) -> Option<&'static str> {
    static LABELS: std::sync::OnceLock<Vec<(u64, String)>> = std::sync::OnceLock::new();
    LABELS
        .get_or_init(|| {
            let mut labels: Vec<(u64, String)> = Vec::new();
            // Keep familiar shared labels R7/F8 and U4/L2 in that order.
            let mut channels = get_all_channels();
            channels.sort_by_key(|c| match c.band {
                FpvBand::Raceband | FpvBand::UltraLow => 0,
                _ => 1,
            });
            for c in channels {
                if let Some((_, label)) = labels.iter_mut().find(|(f, _)| *f == c.frequency_hz) {
                    label.push('/');
                    label.push_str(&c.display_name());
                } else {
                    labels.push((c.frequency_hz, c.display_name()));
                }
            }
            labels
        })
        .iter()
        .find(|(f, _)| *f == frequency_hz)
        .map(|(_, name)| name.as_str())
}

/// Label after rounding a display frequency to the nearest MHz.
pub fn get_fpv_channel_name(freq_mhz: f64) -> Option<&'static str> {
    if !freq_mhz.is_finite() || freq_mhz < 0.0 {
        return None;
    }
    channel_name((freq_mhz.round() * 1e6) as u64)
}

pub const CHANNEL_SNAP_TOLERANCE_MHZ: f64 = 15.0;
/// Candidate carriers in Hz, ordered by distance, with aliases deduplicated.
/// The tolerance is inclusive; equal distances retain catalog order.
pub fn candidate_frequencies(freq_hz: f64, tolerance_hz: f64) -> Vec<f64> {
    if !freq_hz.is_finite() || !tolerance_hz.is_finite() || tolerance_hz < 0.0 {
        return Vec::new();
    }
    let mut candidates: Vec<f64> = channel_catalog()
        .iter()
        .map(|c| c.frequency_hz as f64)
        .filter(|f| (f - freq_hz).abs() <= tolerance_hz)
        .collect();
    candidates.sort_by(|a, b| (a - freq_hz).abs().total_cmp(&(b - freq_hz).abs()));
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|f| seen.insert(*f as u64));
    candidates
}
pub fn get_candidate_fpv_channels(freq_hz: f64) -> Vec<f64> {
    candidate_frequencies(freq_hz, CHANNEL_SNAP_TOLERANCE_MHZ * 1e6)
}
/// Preserve an off-table measurement; snap only strictly inside the tolerance.
/// This is channel identification, not a recommendation to retune the decoder.
pub fn snap_to_nearest_fpv_channel(freq_hz: f64) -> f64 {
    get_candidate_fpv_channels(freq_hz)
        .first()
        .copied()
        .filter(|f| (f - freq_hz).abs() < CHANNEL_SNAP_TOLERANCE_MHZ * 1e6)
        .unwrap_or(freq_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_known_channels() {
        assert_eq!(lookup_channel_by_name("A1"), Some(5_865_000_000));
        assert_eq!(lookup_channel_by_name("a1"), Some(5_865_000_000));
        assert_eq!(lookup_channel_by_name("A8"), Some(5_725_000_000));
        assert_eq!(lookup_channel_by_name("R1"), Some(5_658_000_000));
        assert_eq!(lookup_channel_by_name("R8"), Some(5_917_000_000));
        assert_eq!(lookup_channel_by_name("F1"), Some(5_740_000_000));
        assert_eq!(lookup_channel_by_name("F8"), Some(5_880_000_000));
        assert_eq!(lookup_channel_by_name("B4"), Some(5_790_000_000));
        assert_eq!(lookup_channel_by_name("E1"), Some(5_705_000_000));
        assert_eq!(lookup_channel_by_name("L1"), Some(5_333_000_000));
        assert_eq!(lookup_channel_by_name("L8"), Some(5_613_000_000));
        assert_eq!(lookup_channel_by_name("D1"), Some(5_362_000_000));
        assert_eq!(lookup_channel_by_name("D8"), Some(5_621_000_000));
    }

    #[test]
    fn lookup_invalid_channels() {
        assert_eq!(lookup_channel_by_name("A0"), None);
        assert_eq!(lookup_channel_by_name("A9"), None);
        assert_eq!(lookup_channel_by_name("X1"), None);
        assert_eq!(lookup_channel_by_name(""), None);
        assert_eq!(lookup_channel_by_name("A"), None);
        assert_eq!(lookup_channel_by_name("1A"), None);
    }

    #[test]
    fn lookup_non_ascii_safety() {
        assert_eq!(lookup_channel_by_name("你好"), None);
        assert_eq!(lookup_channel_by_name("A\u{301}"), None);
        assert_eq!(lookup_channel_by_name("東1"), None);
    }

    #[test]
    fn lookup_whitespace_tolerance() {
        assert_eq!(lookup_channel_by_name(" A1 "), Some(5_865_000_000));
        assert_eq!(lookup_channel_by_name("  r4  "), Some(5_769_000_000));
    }

    #[test]
    fn wide_1200_grid_matches_the_vtx_menu() {
        // CH1-CH8 step 40 MHz from 1080; CH9 is the odd 1258 slot.
        for (i, w) in BAND_1200_WIDE_FREQS[..8].windows(2).enumerate() {
            assert_eq!(w[1] - w[0], 40_000_000, "CH{} -> CH{}", i + 1, i + 2);
        }
        assert_eq!(BAND_1200_WIDE_FREQS[0], 1_080_000_000);
        assert_eq!(BAND_1200_WIDE_FREQS[7], 1_360_000_000);
        assert_eq!(BAND_1200_WIDE_FREQS[8], 1_258_000_000);
        let all = get_all_channels();
        let wide: Vec<_> = all
            .iter()
            .filter(|c| c.band == FpvBand::Band1200Wide)
            .collect();
        assert_eq!(wide.len(), 9);
        assert_eq!(wide[8].channel, 9);
        assert_eq!(wide[8].frequency_hz, 1_258_000_000);
        // The scan plan now reaches the low end of the long-range band.
        assert!(all.iter().any(|c| c.frequency_hz == 1_080_000_000));
    }

    #[test]
    fn all_channels_non_empty() {
        let channels = get_all_channels();
        assert!(!channels.is_empty());
        // Every channel should have a non-zero frequency
        for ch in &channels {
            assert!(ch.frequency_hz > 0);
        }
    }
}
