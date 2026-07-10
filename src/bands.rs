use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FpvBand {
    BandA,
    BandB,
    BandE,
    Fatshark, // Often Band F
    Raceband, // Band R
    Lowband,  // Band L (5.3 - 5.6 GHz)
    Band1200, // 1.2GHz - 1.3GHz
    Band3300, // 3.3GHz - 4.8GHz
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

// Standard 48-channel VTX "Lowband" / "L" (Boscam/Aomway L band),
// 37 MHz spacing from 5362 MHz. This matches the table the
// `fpv_viewer` uses for channel naming and coarse-search snapping
// (`FPV_CHANNELS_MHZ` + `get_fpv_channel_name`), so resolving
// `--channel L1` here and labelling a detection there agree. An
// earlier 40 MHz-spaced 5333 MHz grid lived here and silently
// disagreed with the viewer (a tuned `L1` rendered as "Unknown").
pub const LOWBAND_FREQS: [u64; 8] = [
    5_362_000_000,
    5_399_000_000,
    5_436_000_000,
    5_473_000_000,
    5_510_000_000,
    5_547_000_000,
    5_584_000_000,
    5_621_000_000,
];

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

/// 3.3 GHz to 4.8 GHz Band (64 Channels)
pub fn get_3300_freqs() -> Vec<u64> {
    let mut freqs = Vec::new();
    for i in 0..64 {
        freqs.push(3_300_000_000 + (i as u64 * 25_000_000));
    }
    freqs
}

pub fn get_all_channels() -> Vec<FpvChannel> {
    let mut channels = Vec::new();

    for (i, &f) in RACEBAND_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::Raceband,
            channel: (i + 1) as u8,
            frequency_hz: f,
        });
    }
    for (i, &f) in FATSHARK_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::Fatshark,
            channel: (i + 1) as u8,
            frequency_hz: f,
        });
    }
    for (i, &f) in BAND_A_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::BandA,
            channel: (i + 1) as u8,
            frequency_hz: f,
        });
    }
    for (i, &f) in BAND_B_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::BandB,
            channel: (i + 1) as u8,
            frequency_hz: f,
        });
    }
    for (i, &f) in BAND_E_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::BandE,
            channel: (i + 1) as u8,
            frequency_hz: f,
        });
    }
    for (i, &f) in LOWBAND_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::Lowband,
            channel: (i + 1) as u8,
            frequency_hz: f,
        });
    }
    for (i, &f) in BAND_1200_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::Band1200,
            channel: (i + 1) as u8,
            frequency_hz: f,
        });
    }

    // Add 3.3GHz band
    for (i, f) in get_3300_freqs().into_iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::Band3300,
            channel: (i + 1) as u8,
            frequency_hz: f,
        });
    }

    channels
}

/// Resolve a channel name (case-insensitive) to its centre frequency in Hz.
///
/// Accepted formats: `A1`–`A8`, `B1`–`B8`, `E1`–`E8`, `F1`–`F8`,
/// `R1`–`R8`, `L1`–`L8`. Returns `None` for unrecognised names.
///
/// This is the inverse of the `get_fpv_channel_name` lookup in
/// `fpv_viewer.rs` — but lives in the library crate so both the
/// viewer and the main orchestrator can use it.
pub fn lookup_channel_by_name(name: &str) -> Option<u64> {
    let name = name.trim().to_uppercase();
    let mut chars = name.chars();
    let first_char = chars.next()?;
    let channel_str: String = chars.collect();
    if !first_char.is_ascii() {
        return None;
    }
    let band_char = first_char as u8;
    let channel_num: usize = channel_str.parse().ok()?;
    if !(1..=8).contains(&channel_num) {
        return None;
    }
    let idx = channel_num - 1;
    match band_char {
        b'A' => BAND_A_FREQS.get(idx).copied(),
        b'B' => BAND_B_FREQS.get(idx).copied(),
        b'E' => BAND_E_FREQS.get(idx).copied(),
        b'F' => FATSHARK_FREQS.get(idx).copied(),
        b'R' => RACEBAND_FREQS.get(idx).copied(),
        b'L' => LOWBAND_FREQS.get(idx).copied(),
        _ => None,
    }
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
        assert_eq!(lookup_channel_by_name("L1"), Some(5_362_000_000));
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
    fn all_channels_non_empty() {
        let channels = get_all_channels();
        assert!(!channels.is_empty());
        // Every channel should have a non-zero frequency
        for ch in &channels {
            assert!(ch.frequency_hz > 0);
        }
    }
}
