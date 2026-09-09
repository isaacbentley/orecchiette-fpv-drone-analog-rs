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

// Standard 48/56-channel VTX "Lowband" / "L" grid: 40 MHz spacing
// from 5333 MHz — the table real L-band hardware (Boscam/Aomway,
// Eachine, TBS 56CH units) transmits on.
//
// History: this array briefly held the 37 MHz-spaced 5362 grid to
// match the viewer's channel table, but that grid is the *Boscam D
// band* ([`BAND_D_FREQS`] below), not L — the reconciliation went in
// the wrong direction and made `--channel L1` tune 5362 MHz while a
// VTX set to L1 transmits at 5333 MHz, 29 MHz off. Both grids are now
// modelled, under their correct names, here and in fpv-viewer-rs's
// `FPV_CHANNELS_MHZ` / `get_fpv_channel_name` — keep the two repos'
// tables in lockstep when editing either.
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
/// the frequencies are therefore *not* monotonic. fpv-viewer-rs labels
/// CH1–CH5 "1.2G" and CH6–CH9 "1.3G" in `get_fpv_channel_name`; keep
/// its `FPV_CHANNELS_MHZ` and this array in lockstep when editing
/// either.
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
    for (i, &f) in BAND_D_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::BandD,
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
    for (i, &f) in BAND_1200_WIDE_FREQS.iter().enumerate() {
        channels.push(FpvChannel {
            band: FpvBand::Band1200Wide,
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
/// `R1`–`R8`, `L1`–`L8`, `D1`–`D8`. Returns `None` for unrecognised
/// names.
///
/// This is the inverse of the `get_fpv_channel_name` lookup in
/// fpv-viewer-rs's `src/main.rs` — but lives in the library crate so
/// both the viewer and the main orchestrator can use it.
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
        b'D' => BAND_D_FREQS.get(idx).copied(),
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
