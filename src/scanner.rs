use crate::bands::{FpvChannel, get_all_channels};
use crate::detector::FpvDetector;
use crate::types::DetectionResult;
use num_complex::Complex;

#[allow(dead_code)]
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
