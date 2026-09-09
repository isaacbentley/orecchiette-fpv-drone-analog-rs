//! Realistic RF impairment models for weak-signal testing.

use num_complex::Complex;

/// Adds multipath fading (echoes) to the signal.
/// `echoes` is a list of (delay_samples, attenuation_factor, initial_phase_radians, doppler_hz).
pub fn add_multipath(iq: &mut [Complex<f32>], echoes: &[(usize, f32, f32, f32)], sample_rate: f32) {
    // We must work on a copy since echoes depend on older samples
    let original = iq.to_vec();
    for &(delay, atten, initial_phase, doppler_hz) in echoes {
        let omega = 2.0 * std::f32::consts::PI * doppler_hz / sample_rate;
        for i in delay..iq.len() {
            let current_phase = initial_phase + omega * i as f32;
            let p = Complex::new(current_phase.cos(), current_phase.sin());
            iq[i] += original[i - delay] * atten * p;
        }
    }
}

/// Clamps the signal to pure AWGN (burst dropout) for the specified runs.
/// `runs` is a list of (start_index, length).
pub fn add_burst_dropouts(
    iq: &mut [Complex<f32>],
    runs: &[(usize, usize)],
    sigma: f32,
    seed: &mut u64,
) {
    for &(start, len) in runs {
        let end = (start + len).min(iq.len());
        for z in &mut iq[start..end] {
            *z = Complex::new(
                sigma * crate::synthetic::gaussian_noise(seed),
                sigma * crate::synthetic::gaussian_noise(seed),
            );
        }
    }
}

/// Applies a slow fade envelope (e.g. sinusoidal variation in amplitude).
/// `frequency_hz` is the fade frequency, `depth` is the fade depth (0.0 to 1.0).
pub fn add_slow_fade(iq: &mut [Complex<f32>], sample_rate: f32, frequency_hz: f32, depth: f32) {
    let omega = 2.0 * std::f32::consts::PI * frequency_hz / sample_rate;
    for (i, z) in iq.iter_mut().enumerate() {
        let envelope = 1.0 - depth * (1.0 - (omega * i as f32).cos()) * 0.5;
        *z *= envelope;
    }
}

/// Randomly injects high-amplitude spikes to simulate motor/ESC spark noise.
/// `probability` (0.0 to 1.0) is the chance per sample of a spike occurring.
/// `amplitude` is the maximum magnitude of the spike.
pub fn add_impulsive_noise(
    iq: &mut [Complex<f32>],
    probability: f32,
    amplitude: f32,
    seed: &mut u64,
) {
    for z in iq.iter_mut() {
        let rand_val = ((crate::synthetic::gaussian_noise(seed) + 6.0) / 12.0).clamp(0.0, 1.0);
        if rand_val < probability {
            let phase = crate::synthetic::gaussian_noise(seed) * std::f32::consts::PI;
            z.re += amplitude * phase.cos();
            z.im += amplitude * phase.sin();
        }
    }
}
