# Orecchiette Workflow: IQ Capture to Analog FPV Video

This document provides a step-by-step technical walkthrough of how Orecchiette detects analog FPV video signals from raw IQ captures.

---

## Phase 1: Hardware Capture & Scanning (`../../deps/sdr-*-rs` + `../../src/main.rs`)

Phase 1 handles the wideband IQ capture. The hardware-specific code lives in dedicated `SdrSource` implementation crates (`sdr-usrp-rs`, `sdr-aaronia-rs`, `sdr-file-rs`); the orchestrator (`src/main.rs`) selects a backend at runtime and consumes the same `IqPacket` stream regardless of which one is feeding it.

1.  **Multi-Band Orchestration & Auto-Scanning**:
    - **Frequency Pool**: The system can scan a consolidated list of 100+ channels, including Band A, B, E (Boscam), F (FatShark), R (RaceBand), L (LowBand), and the 1.2GHz/3.3GHz ranges.
    - **Tuning**: USRP (`sdr-usrp-rs`) or Aaronia (`sdr-aaronia-rs`) hardware is tuned to each center frequency. For wideband 5.8 GHz captures, **100 MSPS** is used to cover the entire band in a single capture.
    - **Auto-Scanner & Fine-Tuning**: For SDRs with smaller bandwidths (e.g. 25 MSPS), a state machine continuously sweeps the bands. Once a signal is found, it automatically stops scanning and transitions to a fine-tuning mode to snap precisely to the center channel. If the signal is lost for more than 2 seconds, the scanner automatically resumes sweeping.
    - **Scan Modes**: The `fpv_viewer` supports two scan modes:
      - `--scan-mode 58` (default): standard 5.8 GHz FPV band (5.645–5.945 GHz).
      - `--scan-mode ua`: Ukraine theatre — scans all confirmed analog video TX bands (1.2 GHz, 3.3 GHz, 5.3–5.9 GHz, 6–7 GHz) modelled after the Chuyka 3.0 detector and the PEAK THOR T67 VTX evasion band.
    - **Optimized Dwell**: The scan dwell is **10 ms per hop** — just enough for the USRP PLL to settle (~2 ms) and deliver one full 65536-sample chunk (~2.6 ms at 25 MSPS). The detector only needs a single chunk per hop to run the wideband DDC probe sweep. All remaining duplicate-frequency packets are skipped to prevent queue backlog.
    - **No Power Gating**: To maximize sensitivity for weak drone signals, all sync pulse correlation runs regardless of the raw RSSI power level. This guarantees that faint signals below the noise floor are still processed and detected.

2.  **Zero-Allocation Pipeline**:
    - Raw IQ samples are streamed into pre-allocated buffers managed by the backend.
    - Hand-off occurs via lock-free `crossbeam::channel`s from the backend's capture thread to orecchiette's worker pool.
    - **Overrun Protection**: If the downstream pipeline cannot keep up with the SDR stream, frames are dynamically dropped at the dispatcher. In cases of persistent hardware overruns (e.g. `ReceiveErrorKind::Overflow` from UHD), the backend tracks this and automatically steps down the sample rate by 5 MHz (up to 2 times a minute) to restore stability.
    - **Scan-loop backpressure**: The `fpv_viewer` scan loop processes only one packet per frequency hop and skips all remaining packets at the same center frequency. This prevents queue buildup when the detector's ~50 ms DDC sweep can't keep up with the 2.6 ms packet rate.
    - **B210 sample rate limit**: The B210 over USB 3.0 runs clean at 25 MSPS. At 50 MSPS the USB transport saturates (~400 MB/s sustained), producing intermittent hardware FIFO overflows — the scanner still sweeps but some samples are dropped. 25 MSPS is recommended for clean operation.

---

## Phase 2: Sliding DDC Probe Detection (`detector.rs`)

The detector sweeps the entire capture bandwidth to find analog video signals at **any** frequency — no predefined channel table needed.

3.  **Probe Grid Setup**:
    - The bandwidth is divided into 5 MHz steps (e.g., 100 MSPS → ~18 probes).
    - A 5 MHz margin is left at each edge to avoid filter rolloff artifacts.

4.  **DDC + Decimation**:
    - At each probe position, a **Digital Down-Converter** (NCO mixer) shifts the probe frequency to DC.
    - A **63-tap Blackman-windowed-sinc FIR** (> 50 dB stopband) band-limits the signal, then integer-stride decimation drops the rate to 10 MSPS. Cutoff sits at `target_rate / 3` (≈ 3.33 MHz). Nyquist (`target_rate / 2`) leaves band-edge signals at −6 dB which leaks click-noise into the harmonic-consistency check on adjacent probes; pushing tighter to `target_rate / 4 = step/2` opens a detection blind spot at probe-boundary frequencies (−6 dB on both adjacent probes). `target_rate / 3` clears both: adjacent-probe contamination at 5 MHz off lands > 1 MHz into the FIR stopband for > 50 dB rejection, while probe-boundary signals (2.5 MHz off) sit well inside the passband.

5.  **Energy Gating**:
    - Mean power (|I²+Q²|) is computed at each probe position.
    - The **25th percentile** of all probe energies serves as a robust noise floor estimate.
    - Only probes exceeding the noise floor by ≥ 3 dB proceed to sync validation.

---

## Phase 3: FM Demodulation & Sync Detection (`detector.rs`)

6.  **FM Demodulation**:
    - The isolated I/Q at each probe position is FM-demodulated: `arg(z[n] × conj(z[n-1]))`.
    - This recovers the baseband video signal where horizontal sync pulses are encoded as instantaneous frequency excursions.
    - **Why FM, not AM?** Analog FPV video uses FM — it's a constant-envelope modulation. Magnitude-based analysis (|I+jQ|) sees a flat signal and misses the sync information entirely.

7.  **H-Sync Rate Detection (FFT)**:
    - The demodulated signal is DC-blocked (mean subtracted) and Hann-windowed.
    - A single-pass FFT searches for spectral peaks at the known H-sync line rates:
      - **PAL**: 15,625 Hz
      - **NTSC**: 15,734 Hz
    - When FFT resolution is sufficient (bin_hz < 109 Hz, requiring > 9.2 ms of data), the detector classifies PAL vs NTSC straight from the spectrum.
    - At coarse resolution (e.g., 2.6 ms at 100 MSPS, or a 65 k chunk at 25 MSPS ≈ 381 Hz/bin), PAL and NTSC bins collide. The detector then falls back to a **time-domain median sync-tip interval** (`classify_pal_ntsc_time_domain`): it counts sync tips on the demodulated record, takes the median line period, and maps it to a line rate compared against 15625/15734 Hz with a ±30 Hz midpoint dead-band — classifying PAL/NTSC at confidence 0.8. Only if that's inconclusive (too few tips, or median in the dead-band) does it tag `AnalogVideoUnknown` at 0.6 (rather than silently picking one standard). `SignalType::is_analog_video()` returns `true` in all three cases, so downstream consumers gating on "analog FPV present" still see the hit.

8.  **Harmonic-Consistency Check**:
    - H-sync is a ~7% duty-cycle rectangular pulse train, so its FM-demodulated spectrum has the fundamental at the line rate plus a rich harmonic series — for a 7% duty train the first ~14 harmonics are within roughly −3 dB of the fundamental.
    - The detector counts how many of the first 5 harmonics (k = 2..=5) exceed 10% of the fundamental amplitude (and also exceed the weak noise-floor threshold). At least 2 harmonics are required for a positive classification.
    - Threshold is fundamental-relative (not noise-floor-relative) because spectral leakage from a strong fundamental otherwise pulls the noise floor estimate down enough that any FFT-window sidelobe at 2× the fundamental crosses a noise-floor-relative threshold.
    - This rejects narrowband-FM tones and CW interferers that happen to land in the H-sync bin — they have no harmonic structure and fail the count.

9.  **V-Sync Cross-Check**:
    - For borderline cases, the detector checks for the vertical sync rate:
      - **PAL**: 50 Hz
      - **NTSC**: ~59.94 Hz
    - V-sync confirmation raises confidence to 0.6 (only reachable at `bin_hz < 10`, i.e. > 100 ms capture window).

---

## Phase 4: Clustering & Deduplication (`detector.rs`)

10. **Signal Clustering**:
    - All positive detections from the probe sweep are sorted by frequency.
    - Detections within **25 MHz** of each other are grouped into a single cluster.
    - The probe with the **strongest energy** in each cluster becomes the representative detection.
    - This collapses the ~4-5 probes that hit the same ~20 MHz FM signal into a single clean result.

11. **Final Dedup**:
    - A second dedup pass merges any remaining overlapping results from different detection paths.

---

## Phase 5: FM Demodulation for Video Recovery (`demod.rs`)

Once a signal is detected, full FM demodulation recovers the video content:

12. **Quadrature Demodulation**:
    - **The Math**: `arg(iq[n] × conj(iq[n-1]))` — phase difference between consecutive samples.
    - **Implementation**: exact scalar `f32::atan2` (via `Complex::arg`) per sample. The complex multiply + `conj` ahead of the atan2 auto-vectorises cleanly under `-O3` (it's the bulk of the work). The atan2 itself is intentionally NOT replaced with a polynomial approximation — `fast_math::atan2` was tried for edge-device throughput and reverted because image quality wins here: approximate kernels lose precision near ±π, exactly where high-deviation FM operates, and the resulting quadrant errors show up as click-noise sparkles in the reconstructed picture.
    - **Output**: A 1D stream of floating-point values representing the instantaneous frequency (brightness) of the video signal.

---

## Phase 6: Video Frame Reconstruction (`video.rs` + `frame_history.rs`)

This phase turns the 1D frequency stream into a 2D image. Output is
monochrome (luma-only) — colour recovery is currently disabled because
analog FPV chroma bursts on real-world links arrive with σ ≈ 165° of
per-line phase noise, which is below the lock floor of the chroma PLL
that previously drove the colour path. See DESIGN.md §5 for the
empirical justification.

13. **Two-Pass Sync-Tip Alignment**:
    - Pass 1 detects every sync tip — points where the demodulated
      frequency drops below an adaptive threshold — and builds a list of
      raw tip positions.
    - Pass 2 walks the raw list and rejects outliers via a Median +
      MAD (Median Absolute Deviation) test. Surviving tips drive a
      sub-sample TBC via Catmull-Rom cubic interpolation; rejected
      slots are recorded so the dropout-repair stage knows how trusted
      the field is.
    - **Sync-quality score**: `valid_slots / total_slots` is exposed
      via `FrameReconstructor::latest_sync_quality()` and saved into
      the per-field [`FieldMeta`] in the history buffer. A score
      below `DROPOUT_THRESHOLD` (0.5) forces the temporal denoise
      into "static" mode for that field (full blend toward recent
      history), preferring a recently-good frame to current FM static.

14. **2D Mapping & Rescaling**:
    - Samples between H-Sync pulses form a single line of pixels.
    - **Normalization**: Raw FM deviations are mapped to 8-bit grayscale (0–255).
    - **Geometry**: Lines are stacked into frames (720×576 for PAL, 720×480 for NTSC).

14a. **Multi-Field Temporal Denoise** (`frame_history.rs`):
    - Each rendered field is pushed into a fixed-capacity ring buffer
      of recent Y fields (default 5, configurable via
      `FrameReconstructor::with_temporal_window(N)` or the CLI flag
      `--temporal-window N` on `fpv_viewer`).
    - Per-pixel: collect the value from the current field and every
      retained history field; compute the median (kills FM "click"
      sparkles) and the max-absolute motion across history; blend
      `cur` toward `median` by `1 - motion_weight`, where
      `motion_weight ∈ [0, 1]` saturates at `TEMPORAL_MOTION_THRESHOLD`
      (0.10 of full radian-per-volt swing).
    - Static pixels recover ≈ √N noise reduction; moving pixels fall
      back to the current field unblended. Field-parity is preserved
      because the history is keyed per-field, not per-frame.
    - Setting `with_temporal_window(1)` disables denoise — useful for
      batch-mode callers that want single-frame fidelity over noise
      reduction.

---

## Phase 7: Reporting (`main.rs`)

15. **Multi-Signal Reporting**:
    - The orchestrator iterates over **all** detected signals (not just the first).
    - Each signal gets its own frequency-keyed dedup slot (1-second cooldown).
    - A JSON detection record is emitted to `stdout` for each unique signal per second.

16. **JSON Telemetry**:
    ```json
    {
      "frequency_hz": 5775000000.0,
      "protocol": "Analog FPV (NTSC)",
      "source": "drone",
      "rssi_relative_db": -4.3,
      "confidence": 0.8,
      "bandwidth_mhz": "10.0",
      "video_standard": "NTSC",
      "timestamp": 1779763967
    }
    ```
