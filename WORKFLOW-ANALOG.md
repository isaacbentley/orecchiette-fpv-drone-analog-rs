# Orecchiette Workflow: IQ Capture to Analog FPV Video

This document provides a step-by-step technical walkthrough of how Orecchiette detects analog FPV video signals from raw IQ captures.

---

## Phase 1: Hardware Capture & Scanning (`orecchiette-sdr-*-rs` crates + fpv-viewer-rs)

Phase 1 handles the wideband IQ capture. The hardware-specific code lives in dedicated `SdrSource` implementation crates (`orecchiette-sdr-usrp-rs`, `orecchiette-sdr-hackrf-rs`, `sdr-aaronia-rs`, `orecchiette-sdr-file-rs`); the consumer — fpv-viewer-rs's `src/main.rs` — selects a backend at runtime and consumes the same `IqPacket` stream regardless of which one is feeding it.

1.  **Multi-Band Orchestration & Auto-Scanning**:
    - **Frequency Pool**: The system can scan a consolidated list of channels, including Band A, B, E (Boscam), F (FatShark), R (RaceBand), L (LowBand), and D (Boscam D / "5.3G").
    - **Tuning**: USRP (`orecchiette-sdr-usrp-rs`), HackRF (`orecchiette-sdr-hackrf-rs`), or Aaronia (`sdr-aaronia-rs`) hardware is tuned to each center frequency. Typical capture rates are 25 MSPS (USRP B2xx), 20 MSPS (HackRF, USB 2.0 ceiling), and a 61.44 MHz span (Aaronia); the 300 MHz-wide 5.8 GHz band is covered by frequency hopping, not one capture. The detector itself is rate-agnostic and handles captures up to 100 MSPS+.
    - **Auto-Scanner & Fine-Tuning**: For SDRs with smaller bandwidths (e.g. 25 MSPS), a state machine continuously sweeps the 5.8 GHz band (5.645–5.945 GHz). Once a signal is found, it automatically stops scanning and transitions to a fine-tuning mode to snap precisely to the center channel. If the signal is lost for more than 2 seconds, the scanner automatically resumes sweeping.
    - **Optimized Dwell**: The scan dwell is **10 ms per hop** — just enough for the USRP PLL to settle (~2 ms) and deliver one full 65536-sample chunk (~2.6 ms at 25 MSPS). The detector only needs a single chunk per hop to run the wideband DDC probe sweep. All remaining duplicate-frequency packets are skipped to prevent queue backlog.
    - **No Power Gating**: To maximize sensitivity for weak drone signals, all sync pulse correlation runs regardless of the raw RSSI power level. This guarantees that faint signals below the noise floor are still processed and detected.

2.  **Zero-Allocation Pipeline**:
    - Raw IQ samples are streamed into pre-allocated buffers managed by the backend.
    - Hand-off occurs via lock-free `crossbeam::channel`s from the backend's capture thread to the consumer's worker pool.
    - **Overrun Protection**: If the downstream pipeline cannot keep up with the SDR stream, frames are dynamically dropped at the dispatcher. Hardware overruns are surfaced per-packet via `IqPacket.overrun` (set by the USRP backend when UHD reports an overflow); when they persist, the viewer steps its requested sample rate down by 5 MHz to restore stability.
    - **Scan-loop backpressure**: The fpv-viewer scan loop processes only one packet per frequency hop and skips all remaining packets at the same center frequency. This prevents queue buildup when the detector's ~50 ms DDC sweep can't keep up with the 2.6 ms packet rate.
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
    - Only probes exceeding the noise floor by ≥ 3 dB proceed to sync validation — except in *integrated* mode (`detect_from_iq_integrated`), where every probe is classified and accumulated: a signal below one batch's energy floor is exactly what cross-batch integration (DESIGN.md §11) exists to find.

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
    - At coarse resolution (e.g., 2.6 ms at 100 MSPS, or a 65 k chunk at 25 MSPS ≈ 381 Hz/bin), PAL and NTSC bins collide. The detector then falls back to a **time-domain median sync-tip interval** (`classify_pal_ntsc_time_domain`): it counts sync tips on the (smoothed) demodulated record against a robust percentile threshold (`p2 + 0.25·(p50−p2)` — DC-invariant and immune to single FM-click outliers, unlike the global-minimum threshold it replaced), takes the median line period, and maps it to a line rate compared against 15625/15734 Hz with a ±30 Hz midpoint dead-band — classifying PAL/NTSC at confidence 0.8. Only if that's inconclusive (too few tips, or median in the dead-band) does it tag `AnalogVideoUnknown` at 0.6 (rather than silently picking one standard). `SignalType::is_analog_video()` returns `true` in all three cases, so downstream consumers gating on "analog FPV present" still see the hit.
    - The spectral noise floor these checks reference is the **median** of the in-band magnitude bins (a mean let a real signal raise its own detection floor), and in integrated mode all of the spectral checks run against the cross-batch averaged spectrum (DESIGN.md §11) — several dB of sensitivity after ~4 batches at the same tuning.

8.  **Harmonic-Consistency Check**:
    - H-sync is a ~7% duty-cycle rectangular pulse train, so its FM-demodulated spectrum has the fundamental at the line rate plus a rich harmonic series — for a 7% duty train the first ~14 harmonics are within roughly −3 dB of the fundamental.
    - The detector counts how many of the first 5 harmonics (k = 2..=5) exceed 10% of the fundamental amplitude (and also exceed the weak noise-floor threshold). At least 2 harmonics are required for a positive classification.
    - Threshold is fundamental-relative (not noise-floor-relative) because spectral leakage from a strong fundamental otherwise pulls the noise floor estimate down enough that any FFT-window sidelobe at 2× the fundamental crosses a noise-floor-relative threshold.
    - This rejects narrowband-FM tones and CW interferers that happen to land in the H-sync bin — they have no harmonic structure and fail the count.

9.  **Cepstrum Structural Gate**:
    - Any candidate that clears the harmonic check is verified structurally via the cepstrum (IFFT of the log-power spectrum): a true H-sync pulse train's harmonic comb collapses to a single sharp quefrency peak at the line period, which multi-tone interferers (Wi-Fi beacons, BT hopping) cannot mimic.
    - The peak-to-median ratio at the expected quefrency must reach **7×** (`CEPSTRAL_RATIO_THRESHOLD`) or the classification is downgraded to `Unknown`. Real pulse trains measure ~5–20×; the threshold sits inside that range (rather than at its bottom) to reject strong periodic non-video interferers like cellular OFDM frame timing.

9a. **Vertical-Sync (VBI) Confirm Stage** (`vbi.rs`):
    - The line-rate comb and cepstrum only confirm that the *line rate* is present. `confirm_field_sync` additionally parses the demod slice for serrated broad-pulse groups (the vertical-sync structure itself) and checks that consecutive groups land a real field period apart — essentially unfakeable by a non-video interferer.
    - Confidence tiers (`apply_vbi_confidence_tier`):
      - **Boost**: a strong (0.8) hit with confirmed field-sync structure → **0.95**.
      - **Promote**: a standard-ambiguous `AnalogVideoUnknown` hit (0.6) with confirmed structure → **0.75**, clearing the default 0.7 confidence floor.
      - **Demote** (opt-in via `demote_unconfirmed_video`): a 0.8 hit spanning ≥ 2.5 field periods with *zero* confirmed groups → 0.6.

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

12. **Quadrature Demodulation** (or PLL — `demod::PllFmDemod` offers
    measured threshold extension at ≥ 25 MSPS decode rates, selected in
    fpv-viewer via `--demod pll`; see DESIGN.md §11a):
    - **The Math**: `arg(iq[n] × conj(iq[n-1]))` — phase difference between consecutive samples.
    - **Implementation**: exact scalar `f32::atan2` (via `Complex::arg`) per sample. The complex multiply + `conj` ahead of the atan2 auto-vectorises cleanly under `-O3` (it's the bulk of the work). The atan2 itself is intentionally NOT replaced with a polynomial approximation — `fast_math::atan2` was tried for edge-device throughput and reverted because image quality wins here: approximate kernels lose precision near ±π, exactly where high-deviation FM operates, and the resulting quadrant errors show up as click-noise sparkles in the reconstructed picture.
    - **Output**: A 1D stream of floating-point values representing the instantaneous frequency (brightness) of the video signal.

---

## Phase 6: Video Frame Reconstruction (`video.rs` + `frame_history.rs`)

This phase turns the 1D frequency stream into a 2D image. Output is
monochrome (luma-only) — colour recovery is currently disabled; see
DESIGN.md §9 for the rationale and §10 for the conditions under which
the colour path would return.

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
      below `DROPOUT_ENTER_THRESHOLD` (0.5) forces the temporal
      denoise into "static" mode (full blend toward recent history),
      preferring a recently-good frame to current FM static; the mode
      is held until the score recovers past `DROPOUT_EXIT_THRESHOLD`
      (0.6) — hysteresis that stops a field hovering at the boundary
      from flickering the denoise mode frame to frame.

14. **2D Mapping & Rescaling**:
    - Samples between H-Sync pulses form a single line of pixels, low-passed ahead of the sub-sample resampler whenever the resample is decimating (high capture rates), so content above the output Nyquist can't fold into the picture.
    - **Normalization**: Raw FM deviations are mapped to 8-bit grayscale (0–255, rounded) through a line-to-line **smoothed AGC** (EMA'd gain/porch state with clamping — one noisy porch window can no longer flicker a whole line).
    - **Subcarrier notch**: applied only while a Goertzel burst detector actually finds a colour burst on the back porch (3 fields of hysteresis) — burst-free monochrome cameras keep their full luma detail.
    - **Deemphasis**: the viewer applies `demod::Deemphasis` stream-side by default (`--deemphasis-tau`, 0 disables), undoing VTX pre-emphasis so HF noise isn't emphasized in the picture.
    - **Geometry**: Lines are stacked into frames (720×576 for PAL, 720×480 for NTSC).

14a. **Multi-Field Temporal Denoise** (`frame_history.rs`):
    - Each rendered field is pushed into a fixed-capacity ring buffer
      of recent Y fields (default 5, configurable via
      `FrameReconstructor::with_temporal_window(N)` or the CLI flag
      `--temporal-window N` on fpv-viewer).
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

## Phase 7: Consuming Detections

15. **Library boundary**: this crate ends at
    `Vec<DetectionResult>` (frequency, confidence, relative power,
    bandwidth, `SignalType`) plus the reconstructed frames. It is a
    library — it emits no records itself.

16. **Consumers**: fpv-viewer-rs iterates over **all** detected
    signals (not just the first), opens one decode window per signal,
    and prints per-detection lines to the console. `DetectionResult`
    derives `Serialize`, so a headless consumer can emit JSON
    telemetry directly from it if needed.
