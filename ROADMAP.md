# Weak-Signal Recovery Roadmap

Where the analog FPV pipeline sits today, and the concrete next steps to
push weak-NTSC/PAL recovery toward the current state of the art. Ordered
by return-on-effort for this project's actual use case: real-time,
edge-class hardware, often a single inexpensive SDR.

## Baseline (shipped, as of lib 0.2.1 / viewer 0.2.0)

The pipeline is a classically-correct FM video receiver, pushed to a
good point on the *classical* threshold-extension curve and — crucially
— fully instrumented:

- **Demod**: quadrature discriminator + PLL (`demod::PllFmDemod`,
  measured +6–17 dB demod MSE and ~1 σ-step deeper sync survival at
  ≥ 25 MSPS). FMFB documented as the next classical rung (§11a of
  `DESIGN.md`).
- **Detection sensitivity**: noncoherent cross-batch spectral
  integration (`SpectralIntegrator`, ~5 dB, calibrated).
- **Sync/TBC**: two-pass MAD-rejected extraction, robust constant-period
  fit, Catmull-Rom sub-sample resample, pre-TBC anti-alias.
- **Conditioning**: smoothed AGC, burst-gated subcarrier notch,
  multi-field motion-adaptive temporal median, dropout compensation.
- **Telemetry**: `levels::estimate_cnr_db`, PLL lock metric.
- **Measurement discipline**: `examples/weak_signal_sweep.rs` walks a
  signal from clean to below the noise floor and reports per-technique
  detection rate, demod MSE, and reconstruction sync-quality; calibrated
  regression tests pin the wins.

**Honest gap**: we are near the top of the *classical single-channel*
curve, but not using several genuinely current techniques. The four
below are the roadmap. Every phase is gated by the same rule that
retired GPU tiling: **it must beat the baseline on `weak_signal_sweep`
or it does not ship.**

## Phase 0 — Measurement upgrade (prerequisite, small)

Before adding recovery techniques, make the harness score *picture*
recovery, not just sync lock.

- Extend `weak_signal_sweep.rs` (and add a reusable `bench`/test helper)
  to compute a **frame-fidelity metric** — PSNR and a gradient/edge
  correlation against the known `generate_fields` ground truth — per
  technique, per σ. `generate_fields` already emits the exact truth
  field the reconstructor is trying to recover, so this is
  ground-truthed, not eyeballed.
- Report the **σ-cliff** (the noise level at which frame PSNR crosses a
  usability threshold) as the single headline number each technique
  moves.
- Add a small library of **realistic impairment models** beyond AWGN:
  multipath/flutter, burst dropouts, and a slow fade envelope — weak FPV
  links are rarely pure-AWGN, and techniques that only win on AWGN can
  lose on real captures.

Deliverable: one number per technique (σ-cliff) that every later phase
is measured against. ~1 focused change; no new DSP.

## Phase 1 — ld-decode-class TBC + dropout handling (highest ROI)

The **ld-decode / vhs-decode** ecosystem (Domesday86, GPL) is the
reference open-source state of the art for software recovery of degraded
composite video, and it is our closest real-world benchmark. Our
`video.rs` re-derives a subset; their sync/TBC and dropout concealment
are more advanced exactly where low CNR bites.

1. **Matched-filter sync acquisition** (`video.rs`, also feeds detection).
   Replace threshold-crossing sync-tip location with correlation against
   a sync-pulse-shaped template (the pulse geometry is already in
   `vbi::consts`). A matched filter is the optimal detector for a known
   pulse in noise — real processing gain on the sync edge itself, which
   is what the whole TBC grid is pinned to. New
   `video::matched_sync_center` alongside `robust_sync_tip_center`, gated
   by a builder flag until it wins.
2. **Line-locked clock tracking.** Augment the per-field constant-period
   fit with a tracked line clock — a small PLL (or 2-state Kalman on
   `[phase, period]`) that carries lock across fields instead of
   re-fitting each field. At low CNR the current fit starts interpolating
   whole rows; a tracked clock rides through dropouts using the crystal
   stability the fit already assumes.
3. **Confidence-driven dropout detection.** Today DOC thresholds the
   demod rails. Add a per-line/per-pixel **confidence signal** from the
   envelope CNR meter and (when active) the PLL lock metric — both
   already exist — so genuinely-corrupted pixels are marked, not just
   out-of-rail ones.
4. **Better concealment.** Current concealment substitutes the previous
   field wholesale. Adopt ld-decode's approach: for each dropout run,
   choose the best source among (spatial interpolation from adjacent
   good lines in-field) and (previous-field same-line), weighted by
   confidence — spatial when motion is high, temporal when static.

Risk: mostly additive and independently testable; each item is a
separate `weak_signal_sweep` column. Highest ROI because it targets
*our exact signal type* and borrows proven, same-license work.

## Phase 2 — Nonlinear-estimation demodulator (medium ROI)

The modern rung beyond PLL/FMFB. Model FM demod as state estimation and
run an **unscented Kalman filter** rather than a fixed-bandwidth loop.

- New `demod::KalmanFmDemod`, a streaming sibling of `PllFmDemod` (same
  radians/sample output convention, so it drops into the same decode
  slot and the same `--demod` switch).
- State `[phase, freq]` (optionally `+ freq_rate`); measurement is the
  complex sample as a nonlinear (`cos`/`sin`) function of phase — the UKF
  handles that nonlinearity without linearization error. Process model:
  `freq` (the video) as a bandwidth-matched random walk / low-order AR.
- Cost: a 2–3 state UKF is a handful of small matrix ops per sample —
  heavier than the PLL's few MACs but edge-feasible. **Profile against
  the `dsp` bench; if it can't hold real time at 25 MSPS it's
  batch/offline only, and the plan says so.**
- Divergence bounded the same way the PLL clamps `freq`.
- Add as a third demod option and a third `weak_signal_sweep` column;
  clean-signal equivalence + threshold-comparison tests mirror the PLL's.

Risk: tuning the noise covariances; the win over a well-tuned PLL may be
modest at our deviations — the harness decides.

## Phase 3 — Receiver/antenna diversity combining (largest raw dB, largest effort)

This is how real FPV ground stations get range, and it beats every
single-channel demod trick on raw dB — but it needs more than one
capture.

- **Library** (`combine.rs`, new): `DiversityCombiner` takes N complex
  baseband streams (post-DDC), aligns them, and **maximal-ratio
  combines** using per-stream weights from `estimate_cnr_db` (already
  have it). Selection combining is the trivial always-available fallback;
  MRC is the ~3 dB (2-way) / more (N-way) win. Phase alignment via
  cross-correlation or per-stream carrier lock.
- **Viewer**: multi-source capture wiring — the real lift. Manage N
  `SdrSource`s, gate behind a flag, start with two channels / two
  receivers. Software time/phase alignment across independent SDR clocks
  is the hard part (no shared reference on cheap hardware); budget for a
  correlation-based aligner.
- **Measurement**: sweep the same signal through two independent noise
  realizations → show ~3 dB combining gain for 2-way MRC, scaling with N.

Risk: cross-SDR timing/phase alignment is genuinely hard and hardware-
dependent; the library combiner is straightforward and testable in
isolation (synthetic dual streams) even before the viewer plumbing
exists. Ship the library half first.

## Phase 4 — Neural frame restoration (biggest perceptual gain, out-of-core)

Where the largest *perceptual* weak-signal gains live in 2024–2026:
temporal video super-resolution / learned denoisers fed the noisy
reconstructed fields.

- Explicitly **out of scope for the pure-CPU edge crate**. Prototype as
  an *optional post-processor* on the recovered Y fields — export frames,
  run a small temporal denoiser/VSR model on GPU (via `candle`/`ort`),
  behind a hard feature gate.
- Honest caveats: different class of system (needs a model + runtime),
  least edge-friendly, and it only helps *after* the front end has
  recovered enough sync to produce frames — it complements, never
  replaces, Phases 0–3.
- Lowest priority for real-time recon; highest if the goal is best-
  possible offline restoration of a marginal capture.

## Sequencing

```
Phase 0  ─ measurement upgrade ....... prerequisite, do first
Phase 1  ─ ld-decode TBC/dropout ..... highest ROI, our signal type
Phase 2  ─ UKF demod ................. next demod rung (or FMFB first)
Phase 3  ─ diversity (lib → viewer) .. biggest dB, biggest effort
Phase 4  ─ neural restoration ........ optional, off-core, last
```

Each phase lands independently, gated on beating the Phase-0 σ-cliff
baseline, reviewed per the repo's pre-commit rule, released additively
(patch bumps unless the public API breaks).
