"""Train the temporal luma denoiser shipped as
`models/temporal_denoiser.onnx` and consumed by
`src/neural.rs` (behind the `neural-vsr` feature).

Design notes (why this looks the way it does):

- The recurrent net fuses the current noisy field with a carried hidden
  state. If it is trained on *static* sequences (the same frame repeated
  with independent noise) the loss-optimal behaviour is plain temporal
  averaging — which becomes ghosting/smear on anything that moves. So
  every training sequence carries sub-pixel motion (`--max-motion`), and
  the loss includes a gradient term (`--lambda-grad`) so the objective
  penalises edge loss, not just mean pixel error. Together these are what
  keep the model from collapsing into a blur (the failure the GCOR metric
  exposed).
- Robustness to RF interference is bought by domain randomisation: the
  synthetic channel mirrors `src/impairments.rs` (multipath, burst
  dropout, in-frame fade, impulsive/ESC spikes) plus in-band interferers
  (CW tone, swept chirp, OFDM-like block), each applied with independent
  probability per sequence. Train-time impairments must match test-time.
- ~`--clean-prob` of sequences carry no impairment at all so the net
  learns identity on a clean field instead of denoising it (the measured
  GCOR-0.84-on-clean regression).
- The net gets an explicit **noise-level plane** as a second input channel
  (per-frame, spatially constant): the true SNR (via `norm_db`) at train
  time, `levels::estimate_cnr_db` at inference. It learns to modulate
  denoising strength per frame — light on clean fields, aggressive on
  buried ones — instead of applying worst-case smoothing everywhere.
"""

import argparse
import os
import time

import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# Noise-level conditioning is normalised to [0, 1] by this full-scale dB
# span. BOTH sides of the contract must agree: training feeds the true
# per-frame SNR through `norm_db`, and at inference `src/neural.rs`
# feeds `levels::estimate_cnr_db` through the identical map. 1.0 = clean.
CNR_FULL_SCALE_DB = 30.0


def norm_db(db):
    return float(min(max(db / CNR_FULL_SCALE_DB, 0.0), 1.0))


class TemporalDenoiser(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, num_layers=5):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Input concatenates: image (1) + noise-level plane (1) + hidden
        # state (hidden_channels). The noise plane lets the net modulate
        # denoising strength per frame instead of smoothing worst-case.
        first_in = in_channels + 1 + hidden_channels

        layers = []
        layers.append(nn.Conv2d(first_in, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: produces image (1) and new hidden state (hidden_channels)
        layers.append(nn.Conv2d(hidden_channels, in_channels + hidden_channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, input, noise, hidden_in):
        x = torch.cat([input, noise, hidden_in], dim=1)
        out = self.net(x)

        image_out = out[:, :1, :, :]
        # Residual connection: the net predicts a correction to the input,
        # not the image from scratch — biases it toward preserving detail.
        image_out = input + image_out

        hidden_out = out[:, 1:, :, :]
        return image_out, hidden_out


# ── Sub-pixel motion ────────────────────────────────────────────────


def shift_image(img, dx, dy):
    """Sub-pixel translate a (C, H, W) tensor by (dx, dy) pixels via
    bilinear `grid_sample` (reflection padding at the edges). Positive
    dx/dy shift the content right/down."""
    c, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    # To move content by +dx we sample from x - dx (in normalised units).
    grid_x = xs - (2.0 * dx / max(w - 1, 1))
    grid_y = ys - (2.0 * dy / max(h - 1, 1))
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    out = F.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return out.squeeze(0)


# ── Synthetic RF channel (mirrors src/impairments.rs) ───────────────


def _add_multipath(rf, rng):
    """One or two delayed, attenuated, Doppler-rotated echoes → ghosts."""
    n = rf.shape[0]
    out = rf.copy()
    for _ in range(rng.integers(1, 3)):
        delay = int(rng.integers(4, 64))
        atten = rng.uniform(0.15, 0.6)
        doppler = rng.uniform(-0.02, 0.02)  # radians/sample
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        idx = np.arange(n)
        rot = atten * np.exp(1j * (phase0 + doppler * idx))
        out[delay:] += rot[delay:] * rf[:-delay]
    return out


def _add_in_frame_fade(rf, rng):
    """Slow amplitude envelope varying WITHIN the field → horizontal banding."""
    n = rf.shape[0]
    depth = rng.uniform(0.2, 0.8)
    freq = rng.uniform(1.0, 6.0)  # cycles across the field
    phase = rng.uniform(0.0, 2.0 * np.pi)
    env = 1.0 - depth * 0.5 * (1.0 - np.cos(2.0 * np.pi * freq * np.arange(n) / n + phase))
    return rf * env


def _add_burst_dropout(rf, rng, noise_sigma):
    """Replace a short run with pure noise (antenna blockage / hop seam)."""
    n = rf.shape[0]
    out = rf.copy()
    for _ in range(rng.integers(1, 3)):
        length = int(rng.integers(n // 100, n // 20))
        start = int(rng.integers(0, max(n - length, 1)))
        seg = noise_sigma * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
        out[start : start + length] = seg
    return out


def _add_impulsive(rf, rng, amplitude):
    """Sparse high-amplitude spikes (motor/ESC spark noise)."""
    n = rf.shape[0]
    out = rf.copy()
    p = rng.uniform(0.002, 0.02)
    mask = rng.random(n) < p
    phases = rng.uniform(0.0, 2.0 * np.pi, size=mask.sum())
    out[mask] += amplitude * np.exp(1j * phases)
    return out


def _add_interferer(rf, rng):
    """An in-band emitter competing with the video carrier: CW tone,
    swept chirp, or an OFDM-like block of random-phase subcarriers."""
    n = rf.shape[0]
    idx = np.arange(n)
    amp = rng.uniform(0.1, 0.7)
    kind = rng.integers(0, 3)
    if kind == 0:  # CW tone
        w = rng.uniform(-np.pi, np.pi)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        interf = amp * np.exp(1j * (w * idx + ph))
    elif kind == 1:  # linear swept chirp (a sweeping jammer)
        w0 = rng.uniform(-np.pi, 0.0)
        rate = rng.uniform(0.0, 2.0 * np.pi) / n
        interf = amp * np.exp(1j * (w0 * idx + 0.5 * rate * idx * idx))
    else:  # OFDM-like: sum of a few narrow subcarriers
        n_sub = int(rng.integers(3, 8))
        interf = np.zeros(n, dtype=np.complex128)
        for _ in range(n_sub):
            w = rng.uniform(-np.pi, np.pi)
            ph = rng.uniform(0.0, 2.0 * np.pi)
            interf += np.exp(1j * (w * idx + ph))
        # Normalise so total interferer power ~ amp² regardless of count.
        interf *= amp / np.sqrt(n_sub)
    return rf + interf


class AnalogNoiseDataset(Dataset):
    def __init__(self, base_dataset, seq_length=5, patch_size=128, max_motion=3.0,
                 clean_prob=0.15, chroma_prob=0.6, snr_lo=5.0, snr_hi=30.0, seed=0):
        self.base_dataset = base_dataset
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.max_motion = max_motion
        self.clean_prob = clean_prob
        self.chroma_prob = chroma_prob
        self.snr_lo = snr_lo
        self.snr_hi = snr_hi
        self.seed = seed
        # Folded into the per-item seed so every epoch draws fresh
        # impairments (see `__getitem__`). This is a shared-memory value
        # rather than a plain int on purpose: with `persistent_workers`
        # the workers are forked ONCE and hold their own copy of this
        # object, so a normal attribute assignment in the parent would
        # never reach them and the epoch would stay pinned at 0.
        self._epoch = mp.Value("i", 0)

    @property
    def epoch(self):
        return self._epoch.value

    def set_epoch(self, epoch):
        self._epoch.value = int(epoch)

    def __len__(self):
        return len(self.base_dataset)

    def _rf_channel(self, frame, rng, add_chroma):
        """Clean (H, W) luma in [0,1] → (noisy (H, W) luma in [0,1], snr_db)
        via a randomised FM-over-RF round trip. The returned `snr_db` is
        the thermal-noise SNR this frame was generated at — it drives the
        conditioning plane (see `norm_db`)."""
        # Raster-scan the field into a 1-D baseband and FM-modulate.
        baseband = frame.flatten() * 2.0 - 1.0

        if add_chroma:
            w0 = 0.53 * np.pi
            chroma_amp = rng.random() * 0.3
            chroma_phase = rng.random() * 2.0 * np.pi
            t_pixels = np.arange(baseband.shape[0])
            baseband = baseband + chroma_amp * np.sin(w0 * t_pixels + chroma_phase)

        mod_index = 2.0
        rf = np.exp(1j * np.cumsum(baseband * mod_index))

        # ── Randomised RF channel (each impairment independent) ──
        snr_db = rng.uniform(self.snr_lo, self.snr_hi)
        noise_sigma = np.sqrt((1.0 / (10.0 ** (snr_db / 10.0))) / 2.0)

        if rng.random() < 0.5:
            rf = _add_multipath(rf, rng)
        if rng.random() < 0.4:
            rf = _add_in_frame_fade(rf, rng)
        else:
            rf = rf * (1.0 - rng.random() * 0.5)  # flat fade
        if rng.random() < 0.35:
            rf = _add_interferer(rf, rng)

        # Thermal noise always present.
        rf = rf + noise_sigma * (rng.standard_normal(rf.shape) + 1j * rng.standard_normal(rf.shape))

        if rng.random() < 0.25:
            rf = _add_burst_dropout(rf, rng, noise_sigma)
        if rng.random() < 0.3:
            rf = _add_impulsive(rf, rng, amplitude=rng.uniform(2.0, 6.0))

        # FM demod: instantaneous frequency = d/dt of unwrapped phase.
        phase_rx = np.unwrap(np.angle(rf))
        demod = np.diff(phase_rx, prepend=phase_rx[0]) / mod_index
        noisy = np.clip((demod + 1.0) / 2.0, 0.0, 1.0).reshape(frame.shape)
        return noisy.astype(np.float32), float(snr_db)

    def _load_base(self, idx, rng):
        """Fetch a base image, surviving transient storage stalls.

        Reading the image corpus can raise `TimeoutError`/`OSError` when
        the volume hiccups (observed: `[Errno 60] Operation timed out`
        mid-epoch on a network/external disk). A DataLoader worker that
        raises kills the whole run — losing every epoch since the last
        checkpoint — so retry the same index a few times, then fall back
        to other random indices rather than taking the process down.
        """
        last_err = None
        for attempt in range(3):
            try:
                return self.base_dataset[idx]
            except (TimeoutError, OSError) as e:  # noqa: PERF203
                last_err = e
                time.sleep(0.5 * (attempt + 1))  # brief backoff; disks recover
        # Same index keeps failing (stalled or corrupt) — substitute
        # another sample so training continues.
        for _ in range(5):
            alt = int(rng.integers(0, len(self.base_dataset)))
            try:
                return self.base_dataset[alt]
            except (TimeoutError, OSError) as e:
                last_err = e
        raise RuntimeError(f"dataset unreadable near index {idx}: {last_err}")

    def __getitem__(self, idx):
        # Per-item RNG: worker-safe (each item's stream is independent of
        # how the DataLoader shards work), but it MUST also advance per
        # epoch. Seeding on `idx` alone froze one impairment realisation
        # per image for the whole run, so the model saw the same noise on
        # that image every epoch and could memorise it instead of learning
        # to denoise — exactly the augmentation the RF channel exists to
        # provide. `set_epoch` folds the epoch into the seed.
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + idx) * 1_000_033 + self.epoch * 2_654_435_761
        )

        img, _ = self._load_base(idx, rng)
        if img.shape[0] == 3:
            img = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]

        _, h, w = img.shape
        if h > self.patch_size and w > self.patch_size:
            top = int(rng.integers(0, h - self.patch_size))
            left = int(rng.integers(0, w - self.patch_size))
            img = img[:, top : top + self.patch_size, left : left + self.patch_size]
        else:
            img = transforms.functional.resize(img, [self.patch_size, self.patch_size])

        # Cumulative sub-pixel motion (a small random walk) so temporal
        # fusion must be motion-aware, not a static averager.
        step = self.max_motion / max(self.seq_length, 1)
        px = py = 0.0

        clean_seq = torch.zeros(self.seq_length, 1, self.patch_size, self.patch_size)
        noisy_seq = torch.zeros_like(clean_seq)
        # Per-frame conditioning scalar, broadcast to a plane at use.
        noise_seq = torch.zeros(self.seq_length, 1, 1, 1)

        # One clean-passthrough decision per sequence (identity training).
        passthrough = rng.random() < self.clean_prob
        add_chroma = (not passthrough) and (rng.random() < self.chroma_prob)

        for t in range(self.seq_length):
            px += rng.uniform(-step, step)
            py += rng.uniform(-step, step)
            clean_t = shift_image(img, px, py)  # (1, H, W)
            clean_seq[t] = clean_t

            if passthrough:
                noisy_seq[t] = clean_t
                # A clean field reads as very high SNR → plane ≈ 1.
                noise_seq[t, 0, 0, 0] = norm_db(CNR_FULL_SCALE_DB + 10.0)
            else:
                noisy, snr_db = self._rf_channel(clean_t[0].numpy(), rng, add_chroma)
                noisy_seq[t, 0] = torch.from_numpy(noisy)
                noise_seq[t, 0, 0, 0] = norm_db(snr_db)

        return noisy_seq, clean_seq, noise_seq


# ── Loss ────────────────────────────────────────────────────────────


def gradient_l1(pred, target):
    """L1 on forward-difference gradients — trains on the edge structure
    that `metrics::compute_gradient_correlation` scores, so the net can't
    cheaply lower pixel error by smoothing."""
    px = pred[..., :, 1:] - pred[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    tx = target[..., :, 1:] - target[..., :, :-1]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return (px - tx).abs().mean() + (py - ty).abs().mean()


# ── Export + quantization audit ─────────────────────────────────────


def _grad_corr_np(a, b):
    """Gradient correlation between two 2-D arrays — the numpy twin of
    the Rust metric, for the fp32-vs-INT8 audit."""
    ax = np.hypot(np.diff(a, axis=1, append=a[:, -1:]), np.diff(a, axis=0, append=a[-1:, :]))
    bx = np.hypot(np.diff(b, axis=1, append=b[:, -1:]), np.diff(b, axis=0, append=b[-1:, :]))
    ax = ax - ax.mean()
    bx = bx - bx.mean()
    denom = np.sqrt((ax * ax).sum() * (bx * bx).sum())
    return float((ax * bx).sum() / denom) if denom > 0 else 0.0


def export_and_quantize(model, hidden_channels, quantize=True, keep_fp32=True):
    model.eval()

    dummy_input = torch.randn(1, 1, 128, 128)
    dummy_noise = torch.ones(1, 1, 128, 128)
    dummy_hidden = torch.zeros(1, hidden_channels, 128, 128)

    fp32_path = "models/temporal_export_fp32.onnx"
    quant_path = "models/temporal_denoiser.onnx"

    print(f"\nExporting FP32 model to {fp32_path}...")
    torch.onnx.export(
        model,
        (dummy_input, dummy_noise, dummy_hidden),
        fp32_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input", "noise", "hidden_in"],
        output_names=["output", "hidden_out"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "noise": {2: "height", 3: "width"},
            "hidden_in": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
            "hidden_out": {2: "height", 3: "width"},
        },
    )

    if not quantize:
        # Ship fp32 at the canonical path the Rust loads. If the export
        # externalised its weights to a `.data` sidecar, that file must
        # travel too — the `.onnx` references it by name, so copying the
        # graph alone would ship a model with dangling weights. (Small
        # models like this one inline everything, but don't depend on
        # that.)
        import shutil

        shutil.copyfile(fp32_path, quant_path)
        sidecar = fp32_path + ".data"
        if os.path.exists(sidecar):
            shutil.copyfile(sidecar, quant_path + ".data")
            print(f"Copied external weights sidecar → {quant_path}.data")
        print(f"Quantization disabled; wrote FP32 weights to {quant_path}")
        return

    print("Quantizing to INT8...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(model_input=fp32_path, model_output=quant_path, weight_type=QuantType.QInt8)
    print(f"Saved INT8 ONNX model to {quant_path}")

    # ── Audit: does INT8 cost fidelity vs FP32 on a clean field? ──
    try:
        import onnxruntime as ort

        rng = np.random.default_rng(1234)
        clean = rng.random((1, 1, 128, 128)).astype(np.float32)
        noise = np.ones((1, 1, 128, 128), dtype=np.float32)  # clean → plane 1
        hidden = np.zeros((1, hidden_channels, 128, 128), dtype=np.float32)

        def run(path):
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"input": clean, "noise": noise, "hidden_in": hidden})[0]
            return out[0, 0]

        c = clean[0, 0]
        gc_fp32 = _grad_corr_np(run(fp32_path), c)
        gc_int8 = _grad_corr_np(run(quant_path), c)
        print(f"\n[AUDIT] clean-input gradient-corr vs input:  FP32={gc_fp32:.3f}  INT8={gc_int8:.3f}")
        if gc_int8 < gc_fp32 - 0.02:
            print(
                "[AUDIT] INT8 quantization measurably degrades fidelity. Size is "
                "irrelevant here (~13 KB); consider shipping FP32 (--no-quantize) "
                "or quantization-aware training."
            )
    except Exception as e:  # noqa: BLE001 — audit is advisory, never fatal
        print(f"[AUDIT] skipped (onnxruntime unavailable or failed): {e}")

    if not keep_fp32 and os.path.exists(fp32_path):
        os.remove(fp32_path)
        if os.path.exists(fp32_path + ".data"):
            os.remove(fp32_path + ".data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap dataset size, sampled at random across all corpora (0 = all)",
    )
    parser.add_argument(
        "--subset-seed", type=int, default=0, help="seed for --limit's random subset"
    )
    parser.add_argument("--lambda-grad", type=float, default=0.75, help="weight of the gradient loss term")
    parser.add_argument("--clean-prob", type=float, default=0.15, help="fraction of clean-passthrough sequences")
    parser.add_argument("--chroma-prob", type=float, default=0.6, help="fraction of impaired sequences carrying a subcarrier")
    parser.add_argument("--max-motion", type=float, default=3.0, help="total sub-pixel drift (px) across a sequence")
    parser.add_argument("--snr-lo", type=float, default=5.0)
    parser.add_argument("--snr-hi", type=float, default=30.0)
    parser.add_argument("--no-quantize", action="store_true", help="ship FP32 weights instead of INT8")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoint.pt",
        help="per-epoch checkpoint path (training resumes from here)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="ignore an existing checkpoint and train from scratch",
    )
    parser.add_argument("--dummy-data", action="store_true", help="tiny random dataset for a sanity run")
    parser.add_argument("--add-places365", action="store_true", help="Add Places365 val split to the training data for more diversity")
    parser.add_argument(
        "--places-only",
        action="store_true",
        help="train on Places365 val ONLY (native 256px → real 128px crops, no STL10 upscaling)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    if args.dummy_data:
        print("Using dummy random data...")
        base_dataset = [(torch.rand(3, 128, 128), 0) for _ in range(32)]
    elif args.places_only:
        # Places365-small is native 256x256, so a 128px crop is a REAL
        # crop. STL10 is 96px and must be upscaled to 128 — interpolation
        # softens it, which is counterproductive when the whole point of
        # this training run is to stop the model blurring. Prefer real
        # pixels.
        print("Loading Places365 (val split) only...")
        base_dataset = torchvision.datasets.Places365(
            root="./data", split="val", small=True, download=False, transform=transform
        )
    else:
        print("Downloading/Loading STL10 Dataset...")
        base_dataset = torchvision.datasets.STL10(root="./data", split="unlabeled", download=True, transform=transform)
        if args.add_places365:
            print("Downloading/Loading Places365 (val split) Dataset...")
            try:
                places = torchvision.datasets.Places365(root="./data", split="val", small=True, download=True, transform=transform)
                base_dataset = torch.utils.data.ConcatDataset([base_dataset, places])
            except RuntimeError as e:
                print(f"Warning: Failed to load Places365 ({e}). Falling back to STL10 only.")

    if args.limit > 0 and args.limit < len(base_dataset):
        # Sample the subset RANDOMLY, not as a leading range: the corpora
        # are concatenated (STL10 then Places365), so `range(limit)` would
        # draw entirely from the first dataset and silently exclude
        # Places365 — training on none of the high-resolution photos the
        # flag was added to provide. A fixed seed keeps runs comparable.
        g = torch.Generator().manual_seed(args.subset_seed)
        idx = torch.randperm(len(base_dataset), generator=g)[: args.limit].tolist()
        base_dataset = torch.utils.data.Subset(base_dataset, idx)
        print(f"Sampled {len(idx)} images at random from the combined corpus.")

    train_dataset = AnalogNoiseDataset(
        base_dataset,
        seq_length=args.seq_length,
        patch_size=128,
        max_motion=args.max_motion,
        clean_prob=args.clean_prob,
        chroma_prob=args.chroma_prob,
        snr_lo=args.snr_lo,
        snr_hi=args.snr_hi,
    )
    # Measured on an M4 with the corpus on local SSD: JPEG decode is
    # ~1.1 ms/image and a full sequence (decode + crop + motion + FM
    # round trip) ~7.6 ms, so even 3 workers feed batches several times
    # faster than the GPU consumes them — training is GPU-bound, which is
    # where we want it. These flags keep it that way:
    #  - persistent_workers: don't tear down and respawn the worker pool
    #    every epoch (20 epochs = 20 needless respawns, each re-importing
    #    torch and re-opening the dataset).
    #  - prefetch_factor: keep batches queued so the loader stays ahead of
    #    the ~232 ms model step instead of being sampled just-in-time.
    #  - pin_memory: page-locked staging buffers for faster host→device
    #    copies. CUDA only — MPS ignores it and warns, so don't ask for
    #    it there.
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        **loader_kwargs,
    )

    model = TemporalDenoiser(
        in_channels=1, hidden_channels=args.hidden_channels, num_layers=args.num_layers
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    l1 = nn.L1Loss()

    # Resume from a checkpoint if one exists, so a crash (or a stalled
    # disk taking a worker down) costs at most the current epoch rather
    # than the whole run.
    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from {args.checkpoint} at epoch {start_epoch}")

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Advance the impairment RNG stream so this epoch draws fresh
        # noise/fades/interferers for every image (shared-memory value, so
        # it reaches persistent workers too). Set BEFORE iterating.
        train_dataset.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, (noisy_seq, clean_seq, noise_seq) in enumerate(train_loader):
            noisy_seq = noisy_seq.to(device)  # (B, T, C, H, W)
            clean_seq = clean_seq.to(device)
            noise_seq = noise_seq.to(device)  # (B, T, 1, 1, 1)
            b, t_len, _, h, w = noisy_seq.shape

            optimizer.zero_grad()
            hidden = torch.zeros(b, args.hidden_channels, h, w, device=device)
            loss = 0.0
            for t in range(t_len):
                noise_plane = noise_seq[:, t].expand(b, 1, h, w)
                img_out, hidden = model(noisy_seq[:, t], noise_plane, hidden)
                target = clean_seq[:, t]
                loss = loss + l1(img_out, target) + args.lambda_grad * gradient_l1(img_out, target)
            loss = loss / t_len

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"==> Epoch {epoch+1} Average Loss: {epoch_loss / max(len(train_loader), 1):.4f}")

        # Checkpoint after every epoch (atomic rename so an interrupted
        # write can't leave a corrupt file behind).
        tmp = args.checkpoint + ".tmp"
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "hidden_channels": args.hidden_channels,
                "num_layers": args.num_layers,
            },
            tmp,
        )
        os.replace(tmp, args.checkpoint)
        print(f"    checkpoint → {args.checkpoint}")

    print("Training complete.")
    export_and_quantize(model, hidden_channels=args.hidden_channels, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()
