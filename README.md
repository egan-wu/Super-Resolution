# DLSS-style Temporal Super Resolution

A PyTorch implementation exploring the core ideas behind **DLSS / PSSR** — using temporal information across frames to reconstruct high-resolution video from low-resolution input. Three progressively more sophisticated architectures are implemented.

---

## Architecture Overview

| Phase | Model | Key Idea | Best Val PSNR |
|-------|-------|----------|--------------|
| 1 | `TemporalSRResNet` | Early fusion — stack LR(t-1) + LR(t) as 6-ch input | 29.01 dB |
| 2 | `WarpTSRNet` | Warp-then-fuse — backward-warp LR(t-1) using known flow before fusion | **33.37 dB** |
| 3 | `RecurrentTSRNet` | DLSS-style feedback — warp HR(t-1) at HR space; Halton jitter; true recurrent loop at inference | 28.69 dB* |

*Phase 3 trained on 5 images; improves significantly with DIV2K (100 images).

### How temporal upscaling works (without a game engine)

Real upscalers (DLSS, PSSR) receive per-frame motion vectors from the GPU rasterizer. Since we work with 2D image datasets, we simulate a virtual panning camera: each frame is a slightly offset crop of the same high-resolution image, with a known pixel displacement.

```
Full HR image
  ├── Frame t-1  crop at (x + offset, y + offset)  →  downsample  →  LR(t-1)
  └── Frame t    crop at (x, y)                     →  downsample  →  LR(t)
                                                          known flow ↗
```

**Phase 3** additionally applies **Halton quasi-random jitter** — each frame gets a different sub-pixel offset, giving the recurrent network diverse samples of the scene to accumulate into a sharper HR output (the same principle DLSS uses with its jitter pattern).

---

## Setup

```bash
pip install torch torchvision pillow requests tqdm matplotlib opencv-python scikit-image
```

---

## Dataset

### Option A — Quick start (5 sample images, auto-downloaded)

The training script downloads 5 Unsplash images automatically if no `--div2k` flag is given.

### Option B — DIV2K (recommended)

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is the standard SR benchmark. The `--div2k` flag downloads the **validation set** (100 high-resolution images, ~770 MB) automatically on first run.

```bash
# Downloads DIV2K on first use, then trains
python src/train.py --phase 3 --div2k --epochs 5000
```

If you already have a local dataset directory, use `--data-dir`:

```bash
python src/train.py --phase 3 --data-dir /path/to/images --epochs 5000
```

---

## Training

All three phases share a single training script.

```bash
python src/train.py --phase <1|2|3> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `3` | Which architecture to train (1, 2, or 3) |
| `--epochs` | `5000` | Number of training epochs |
| `--batch-size` | `4` | Batch size (`8` recommended with DIV2K) |
| `--lr` | `1e-4` | Learning rate |
| `--val-every` | `500` | Run validation and save checkpoint every N epochs |
| `--seq-len` | `4` | Sequence length for Phase 3 (ignored by Phase 1/2) |
| `--div2k` | off | Download and use DIV2K validation HR dataset |
| `--data-dir` | `""` | Custom image directory (overrides `--div2k`) |
| `--save-dir` | `checkpoints` | Directory to save checkpoints |

### Examples

```bash
# Phase 1 — Early Fusion baseline
python src/train.py --phase 1 --epochs 5000 --batch-size 8 --div2k

# Phase 2 — Warp-then-Fuse
python src/train.py --phase 2 --epochs 5000 --batch-size 8 --div2k

# Phase 3 — Recurrent DLSS-style (recommended)
python src/train.py --phase 3 --epochs 5000 --batch-size 4 --div2k --seq-len 4
```

### Checkpoints

Each phase saves:
- `checkpoints/p{1|2|3}_best.pth` — best validation PSNR checkpoint (use this for inference)
- `checkpoints/p{1|2|3}_epoch_<N>.pth` — periodic milestone checkpoint

---

## Inference

### Phase 1 / 2 — Single image, two-frame input

```bash
python src/inference.py \
  --phase 2 \
  --checkpoint checkpoints/p2_best.pth \
  --image data/samples/sample_00.jpg \
  --output assets/output.png
```

The script crops the center of the image, simulates a 5-pixel camera offset for the previous frame, runs the model, and saves a 4-panel comparison:

```
[ LR Frame t-1 ] [ LR Frame t ] [ Bicubic ] [ SR Output ]
                                  PSNR / SSIM printed in title and console
```

### Phase 3 — Recurrent multi-frame inference

```bash
python src/inference.py \
  --phase 3 \
  --checkpoint checkpoints/p3_best.pth \
  --image data/samples/sample_00.jpg \
  --output assets/recurrent_output.png \
  --num-frames 6
```

This runs true recurrent inference: the model feeds its own HR output back as history for each subsequent frame, accumulating detail across `--num-frames` frames. The output shows how quality improves as more frames are processed.

### Inference options

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `3` | Which model architecture to use |
| `--checkpoint` | `checkpoints/p3_best.pth` | Path to model checkpoint |
| `--image` | `data/samples/sample_00.jpg` | Input image |
| `--output` | `assets/output.png` | Output comparison image path |
| `--offset` | `5` | Simulated camera pan offset in pixels (Phase 1/2) |
| `--num-frames` | `6` | Number of recurrent frames to accumulate (Phase 3) |

---

## Video Inference (Phase 1 / 2)

Generate a panning low-resolution video from an image, then upscale it frame-by-frame:

```bash
python src/video_inference.py \
  --checkpoint checkpoints/p2_best.pth \
  --image_source data/samples/sample_00.jpg \
  --output_video assets/upscaled.mp4
```

Outputs a side-by-side comparison video: **Bicubic** (left) vs **SR output** (right).

---

## Results

Tested on `sample_00.jpg` (800×800), 4× upscaling:

| Method | PSNR | SSIM |
|--------|------|------|
| Bicubic | 35.85 dB | 0.931 |
| Phase 1 — EarlyFusion | 31.07 dB | 0.885 |
| Phase 2 — WarpThenFuse | 35.03 dB | 0.921 |
| Phase 3 — Recurrent (4 frames) | 30.09 dB | 0.917 |

Phase 2 nearly matches Bicubic on a still-image test; Phase 3 shows its advantage on true video sequences where cross-frame information accumulates.

---

## Project Structure

```
src/
  dataset.py       — SRTemporalDataset (Phase 1/2), SRSequenceDataset (Phase 3), DIV2K downloader
  model.py         — TemporalSRResNet, WarpTSRNet, RecurrentTSRNet
  warp.py          — backward_warp() using grid_sample + occlusion mask
  utils.py         — get_device() (CUDA→MPS→CPU), compute_psnr(), compute_ssim()
  train.py         — unified training script (--phase 1/2/3)
  inference.py     — image inference with PSNR/SSIM output
  video_inference.py — frame-by-frame video upscaling

checkpoints/       — saved model weights
data/samples/      — auto-downloaded sample images
data/div2k/        — DIV2K HR images (created on first --div2k run)
assets/            — output comparison images and training logs
```
