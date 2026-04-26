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

The training script downloads 5 Unsplash images automatically if no dataset flag is given.

### Option B — DIV2K (recommended minimum)

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is the standard SR benchmark. The `--div2k` flag downloads the **validation set** (100 high-resolution images, ~770 MB) automatically on first run.

```bash
python src/train.py --phase 3 --div2k --epochs 5000
```

### Option C — DIV2K + Flickr2K (best quality)

[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (~2650 images, ~3.2 GB) is the companion corpus used by EDSR, BasicVSR, and most modern SR models. Combined with DIV2K it gives ~3450 training images.

```bash
python src/train.py --phase 3 --div2k --flickr2k --epochs 5000
```

Both datasets are downloaded automatically. If Flickr2K download fails, download `Flickr2K.tar` manually and place it at `data/Flickr2K.tar`.

### Option D — Custom directory

```bash
python src/train.py --phase 3 --data-dir /path/to/images --epochs 5000
```

---

## Training

All three phases share a single training script.

```bash
python src/train.py --phase <1|2|3> [options]
```

### Core options

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `3` | Architecture to train (1, 2, or 3) |
| `--epochs` | `5000` | Number of training epochs |
| `--batch-size` | `4` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--val-every` | `500` | Validate and save checkpoint every N epochs |
| `--seq-len` | `4` | Initial sequence length (Phase 3 only) |
| `--save-dir` | `checkpoints` | Checkpoint output directory |
| `--div2k` | off | Download and use DIV2K validation HR (100 images) |
| `--flickr2k` | off | Download and use Flickr2K HR (~2650 images) |
| `--data-dir` | `""` | Custom image directory (overrides `--div2k`/`--flickr2k`) |

### Phase 3 enhancement options

| Flag | Default | Description |
|------|---------|-------------|
| `--sched-sampling` | off | Enable scheduled sampling to fix exposure bias |
| `--sched-sample-ramp` | `2000` | Epochs to ramp self-feedback probability 0 → 0.9 |
| `--curriculum` | off | Progressive seq_len: 4 → 6 → 8 across training |
| `--temp-loss-weight` | `0.0` | Temporal consistency loss weight (try 0.05–0.2) |
| `--grad-clip` | `1.0` | Max gradient norm (0 = disabled) |
| `--grad-accum` | `1` | Gradient accumulation steps (OOM workaround) |

### Examples

```bash
# Phase 1 — Early Fusion baseline
python src/train.py --phase 1 --epochs 5000 --batch-size 8 --div2k

# Phase 2 — Warp-then-Fuse
python src/train.py --phase 2 --epochs 5000 --batch-size 8 --div2k

# Phase 3 — Basic (same as before)
python src/train.py --phase 3 --epochs 5000 --batch-size 4 --div2k --seq-len 4

# Phase 3 — Full enhanced config (recommended)
python src/train.py --phase 3 --epochs 5000 --batch-size 4 \
    --div2k --flickr2k \
    --sched-sampling --curriculum \
    --temp-loss-weight 0.1 --grad-clip 1.0

# Phase 3 — OOM fallback (simulate batch-size 8 with 1 GPU sample at a time)
python src/train.py --phase 3 --epochs 5000 \
    --batch-size 1 --grad-accum 8 \
    --div2k --sched-sampling --curriculum --temp-loss-weight 0.1
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

## Video Inference

All three phases support frame-by-frame video upscaling. The output is a side-by-side comparison: **Bicubic** (left) vs **SR output** (right).

### How motion is estimated on real video

For Phase 2 and 3, the model needs motion vectors between frames. On real video, these are estimated automatically using **OpenCV Farneback dense optical flow** — no game engine or synthetic data needed.

```
Frame t-1 ──┐
             ├─► Farneback optical flow ─► dense flow (H×W×2)
Frame t   ──┘         │
                       ▼
              backward_warp(Frame t-1, flow) ─► aligned history
```

Phase 1 uses early fusion (no flow estimation needed).

### Test video sources

**Option A — Download an open-source video automatically:**

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from dataset import download_test_video
download_test_video('data/test_video.mp4', resolution='360')
"
```

This tries the following sources in order (all Creative Commons):
- [Big Buck Bunny](https://peach.blender.org/) — Blender Foundation
- [Sintel trailer](https://durian.blender.org/) — Blender Foundation
- Falls back to generating a synthetic panning video from `data/samples/sample_00.jpg`

**Option B — Use any local video:**

Any `.mp4` file works. For best results use a clean, low-compression source at a resolution divisible by the scale factor (4).

Recommended: [Blender Open Movies](https://download.blender.org/peach/bigbuckbunny_movies/) · [Xiph.org test media](https://media.xiph.org/video/derf/)

### Usage

```bash
python src/video_inference.py --phase <1|2|3> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `2` | Which model to use (1 = no flow, 2 = Farneback+warp, 3 = Farneback+recurrent) |
| `--checkpoint` | `checkpoints/p2_best.pth` | Path to model checkpoint |
| `--input-video` | *(auto-generate)* | Path to input LR video; omit to use synthetic fallback |
| `--output-video` | `assets/video_sr_output.mp4` | Output side-by-side comparison video |
| `--image-source` | `data/samples/sample_00.jpg` | HR image for synthetic video generation (fallback only) |
| `--lr-width` | `320` | Width of synthetic LR video (fallback only) |
| `--lr-height` | `240` | Height of synthetic LR video (fallback only) |
| `--frames` | `120` | Number of frames for synthetic video (fallback only) |

### Examples

```bash
# Phase 1 — no optical flow (fastest)
python src/video_inference.py \
  --phase 1 \
  --checkpoint checkpoints/p1_best.pth \
  --input-video data/test_video.mp4 \
  --output-video assets/video_p1.mp4

# Phase 2 — Farneback flow + warp (recommended for real video)
python src/video_inference.py \
  --phase 2 \
  --checkpoint checkpoints/p2_best.pth \
  --input-video data/test_video.mp4 \
  --output-video assets/video_p2.mp4

# Phase 3 — Farneback flow + recurrent HR feedback (best quality, slowest)
python src/video_inference.py \
  --phase 3 \
  --checkpoint checkpoints/p3_best.pth \
  --input-video data/test_video.mp4 \
  --output-video assets/video_p3.mp4

# No input video — auto-generates synthetic panning clip from an image
python src/video_inference.py --phase 2 --checkpoint checkpoints/p2_best.pth
```

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
  dataset.py        — SRTemporalDataset (Phase 1/2), SRSequenceDataset + Halton jitter (Phase 3)
                      Both accept single dir or list of dirs (multi-dataset support)
                      download_sample_images(), download_div2k(), download_flickr2k(),
                      download_test_video()
  model.py          — TemporalSRResNet (P1), WarpTSRNet (P2), RecurrentTSRNet (P3)
                      All models use ICNR initialization on PixelShuffle layers
  warp.py           — backward_warp(): grid_sample + occlusion mask
                      supports rigid (B,2) and dense (B,2,H,W) flow
  utils.py          — get_device() (CUDA→MPS→CPU), compute_psnr(), compute_ssim()
  train.py          — unified training script (--phase 1/2/3)
                      Phase 3: Charbonnier loss, temporal consistency loss,
                      scheduled sampling, curriculum seq_len, grad clipping/accumulation
  inference.py      — image inference with PSNR/SSIM output (--phase 1/2/3)
  video_inference.py — video SR with Farneback optical flow (--phase 1/2/3)

checkpoints/        — saved model weights (p1_best.pth, p2_best.pth, p3_best.pth)
data/samples/       — auto-downloaded sample images (5 Unsplash photos)
data/div2k/         — DIV2K HR images (created on first --div2k run, ~770 MB)
data/flickr2k/      — Flickr2K HR images (created on first --flickr2k run, ~3.2 GB)
data/test_video.mp4 — test video (downloaded or synthetic)
assets/             — output comparison images and training logs
revision_history.md — detailed log of all architectural changes and their rationale
```
