# Revision History

A concise record of each improvement round, its motivation, approach, and core idea.

---

## Capacity & Quality Round (2026-05-02)

Three improvements targeting limited model capacity, boundary artefacts, and
training efficiency.

---

### 11. Configurable Hidden Channels (`src/model.py`)

**Problem:** 64 hidden channels across all ResBlocks limits the model's
representational capacity. With larger datasets (DIV2K + Flickr2K) the model
finds a PixelShuffle shortcut (striping) instead of learning genuine detail
because it lacks capacity for the latter.

**Fix:** All three models (`TemporalSRResNet`, `WarpTSRNet`, `RecurrentTSRNet`)
accept `hidden_channels` parameter (default 64 for backward compat). Setting
`--hidden-channels 128` gives ~4× parameters and capacity.

**Key choices:**
- Parameter exposed via `--hidden-channels` flag (default 64).
- `ch * 4` used for PixelShuffle conv (instead of hardcoded 256).
- Fusion layer uses `ch * 2` input channels (instead of hardcoded 128).
- Checkpoints are incompatible across different `hidden_channels` values.

**Core idea:** More capacity → model can represent fine detail without resorting
to regular-pattern shortcuts.

---

### 12. Reflection Padding for Warp (`src/warp.py`, `src/model.py`)

**Problem:** `grid_sample` with `padding_mode="zeros"` fills out-of-bounds
samples with black, creating dark boundary artefacts. `"border"` repeats edge
pixels, which can smear. Neither is ideal for the occlusion-masked warp.

**Fix:** `backward_warp()` now accepts `padding_mode` parameter. Models
(`WarpTSRNet`, `RecurrentTSRNet`) store and pass `pad_mode` through to warp.
Default changed from `"border"` to `"reflection"` which mirrors content
smoothly across boundaries.

**Key choices:**
- `--pad-mode` flag: `reflection` (default, recommended), `border`, `zeros`.
- Applied to Phase 2 and 3 (Phase 1 doesn't use warp).
- Occlusion mask still correctly marks out-of-bounds pixels.

**Core idea:** Reflection padding provides natural-looking content at boundaries
instead of black fill or edge repetition.

---

### 13. Data Augmentation (`src/dataset.py`)

**Problem:** Even with ~2750 images, the model sees each crop in only one
orientation. Random spatial transforms effectively multiply the dataset ×8
(4 rotations × 2 flips) and force rotational invariance.

**Fix:** Both `SRTemporalDataset` and `SRSequenceDataset` accept `augment=True`.
When enabled, each sample is randomly flipped horizontally, vertically, and/or
transposed (90° rotation). Transforms are applied consistently to all frames
in a sequence.

**Key choices:**
- `--augment` flag (default off, backward compat).
- Flow vectors become approximate after spatial transforms — this is intentional;
  the model learns robustness to flow noise, matching real-world optical flow.
- Applied at data loading time, zero compute overhead.

**Core idea:** ×8 effective dataset via geometry transforms, forcing the model
to learn orientation-invariant features.

---

### 14. Cosine Annealing LR Scheduler (`src/train.py`)

**Problem:** Constant learning rate throughout training. In later epochs the
model overshoots fine details because the step size is too large for the
flatter loss landscape.

**Fix:** `--cosine-anneal` enables `CosineAnnealingLR` with `T_max=epochs`
and `eta_min = lr × 0.01`. Learning rate smoothly decays from `lr` to near
zero following a cosine curve.

**Key choices:**
- Optional flag for backward compat.
- `eta_min = lr * 0.01` (not zero) to avoid complete stagnation.
- Applied to all three phases.
- Gradient clipping now also applied to Phase 1 and 2 (was Phase 3 only).

**Core idea:** Cosine schedule = aggressive early learning + fine late
refinement. Standard in modern SR (EDSR, SwinIR, HAT).

---

### Updated Summary Table

| Change | Files | Key Flag / API |
|---|---|---|
| Configurable hidden channels | `model.py` | `--hidden-channels 128` |
| Reflection padding for warp | `warp.py`, `model.py` | `--pad-mode reflection` |
| Data augmentation (flip+rot) | `dataset.py` | `--augment` |
| Cosine annealing LR | `train.py` | `--cosine-anneal` |

---

## Anti-Striping Round (2026-04-30)

Addresses the **black-grid / striping artefact** observed in Phase 3 outputs after
val-PSNR climbed past ~30 dB on DIV2K + Flickr2K. Root cause analysis (architectural):
the model was finding the **cheapest mathematical shortcut** to push PSNR — regular
high-frequency oscillations on the PixelShuffle sub-pixel grid — instead of true
high-frequency detail.

Two changes target this directly.

---

### 9. Post-shuffle Smoothing Conv (`src/model.py`)

**Problem:** ICNR initialisation only ensures the *initial* state of the four
sub-pixel kernels is symmetric. As training proceeds the kernels drift apart,
and any persistent asymmetry produces a regular checkerboard / striping pattern
that PSNR actually *rewards* (it counts as legitimate high-frequency energy).

**Fix:** After each `PixelShuffle(2)` insert a lightweight 3×3 conv:
```
Conv2d(64, 256, 3) → PixelShuffle(2) → Conv2d(64, 64, 3) → PReLU
                                       ^^^ NEW post-smooth ^^^
```
The smooth conv blends the four sub-pixels of every 2×2 output tile, so any
sub-pixel imbalance gets averaged away before it propagates further.

**Key choices:**
- 3×3 kernel: enough receptive field to mix all four sub-pixels plus a small
  neighbourhood; 1×1 would only re-weight per-pixel and can't truly smooth.
- Kept 64→64 channels: no capacity blow-up.
- ICNR is *only* applied to the conv immediately before each PixelShuffle.
  The new smoothing convs use default Kaiming init (they're not the symmetry
  bottleneck).
- Applied to `TemporalSRResNet` (P1), `WarpTSRNet` (P2) and
  `RecurrentTSRNet` (P3) — all three use PixelShuffle.

**Core idea:** Treat PixelShuffle as a *signal-processing* step, not a learning
step. Add an explicit anti-aliasing layer right after sub-pixel rearrangement
so checkerboard patterns are absorbed before they ever reach the loss.

⚠️ **Architecture change** — old checkpoints (without the smoothing conv)
cannot be loaded into the new model.

---

### 10. Total Variation (TV) Loss (`src/train.py`)

**Problem:** Pixel-wise loss alone (Charbonnier / MSE) is *agnostic* to the
*pattern* of error. A regular striping pattern and a smooth gradient with the
same MSE look identical to the loss — but only one is what we want.

**Fix:** Added `tv_loss(x)` (anisotropic L1 form):
```
L_TV = mean(|x[i+1,j] − x[i,j]|) + mean(|x[i,j+1] − x[i,j]|)
```
A small weighted TV term is added to the per-frame pixel loss in all three
phases:
```
loss = pixel_loss + tv_loss_weight * L_TV(out)   [+ temp_loss for P3]
```

**Key choices:**
- **L1 form (not L2):** L2 over-penalises real edges; L1 only suppresses
  high-frequency *low-amplitude* oscillation, which is exactly the striping.
- **Recommended weight 1e-6 to 1e-4:** below this no effect, above ~1e-3
  outputs become visibly soft. Default 0.0 (opt-in).
- **Applies to all phases:** PixelShuffle striping is an architecture-level
  problem, not Phase-3-specific.
- Controlled by `--tv-loss-weight` (e.g. `--tv-loss-weight 1e-5`).

**Core idea:** Pattern-aware regularisation. Make "smooth" cheaper than
"striped" in loss space, so the model can no longer exploit the
sub-pixel-grid shortcut to inflate PSNR.

---

### Recommended config for retraining (anti-striping)

```bash
python src/train.py --phase 3 --epochs 600 \
    --batch-size 4 --seq-len 4 --div2k --flickr2k \
    --sched-sampling --curriculum \
    --temp-loss-weight 0.1 \
    --tv-loss-weight  1e-5 \
    --grad-clip 1.0
```

Start TV weight at `1e-5`. If striping still visible after ~300 epochs, raise
to `5e-5`. If output looks washed out, drop to `1e-6` or disable.

---

## Phase 3 Enhancement (2026-04-26)

Addresses two main failure modes observed in Phase 3:
- **Temporal flickering / banding stripes** in output video
- **Exposure bias**: model trained on GT history but tested with own (imperfect) output

The following changes were applied in priority order (highest impact first).

---

### 1. Temporal Consistency Loss (`src/train.py`)

**Problem:** The pixel-wise loss (MSE) treats each output frame independently.
There is no explicit constraint that consecutive frames must be geometrically consistent —
this allows the model to produce frames that look good individually but flicker when played back.

**Fix:** Added `temporal_consistency_loss`:
```
L_temp = mean( |out_curr − warp(out_prev, flow_hr)| × occlusion_mask )
```
If the model's previous output is warped to align with the current frame, it should
resemble the current output. Any residual difference is penalised.

**Key choices:**
- **L1/Charbonnier (not MSE):** MSE over-penalises legitimate occlusion boundaries.
- **Occlusion mask:** pixels that warp outside the frame are excluded.
- **`out_prev.detach()`:** prevents gradients from propagating back through time — avoids explosive gradients.
- Weight controlled by `--temp-loss-weight` (recommended 0.05–0.2, default 0.0 for backward compat).

**Core idea:** Make temporal coherence an explicit training objective, not just an emergent property.

---

### 2. Gradient Clipping (`src/train.py`)

**Problem:** Long sequences (T ≥ 6) unroll a deep computation graph through the recurrent loop.
Exploding gradients become increasingly likely, causing sudden loss spikes and divergence.

**Fix:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)` before every `optimizer.step()`.

**Core idea:** Bound gradient magnitude so no single bad batch can derail training.
Default `--grad-clip 1.0` (0 = disabled).

---

### 3. Charbonnier Loss (`src/train.py`)

**Problem:** MSE loss heavily penalises large errors and is known to produce blurry outputs
because averaging is the MSE-optimal prediction for uncertain regions.

**Fix:** Replaced `nn.MSELoss` with `charbonnier_loss(pred, target, eps=1e-3)`:
```
L = mean( sqrt((pred − target)^2 + eps^2) )
```
Applied to **all three phases** (P1, P2, P3).

**Core idea:** Charbonnier is a smooth approximation of L1. It is robust to outliers
like MSE but encourages sharper predictions because it does not reward extreme blurring.
Used by EDSR, BasicVSR, and most modern SR networks.

---

### 4. Scheduled Sampling (`src/train.py`)

**Problem (Exposure Bias):** During training, the model always receives ground-truth HR history (`hr_prev = hr_seq[:, t-1]`).
At inference, it receives its own (imperfect) previous output. This distribution mismatch
causes quality to degrade more than expected during true recurrent inference.

**Fix:** Introduced `scheduled_sampling_prob(epoch, ramp_epochs, max_prob)` — a linear ramp:
- Epoch 0: `p_self = 0` → pure teacher-forcing (stable early training)
- Epoch `ramp_epochs`: `p_self → 0.9` → mostly self-feedback

Per time step, with probability `p_self` the model receives its own detached previous
output; otherwise it receives GT.

**Core idea:** Gradually expose the model to its own errors during training so it learns
to be robust to them. Avoids cold-start instability of jumping straight to self-feedback.
Controlled by `--sched-sampling` (flag) and `--sched-sample-ramp` (epochs, default 2000).

---

### 5. Curriculum Sequence Length (`src/train.py`, `src/dataset.py`)

**Problem:** Training with long sequences (T=8+) from the start is expensive and can
destabilise early learning because the model has not yet learned reliable per-frame SR.

**Fix:** Added `get_curriculum_seq_len(epoch, total_epochs)` — a three-stage schedule:

| Training progress | `seq_len` | Purpose |
|---|---|---|
| 0 – 20% | 4 | Fast convergence, simple temporal patterns |
| 20% – 60% | 6 | Medium-range dependencies |
| 60% – 100% | 8 | Long-range drift and flicker suppression |

`SRSequenceDataset.seq_len` is updated in-place between epochs — no DataLoader rebuild needed.
Enabled by `--curriculum` flag. `num_workers=0` is required (workers would cache the old `seq_len`).

**Core idea:** Curriculum learning: start with easier (shorter) tasks, progressively increase difficulty.

---

### 6. Gradient Accumulation (`src/train.py`)

**Problem:** Longer sequences (T=8) at the same batch size require proportionally more VRAM.
Users with limited GPU memory would need to reduce batch size to 1, losing training stability.

**Fix:** `--grad-accum N` accumulates gradients over N batches before calling `optimizer.step()`.
`effective_batch = batch_size × grad_accum`. Loss is normalised by `grad_accum` to keep gradient
magnitude consistent regardless of accumulation factor.

**Core idea:** Trade compute time for memory — run N smaller batches but treat them as one large batch.

---

### 7. ICNR Initialization for PixelShuffle (`src/model.py`)

**Problem:** The default random initialisation of Conv2d layers before PixelShuffle assigns
independent weights to each of the `r^2` sub-pixel outputs. This initial asymmetry manifests
as a checkerboard pattern in the output at epoch 0 and can persist if not corrected by training.

**Fix:** `icnr_init(conv, scale_factor)` — the `r^2` sub-kernels are all initialised to the same
Kaiming-normal values:
```
sub_kernel = kaiming_normal(out_ch / r^2, in_ch, kH, kW)
full_kernel = tile(sub_kernel, times=r^2, dim=0)
```
Applied to all upsampling `Conv2d` layers in `TemporalSRResNet`, `WarpTSRNet`, and `RecurrentTSRNet`.

**Core idea:** Equal sub-pixel initialisation → PixelShuffle outputs uniform tiles → no checkerboard bias.

---

### 8. Dataset Expansion Support (`src/dataset.py`, `src/train.py`)

**Problem:** 5–100 training images are insufficient for a generalisable SR model. Overfitting
causes high validation variance and poor PSNR on unseen content.

**Fix:**
- `SRTemporalDataset` and `SRSequenceDataset` now accept a **list of directories** in addition
  to a single path. Images from all directories are merged into one flat pool.
- Added `download_flickr2k(output_dir)` — downloads Flickr2K (~2650 images, ~3.2 GB).
- New `--flickr2k` flag in `train.py`. Can be combined with `--div2k` for ~3450 total images
  (the standard SR training corpus used by EDSR, BasicVSR, RealESRGAN).

**Core idea:** More diverse training data → better generalisation, more stable validation metrics.

---

## Summary Table

| Change | Files | Key Flag / API |
|---|---|---|
| Temporal consistency loss | `train.py` | `--temp-loss-weight 0.1` |
| Gradient clipping | `train.py` | `--grad-clip 1.0` |
| Charbonnier loss | `train.py` | (always on) |
| Scheduled sampling | `train.py` | `--sched-sampling --sched-sample-ramp 2000` |
| Curriculum seq_len | `train.py`, `dataset.py` | `--curriculum` |
| Gradient accumulation | `train.py` | `--grad-accum 4` |
| ICNR init | `model.py` | (always on) |
| Multi-dir + Flickr2K | `dataset.py`, `train.py` | `--flickr2k` |
| Post-shuffle smoothing conv | `model.py` | (always on, all phases) |
| Total Variation (TV) loss | `train.py` | `--tv-loss-weight 1e-5` |
