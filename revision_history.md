# Revision History

A concise record of each improvement round, its motivation, approach, and core idea.

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
