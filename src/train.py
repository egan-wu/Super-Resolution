"""
Unified training script for all three phases.

Phase 1 (EarlyFusion):
  python src/train.py --phase 1 --epochs 5000 --div2k

Phase 2 (WarpThenFuse):
  python src/train.py --phase 2 --epochs 5000 --div2k

Phase 3 (Recurrent, DLSS-style) — recommended full config:
  python src/train.py --phase 3 --epochs 5000 --div2k --flickr2k \
      --sched-sampling --curriculum --temp-loss-weight 0.1 \
      --grad-clip 1.0 --grad-accum 1 --seq-len 4

OOM fallback (simulate larger batch with accumulation):
  python src/train.py --phase 3 --epochs 5000 --div2k \
      --batch-size 1 --grad-accum 4
"""

import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import (SRTemporalDataset, SRSequenceDataset,
                     download_sample_images, download_div2k, download_flickr2k)
from model import TemporalSRResNet, WarpTSRNet, RecurrentTSRNet
from utils import get_device, compute_psnr, compute_ssim
from warp import backward_warp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-3) -> torch.Tensor:
    """
    Charbonnier loss: sqrt((pred - target)^2 + eps^2).
    Smooth L1 variant — less sensitive to outliers than MSE,
    preserves sharp edges better than pure L2.
    """
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Total Variation (TV) loss — anisotropic L1 form:
        L_TV = mean( |x[i+1,j] - x[i,j]| ) + mean( |x[i,j+1] - x[i,j]| )

    Penalises abrupt pixel-to-pixel jumps. The PixelShuffle checkerboard /
    striping artefact is exactly that — a regular high-frequency oscillation
    on the sub-pixel grid — so a small TV weight strongly suppresses it
    without softening real edges (those are infrequent isolated jumps,
    while artefacts are dense regular patterns).

    Recommended weight: 1e-6 to 1e-3 (start at 1e-5 and tune).
    Too high → blurry output; too low → no effect.
    """
    diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()


def temporal_consistency_loss(out_curr: torch.Tensor,
                               out_prev: torch.Tensor,
                               flow_lr: torch.Tensor,
                               scale: int,
                               eps: float = 1e-3) -> torch.Tensor:
    """
    Temporal consistency loss:
        L_temp = mean( |out_curr - warp(out_prev, flow_hr)| * occlusion_mask )

    Penalises flickering: if we warp the previous HR output to align with the
    current frame, it should look like the current output. Only non-occluded
    pixels (mask=1) are penalised — out-of-frame regions are excluded.

    Using Charbonnier (not MSE) so occlusion-boundary blur is tolerated.
    """
    # Scale flow from LR pixel space to HR pixel space
    if flow_lr.dim() == 2:
        flow_hr = flow_lr * scale                           # (B, 2)
    else:
        flow_hr = F.interpolate(
            flow_lr, scale_factor=scale,
            mode="bilinear", align_corners=False
        ) * scale                                           # (B, 2, H_hr, W_hr)

    warped_prev, mask = backward_warp(out_prev, flow_hr)
    diff = (out_curr - warped_prev) * mask
    return torch.sqrt(diff * diff + eps * eps).mean()


# ---------------------------------------------------------------------------
# Curriculum & scheduled-sampling helpers
# ---------------------------------------------------------------------------

def get_curriculum_seq_len(epoch: int, total_epochs: int) -> int:
    """
    Progressive sequence length schedule:
      0  –  20% of total : seq_len = 4   (fast convergence, simple patterns)
      20% – 60% of total : seq_len = 6   (medium-range temporal dependencies)
      60% – 100%         : seq_len = 8   (long-range drift suppression)

    seq_len=8 is used as the cap (vs the original plan's 16) because
    LR crop_size=128 → HR=512, and seq_len=16 requires ~8× more VRAM
    than seq_len=4 with the same batch size. Use --grad-accum if needed.
    """
    progress = epoch / total_epochs
    if progress < 0.20:
        return 4
    elif progress < 0.60:
        return 6
    else:
        return 8


def scheduled_sampling_prob(epoch: int, ramp_epochs: int = 2000,
                             max_prob: float = 0.9) -> float:
    """
    Linear ramp: 0.0 at epoch 0 → max_prob at ramp_epochs.
    After ramp_epochs the probability stays at max_prob.

    Why not jump straight to self-feedback?
    Early in training the model output is noisy — feeding it back compounds
    errors and destabilises learning. The ramp lets the model first learn a
    reasonable mapping from GT history, then gradually adapt to its own
    (increasingly good) outputs, closing the train-inference gap.
    """
    return min(epoch / max(ramp_epochs, 1), 1.0) * max_prob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


def _log(epoch, total, train_loss, val_loss=None, val_psnr=None,
         val_ssim=None, is_best=False, extra=""):
    if val_psnr is not None:
        msg = (f"Epoch {epoch:5d}/{total} | Train: {train_loss:.4f} | "
               f"Val: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")
        if extra:
            msg += f" | {extra}"
        if is_best:
            msg += "  ★ best"
        print(msg)
    else:
        msg = f"Epoch {epoch:5d}/{total} | Train: {train_loss:.4f}"
        if extra:
            msg += f" | {extra}"
        print(msg)


# ---------------------------------------------------------------------------
# Validation loops
# ---------------------------------------------------------------------------

def validate_p1(model, loader, device):
    model.eval()
    loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for lr_prev, lr_curr, hr_curr, _ in loader:
            lr_prev, lr_curr, hr_curr = (
                lr_prev.to(device), lr_curr.to(device), hr_curr.to(device))
            out = model(lr_prev, lr_curr)
            loss_sum += charbonnier_loss(out, hr_curr).item()
            for i in range(out.size(0)):
                psnr_sum += compute_psnr(out[i].cpu(), hr_curr[i].cpu())
                ssim_sum += compute_ssim(out[i].cpu(), hr_curr[i].cpu())
                count += 1
    n = max(len(loader), 1)
    return loss_sum / n, psnr_sum / max(count, 1), ssim_sum / max(count, 1)


def validate_p2(model, loader, device):
    model.eval()
    loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for lr_prev, lr_curr, hr_curr, flow in loader:
            lr_prev, lr_curr, hr_curr, flow = (
                lr_prev.to(device), lr_curr.to(device),
                hr_curr.to(device), flow.to(device))
            out = model(lr_prev, lr_curr, flow)
            loss_sum += charbonnier_loss(out, hr_curr).item()
            for i in range(out.size(0)):
                psnr_sum += compute_psnr(out[i].cpu(), hr_curr[i].cpu())
                ssim_sum += compute_ssim(out[i].cpu(), hr_curr[i].cpu())
                count += 1
    n = max(len(loader), 1)
    return loss_sum / n, psnr_sum / max(count, 1), ssim_sum / max(count, 1)


def validate_p3(model, loader, device, scale_factor):
    """
    Validation uses teacher-forcing (GT hr_prev) for stable, comparable metrics.
    Inference uses self-feedback — the train-val gap will shrink as scheduled
    sampling probability rises during training.
    """
    model.eval()
    loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for lr_seq, hr_seq, flow_seq in loader:
            lr_seq   = lr_seq.to(device)
            hr_seq   = hr_seq.to(device)
            flow_seq = flow_seq.to(device)
            B, T = lr_seq.shape[:2]
            for t in range(T):
                hr_prev = (bicubic_upsample(lr_seq[:, 0], scale_factor)
                           if t == 0 else hr_seq[:, t - 1])
                out = model(lr_seq[:, t], hr_prev, flow_seq[:, t])
                loss_sum += charbonnier_loss(out, hr_seq[:, t]).item()
                for i in range(B):
                    psnr_sum += compute_psnr(out[i].cpu(), hr_seq[i, t].cpu())
                    ssim_sum += compute_ssim(out[i].cpu(), hr_seq[i, t].cpu())
                    count += 1
    n = max(len(loader) * T, 1)
    return loss_sum / n, psnr_sum / max(count, 1), ssim_sum / max(count, 1)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_phase1(args, device, train_loader, val_loader):
    model     = TemporalSRResNet(scale_factor=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_psnr, best_path = 0.0, os.path.join(args.save_dir, "p1_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for lr_prev, lr_curr, hr_curr, _ in tqdm(train_loader, desc=f"P1 Ep{epoch}", leave=False):
            lr_prev, lr_curr, hr_curr = (
                lr_prev.to(device), lr_curr.to(device), hr_curr.to(device))
            out  = model(lr_prev, lr_curr)
            loss = charbonnier_loss(out, hr_curr)
            if args.tv_loss_weight > 0:
                loss = loss + args.tv_loss_weight * tv_loss(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg = loss_sum / len(train_loader)
        is_milestone = epoch % args.val_every == 0 or epoch == args.epochs
        if is_milestone:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"p1_epoch_{epoch}.pth"))
            vl, vp, vs = validate_p1(model, val_loader, device)
            is_best = vp > best_psnr
            if is_best:
                best_psnr = vp
                torch.save(model.state_dict(), best_path)
            _log(epoch, args.epochs, avg, vl, vp, vs, is_best)
        else:
            _log(epoch, args.epochs, avg)

    print(f"\nPhase 1 done. Best Val PSNR: {best_psnr:.2f} dB  →  {best_path}")


def train_phase2(args, device, train_loader, val_loader):
    model     = WarpTSRNet(scale_factor=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_psnr, best_path = 0.0, os.path.join(args.save_dir, "p2_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for lr_prev, lr_curr, hr_curr, flow in tqdm(train_loader, desc=f"P2 Ep{epoch}", leave=False):
            lr_prev, lr_curr, hr_curr, flow = (
                lr_prev.to(device), lr_curr.to(device),
                hr_curr.to(device), flow.to(device))
            out  = model(lr_prev, lr_curr, flow)
            loss = charbonnier_loss(out, hr_curr)
            if args.tv_loss_weight > 0:
                loss = loss + args.tv_loss_weight * tv_loss(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg = loss_sum / len(train_loader)
        is_milestone = epoch % args.val_every == 0 or epoch == args.epochs
        if is_milestone:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"p2_epoch_{epoch}.pth"))
            vl, vp, vs = validate_p2(model, val_loader, device)
            is_best = vp > best_psnr
            if is_best:
                best_psnr = vp
                torch.save(model.state_dict(), best_path)
            _log(epoch, args.epochs, avg, vl, vp, vs, is_best)
        else:
            _log(epoch, args.epochs, avg)

    print(f"\nPhase 2 done. Best Val PSNR: {best_psnr:.2f} dB  →  {best_path}")


def train_phase3(args, device, train_loader, val_loader, full_ds):
    """
    Phase 3 training with four key improvements:

    1. Charbonnier loss       — replaces MSE; better edge detail
    2. Scheduled sampling     -- linear ramp from teacher-forcing → self-feedback,
                                 closing the exposure-bias gap
    3. Temporal consistency   -- L_temp penalises frame-to-frame flicker directly
    4. Gradient clipping      -- prevents exploding gradients with long sequences
    5. Gradient accumulation  -- simulates large batch when VRAM is limited
    6. Curriculum seq_len     -- progressively longer sequences (4 → 6 → 8)
    """
    scale     = 4
    model     = RecurrentTSRNet(scale_factor=scale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_psnr, best_path = 0.0, os.path.join(args.save_dir, "p3_best.pth")

    current_seq_len = args.seq_len

    for epoch in range(1, args.epochs + 1):

        # ------------------------------------------------------------------
        # Curriculum: update dataset seq_len at stage boundaries
        # (works because SRSequenceDataset reads self.seq_len in __getitem__)
        # ------------------------------------------------------------------
        if args.curriculum:
            new_seq_len = get_curriculum_seq_len(epoch, args.epochs)
            if new_seq_len != current_seq_len:
                current_seq_len = new_seq_len
                full_ds.seq_len = current_seq_len
                print(f"\n[Curriculum] seq_len → {current_seq_len} at epoch {epoch}")

        # ------------------------------------------------------------------
        # Scheduled sampling: probability of using model's own output as hr_prev
        # 0.0 = pure teacher-forcing; 0.9 = mostly self-feedback
        # ------------------------------------------------------------------
        p_self = (scheduled_sampling_prob(epoch, args.sched_sample_ramp)
                  if args.sched_sampling else 0.0)

        model.train()
        loss_sum = 0.0
        optimizer.zero_grad()

        for step, (lr_seq, hr_seq, flow_seq) in enumerate(
                tqdm(train_loader, desc=f"P3 Ep{epoch} T={current_seq_len}", leave=False)):

            lr_seq   = lr_seq.to(device)    # (B, T, 3, H_lr, W_lr)
            hr_seq   = hr_seq.to(device)    # (B, T, 3, H_hr, W_hr)
            flow_seq = flow_seq.to(device)  # (B, T, 2)
            B, T     = lr_seq.shape[:2]

            seq_loss = torch.zeros(1, device=device)
            out_prev = None                  # model's own previous output

            for t in range(T):
                # --- Select hr_prev input ---
                if t == 0:
                    # Frame 0: no history — use bicubic as warm-start anchor
                    hr_prev_in = bicubic_upsample(lr_seq[:, 0], scale)
                else:
                    # t ≥ 1: probabilistically mix GT (teacher) vs self-feedback
                    use_self = (out_prev is not None) and (random.random() < p_self)
                    hr_prev_in = out_prev.detach() if use_self else hr_seq[:, t - 1]

                out = model(lr_seq[:, t], hr_prev_in, flow_seq[:, t])

                # --- Pixel loss (Charbonnier) ---
                pix_loss = charbonnier_loss(out, hr_seq[:, t])

                # --- TV loss (suppress PixelShuffle checkerboard / striping) ---
                if args.tv_loss_weight > 0:
                    pix_loss = pix_loss + args.tv_loss_weight * tv_loss(out)

                # --- Temporal consistency loss (t ≥ 1) ---
                if t > 0 and out_prev is not None and args.temp_loss_weight > 0:
                    t_loss = temporal_consistency_loss(
                        out, out_prev.detach(), flow_seq[:, t], scale
                    )
                    pix_loss = pix_loss + args.temp_loss_weight * t_loss

                seq_loss = seq_loss + pix_loss
                out_prev = out                   # keep for next frame (detach in t+1)

            # Average over time steps; normalise for gradient accumulation
            loss = seq_loss / T / args.grad_accum
            loss.backward()

            # Step after every grad_accum batches (or at end of epoch)
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            loss_sum += (seq_loss / T).item()   # log unscaled loss

        avg = loss_sum / len(train_loader)
        is_milestone = epoch % args.val_every == 0 or epoch == args.epochs
        if is_milestone:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"p3_epoch_{epoch}.pth"))
            vl, vp, vs = validate_p3(model, val_loader, device, scale)
            is_best = vp > best_psnr
            if is_best:
                best_psnr = vp
                torch.save(model.state_dict(), best_path)
            extra = f"p_self={p_self:.2f} T={current_seq_len}"
            _log(epoch, args.epochs, avg, vl, vp, vs, is_best, extra=extra)
        else:
            extra = f"p_self={p_self:.2f}"
            _log(epoch, args.epochs, avg, extra=extra)

    print(f"\nPhase 3 done. Best Val PSNR: {best_psnr:.2f} dB  →  {best_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SR model (Phase 1 / 2 / 3)")

    # --- Core ---
    parser.add_argument("--phase",      type=int,   default=3, choices=[1, 2, 3])
    parser.add_argument("--epochs",     type=int,   default=5000)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val-every",  type=int,   default=500)
    parser.add_argument("--seq-len",    type=int,   default=4,
                        help="Initial sequence length (Phase 3 only)")
    parser.add_argument("--save-dir",   type=str,   default="checkpoints")

    # --- Dataset ---
    parser.add_argument("--data-dir",  type=str, default="",
                        help="Custom dataset directory (overrides --div2k / --flickr2k)")
    parser.add_argument("--div2k",     action="store_true",
                        help="Download and use DIV2K validation HR (~770 MB, 100 images)")
    parser.add_argument("--flickr2k",  action="store_true",
                        help="Download and use Flickr2K HR (~3.2 GB, 2650 images). "
                             "Combined with --div2k for ~3450 image training pool.")

    # --- Phase 3 specific ---
    parser.add_argument("--sched-sampling",   action="store_true",
                        help="Enable scheduled sampling (exposure-bias fix)")
    parser.add_argument("--sched-sample-ramp", type=int, default=2000,
                        help="Epoch at which scheduled-sampling probability reaches 0.9")
    parser.add_argument("--curriculum",       action="store_true",
                        help="Enable curriculum seq_len: 4 → 6 → 8 across training")
    parser.add_argument("--temp-loss-weight", type=float, default=0.0,
                        help="Weight for temporal consistency loss (0 = disabled). "
                             "Recommended: 0.05–0.2")
    parser.add_argument("--tv-loss-weight",   type=float, default=0.0,
                        help="Weight for Total Variation (TV) loss to suppress "
                             "PixelShuffle checkerboard/striping artefacts. "
                             "0 = disabled. Recommended: 1e-6 to 1e-4. "
                             "Applies to all phases.")
    parser.add_argument("--grad-clip",  type=float, default=1.0,
                        help="Max gradient norm (0 = disabled)")
    parser.add_argument("--grad-accum", type=int,   default=1,
                        help="Gradient accumulation steps (OOM fallback). "
                             "Effective batch = batch-size × grad-accum")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()
    print(f"Phase {args.phase} | device: {device} | epochs: {args.epochs} | "
          f"batch: {args.batch_size} × accum {args.grad_accum}")

    # --- Dataset setup ---
    if args.data_dir:
        data_dirs = [args.data_dir]
    else:
        data_dirs = []
        if args.div2k:
            download_div2k(output_dir="data/div2k", split="valid", max_images=100)
            data_dirs.append("data/div2k")
        if args.flickr2k:
            download_flickr2k(output_dir="data/flickr2k")
            data_dirs.append("data/flickr2k")
        if not data_dirs:
            download_sample_images(output_dir="data/samples")
            data_dirs = ["data/samples"]

    if args.phase in (1, 2):
        full_ds = SRTemporalDataset(data_dirs, scale_factor=4,
                                    crop_size=128, max_offset=8)
    else:
        full_ds = SRSequenceDataset(data_dirs, scale_factor=4, crop_size=128,
                                    max_offset=8, seq_len=args.seq_len,
                                    use_jitter=True)

    val_size   = max(1, int(0.2 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # num_workers=0 required for curriculum seq_len updates to propagate
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    total_images = sum(len(os.listdir(d)) for d in data_dirs
                       if os.path.isdir(d))
    print(f"Dataset dirs: {data_dirs}  |  ~{total_images} images  "
          f"|  train={train_size}  val={val_size}")

    if args.phase == 3:
        print(f"Phase 3 options: sched_sampling={args.sched_sampling} "
              f"(ramp={args.sched_sample_ramp}) | curriculum={args.curriculum} | "
              f"temp_loss_weight={args.temp_loss_weight} | "
              f"tv_loss_weight={args.tv_loss_weight} | grad_clip={args.grad_clip}")

    # --- Train ---
    if args.phase == 1:
        train_phase1(args, device, train_loader, val_loader)
    elif args.phase == 2:
        train_phase2(args, device, train_loader, val_loader)
    else:
        train_phase3(args, device, train_loader, val_loader, full_ds)


if __name__ == "__main__":
    main()
