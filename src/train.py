"""
Unified training script for all three phases.

Usage:
  Phase 1 (EarlyFusion):
    python src/train.py --phase 1 --epochs 5000 --div2k

  Phase 2 (WarpThenFuse):
    python src/train.py --phase 2 --epochs 5000 --div2k

  Phase 3 (Recurrent, DLSS-style):
    python src/train.py --phase 3 --epochs 5000 --div2k --seq-len 4
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import SRTemporalDataset, SRSequenceDataset, download_sample_images, download_div2k
from model import TemporalSRResNet, WarpTSRNet, RecurrentTSRNet
from utils import get_device, compute_psnr, compute_ssim
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


# ---------------------------------------------------------------------------
# Validation loops (one per phase)
# ---------------------------------------------------------------------------

def validate_p1(model, loader, criterion, device):
    model.eval()
    loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for lr_prev, lr_curr, hr_curr, _ in loader:
            lr_prev = lr_prev.to(device)
            lr_curr = lr_curr.to(device)
            hr_curr = hr_curr.to(device)
            out = model(lr_prev, lr_curr)
            loss_sum += criterion(out, hr_curr).item()
            for i in range(out.size(0)):
                psnr_sum += compute_psnr(out[i].cpu(), hr_curr[i].cpu())
                ssim_sum += compute_ssim(out[i].cpu(), hr_curr[i].cpu())
                count += 1
    n = len(loader)
    return loss_sum / n, psnr_sum / count, ssim_sum / count


def validate_p2(model, loader, criterion, device):
    model.eval()
    loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for lr_prev, lr_curr, hr_curr, flow in loader:
            lr_prev = lr_prev.to(device)
            lr_curr = lr_curr.to(device)
            hr_curr = hr_curr.to(device)
            flow    = flow.to(device)
            out = model(lr_prev, lr_curr, flow)
            loss_sum += criterion(out, hr_curr).item()
            for i in range(out.size(0)):
                psnr_sum += compute_psnr(out[i].cpu(), hr_curr[i].cpu())
                ssim_sum += compute_ssim(out[i].cpu(), hr_curr[i].cpu())
                count += 1
    n = len(loader)
    return loss_sum / n, psnr_sum / count, ssim_sum / count


def validate_p3(model, loader, criterion, device, scale_factor):
    model.eval()
    loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for lr_seq, hr_seq, flow_seq in loader:
            lr_seq   = lr_seq.to(device)
            hr_seq   = hr_seq.to(device)
            flow_seq = flow_seq.to(device)
            B, T     = lr_seq.shape[:2]
            hr_prev  = bicubic_upsample(lr_seq[:, 0], scale_factor)
            for t in range(T):
                if t > 0:
                    hr_prev = hr_seq[:, t - 1]
                out = model(lr_seq[:, t], hr_prev, flow_seq[:, t])
                loss_sum += criterion(out, hr_seq[:, t]).item()
                for i in range(B):
                    psnr_sum += compute_psnr(out[i].cpu(), hr_seq[i, t].cpu())
                    ssim_sum += compute_ssim(out[i].cpu(), hr_seq[i, t].cpu())
                    count += 1
    n = len(loader) * T
    return loss_sum / n, psnr_sum / count, ssim_sum / count


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _log(epoch, total, train_loss, val_loss=None, val_psnr=None, val_ssim=None, is_best=False):
    if val_psnr is not None:
        msg = (f"Epoch {epoch:5d}/{total} | Train: {train_loss:.4f} | "
               f"Val: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")
        if is_best:
            msg += "  ★ best"
        print(msg)
    else:
        print(f"Epoch {epoch:5d}/{total} | Train: {train_loss:.4f}")


def train_phase1(args, device, train_loader, val_loader):
    model     = TemporalSRResNet(scale_factor=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_psnr, best_path = 0.0, os.path.join(args.save_dir, "p1_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for lr_prev, lr_curr, hr_curr, _ in tqdm(train_loader, desc=f"P1 Ep{epoch}", leave=False):
            lr_prev, lr_curr, hr_curr = lr_prev.to(device), lr_curr.to(device), hr_curr.to(device)
            out  = model(lr_prev, lr_curr)
            loss = criterion(out, hr_curr)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()

        avg = loss_sum / len(train_loader)
        is_milestone = epoch % args.val_every == 0 or epoch == args.epochs
        if is_milestone:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"p1_epoch_{epoch}.pth"))
            vl, vp, vs = validate_p1(model, val_loader, criterion, device)
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_psnr, best_path = 0.0, os.path.join(args.save_dir, "p2_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for lr_prev, lr_curr, hr_curr, flow in tqdm(train_loader, desc=f"P2 Ep{epoch}", leave=False):
            lr_prev, lr_curr, hr_curr, flow = (
                lr_prev.to(device), lr_curr.to(device), hr_curr.to(device), flow.to(device))
            out  = model(lr_prev, lr_curr, flow)
            loss = criterion(out, hr_curr)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()

        avg = loss_sum / len(train_loader)
        is_milestone = epoch % args.val_every == 0 or epoch == args.epochs
        if is_milestone:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"p2_epoch_{epoch}.pth"))
            vl, vp, vs = validate_p2(model, val_loader, criterion, device)
            is_best = vp > best_psnr
            if is_best:
                best_psnr = vp
                torch.save(model.state_dict(), best_path)
            _log(epoch, args.epochs, avg, vl, vp, vs, is_best)
        else:
            _log(epoch, args.epochs, avg)

    print(f"\nPhase 2 done. Best Val PSNR: {best_psnr:.2f} dB  →  {best_path}")


def train_phase3(args, device, train_loader, val_loader):
    scale     = 4
    model     = RecurrentTSRNet(scale_factor=scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_psnr, best_path = 0.0, os.path.join(args.save_dir, "p3_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for lr_seq, hr_seq, flow_seq in tqdm(train_loader, desc=f"P3 Ep{epoch}", leave=False):
            lr_seq, hr_seq, flow_seq = lr_seq.to(device), hr_seq.to(device), flow_seq.to(device)
            B, T = lr_seq.shape[:2]
            seq_loss = torch.tensor(0.0, device=device)
            for t in range(T):
                hr_prev = (bicubic_upsample(lr_seq[:, 0], scale)
                           if t == 0 else hr_seq[:, t - 1])
                out = model(lr_seq[:, t], hr_prev, flow_seq[:, t])
                seq_loss = seq_loss + criterion(out, hr_seq[:, t])
            loss = seq_loss / T
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()

        avg = loss_sum / len(train_loader)
        is_milestone = epoch % args.val_every == 0 or epoch == args.epochs
        if is_milestone:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"p3_epoch_{epoch}.pth"))
            vl, vp, vs = validate_p3(model, val_loader, criterion, device, scale)
            is_best = vp > best_psnr
            if is_best:
                best_psnr = vp
                torch.save(model.state_dict(), best_path)
            _log(epoch, args.epochs, avg, vl, vp, vs, is_best)
        else:
            _log(epoch, args.epochs, avg)

    print(f"\nPhase 3 done. Best Val PSNR: {best_psnr:.2f} dB  →  {best_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SR model (Phase 1 / 2 / 3)")
    parser.add_argument("--phase",      type=int,   default=3,          choices=[1, 2, 3])
    parser.add_argument("--epochs",     type=int,   default=5000)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val-every",  type=int,   default=500)
    parser.add_argument("--seq-len",    type=int,   default=4,          help="Phase 3 only")
    parser.add_argument("--save-dir",   type=str,   default="checkpoints")
    parser.add_argument("--data-dir",   type=str,   default="",
                        help="Custom dataset directory (overrides --div2k)")
    parser.add_argument("--div2k",      action="store_true",
                        help="Download and use DIV2K validation HR (100 images, ~770 MB)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()
    print(f"Phase {args.phase} | device: {device} | epochs: {args.epochs} | batch: {args.batch_size}")

    # --- dataset setup ---
    if args.data_dir:
        data_dir = args.data_dir
    elif args.div2k:
        data_dir = "data/div2k"
        download_div2k(output_dir=data_dir, split="valid", max_images=100)
    else:
        data_dir = "data/samples"
        download_sample_images(output_dir=data_dir)

    if args.phase in (1, 2):
        full_ds = SRTemporalDataset(data_dir, scale_factor=4, crop_size=128, max_offset=8)
    else:
        full_ds = SRSequenceDataset(data_dir, scale_factor=4, crop_size=128,
                                    max_offset=8, seq_len=args.seq_len, use_jitter=True)

    val_size   = max(1, int(0.2 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: {data_dir} | train={train_size} val={val_size} images")

    # --- train ---
    if args.phase == 1:
        train_phase1(args, device, train_loader, val_loader)
    elif args.phase == 2:
        train_phase2(args, device, train_loader, val_loader)
    else:
        train_phase3(args, device, train_loader, val_loader)


if __name__ == "__main__":
    main()
