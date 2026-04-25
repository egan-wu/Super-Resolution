"""
Unified inference script for all three phases.

Phase 1 / 2 — single image, two-frame input:
  python src/inference.py --phase 2 --checkpoint checkpoints/p2_best.pth

Phase 3 — recurrent multi-frame accumulation:
  python src/inference.py --phase 3 --checkpoint checkpoints/p3_best.pth --num-frames 6
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import TemporalSRResNet, WarpTSRNet, RecurrentTSRNet
from utils import get_device, compute_psnr, compute_ssim
import matplotlib.pyplot as plt


def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


# ---------------------------------------------------------------------------
# Phase 1 / 2 inference
# ---------------------------------------------------------------------------

def run_inference_p12(phase, model_path, image_path, output_path, scale_factor=4, offset=5):
    device = get_device()
    print(f"Using device: {device}")

    if phase == 1:
        model = TemporalSRResNet(scale_factor=scale_factor).to(device)
        label = "Phase 1 — EarlyFusion"
    else:
        model = WarpTSRNet(scale_factor=scale_factor).to(device)
        label = "Phase 2 — WarpThenFuse"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    crop_size = (min(img.size) - 20) // scale_factor * scale_factor
    cx = img.size[0] // 2 - crop_size // 2
    cy = img.size[1] // 2 - crop_size // 2

    curr_box = (cx, cy, cx + crop_size, cy + crop_size)
    prev_box = (cx - offset, cy - offset, cx - offset + crop_size, cy - offset + crop_size)
    hr_curr  = img.crop(curr_box)
    hr_prev  = img.crop(prev_box)

    lr_tf = transforms.Resize(
        (crop_size // scale_factor,) * 2,
        interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
    )
    lr_curr = lr_tf(hr_curr)
    lr_prev = lr_tf(hr_prev)
    bicubic = transforms.Resize(
        (crop_size,) * 2,
        interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
    )(lr_curr)

    tp = to_tensor(lr_prev).unsqueeze(0).to(device)
    tc = to_tensor(lr_curr).unsqueeze(0).to(device)
    hr = to_tensor(hr_curr)
    bic_t = to_tensor(bicubic)

    with torch.no_grad():
        if phase == 1:
            out = model(tp, tc)
        else:
            flow = torch.tensor([[offset / scale_factor, offset / scale_factor]],
                                dtype=torch.float32).to(device)
            out = model(tp, tc, flow)

    out = out.squeeze(0).cpu().clamp(0, 1)

    sr_psnr  = compute_psnr(out,   hr);     sr_ssim  = compute_ssim(out,   hr)
    bic_psnr = compute_psnr(bic_t, hr);     bic_ssim = compute_ssim(bic_t, hr)

    print(f"Bicubic  — PSNR: {bic_psnr:.2f} dB | SSIM: {bic_ssim:.4f}")
    print(f"{label} — PSNR: {sr_psnr:.2f} dB | SSIM: {sr_ssim:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(label, fontsize=13, fontweight="bold")
    axes[0].imshow(lr_prev);          axes[0].set_title(f"Frame t-1 (offset {offset}px)\n{lr_prev.size}"); axes[0].axis("off")
    axes[1].imshow(lr_curr);          axes[1].set_title(f"Frame t (input)\n{lr_curr.size}");              axes[1].axis("off")
    axes[2].imshow(bicubic);          axes[2].set_title(f"Bicubic\nPSNR: {bic_psnr:.2f} dB | SSIM: {bic_ssim:.4f}"); axes[2].axis("off")
    axes[3].imshow(to_pil(out));      axes[3].set_title(f"SR Output\nPSNR: {sr_psnr:.2f} dB | SSIM: {sr_ssim:.4f}");  axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# Phase 3 recurrent inference
# ---------------------------------------------------------------------------

def run_inference_p3(model_path, image_path, output_path, scale_factor=4,
                     offset=5, num_frames=6):
    device = get_device()
    print(f"Using device: {device}")

    model = RecurrentTSRNet(scale_factor=scale_factor).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    raw       = min(img.size) - offset * num_frames - 4
    crop_size = (raw // scale_factor) * scale_factor
    cx        = img.size[0] // 2 - crop_size // 2
    cy        = img.size[1] // 2 - crop_size // 2

    lr_tf = transforms.Resize(
        (crop_size // scale_factor,) * 2,
        interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
    )

    hr_frames, sr_frames, bic_frames = [], [], []
    hr_prev_tensor = None

    for t in range(num_frames):
        pan = t * offset
        box    = (cx + pan, cy + pan, cx + pan + crop_size, cy + pan + crop_size)
        hr_pil = img.crop(box)
        lr_pil = lr_tf(hr_pil)

        hr_t  = to_tensor(hr_pil)
        lr_t  = to_tensor(lr_pil)
        bic_t = bicubic_upsample(lr_t.unsqueeze(0), scale_factor).squeeze(0).clamp(0, 1)

        lr_in = lr_t.unsqueeze(0).to(device)

        if t == 0:
            hr_prev_tensor = bicubic_upsample(lr_in, scale_factor)
            flow = torch.zeros(1, 2, device=device)
        else:
            flow_px = offset / scale_factor
            flow = torch.tensor([[flow_px, flow_px]], device=device)

        with torch.no_grad():
            hr_pred = model(lr_in, hr_prev_tensor, flow).clamp(0, 1)

        hr_prev_tensor = hr_pred
        hr_frames.append(hr_t)
        sr_frames.append(hr_pred.squeeze(0).cpu())
        bic_frames.append(bic_t)

    # Metrics on final frame (most accumulated history)
    tsr_psnr = compute_psnr(sr_frames[-1],  hr_frames[-1])
    tsr_ssim = compute_ssim(sr_frames[-1],  hr_frames[-1])
    bic_psnr = compute_psnr(bic_frames[-1], hr_frames[-1])
    bic_ssim = compute_ssim(bic_frames[-1], hr_frames[-1])

    print(f"Final frame (t={num_frames-1}) metrics:")
    print(f"  Bicubic       — PSNR: {bic_psnr:.2f} dB | SSIM: {bic_ssim:.4f}")
    print(f"  RecurrentTSR  — PSNR: {tsr_psnr:.2f} dB | SSIM: {tsr_ssim:.4f}")

    show = min(num_frames, 4)
    fig, axes = plt.subplots(2, show, figsize=(6 * show, 10))
    fig.suptitle(f"Phase 3 — RecurrentTSRNet ({num_frames}-frame accumulation)",
                 fontsize=13, fontweight="bold")

    for i in range(show):
        axes[0, i].imshow(to_pil(bic_frames[i]))
        axes[0, i].set_title(f"Bicubic frame {i}")
        axes[0, i].axis("off")

        p = compute_psnr(sr_frames[i], hr_frames[i])
        axes[1, i].imshow(to_pil(sr_frames[i]))
        axes[1, i].set_title(f"RecurTSR frame {i}\nPSNR: {p:.2f} dB")
        axes[1, i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SR inference (Phase 1 / 2 / 3)")
    parser.add_argument("--phase",      type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/p3_best.pth")
    parser.add_argument("--image",      type=str, default="data/samples/sample_00.jpg")
    parser.add_argument("--output",     type=str, default="assets/output.png")
    parser.add_argument("--offset",     type=int, default=5,
                        help="Simulated camera pan in pixels (Phase 1/2)")
    parser.add_argument("--num-frames", type=int, default=6,
                        help="Number of recurrent frames to accumulate (Phase 3)")
    args = parser.parse_args()

    if args.phase in (1, 2):
        run_inference_p12(args.phase, args.checkpoint, args.image, args.output,
                          offset=args.offset)
    else:
        run_inference_p3(args.checkpoint, args.image, args.output,
                         offset=args.offset, num_frames=args.num_frames)
