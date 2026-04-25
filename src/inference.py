import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import RecurrentTSRNet
from utils import get_device, compute_psnr, compute_ssim
import matplotlib.pyplot as plt


def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


def run_inference(model_path, image_path, output_path, scale_factor=4,
                  offset=5, num_frames=6):
    """
    True recurrent inference: simulate `num_frames` frames of panning video.
    The model feeds its own HR output back as history each frame (no teacher forcing).
    """
    device = get_device()
    print(f"Using device: {device}")

    model = RecurrentTSRNet(scale_factor=scale_factor).to(device)
    if not os.path.exists(model_path):
        print(f"Error: checkpoint not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    raw = min(img.size) - offset * num_frames - 4
    crop_size = (raw // scale_factor) * scale_factor  # must be divisible by scale_factor
    cx = img.size[0] // 2 - crop_size // 2
    cy = img.size[1] // 2 - crop_size // 2

    lr_tf = transforms.Resize(
        (crop_size // scale_factor,) * 2,
        interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
    )

    hr_frames, sr_frames, bic_frames = [], [], []
    hr_prev_tensor = None

    for t in range(num_frames):
        pan = t * offset
        box = (cx + pan, cy + pan, cx + pan + crop_size, cy + pan + crop_size)
        hr_pil = img.crop(box)
        lr_pil = lr_tf(hr_pil)

        hr_t  = to_tensor(hr_pil)
        lr_t  = to_tensor(lr_pil)
        bic_t = bicubic_upsample(lr_t.unsqueeze(0), scale_factor).squeeze(0).clamp(0, 1)

        lr_input = lr_t.unsqueeze(0).to(device)

        if t == 0:
            # Bootstrap: no history, use bicubic as initial HR prev
            hr_prev_tensor = bicubic_upsample(lr_input, scale_factor)
            flow = torch.zeros(1, 2, device=device)
        else:
            # True recurrent: flow is constant diagonal pan
            flow_px = offset / scale_factor  # in LR pixels
            flow = torch.tensor([[flow_px, flow_px]], device=device)

        with torch.no_grad():
            hr_pred = model(lr_input, hr_prev_tensor, flow)
            hr_pred = hr_pred.clamp(0, 1)

        hr_prev_tensor = hr_pred  # feed back own output

        hr_frames.append(hr_t)
        sr_frames.append(hr_pred.squeeze(0).cpu())
        bic_frames.append(bic_t)

    # Compute metrics on the final frame (most accumulated history)
    hr_ref = hr_frames[-1]
    sr_out = sr_frames[-1]
    bic_out = bic_frames[-1]

    tsr_psnr = compute_psnr(sr_out, hr_ref)
    tsr_ssim = compute_ssim(sr_out, hr_ref)
    bic_psnr = compute_psnr(bic_out, hr_ref)
    bic_ssim = compute_ssim(bic_out, hr_ref)

    print(f"Final frame (t={num_frames-1}) metrics:")
    print(f"  Bicubic     — PSNR: {bic_psnr:.2f} dB | SSIM: {bic_ssim:.4f}")
    print(f"  RecurrentTSR— PSNR: {tsr_psnr:.2f} dB | SSIM: {tsr_ssim:.4f}")

    # Show first LR, bicubic baseline, and first 4 SR outputs to show accumulation
    show_frames = min(num_frames, 4)
    fig, axes = plt.subplots(2, show_frames, figsize=(6 * show_frames, 10))
    fig.suptitle(f"Phase 3 — RecurrentTSRNet ({num_frames}-frame accumulation)", fontsize=13, fontweight="bold")

    for i in range(show_frames):
        axes[0, i].imshow(to_pil(bic_frames[i]))
        axes[0, i].set_title(f"Bicubic frame {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(to_pil(sr_frames[i]))
        p = compute_psnr(sr_frames[i], hr_frames[i])
        axes[1, i].set_title(f"RecurTSR frame {i}\nPSNR: {p:.2f} dB")
        axes[1, i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 RecurrentTSRNet inference")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/recurrent_tsr_best.pth")
    parser.add_argument("--image",       type=str, default="data/samples/sample_00.jpg")
    parser.add_argument("--output",      type=str, default="assets/recurrent_comparison.png")
    parser.add_argument("--offset",      type=int, default=5)
    parser.add_argument("--num-frames",  type=int, default=6)
    args = parser.parse_args()

    run_inference(args.checkpoint, args.image, args.output,
                  offset=args.offset, num_frames=args.num_frames)
