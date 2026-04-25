import torch
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB. Expects tensors in [0, 1], any shape."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM. Expects (C, H, W) tensors in [0, 1]."""
    pred_np = pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    target_np = target.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return skimage_ssim(pred_np, target_np, data_range=1.0, channel_axis=2)
