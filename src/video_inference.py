"""
Video Super-Resolution inference for Phase 1, 2, and 3.

Phase 1 — Early Fusion (no optical flow):
  python src/video_inference.py --phase 1 --checkpoint checkpoints/p1_best.pth

Phase 2 — Warp-then-Fuse (Farneback optical flow at LR space):
  python src/video_inference.py --phase 2 --checkpoint checkpoints/p2_best.pth

Phase 3 — Recurrent DLSS-style (Farneback + HR feedback loop):
  python src/video_inference.py --phase 3 --checkpoint checkpoints/p3_best.pth

To download a test video first:
  python -c "import sys; sys.path.insert(0,'src'); from dataset import download_test_video; download_test_video()"
"""

import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from model import TemporalSRResNet, WarpTSRNet, RecurrentTSRNet
from utils import get_device


# ---------------------------------------------------------------------------
# Optical flow helper
# ---------------------------------------------------------------------------

def farneback_backward_flow(prev_gray: np.ndarray,
                             curr_gray: np.ndarray) -> torch.Tensor:
    """
    Estimate dense backward flow from prev_gray → curr_gray using Farneback.

    Returns (2, H, W) float32 tensor [dx, dy] in pixel space (backward convention:
    for output pixel (i,j), sample prev at (j+dx, i+dy)).
    """
    fwd = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )  # (H, W, 2)  forward flow: prev → curr
    bwd = -fwd   # negate → backward flow: curr samples from prev
    return torch.from_numpy(bwd.transpose(2, 0, 1)).float()  # (2, H, W)


# ---------------------------------------------------------------------------
# Video SR pipeline
# ---------------------------------------------------------------------------

def run_video_sr(phase: int,
                 model_path: str,
                 input_video: str,
                 output_video: str,
                 scale_factor: int = 4,
                 hidden_channels: int = 64,
                 pad_mode: str = "reflection"):
    """
    Run frame-by-frame SR on `input_video` and write side-by-side comparison
    (Bicubic | SR) to `output_video`.
    """
    device = get_device()
    print(f"Phase {phase} video SR | device: {device} | scale: {scale_factor}×")

    # Load model
    if phase == 1:
        model = TemporalSRResNet(scale_factor=scale_factor,
                                  hidden_channels=hidden_channels).to(device)
    elif phase == 2:
        model = WarpTSRNet(scale_factor=scale_factor,
                            hidden_channels=hidden_channels,
                            pad_mode=pad_mode).to(device)
    else:
        model = RecurrentTSRNet(scale_factor=scale_factor,
                                 hidden_channels=hidden_channels,
                                 pad_mode=pad_mode).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_W, out_H = W * scale_factor, H * scale_factor

    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (out_W * 2, out_H))

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    # Per-phase state
    prev_tensor = None
    prev_gray   = None
    hr_prev     = None   # Phase 3 only

    print(f"Input:  {input_video}  ({W}×{H}, {total_frames} frames)")
    print(f"Output: {output_video}  ({out_W * 2}×{out_H} side-by-side)")

    pbar = tqdm(total=total_frames, desc=f"Phase {phase} SR")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        curr_pil    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        curr_tensor = to_tensor(curr_pil).unsqueeze(0).to(device)
        curr_gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Bootstrap first frame
        if prev_tensor is None:
            prev_tensor = curr_tensor
            prev_gray   = curr_gray
            if phase == 3:
                hr_prev = F.interpolate(
                    curr_tensor, scale_factor=scale_factor,
                    mode="bicubic", align_corners=False
                )

        with torch.no_grad():
            if phase == 1:
                # No optical flow — concatenate prev + curr
                sr = model(prev_tensor, curr_tensor)

            elif phase == 2:
                # Dense Farneback flow at LR space
                flow = farneback_backward_flow(prev_gray, curr_gray)
                flow = flow.unsqueeze(0).to(device)          # (1, 2, H, W)
                sr = model(prev_tensor, curr_tensor, flow)

            else:  # Phase 3
                # Dense flow at LR; RecurrentTSRNet scales it to HR internally
                flow = farneback_backward_flow(prev_gray, curr_gray)
                flow = flow.unsqueeze(0).to(device)          # (1, 2, H, W)
                sr   = model(curr_tensor, hr_prev, flow)
                hr_prev = sr.clamp(0, 1)                     # feed back own output

        sr_clamp = sr.squeeze(0).cpu().clamp(0, 1)
        sr_cv    = cv2.cvtColor(np.array(to_pil(sr_clamp)), cv2.COLOR_RGB2BGR)

        # Bicubic reference at same output size
        bic = F.interpolate(curr_tensor, scale_factor=scale_factor,
                            mode="bicubic", align_corners=False)
        bic_cv = cv2.cvtColor(
            np.array(to_pil(bic.squeeze(0).cpu().clamp(0, 1))), cv2.COLOR_RGB2BGR
        )

        # Side-by-side: Bicubic (left) | SR (right)
        combined = np.hstack([bic_cv, sr_cv])
        cv2.putText(combined, "Bicubic",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, f"Phase {phase} SR",
                    (out_W + 10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(combined)

        prev_tensor = curr_tensor
        prev_gray   = curr_gray
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    print(f"Saved: {output_video}")


# ---------------------------------------------------------------------------
# Synthetic test video generator (no external video needed)
# ---------------------------------------------------------------------------

def generate_synthetic_lr_video(image_path: str,
                                 output_path: str,
                                 lr_size: tuple = (320, 240),
                                 frames: int = 120,
                                 fps: float = 30.0):
    """
    Simulate a panning LR video from a single HR image.
    Useful when no real video is available.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    window = min(w, h) - 10
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, lr_size)
    for i in range(frames):
        t  = i / max(frames - 1, 1)
        px = int(t * (w - window))
        py = int(t * (h - window))
        crop  = img.crop((px, py, px + window, py + window))
        frame = crop.resize(lr_size, Image.Resampling.BICUBIC)
        writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Synthetic LR video saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video SR — Phase 1 / 2 / 3")
    parser.add_argument("--phase",       type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/p2_best.pth")
    parser.add_argument("--input-video", type=str, default="",
                        help="Path to input LR video (.mp4). "
                             "Leave empty to auto-generate from --image-source.")
    parser.add_argument("--output-video",type=str, default="assets/video_sr_output.mp4")
    parser.add_argument("--image-source",type=str, default="data/samples/sample_00.jpg",
                        help="HR image used to generate synthetic LR video (fallback)")
    parser.add_argument("--lr-width",   type=int, default=320)
    parser.add_argument("--lr-height",  type=int, default=240)
    parser.add_argument("--frames",     type=int, default=120)
    parser.add_argument("--hidden-channels", type=int, default=64,
                        help="Hidden channel width (must match training checkpoint)")
    parser.add_argument("--pad-mode", type=str, default="reflection",
                        choices=["zeros", "border", "reflection"],
                        help="grid_sample padding mode (must match training)")
    args = parser.parse_args()

    input_video = args.input_video
    if not input_video:
        input_video = "data/synthetic_lr.mp4"
        if not os.path.exists(input_video):
            generate_synthetic_lr_video(
                args.image_source, input_video,
                lr_size=(args.lr_width, args.lr_height),
                frames=args.frames
            )

    run_video_sr(
        phase       = args.phase,
        model_path  = args.checkpoint,
        input_video = input_video,
        output_video= args.output_video,
        hidden_channels=args.hidden_channels,
        pad_mode    = args.pad_mode,
    )
