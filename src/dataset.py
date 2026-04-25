import os
import zipfile
import requests
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Halton quasi-random sequence (for structured sub-pixel jitter)
# ---------------------------------------------------------------------------

def _halton(index: int, base: int) -> float:
    result, f, i = 0.0, 1.0 / base, index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def halton_2d(index: int):
    """Return (x, y) ∈ [0, 1) using Halton bases (2, 3)."""
    return _halton(index, 2), _halton(index, 3)


# ---------------------------------------------------------------------------
# Phase 1 / 2 dataset — single frame-pair
# ---------------------------------------------------------------------------

class SRTemporalDataset(Dataset):
    """
    Generates frame pairs by simulating a panning camera over a high-res image.
    Returns: LR(t-1), LR(t), HR(t), backward-warp flow (LR pixel space)
    """

    def __init__(self, image_dir, scale_factor=4, crop_size=128, max_offset=4):
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.max_offset = max_offset

        self.image_filenames = [
            os.path.join(image_dir, x) for x in os.listdir(image_dir)
            if x.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        full_image = Image.open(self.image_filenames[idx]).convert('RGB')
        w, h = full_image.size

        padded = self.crop_size + self.max_offset
        if w < padded or h < padded:
            full_image = full_image.resize((max(w, padded), max(h, padded)))
            w, h = full_image.size

        x_t = random.randint(0, w - self.crop_size)
        y_t = random.randint(0, h - self.crop_size)

        offset_x = random.randint(-self.max_offset, self.max_offset)
        offset_y = random.randint(-self.max_offset, self.max_offset)

        x_tm1 = max(0, min(w - self.crop_size, x_t + offset_x))
        y_tm1 = max(0, min(h - self.crop_size, y_t + offset_y))

        hr_t   = self.to_tensor(full_image.crop((x_t,   y_t,   x_t   + self.crop_size, y_t   + self.crop_size)))
        hr_tm1 = self.to_tensor(full_image.crop((x_tm1, y_tm1, x_tm1 + self.crop_size, y_tm1 + self.crop_size)))

        lr_size = self.crop_size // self.scale_factor
        lr_tf = transforms.Resize((lr_size, lr_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        lr_t   = lr_tf(hr_t)
        lr_tm1 = lr_tf(hr_tm1)

        # Backward warp flow in LR pixel space
        flow_x = -(x_tm1 - x_t) / self.scale_factor
        flow_y = -(y_tm1 - y_t) / self.scale_factor
        flow = torch.tensor([flow_x, flow_y], dtype=torch.float32)

        return lr_tm1, lr_t, hr_t, flow


# ---------------------------------------------------------------------------
# Phase 3 dataset — N-frame sequences with Halton jitter
# ---------------------------------------------------------------------------

class SRSequenceDataset(Dataset):
    """
    Generates N-frame panning sequences with structured sub-pixel Halton jitter.

    Jitter: each frame t gets a Halton-based offset in [0, scale_factor) HR pixels.
    This is sub-pixel at LR scale (scale_factor=4 → 1 HR px = 0.25 LR px),
    giving the recurrent network diverse sub-pixel views to accumulate detail.

    Returns (stacked tensors so DataLoader can batch directly):
        lr_seq   : (T, 3, H_lr, W_lr)
        hr_seq   : (T, 3, H_hr, W_hr)
        flow_seq : (T, 2)  — LR-pixel backward warp flow; flow[0] = (0,0) (no prev)
    """

    def __init__(self, image_dir, scale_factor=4, crop_size=128,
                 max_offset=8, seq_len=4, use_jitter=True):
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.max_offset = max_offset
        self.seq_len = seq_len
        self.use_jitter = use_jitter

        self.image_filenames = [
            os.path.join(image_dir, x) for x in os.listdir(image_dir)
            if x.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.to_tensor = transforms.ToTensor()

        lr_size = crop_size // scale_factor
        self.lr_tf = transforms.Resize(
            (lr_size, lr_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        full_image = Image.open(self.image_filenames[idx]).convert('RGB')
        w, h = full_image.size

        # Ensure image is large enough for sequence panning + jitter headroom
        headroom = self.max_offset * self.seq_len + self.scale_factor
        padded = self.crop_size + headroom
        if w < padded or h < padded:
            full_image = full_image.resize((max(w, padded), max(h, padded)))
            w, h = full_image.size

        # Generate random start position with enough margin for the full sequence
        margin = self.max_offset * self.seq_len + self.scale_factor
        x = random.randint(margin, w - self.crop_size - margin)
        y = random.randint(margin, h - self.crop_size - margin)

        lr_frames, hr_frames, flows = [], [], []
        x_prev, y_prev = x, y

        for t in range(self.seq_len):
            # Halton jitter in HR pixels (0 … scale_factor-1), giving sub-pixel diversity at LR
            if self.use_jitter and t > 0:
                jx = int(_halton(t, 2) * self.scale_factor)
                jy = int(_halton(t, 3) * self.scale_factor)
            else:
                jx, jy = 0, 0

            if t == 0:
                # First frame — no motion
                flow = torch.zeros(2, dtype=torch.float32)
            else:
                # Random pan from previous position
                dx = random.randint(-self.max_offset, self.max_offset)
                dy = random.randint(-self.max_offset, self.max_offset)
                x_new = max(margin, min(w - self.crop_size - margin, x + dx))
                y_new = max(margin, min(h - self.crop_size - margin, y + dy))

                # Total motion = pan + jitter difference; backward flow in LR pixels
                # prev crop anchor: (x_prev + jx_prev, y_prev + jy_prev)
                # curr crop anchor: (x_new  + jx,      y_new  + jy)
                # backward flow_x = (x_curr - x_prev) / scale (with jitter included)
                jx_prev = int(_halton(t - 1, 2) * self.scale_factor) if t > 1 else 0
                jy_prev = int(_halton(t - 1, 3) * self.scale_factor) if t > 1 else 0
                dx_total = (x_new + jx) - (x_prev + jx_prev)
                dy_total = (y_new + jy) - (y_prev + jy_prev)
                flow = torch.tensor(
                    [dx_total / self.scale_factor, dy_total / self.scale_factor],
                    dtype=torch.float32
                )
                x_prev, y_prev = x_new, y_new
                x = x_new

            # Crop HR frame (with jitter offset)
            cx, cy = x + jx, y + jy
            hr_pil = full_image.crop((cx, cy, cx + self.crop_size, cy + self.crop_size))
            hr_t = self.to_tensor(hr_pil)
            lr_t = self.lr_tf(hr_t)

            lr_frames.append(lr_t)
            hr_frames.append(hr_t)
            flows.append(flow)

        return (
            torch.stack(lr_frames),    # (T, 3, H_lr, W_lr)
            torch.stack(hr_frames),    # (T, 3, H_hr, W_hr)
            torch.stack(flows),        # (T, 2)
        )


# ---------------------------------------------------------------------------

def download_div2k(output_dir="data/div2k", split="valid", max_images=100):
    """
    Downloads DIV2K HR images from the official ETH source.

    split="valid"  → 100 images, ~770 MB  (recommended for quick setup)
    split="train"  → 800 images, ~7.1 GB

    Images are extracted flat into output_dir as *.png files.
    """
    os.makedirs(output_dir, exist_ok=True)

    existing = [f for f in os.listdir(output_dir) if f.lower().endswith(".png")]
    if len(existing) >= max_images:
        print(f"DIV2K {split} already ready ({len(existing)} images in {output_dir}).")
        return

    urls = {
        "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    }
    url = urls[split]
    zip_path = os.path.join("data", f"DIV2K_{split}_HR.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading DIV2K {split} HR from {url}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"DIV2K_{split}_HR.zip"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Extracting up to {max_images} images to {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        png_members = sorted(m for m in zf.namelist() if m.lower().endswith(".png"))
        for member in png_members[:max_images]:
            fname = os.path.basename(member)
            dest  = os.path.join(output_dir, fname)
            if not os.path.exists(dest):
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    extracted = len([f for f in os.listdir(output_dir) if f.lower().endswith(".png")])
    print(f"DIV2K {split}: {extracted} images ready in {output_dir}")


def download_sample_images(output_dir="data/samples", num_images=5):
    os.makedirs(output_dir, exist_ok=True)
    urls = [
        "https://images.unsplash.com/photo-1494548162494-384bba4ab999?w=800",
        "https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=800",
        "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800",
        "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=800",
        "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=800",
    ]
    for i, url in enumerate(urls[:num_images]):
        filepath = os.path.join(output_dir, f"sample_{i:02d}.jpg")
        if not os.path.exists(filepath):
            print(f"Downloading sample {i}...")
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
    print(f"Sample images ready in {output_dir}")


if __name__ == "__main__":
    download_sample_images()
    ds = SRSequenceDataset("data/samples", seq_len=4)
    lr_seq, hr_seq, flow_seq = ds[0]
    print(f"LR seq: {lr_seq.shape}, HR seq: {hr_seq.shape}, Flow seq: {flow_seq.shape}")
