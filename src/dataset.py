import os
import requests
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SRTemporalDataset(Dataset):
    """
    Temporal Super-Resolution Dataset.
    Generates sequences of frames by simulating a panning camera over a high-res image.
    Outputs: LR(t-1), LR(t), HR(t)
    """
    def __init__(self, image_dir, scale_factor=4, crop_size=128, max_offset=4):
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.max_offset = max_offset # max pixels to "pan" between frames

        self.image_filenames = [
            os.path.join(image_dir, x) for x in os.listdir(image_dir)
            if x.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        full_image = Image.open(img_path).convert('RGB')

        w, h = full_image.size

        # Ensure image is large enough for crop + offset
        padded_size = self.crop_size + self.max_offset
        if w < padded_size or h < padded_size:
            # Resize image if it's too small
            full_image = full_image.resize((max(w, padded_size), max(h, padded_size)))
            w, h = full_image.size

        # Random base coordinates for Frame t
        x_t = random.randint(0, w - self.crop_size)
        y_t = random.randint(0, h - self.crop_size)

        # Simulate motion by picking an offset for Frame t-1
        offset_x = random.randint(-self.max_offset, self.max_offset)
        offset_y = random.randint(-self.max_offset, self.max_offset)

        # Prevent going out of bounds for Frame t-1
        x_t_minus_1 = max(0, min(w - self.crop_size, x_t + offset_x))
        y_t_minus_1 = max(0, min(h - self.crop_size, y_t + offset_y))

        # Crop the High-Res frames
        hr_img_t = full_image.crop((x_t, y_t, x_t + self.crop_size, y_t + self.crop_size))
        hr_img_t_minus_1 = full_image.crop((x_t_minus_1, y_t_minus_1, x_t_minus_1 + self.crop_size, y_t_minus_1 + self.crop_size))

        hr_tensor_t = self.to_tensor(hr_img_t)
        hr_tensor_t_minus_1 = self.to_tensor(hr_img_t_minus_1)

        # Generate Low-Res frames
        lr_size = self.crop_size // self.scale_factor
        lr_transform = transforms.Resize(
            (lr_size, lr_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

        lr_tensor_t = lr_transform(hr_tensor_t)
        lr_tensor_t_minus_1 = lr_transform(hr_tensor_t_minus_1)

        # We return:
        # 1. Previous LR frame
        # 2. Current LR frame
        # 3. Current HR frame (the ground truth target)
        return lr_tensor_t_minus_1, lr_tensor_t, hr_tensor_t


def download_sample_images(output_dir="data/samples", num_images=5):
    """
    Downloads a few sample high-resolution images to act as a mini-dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    urls = [
        "https://images.unsplash.com/photo-1494548162494-384bba4ab999?w=800", # sunrise
        "https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=800", # bridge
        "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800", # nature
        "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=800", # mountains
        "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=800"  # forest
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
    dataset = SRTemporalDataset("data/samples")
    lr_prev, lr_curr, hr_curr = dataset[0]
    print(f"LR Prev Shape: {lr_prev.shape}, LR Curr Shape: {lr_curr.shape}, HR Curr Shape: {hr_curr.shape}")
