import os
import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SRDataset(Dataset):
    """
    Super-Resolution Dataset that generates Low-Resolution (LR) and
    High-Resolution (HR) image pairs on the fly.
    """
    def __init__(self, image_dir, scale_factor=4, crop_size=128):
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.crop_size = crop_size

        # We only want actual images
        self.image_filenames = [
            os.path.join(image_dir, x) for x in os.listdir(image_dir)
            if x.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # HR Transform: random crop and to tensor
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load HR image
        img_path = self.image_filenames[idx]
        hr_image = Image.open(img_path).convert('RGB')

        # Apply HR transform (Crop & Tensor)
        hr_tensor = self.hr_transform(hr_image)

        # LR Transform: we take the HR tensor, downsample it to simulate low resolution
        # Then, importantly, we don't scale it back up with bicubic here.
        # The model will take the small (e.g. 32x32) tensor and output a 128x128 tensor.
        lr_size = self.crop_size // self.scale_factor

        # We use PIL for accurate bicubic downsampling, as it's standard in SR literature
        # To do this cleanly, let's revert the cropped HR tensor back to PIL briefly
        # or just do the downsampling via torchvision transforms on the tensor.

        # Let's use torchvision transforms on the tensor for simplicity and speed
        lr_transform = transforms.Resize(
            (lr_size, lr_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

        lr_tensor = lr_transform(hr_tensor)

        return lr_tensor, hr_tensor


def download_sample_images(output_dir="data/samples", num_images=5):
    """
    Downloads a few sample high-resolution images to act as a mini-dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Using some sample Unsplash images as placeholders for high-res images
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
    dataset = SRDataset("data/samples")
    lr, hr = dataset[0]
    print(f"LR Shape: {lr.shape}, HR Shape: {hr.shape}")
