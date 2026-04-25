import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import SRResNet
import matplotlib.pyplot as plt

def run_inference(model_path, image_path, output_path, scale_factor=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = SRResNet(scale_factor=scale_factor).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    # Strict=False in case of minor architecture differences, but should match
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Image
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    print(f"Original Image Size: {img.size}")

    # Prepare transforms
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # We will simulate a low-resolution input by taking the original image,
    # shrinking it, and then running our model on the shrunk version.
    lr_size = (img.size[1] // scale_factor, img.size[0] // scale_factor)

    # Shrink the image using bicubic interpolation
    lr_transform = transforms.Resize(
        lr_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    )

    lr_img = lr_transform(img)
    print(f"Low Resolution Input Size: {lr_img.size}")

    # Also create a simple bicubic upscaled version for comparison
    bicubic_upscale = transforms.Resize(
        (img.size[1], img.size[0]),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    )
    bicubic_img = bicubic_upscale(lr_img)

    # Run through SR model
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_tensor = sr_tensor.squeeze(0).cpu()
    # Clamp values just in case
    sr_tensor = torch.clamp(sr_tensor, 0, 1)

    sr_img = to_pil(sr_tensor)
    print(f"Super Resolved Output Size: {sr_img.size}")

    # Save the side-by-side comparison
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr_img)
    axes[0].set_title(f"Low Res Input\n{lr_img.size}")
    axes[0].axis('off')

    axes[1].imshow(bicubic_img)
    axes[1].set_title(f"Bicubic Upscale\n{bicubic_img.size}")
    axes[1].axis('off')

    axes[2].imshow(sr_img)
    axes[2].set_title(f"AI Super Resolution\n{sr_img.size}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved comparison to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using trained Super-Resolution Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sr_model_epoch_5.pth", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default="data/samples/sample_00.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="assets/example_comparison.png", help="Path to save output comparison")
    args = parser.parse_args()

    run_inference(args.checkpoint, args.image, args.output)
