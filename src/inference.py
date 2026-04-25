import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import TemporalSRResNet
import matplotlib.pyplot as plt

def run_inference(model_path, image_path, output_path, scale_factor=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TemporalSRResNet(scale_factor=scale_factor).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    print(f"Original Image Size: {img.size}")

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # To test the temporal model, we need TWO frames.
    # We will simulate motion just like in the dataset by taking the image and shifting it.
    # Let's crop a center region for the "current" frame
    crop_size = min(img.size) - 20 # Leave 20px padding for the "pan"
    center_x = img.size[0] // 2
    center_y = img.size[1] // 2

    curr_box = (
        center_x - crop_size // 2,
        center_y - crop_size // 2,
        center_x + crop_size // 2,
        center_y + crop_size // 2
    )
    hr_curr = img.crop(curr_box)

    # Let's offset by 5 pixels diagonally for the "previous" frame
    offset = 5
    prev_box = (
        curr_box[0] - offset,
        curr_box[1] - offset,
        curr_box[2] - offset,
        curr_box[3] - offset
    )
    hr_prev = img.crop(prev_box)

    # Shrink both images to create the low-res inputs
    lr_size = (crop_size // scale_factor, crop_size // scale_factor)
    lr_transform = transforms.Resize(
        lr_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    )

    lr_prev = lr_transform(hr_prev)
    lr_curr = lr_transform(hr_curr)

    # Create bicubic upscale of the CURRENT frame for comparison
    bicubic_upscale = transforms.Resize(
        (crop_size, crop_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    )
    bicubic_img = bicubic_upscale(lr_curr)

    # Run through TSR model
    tensor_prev = to_tensor(lr_prev).unsqueeze(0).to(device)
    tensor_curr = to_tensor(lr_curr).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(tensor_prev, tensor_curr)

    sr_tensor = sr_tensor.squeeze(0).cpu()
    sr_tensor = torch.clamp(sr_tensor, 0, 1)
    sr_img = to_pil(sr_tensor)

    # Plot results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # We will show the previous frame as well to demonstrate the motion
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(lr_prev)
    axes[0].set_title(f"Frame t-1 (Offset)\n{lr_prev.size}")
    axes[0].axis('off')

    axes[1].imshow(lr_curr)
    axes[1].set_title(f"Frame t (Target)\n{lr_curr.size}")
    axes[1].axis('off')

    axes[2].imshow(bicubic_img)
    axes[2].set_title(f"Bicubic Upscale (Frame t)\n{bicubic_img.size}")
    axes[2].axis('off')

    axes[3].imshow(sr_img)
    axes[3].set_title(f"TSR Output\n{sr_img.size}")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved temporal comparison to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using trained Temporal Super-Resolution Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tsr_model_epoch_5.pth", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default="data/samples/sample_00.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="assets/temporal_example_comparison.png", help="Path to save output comparison")
    args = parser.parse_args()

    run_inference(args.checkpoint, args.image, args.output)
