import os
import argparse
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from model import TemporalSRResNet

def generate_test_video(image_path, output_video, lr_size=(100, 100), frames=60):
    """
    Simulates a panning camera to create a Low-Resolution video from a High-Res image.
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    # We will pan a window diagonally across the image
    window_size = 400
    start_x, start_y = 0, 0
    end_x, end_y = w - window_size, h - window_size

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, lr_size)

    print(f"Generating test video: {output_video}")
    for i in range(frames):
        # Calculate current position (linear interpolation)
        progress = i / (frames - 1)
        curr_x = int(start_x + (end_x - start_x) * progress)
        curr_y = int(start_y + (end_y - start_y) * progress)

        # Crop and resize to Low-Resolution
        crop = img.crop((curr_x, curr_y, curr_x + window_size, curr_y + window_size))
        lr_frame = crop.resize(lr_size, Image.Resampling.BICUBIC)

        # Convert PIL (RGB) to OpenCV (BGR)
        cv_img = cv2.cvtColor(np.array(lr_frame), cv2.COLOR_RGB2BGR)
        out.write(cv_img)

    out.release()
    print("Test video generated successfully.")

def run_video_inference(model_path, input_video, output_video, scale_factor=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = TemporalSRResNet(scale_factor=scale_factor).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Open Video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output will be scaled up
    out_width = width * scale_factor
    out_height = height * scale_factor

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (out_width * 2, out_height)) # Width * 2 for Side-by-Side

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    print(f"Processing video {input_video} ({width}x{height}) -> ({out_width}x{out_height}) side-by-side")

    prev_tensor = None

    pbar = tqdm(total=frame_count, desc="Upscaling Frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV BGR to PIL RGB
        curr_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_tensor = to_tensor(curr_pil).unsqueeze(0).to(device)

        # If it's the very first frame, we don't have a previous frame, so we duplicate it.
        if prev_tensor is None:
            prev_tensor = curr_tensor

        # Run Inference
        with torch.no_grad():
            sr_tensor = model(prev_tensor, curr_tensor)

        sr_tensor = sr_tensor.squeeze(0).cpu()
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
        sr_pil = to_pil(sr_tensor)

        # Create Bicubic for side-by-side comparison
        bicubic_pil = curr_pil.resize((out_width, out_height), Image.Resampling.BICUBIC)

        # Convert back to OpenCV BGR for saving
        sr_cv = cv2.cvtColor(np.array(sr_pil), cv2.COLOR_RGB2BGR)
        bicubic_cv = cv2.cvtColor(np.array(bicubic_pil), cv2.COLOR_RGB2BGR)

        # Concatenate side-by-side (Bicubic on Left, TSR on Right)
        combined = np.hstack((bicubic_cv, sr_cv))

        # Add text labels
        cv2.putText(combined, 'Bicubic', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, 'TSR Output', (out_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(combined)

        # Update previous frame for the next loop
        prev_tensor = curr_tensor
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"Saved upscaled video to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video inference using Temporal Super-Resolution Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tsr_model_epoch_5.pth", help="Path to model checkpoint")
    parser.add_argument("--image_source", type=str, default="data/samples/sample_00.jpg", help="High res image to generate test video from")
    parser.add_argument("--input_video", type=str, default="data/test_lr_video.mp4", help="Path to low-res input video")
    parser.add_argument("--output_video", type=str, default="assets/tsr_upscaled_video.mp4", help="Path to save side-by-side comparison video")

    args = parser.parse_args()

    # 1. Generate the dummy video if it doesn't exist
    if not os.path.exists(args.input_video):
        generate_test_video(args.image_source, args.input_video)

    # 2. Run inference on the video
    run_video_inference(args.checkpoint, args.input_video, args.output_video)
