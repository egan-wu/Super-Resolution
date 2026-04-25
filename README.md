# Super Resolution PyTorch Model

This project is a simple Super-Resolution Convolutional Neural Network (SRResNet) to explore the spatial upscaling concepts behind technologies like DLSS and PSSR. It demonstrates how to train a machine learning model to upscale low-resolution images into high-resolution images.

## Setup

First, ensure you have the required dependencies installed:

```bash
pip install torch torchvision pillow requests tqdm matplotlib
```

## How to Train

You can train the model using the provided training script. The script downloads a small sample dataset automatically.

For quick testing or CPU execution:
```bash
python3 src/train.py --epochs 10
```

**To achieve high-quality results (like those seen in DLSS/PSSR), you need to train the model for thousands of epochs on a GPU.**

To train for 1000 or 5000 epochs (A CUDA-capable GPU is highly recommended):
```bash
# This will take a long time and save checkpoints every 50 epochs
python3 src/train.py --epochs 1000
# OR
python3 src/train.py --epochs 5000
```

### What to expect at higher epochs:
* **10-100 Epochs:** The model learns basic colors but output remains blurry and washed out.
* **1000 Epochs:** The model will begin to reconstruct hard edges and the color accuracy will match the original high-resolution image much closer than basic bicubic upscaling.
* **5000+ Epochs:** High-frequency details (like textures on leaves or brick lines) will start to synthesize, and the result will visibly outperform standard interpolation.

## How to Run Inference

After training, you can run the inference script to generate a side-by-side visual comparison (Low Res, Bicubic, AI Super Resolution).

```bash
# Example for running inference using the 1000 epoch checkpoint
python3 src/inference.py --checkpoint checkpoints/sr_model_epoch_1000.pth --output assets/epoch_1000_comparison.png
```

The resulting comparison image will be saved in the `assets/` folder.
