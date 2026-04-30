#!/bin/bash
# setup_runpod.sh — RunPod pod 環境安裝（PyTorch SR 專案）
# Usage: bash setup_runpod.sh
#
# 假設使用 RunPod 的 PyTorch image，例如:
#   runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# 這類 image 已經預裝 torch + CUDA，所以我們只裝缺的套件。

set -e

echo "================================================"
echo " RunPod 環境安裝（Super-Resolution）"
echo "================================================"

# ── 1. Pre-install GPU check ─────────────────────────
echo ""
echo "=== Pre-install GPU check ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'CUDA version: {torch.version.cuda}')
"

# ── 2. 確認在 volume (/workspace) 上 ──────────────────
echo ""
echo "=== Disk usage ==="
df -h /workspace 2>/dev/null || echo "WARNING: /workspace not mounted as volume!"
df -h /

# ── 3. 安裝相依套件 ──────────────────────────────────
echo ""
echo "=== Installing dependencies ==="
# 不動 torch / torchvision（image 已經裝好對應 CUDA 版本）
pip install --no-deps \
    pillow \
    requests \
    tqdm \
    matplotlib \
    opencv-python-headless \
    scikit-image \
    numpy \
    scipy \
    imageio \
    networkx \
    packaging \
    tifffile \
    lazy-loader \
    PyWavelets

# ── 4. Post-install GPU check ────────────────────────
echo ""
echo "=== Post-install GPU check ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

# ── 5. 驗證 imports ──────────────────────────────────
echo ""
echo "=== Verify imports ==="
python3 -c "
import torch
import torchvision
import cv2
import skimage
import PIL
import matplotlib
import requests
import tqdm
print(f'torch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'opencv: {cv2.__version__}')
print(f'skimage: {skimage.__version__}')
print('All imports OK')
"

echo ""
echo "================================================"
echo " Setup complete"
echo "================================================"
