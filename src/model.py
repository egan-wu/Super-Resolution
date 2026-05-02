import torch
import torch.nn as nn
import torch.nn.functional as F
from warp import backward_warp


# ---------------------------------------------------------------------------
# ICNR Initialization (prevents PixelShuffle checkerboard artifacts)
# ---------------------------------------------------------------------------

def icnr_init(conv: nn.Conv2d, scale_factor: int = 2):
    """
    ICNR (Initialized to Convolution NearestResampled) initialization.
    For the conv layer immediately before PixelShuffle(r):
      - Creates a sub-kernel of shape (out_ch / r^2, in_ch, kH, kW)
      - Initializes it with Kaiming normal
      - Tiles it r^2 times along dim=0

    Result: after PixelShuffle all r^2 sub-pixels share the same initial
    weights → no systematic bias → no checkerboard pattern at epoch 0.
    """
    out_ch, in_ch, kH, kW = conv.weight.shape
    sub_out = out_ch // (scale_factor ** 2)
    tmp = torch.empty(sub_out, in_ch, kH, kW)
    nn.init.kaiming_normal_(tmp, nonlinearity="relu")
    kernel = tmp.repeat_interleave(scale_factor ** 2, dim=0)
    conv.weight.data.copy_(kernel)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class TemporalSRResNet(nn.Module):
    """
    Temporal Super-Resolution ResNet.
    Accepts concatenated (Frame t-1, Frame t) as input (6 channels).
    """
    def __init__(self, in_channels=6, num_res_blocks=16, scale_factor=4,
                 hidden_channels=64):
        super(TemporalSRResNet, self).__init__()
        ch = hidden_channels

        # Initial Feature Extraction (accepts 6 channels instead of 3)
        self.conv1 = nn.Conv2d(in_channels, ch, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()

        # Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(ch) for _ in range(num_res_blocks)]
        )

        # Post-Residual Convolution
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

        # Upsampling (PixelShuffle + post-smooth)
        # For scale factor 4, two 2× stages.
        # Each stage: pre-shuffle conv (ICNR init) → PixelShuffle → smooth conv → PReLU
        # The smooth conv "tidies up" the rearranged sub-pixels, suppressing
        # checkerboard / striping artefacts (expert tip — anti-aliasing post-shuffle).
        upsample_blocks = []
        for _ in range(2):
            upsample_blocks.append(nn.Conv2d(ch, ch * 4, kernel_size=3, padding=1))  # ICNR
            upsample_blocks.append(nn.PixelShuffle(2))
            upsample_blocks.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))      # smooth
            upsample_blocks.append(nn.PReLU())

        self.upsample = nn.Sequential(*upsample_blocks)

        # Final Output Layer (outputs 3 channels: RGB)
        self.conv3 = nn.Conv2d(ch, 3, kernel_size=9, padding=4)
        self._init_icnr()

    def _init_icnr(self):
        # Only ICNR-init the Conv2d immediately *before* a PixelShuffle.
        # The post-shuffle smoothing convs use default Kaiming init.
        modules = list(self.upsample)
        for i, m in enumerate(modules):
            if isinstance(m, nn.Conv2d) and i + 1 < len(modules) \
                    and isinstance(modules[i + 1], nn.PixelShuffle):
                icnr_init(m, scale_factor=2)

    def forward(self, x_prev, x_curr):
        # Concatenate temporal frames along the channel dimension
        # Shape: (Batch, 3, H, W) + (Batch, 3, H, W) -> (Batch, 6, H, W)
        x = torch.cat((x_prev, x_curr), dim=1)

        # Extract features
        out1 = self.prelu1(self.conv1(x))

        # Deep residual processing
        res = self.res_blocks(out1)

        # Skip connection
        out2 = self.bn2(self.conv2(res))
        out = out1 + out2

        # Upscale
        out = self.upsample(out)

        # Final image reconstruction
        out = self.conv3(out)

        # Ensure values stay roughly in valid range [0, 1] during initial training
        # We can use sigmoid or just clamp later in inference. No strict activation here is typical.
        return out

class WarpTSRNet(nn.Module):
    """
    Phase 2: Warp-then-Fuse Temporal Super-Resolution.

    Instead of naively stacking t-1 and t (early fusion), we:
      1. Backward-warp LR(t-1) onto LR(t) using the known motion flow.
      2. Concatenate [warped_prev (3ch), lr_curr (3ch), occlusion_mask (1ch)] = 7ch.
      3. Run the same ResNet + PixelShuffle upsampler.

    This gives the network a geometrically-aligned history signal rather than
    a misaligned one, which is the core DLSS trick.
    """

    def __init__(self, in_channels=7, num_res_blocks=16, scale_factor=4,
                 hidden_channels=64, pad_mode="border"):
        super().__init__()
        self.pad_mode = pad_mode
        ch = hidden_channels

        self.conv1 = nn.Conv2d(in_channels, ch, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(ch) for _ in range(num_res_blocks)]
        )

        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

        # PixelShuffle ×4 with post-shuffle smoothing convs (anti-checkerboard)
        upsample_blocks = []
        for _ in range(2):  # 2x * 2x = 4x
            upsample_blocks.append(nn.Conv2d(ch, ch * 4, kernel_size=3, padding=1))  # ICNR
            upsample_blocks.append(nn.PixelShuffle(2))
            upsample_blocks.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))      # smooth
            upsample_blocks.append(nn.PReLU())
        self.upsample = nn.Sequential(*upsample_blocks)

        self.conv3 = nn.Conv2d(ch, 3, kernel_size=9, padding=4)
        self._init_icnr()

    def _init_icnr(self):
        modules = list(self.upsample)
        for i, m in enumerate(modules):
            if isinstance(m, nn.Conv2d) and i + 1 < len(modules) \
                    and isinstance(modules[i + 1], nn.PixelShuffle):
                icnr_init(m, scale_factor=2)

    def forward(self, x_prev, x_curr, flow):
        """
        x_prev : (B, 3, H, W) — LR frame t-1
        x_curr : (B, 3, H, W) — LR frame t
        flow   : (B, 2)        — backward warp flow in LR pixels
        """
        warped_prev, mask = backward_warp(x_prev, flow, padding_mode=self.pad_mode)

        x = torch.cat([warped_prev, x_curr, mask], dim=1)  # (B, 7, H, W)

        out1 = self.prelu1(self.conv1(x))
        res = self.res_blocks(out1)
        out2 = self.bn2(self.conv2(res))
        out = out1 + out2
        out = self.upsample(out)
        return self.conv3(out)


class RecurrentTSRNet(nn.Module):
    """
    Phase 3: Recurrent Temporal Super-Resolution (DLSS-style feedback loop).

    Architecture:
      LR branch  — feature extraction at LR resolution + PixelShuffle ×4 → 64ch @ HR
      Hist branch — warp HR(t-1) at HR space, extract features               → 64ch @ HR
      Fusion      — concat (128ch) → refine → HR output

    At inference, HR(t-1) is the model's own previous output (true recurrence).
    During training, HR(t-1) is the GT frame (teacher forcing).

    The Halton jitter in SRSequenceDataset gives each frame a different sub-pixel
    view of the scene. The recurrent feedback accumulates these shifted samples into
    increasingly sharp HR outputs — the same principle as DLSS.
    """

    def __init__(self, scale_factor=4, lr_res_blocks=8, hist_res_blocks=4,
                 fuse_res_blocks=4, hidden_channels=64, pad_mode="border"):
        super().__init__()
        self.scale_factor = scale_factor
        self.pad_mode = pad_mode
        ch = hidden_channels

        # --- LR branch (operates at LR resolution) ---
        self.lr_entry = nn.Sequential(nn.Conv2d(3, ch, 9, padding=4), nn.PReLU())
        self.lr_res   = nn.Sequential(*[ResidualBlock(ch) for _ in range(lr_res_blocks)])
        self.lr_post  = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch))
        # PixelShuffle ×4: two ×2 stages, each followed by a smoothing conv
        # to suppress checkerboard / striping artefacts on the sub-pixel grid.
        self.lr_up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2),  # pre-shuffle (ICNR)
            nn.Conv2d(ch, ch,     3, padding=1), nn.PReLU(),           # post-shuffle smooth
            nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2),  # pre-shuffle (ICNR)
            nn.Conv2d(ch, ch,     3, padding=1), nn.PReLU(),           # post-shuffle smooth
        )  # output: (B, ch, H_hr, W_hr)

        # --- History branch (operates at HR resolution) ---
        # Input: warped HR(t-1) [3ch] + occlusion mask [1ch] = 4ch
        self.hist_entry = nn.Sequential(nn.Conv2d(4, ch, 3, padding=1), nn.PReLU())
        self.hist_res   = nn.Sequential(*[ResidualBlock(ch) for _ in range(hist_res_blocks)])

        # --- Fusion (at HR resolution) ---
        self.fuse_entry = nn.Sequential(nn.Conv2d(ch * 2, ch, 3, padding=1), nn.PReLU())
        self.fuse_res   = nn.Sequential(*[ResidualBlock(ch) for _ in range(fuse_res_blocks)])
        self.output     = nn.Conv2d(ch, 3, 9, padding=4)
        self._init_icnr()

    def _init_icnr(self):
        # Only ICNR-init the Conv2d immediately *before* a PixelShuffle.
        # Post-shuffle smoothing convs use default Kaiming init.
        modules = list(self.lr_up)
        for i, m in enumerate(modules):
            if isinstance(m, nn.Conv2d) and i + 1 < len(modules) \
                    and isinstance(modules[i + 1], nn.PixelShuffle):
                icnr_init(m, scale_factor=2)

    def forward(self, lr_curr, hr_prev, flow_lr):
        """
        lr_curr  : (B, 3, H_lr, W_lr) — current low-res frame
        hr_prev  : (B, 3, H_hr, W_hr) — previous HR frame (GT during training, model output at inference)
        flow_lr  : (B, 2)              — rigid backward flow in LR pixel space  [synthetic training]
               OR (B, 2, H_lr, W_lr)  — dense backward flow from Farneback     [real video]
        """
        s = self.scale_factor

        if flow_lr.dim() == 2:
            flow_hr = flow_lr * s
        else:
            flow_hr = F.interpolate(
                flow_lr, scale_factor=s, mode="bilinear", align_corners=False
            ) * s

        warped_hr, mask = backward_warp(hr_prev, flow_hr, padding_mode=self.pad_mode)

        # LR branch → upsample to HR feature space
        x = self.lr_entry(lr_curr)
        x = x + self.lr_post(self.lr_res(x))
        x = self.lr_up(x)                          # (B, ch, H_hr, W_hr)

        # History branch
        hist_in = torch.cat([warped_hr, mask], dim=1)  # (B, 4, H_hr, W_hr)
        h = self.hist_entry(hist_in)
        h = self.hist_res(h)                            # (B, ch, H_hr, W_hr)

        # Fusion
        f = self.fuse_entry(torch.cat([x, h], dim=1))  # (B, ch, H_hr, W_hr)
        f = self.fuse_res(f)
        return self.output(f)                           # (B, 3, H_hr, W_hr)


if __name__ == "__main__":
    model = RecurrentTSRNet(scale_factor=4)
    lr = torch.randn(1, 3, 32, 32)
    hr_prev = torch.randn(1, 3, 128, 128)
    flow = torch.zeros(1, 2)
    out = model(lr, hr_prev, flow)
    print(f"LR: {lr.shape}, HR_prev: {hr_prev.shape} → HR_out: {out.shape}")
