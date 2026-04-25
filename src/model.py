import torch
import torch.nn as nn

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
    def __init__(self, in_channels=6, num_res_blocks=16, scale_factor=4):
        super(TemporalSRResNet, self).__init__()

        # Initial Feature Extraction (accepts 6 channels instead of 3)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()

        # Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        # Post-Residual Convolution
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Upsampling (PixelShuffle)
        # For scale factor 4, we need two 2x upsampling blocks
        upsample_blocks = []
        for _ in range(2):
            upsample_blocks.append(nn.Conv2d(64, 256, kernel_size=3, padding=1))
            upsample_blocks.append(nn.PixelShuffle(2))
            upsample_blocks.append(nn.PReLU())

        self.upsample = nn.Sequential(*upsample_blocks)

        # Final Output Layer (outputs 3 channels: RGB)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

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

if __name__ == "__main__":
    # Test model with dummy temporal input
    model = TemporalSRResNet(scale_factor=4)
    # Dummy tensors (Batch Size 1, Channels 3, Height 32, Width 32)
    dummy_prev = torch.randn(1, 3, 32, 32)
    dummy_curr = torch.randn(1, 3, 32, 32)

    output = model(dummy_prev, dummy_curr)
    print(f"Input Shape: Prev {dummy_prev.shape}, Curr {dummy_curr.shape}")
    print(f"Output Shape: {output.shape}")
