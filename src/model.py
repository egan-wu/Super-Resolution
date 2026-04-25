import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard Residual Block as used in SRResNet.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

class UpsampleBlock(nn.Module):
    """
    Upsampling Block using PixelShuffle.
    Increases the spatial resolution by 2x.
    """
    def __init__(self, in_channels, up_scale=2):
        super(UpsampleBlock, self).__init__()
        # To upscale by 2x, we need 4x the channels (2^2) for PixelShuffle
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class SRResNet(nn.Module):
    """
    A lightweight Super-Resolution ResNet model.
    It takes a low-resolution image, extracts features, applies residual blocks,
    and then upsamples the image to the target high resolution.
    """
    def __init__(self, scale_factor=4, num_channels=3, num_res_blocks=4, num_features=64):
        super(SRResNet, self).__init__()

        # Initial feature extraction
        self.conv_in = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(num_features))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Post-residual convolution
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features)
        )

        # Upsampling (Assuming scale_factor is 2, 4, or 8)
        upsample_blocks = []
        import math
        if scale_factor in [2, 4, 8]:
            num_upsample_blocks = int(math.log2(scale_factor))
            for _ in range(num_upsample_blocks):
                upsample_blocks.append(UpsampleBlock(num_features, up_scale=2))
        else:
            raise ValueError("Scale factor must be 2, 4, or 8")

        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        # Final output layer to map back to 3 channels (RGB)
        self.conv_out = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)

    def forward(self, x):
        # Initial feature extraction
        out1 = self.conv_in(x)

        # Residual blocks
        res = self.res_blocks(out1)
        res = self.conv_mid(res)

        # Skip connection over the residual blocks
        out2 = out1 + res

        # Upsampling
        out = self.upsample_blocks(out2)

        # Final output
        out = self.conv_out(out)

        # We can use torch.sigmoid to ensure outputs are between 0 and 1,
        # or just let the loss function handle it. Since we are using standard tensors,
        # we clamp/sigmoid it. Sigmoid helps keep it strictly in [0, 1] range of image tensors.
        return torch.sigmoid(out)

if __name__ == "__main__":
    # Test the model with dummy data
    model = SRResNet(scale_factor=4)
    # Batch Size 1, 3 Channels, 32x32 LR image
    dummy_lr = torch.randn(1, 3, 32, 32)
    output_hr = model(dummy_lr)

    print(f"Input LR shape: {dummy_lr.shape}")
    print(f"Output HR shape: {output_hr.shape}")
