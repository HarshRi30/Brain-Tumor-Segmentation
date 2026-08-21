"""
unet3d.py — Baseline 3D U-Net Architecture

Standard 3D U-Net encoder-decoder with skip connections.
Used as the baseline model for comparison against the Attention U-Net.

Reference: Çiçek et al. "3D U-Net: Learning Dense Volumetric Segmentation
           from Sparse Annotation" (MICCAI 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv3D(nn.Module):
    """
    Two consecutive (Conv3D → BatchNorm → ReLU) blocks.
    The fundamental unit of both the encoder and decoder.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """MaxPool3D downsampling → DoubleConv3D."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool  = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv  = DoubleConv3D(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    """
    TransposedConv3D upsampling → concatenate skip → DoubleConv3D.
    If sizes don't match exactly (odd dims), we pad the skip connection.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                        kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims differ (handles odd-sized inputs)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear",
                              align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# 3D U-Net
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    Baseline 3D U-Net for brain tumor segmentation.

    Architecture:
        Encoder: 4 DoubleConv blocks with MaxPool downsampling
        Bottleneck: DoubleConv
        Decoder: 4 DoubleConv blocks with TransposedConv upsampling + skip connections
        Head: 1×1×1 Conv → num_classes

    Args:
        in_channels  : number of input modalities (4 for BraTS)
        out_channels : number of segmentation classes (4 for BraTS 2023)
        init_features: number of feature maps in first encoder layer (doubled each level)
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 init_features: int = 32):
        super().__init__()
        f = init_features

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, f)        # f  → skip1
        self.enc2 = EncoderBlock(f,     f * 2)           # f*2  → skip2
        self.enc3 = EncoderBlock(f * 2, f * 4)           # f*4  → skip3
        self.enc4 = EncoderBlock(f * 4, f * 8)           # f*8  → skip4

        # Bottleneck
        self.bottleneck = EncoderBlock(f * 8, f * 16)   # f*16

        # Decoder
        self.dec4 = DecoderBlock(f * 16, f * 8,  f * 8)
        self.dec3 = DecoderBlock(f * 8,  f * 4,  f * 4)
        self.dec2 = DecoderBlock(f * 4,  f * 2,  f * 2)
        self.dec1 = DecoderBlock(f * 2,  f,      f)

        # Segmentation head
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 4, H, W, D)
        Returns:
            logits : (B, num_classes, H, W, D)
        """
        # Encoder
        s1 = self.enc1(x)         # (B, f,    H,   W,   D)
        s2 = self.enc2(s1)        # (B, f*2,  H/2, W/2, D/2)
        s3 = self.enc3(s2)        # (B, f*4,  H/4, W/4, D/4)
        s4 = self.enc4(s3)        # (B, f*8,  H/8, W/8, D/8)

        # Bottleneck
        b  = self.bottleneck(s4)  # (B, f*16, H/16,W/16,D/16)

        # Decoder
        d4 = self.dec4(b,  s4)   # (B, f*8,  H/8, W/8, D/8)
        d3 = self.dec3(d4, s3)   # (B, f*4,  H/4, W/4, D/4)
        d2 = self.dec2(d3, s2)   # (B, f*2,  H/2, W/2, D/2)
        d1 = self.dec1(d2, s1)   # (B, f,    H,   W,   D)

        return self.head(d1)      # (B, num_classes, H, W, D)


def count_parameters(model: nn.Module) -> str:
    """Return a human-readable parameter count string."""
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {train:,}"


if __name__ == "__main__":
    # Quick sanity check
    model = UNet3D(in_channels=4, out_channels=4, init_features=32)
    print("3D U-Net (Baseline)")
    print(count_parameters(model))
    x = torch.randn(1, 4, 96, 96, 96)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")   # Should be (1, 4, 96, 96, 96)
