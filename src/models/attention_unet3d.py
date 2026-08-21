"""
attention_unet3d.py — Proposed 3D Attention-Augmented U-Net

Integrates two complementary attention mechanisms into the U-Net backbone:
  1. Squeeze-and-Excitation (SE) blocks — Channel Attention
     "Which feature maps are important?"
  2. Spatial Attention Gates — Spatial Attention
     "Which spatial locations are important?"

Reference:
  - Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    (MIDL 2018) — for spatial attention gates
  - Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018) — for SE blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Squeeze-and-Excitation Block (Channel Attention)
# ---------------------------------------------------------------------------

class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps.

    Mechanism:
      ① Squeeze  : Global Average Pooling → (B, C, 1, 1, 1)
      ② Excite   : FC → ReLU → FC → Sigmoid → (B, C, 1, 1, 1)
      ③ Scale    : Multiply input feature maps by channel weights

    Args:
        channels    : number of input/output channels
        reduction   : reduction ratio for the bottleneck FC layer (default 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 4)  # never below 4
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                            # Squeeze
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),        # Excite (down)
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),        # Excite (up)
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        weights = self.se(x).view(B, C, 1, 1, 1)    # (B, C, 1, 1, 1)
        return x * weights                            # broadcast multiply


# ---------------------------------------------------------------------------
# 2. Spatial Attention Gate (Spatial Attention)
# ---------------------------------------------------------------------------

class SpatialAttentionGate3D(nn.Module):
    """
    Attention gate that focuses the decoder on relevant spatial regions
    by using a gating signal from the deeper (coarser) decoder feature map.

    Mechanism:
      g  (gating signal) : coarser feature from decoder
      x  (skip signal)   : finer feature from encoder
      ① Both projected to intermediate space F_int
      ② Added → ReLU → 1×1×1 Conv → Sigmoid → attention coefficient α
      ③ x × α  (attended skip connection)

    Args:
        F_g     : channels in gating signal g
        F_l     : channels in skip connection x
        F_int   : intermediate channels (bottleneck)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g : gating signal  (B, F_g, H/2, W/2, D/2)  — from decoder
            x : skip connection (B, F_l, H, W, D)        — from encoder
        Returns:
            attended x : same shape as x
        """
        g1 = self.W_g(g)

        # Upsample g1 to match x spatial size
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:],
                               mode="trilinear", align_corners=True)

        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)                # attention map  (B, 1, H, W, D)
        return x * psi                     # reweight skip


# ---------------------------------------------------------------------------
# Building blocks (reuse + SE)
# ---------------------------------------------------------------------------

class DoubleConv3D(nn.Module):
    """Conv3D→BN→ReLU ×2, optionally with SE channel attention at the end."""

    def __init__(self, in_ch: int, out_ch: int,
                 use_se: bool = True, se_reduction: int = 16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock3D(out_ch, se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.block(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv = DoubleConv3D(in_ch, out_ch, use_se=use_se)

    def forward(self, x):
        return self.conv(self.pool(x))


class AttentionDecoderBlock(nn.Module):
    """
    Decoder block with:
      ① Spatial Attention Gate applied to the skip connection
      ② TransposedConv upsampling
      ③ Concatenate attended skip + upsampled feature
      ④ DoubleConv3D (with SE)
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 use_spatial_attn: bool = True, use_se: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.attn_gate = (
            SpatialAttentionGate3D(F_g=in_ch // 2, F_l=skip_ch, F_int=skip_ch // 2)
            if use_spatial_attn else None
        )

        self.conv = DoubleConv3D(in_ch // 2 + skip_ch, out_ch, use_se=use_se)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Resize if needed
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="trilinear", align_corners=True)

        # Apply spatial attention gate to skip connection
        if self.attn_gate is not None:
            skip = self.attn_gate(g=x, x=skip)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Attention U-Net 3D (proposed model)
# ---------------------------------------------------------------------------

class AttentionUNet3D(nn.Module):
    """
    3D Attention-Augmented U-Net — the proposed model.

    Adds two types of attention to the standard 3D U-Net:
      • SE blocks (channel attention) after every DoubleConv in encoder + decoder
      • Spatial Attention Gates in all 4 decoder stages

    This gives the model interpretable attention maps showing:
      - Which feature channels are most discriminative (SE)
      - Which spatial regions within the MRI the model focuses on (attention gates)

    Args:
        in_channels         : 4 (FLAIR, T1, T1ce, T2)
        out_channels        : 4 (BG, NCR, ED, ET)
        init_features       : 32 (feature maps at first encoder level)
        use_channel_attn    : enable SE blocks (channel attention)
        use_spatial_attn    : enable spatial attention gates
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 init_features: int = 32,
                 use_channel_attn: bool = True,
                 use_spatial_attn: bool = True):
        super().__init__()
        f = init_features
        self.use_channel_attn = use_channel_attn
        self.use_spatial_attn = use_spatial_attn

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc1 = DoubleConv3D(in_channels, f,      use_se=use_channel_attn)
        self.enc2 = EncoderBlock(f,     f * 2,         use_se=use_channel_attn)
        self.enc3 = EncoderBlock(f * 2, f * 4,         use_se=use_channel_attn)
        self.enc4 = EncoderBlock(f * 4, f * 8,         use_se=use_channel_attn)

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.bottleneck = EncoderBlock(f * 8, f * 16,  use_se=use_channel_attn)

        # ── Decoder (with spatial attention gates) ─────────────────────────
        self.dec4 = AttentionDecoderBlock(f * 16, f * 8,  f * 8,
                                          use_spatial_attn, use_channel_attn)
        self.dec3 = AttentionDecoderBlock(f * 8,  f * 4,  f * 4,
                                          use_spatial_attn, use_channel_attn)
        self.dec2 = AttentionDecoderBlock(f * 4,  f * 2,  f * 2,
                                          use_spatial_attn, use_channel_attn)
        self.dec1 = AttentionDecoderBlock(f * 2,  f,      f,
                                          use_spatial_attn, use_channel_attn)

        # ── Segmentation head ──────────────────────────────────────────────
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 4, H, W, D)
        Returns:
            logits : (B, 4, H, W, D)
        """
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b  = self.bottleneck(s4)

        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.head(d1)

    def get_attention_maps(self, x: torch.Tensor):
        """
        Forward pass that also returns spatial attention maps
        from all 4 decoder stages (for visualisation notebook).

        Returns:
            logits        : (B, 4, H, W, D)
            attention_maps: list of 4 tensors, each (B, 1, H', W', D')
        """
        if not self.use_spatial_attn:
            raise ValueError("Model was built without spatial attention gates.")

        attention_maps = []

        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)

        # Manually run decoder with attention map capture
        def decode_and_capture(decoder_block, x_in, skip):
            x_up = decoder_block.up(x_in)
            if x_up.shape[2:] != skip.shape[2:]:
                x_up = F.interpolate(x_up, size=skip.shape[2:],
                                     mode="trilinear", align_corners=True)
            if decoder_block.attn_gate is not None:
                g1  = decoder_block.attn_gate.W_g(x_up)
                if g1.shape[2:] != skip.shape[2:]:
                    g1 = F.interpolate(g1, size=skip.shape[2:],
                                       mode="trilinear", align_corners=True)
                x1  = decoder_block.attn_gate.W_x(skip)
                psi = decoder_block.attn_gate.relu(g1 + x1)
                attn = decoder_block.attn_gate.psi(psi)
                attended_skip = skip * attn
                attention_maps.append(attn.detach().cpu())
            else:
                attended_skip = skip
            out = decoder_block.conv(torch.cat([attended_skip, x_up], dim=1))
            return out

        d4 = decode_and_capture(self.dec4, b,  s4)
        d3 = decode_and_capture(self.dec3, d4, s3)
        d2 = decode_and_capture(self.dec2, d3, s2)
        d1 = decode_and_capture(self.dec1, d2, s1)

        logits = self.head(d1)
        return logits, attention_maps


def count_parameters(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {train:,}"


if __name__ == "__main__":
    # Sanity check all 3 ablation variants
    configs = [
        ("Channel Attn only",  True,  False),
        ("Spatial Attn only",  False, True),
        ("Full Attn (proposed)", True, True),
    ]
    x = torch.randn(1, 4, 96, 96, 96)
    for name, ch, sp in configs:
        model = AttentionUNet3D(use_channel_attn=ch, use_spatial_attn=sp)
        y = model(x)
        print(f"{name:30s} | {count_parameters(model)} | Output: {y.shape}")
