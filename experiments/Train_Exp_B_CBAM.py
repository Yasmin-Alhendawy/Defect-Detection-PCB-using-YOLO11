"""
cbam_module.py — Convolutional Block Attention Module (Woo et al., ECCV 2018)

Designed for integration with Ultralytics YOLOv11 via setup_cbam.py.
CBAM is channel-preserving: output channels == input channels.

Usage in YAML:
    - [-1, 1, CBAM, [512]]   # 512 = unscaled channels (will be width-scaled by parse_model)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM.

    Uses both average-pooling and max-pooling paths through a shared MLP,
    then combines with sigmoid gating.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module of CBAM.

    Computes channel-wise avg- and max-pool, concatenates, then applies a
    single conv + sigmoid gate over the spatial map.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Applies sequential channel attention then spatial attention.
    Channel-preserving: input and output feature maps have identical shape.

    Args:
        c1 (int): Input channels. Passed automatically by parse_model.
        c2 (int | None): Output channels — must equal c1 (ignored, for API compat).
        reduction (int): Channel reduction ratio for the MLP. Default: 16.
        kernel_size (int): Spatial attention conv kernel (3 or 7). Default: 7.
    """

    def __init__(self, c1: int, c2: int = None, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        if c2 is not None and c2 != c1:
            raise ValueError(f"CBAM is channel-preserving: c1={c1} must equal c2={c2}")
        self.channel_attn = ChannelAttention(c1, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
