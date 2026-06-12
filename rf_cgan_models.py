"""
Minimal cGAN modules for the RF_Image_GAN smoke test.

This file intentionally contains only the discriminator-side pieces needed for
the first loop-stability smoke test. The generator is reused from the frozen
RF_Image regression code during this stage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Envelope2DPatchDiscriminator(nn.Module):
    """
    2D PatchGAN discriminator on linear envelope.

    Input:
        [B, 2, Z, X]
        channel 0 = candidate envelope slice
        channel 1 = baseline envelope slice (condition)

    Output:
        [B, 1, H', W'] patch-level real/fake scores.
    """

    def __init__(self, in_channels: int = 2, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BMode3DPatchDiscriminator(nn.Module):
    """
    3D PatchGAN discriminator for B-mode adversarial smoke tests.

    Input:
        [B, 2, 64, 32, 32]
        channel 0 = candidate B-mode (pred or GT)
        channel 1 = baseline B-mode condition

    Output:
        [B, 1, Z', X', Y'] raw LSGAN patch scores. No sigmoid here.
    """

    def __init__(self, in_channels: int = 2, ndf: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError(f"Expected [B,2,Z,X,Y], got {tuple(x.shape)}")
        return self.net(x)


class DoubleConv3D(nn.Module):
    """Conv3d -> BN -> ReLU -> Conv3d -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch, momentum=0.9),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Light3DUNet(nn.Module):
    """
    Lightweight 3D U-Net for direct RF-to-B-mode prediction.

    Input:  [B, 1536, 64, 32, 32]
    Output: [B,    1, 64, 32, 32] in [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv3d(1536, 64, kernel_size=1, bias=False),
            nn.BatchNorm3d(64, momentum=0.9),
            nn.ReLU(inplace=True),
        )

        self.enc1_down = nn.Sequential(
            nn.Conv3d(64, 48, kernel_size=3, stride=(2, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(48, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.enc1_conv = DoubleConv3D(48, 48)

        self.enc2_down = nn.Sequential(
            nn.Conv3d(48, 56, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(56, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.enc2_conv = DoubleConv3D(56, 56)

        self.enc3_down = nn.Sequential(
            nn.Conv3d(56, 64, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(64, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.enc3_conv = DoubleConv3D(64, 64)

        self.bottleneck = DoubleConv3D(64, 64)

        self.dec3_up = nn.Sequential(
            nn.Conv3d(120, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(56, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.dec3_conv = DoubleConv3D(56, 56)

        self.dec2_up = nn.Sequential(
            nn.Conv3d(104, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(48, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.dec2_conv = DoubleConv3D(48, 48)

        self.dec1_up = nn.Sequential(
            nn.Conv3d(112, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.dec1_conv = DoubleConv3D(64, 64)

        self.output = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.kaiming_normal_(self.output[0].weight, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.output[0].bias, -2.0)

    @staticmethod
    def _assert_same_spatial(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
        if left.shape[2:] != right.shape[2:]:
            raise RuntimeError(f"{name} spatial mismatch: {tuple(left.shape[2:])} vs {tuple(right.shape[2:])}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.entry(x)              # [B, 64, 64, 32, 32]

        e1 = self.enc1_down(e0)         # [B, 48, 32, 32, 32]
        e1 = self.enc1_conv(e1)

        e2 = self.enc2_down(e1)         # [B, 56, 16, 16, 16]
        e2 = self.enc2_conv(e2)

        e3 = self.enc3_down(e2)         # [B, 64, 8, 8, 8]
        e3 = self.enc3_conv(e3)

        b = self.bottleneck(e3)         # [B, 64, 8, 8, 8]

        d3 = F.interpolate(b, scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)
        self._assert_same_spatial("Dec3/skip2", d3, e2)
        d3 = torch.cat([d3, e2], dim=1) # [B, 120, 16, 16, 16]
        d3 = self.dec3_up(d3)
        d3 = self.dec3_conv(d3)

        d2 = F.interpolate(d3, scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)
        self._assert_same_spatial("Dec2/skip1", d2, e1)
        d2 = torch.cat([d2, e1], dim=1) # [B, 104, 32, 32, 32]
        d2 = self.dec2_up(d2)
        d2 = self.dec2_conv(d2)

        d1 = F.interpolate(d2, scale_factor=(2, 1, 1), mode="trilinear", align_corners=False)
        self._assert_same_spatial("Dec1/e0", d1, e0)
        d1 = torch.cat([d1, e0], dim=1) # [B, 112, 64, 32, 32]
        d1 = self.dec1_up(d1)
        d1 = self.dec1_conv(d1)

        return self.output(d1)


def extract_envelope_slice(complex_volume: torch.Tensor, y_idx: int | None = None) -> torch.Tensor:
    """
    Extract a linear-envelope XZ slice from a complex volume.

    Input:
        [B, 2, Z, X, Y] complex volume (channel 0=real, 1=imag).

    Output:
        [B, 1, Z, X] linear envelope at middle Y, or at y_idx when provided.
    """

    if complex_volume.ndim != 5 or complex_volume.shape[1] != 2:
        raise ValueError(f"Expected [B,2,Z,X,Y], got {tuple(complex_volume.shape)}")
    y_size = complex_volume.shape[-1]
    if y_idx is None:
        y_idx = y_size // 2
    if not 0 <= y_idx < y_size:
        raise IndexError(f"y_idx={y_idx} outside valid range [0,{y_size})")

    real = complex_volume[:, 0]
    imag = complex_volume[:, 1]
    envelope = torch.sqrt(real.square() + imag.square() + 1e-12)
    return envelope[..., y_idx].unsqueeze(1)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    D = Envelope2DPatchDiscriminator()
    x = torch.randn(2, 2, 64, 32)
    out = D(x)
    assert out.shape[1] == 1, f"Expected 1 output channel, got {out.shape}"
    D3 = BMode3DPatchDiscriminator()
    x3 = torch.randn(2, 2, 64, 32, 32)
    out3 = D3(x3)
    assert out3.shape[1] == 1, f"Expected 1 output channel, got {out3.shape}"
    assert torch.isfinite(out3).all(), "BMode3DPatchDiscriminator produced non-finite output"
    print(f"PASS: 2D D output shape {out.shape}")
    print(f"PASS: 3D B-mode D output shape {out3.shape}")
    print(f"3D B-mode D params: {count_trainable_params(D3):,}")
