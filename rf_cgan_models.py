"""
Minimal cGAN modules for the RF_Image_GAN smoke test.

This file intentionally contains only the discriminator-side pieces needed for
the first loop-stability smoke test. The generator is reused from the frozen
RF_Image regression code during this stage.
"""

from __future__ import annotations

import torch
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
    print(f"PASS: D output shape {out.shape}")
