from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from rf_cgan_models import extract_envelope_slice

STRUCT_NORMALIZATION = "label_mean_envelope"
STRUCT_NORMALIZATION_EPS = 1e-8


def complex_envelope(volume: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert [B,2,Z,X,Y] real/imag volume to [B,1,Z,X,Y] linear envelope.
    """

    if volume.ndim != 5 or volume.shape[1] != 2:
        raise ValueError(f"Expected [B,2,Z,X,Y], got {tuple(volume.shape)}")
    env = torch.sqrt(volume[:, 0].square() + volume[:, 1].square() + eps)
    return env.unsqueeze(1)


def _gaussian_kernel1d(sigma: float, truncate: float = 3.0, device=None, dtype=None) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(int(math.ceil(truncate * float(sigma))), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / float(sigma)).square())
    kernel = kernel / kernel.sum()
    return kernel


class AnisotropicGaussianLowpass3D(nn.Module):
    """
    Separable 3D Gaussian low-pass for [B,C,Z,X,Y].

    sigma is in voxel units ordered as (z, x, y).
    """

    def __init__(self, sigma_zyx: tuple[float, float, float] = (6.0, 3.0, 3.0), truncate: float = 3.0):
        super().__init__()
        self.sigma_zyx = tuple(float(x) for x in sigma_zyx)
        self.truncate = float(truncate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,Z,X,Y], got {tuple(x.shape)}")

        out = x
        channels = x.shape[1]
        for axis, sigma in enumerate(self.sigma_zyx):
            kernel = _gaussian_kernel1d(sigma, self.truncate, device=x.device, dtype=x.dtype)
            radius = kernel.numel() // 2
            if axis == 0:
                weight = kernel.view(1, 1, -1, 1, 1).repeat(channels, 1, 1, 1, 1)
                padding = (radius, 0, 0)
            elif axis == 1:
                weight = kernel.view(1, 1, 1, -1, 1).repeat(channels, 1, 1, 1, 1)
                padding = (0, radius, 0)
            else:
                weight = kernel.view(1, 1, 1, 1, -1).repeat(channels, 1, 1, 1, 1)
                padding = (0, 0, radius)
            out = F.conv3d(out, weight, padding=padding, groups=channels)
        return out


def random_y_index(volume: torch.Tensor, rng: random.Random | None = None) -> int:
    y_size = int(volume.shape[-1])
    if y_size <= 0:
        raise ValueError("Cannot sample y index from empty Y dimension")
    if rng is None:
        return random.randrange(y_size)
    return rng.randrange(y_size)


def discriminator_lsgan_loss(
    discriminator: nn.Module,
    pred: torch.Tensor,
    label: torch.Tensor,
    baseline: torch.Tensor,
    y_idx: int,
    criterion: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    criterion = criterion or nn.MSELoss()
    label_env = extract_envelope_slice(label, y_idx=y_idx)
    baseline_env = extract_envelope_slice(baseline, y_idx=y_idx)
    pred_env = extract_envelope_slice(pred.detach(), y_idx=y_idx)

    real_out = discriminator(torch.cat([label_env, baseline_env], dim=1))
    fake_out = discriminator(torch.cat([pred_env, baseline_env], dim=1))
    real_loss = criterion(real_out, torch.ones_like(real_out))
    fake_loss = criterion(fake_out, torch.zeros_like(fake_out))
    loss = 0.5 * (real_loss + fake_loss)
    return loss, {
        "d_real_loss": real_loss.detach(),
        "d_fake_loss": fake_loss.detach(),
        "d_real_score_sigmoid": torch.sigmoid(real_out.detach().float()).mean(),
        "d_fake_score_sigmoid": torch.sigmoid(fake_out.detach().float()).mean(),
    }


def generator_lsgan_struct_carrier_loss(
    discriminator: nn.Module,
    pred: torch.Tensor,
    label: torch.Tensor,
    baseline: torch.Tensor,
    lowpass: AnisotropicGaussianLowpass3D,
    y_idx: int,
    lambda_adv: float = 1.0,
    lambda_struct: float = 10.0,
    lambda_carrier: float = 1.0,
    criterion: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    criterion = criterion or nn.MSELoss()

    pred_env_slice = extract_envelope_slice(pred, y_idx=y_idx)
    baseline_env_slice = extract_envelope_slice(baseline, y_idx=y_idx)
    adv_out = discriminator(torch.cat([pred_env_slice, baseline_env_slice], dim=1))
    adv = criterion(adv_out, torch.ones_like(adv_out))

    pred_env = complex_envelope(pred)
    label_env = complex_envelope(label)
    # pred and label share one label-derived scalar scale. This is exactly
    # equivalent to the previous structural term divided by scale, because
    # Gaussian low-pass is linear: LP((pred-label)/s) == LP(pred-label)/s.
    struct_scale = label_env.detach().abs().mean() + STRUCT_NORMALIZATION_EPS
    struct_delta = lowpass((pred_env - label_env) / struct_scale)
    struct = torch.mean(torch.abs(struct_delta))
    struct_unnormalized = struct * struct_scale
    if float(lambda_carrier) == 0.0:
        carrier = pred.detach().new_zeros(())
        carrier_unnormalized = pred.detach().new_zeros(())
        carrier_weighted = pred.detach().new_zeros(())
        total = lambda_adv * adv + lambda_struct * struct
        carrier_skipped = True
    else:
        # Carrier uses the same label-derived scalar as struct. This is
        # exactly the previous complex L1 carrier divided by s:
        # L1(pred/s, label/s) == L1(pred, label) / s.
        carrier = F.l1_loss(pred / struct_scale, label / struct_scale)
        carrier_unnormalized = carrier * struct_scale
        carrier_weighted = lambda_carrier * carrier
        total = lambda_adv * adv + lambda_struct * struct + lambda_carrier * carrier
        carrier_skipped = False

    return total, {
        "g_adv_raw": adv.detach(),
        "g_struct_raw": struct.detach(),
        "g_struct_unnormalized_raw": struct_unnormalized.detach(),
        "g_struct_scale": struct_scale.detach(),
        "g_carrier_raw": carrier.detach(),
        "g_carrier_unnormalized_raw": carrier_unnormalized.detach(),
        "g_carrier_scale": struct_scale.detach(),
        "g_adv_weighted": (lambda_adv * adv).detach(),
        "g_struct_weighted": (lambda_struct * struct).detach(),
        "g_carrier_weighted": carrier_weighted.detach(),
        "g_total": total.detach(),
        "g_carrier_skipped": torch.tensor(float(carrier_skipped), device=pred.device),
        "y_idx": torch.tensor(float(y_idx), device=pred.device),
    }
