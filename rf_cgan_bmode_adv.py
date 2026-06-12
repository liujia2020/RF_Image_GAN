from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _condition(candidate_bmode: torch.Tensor, baseline_bmode: torch.Tensor) -> torch.Tensor:
    if candidate_bmode.shape != baseline_bmode.shape:
        raise ValueError(f"candidate/baseline shape mismatch: {candidate_bmode.shape} vs {baseline_bmode.shape}")
    if candidate_bmode.ndim != 5 or candidate_bmode.shape[1] != 1:
        raise ValueError(f"Expected [B,1,Z,X,Y], got {tuple(candidate_bmode.shape)}")
    return torch.cat([candidate_bmode, baseline_bmode], dim=1)


def d_lsgan_loss(
    D: nn.Module,
    pred_bmode: torch.Tensor,
    gt_bmode: torch.Tensor,
    baseline_bmode: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Discriminator objective for LSGAN.

    Sign convention:
    - Real GT conditioned on baseline should be scored as 1.
    - Fake pred conditioned on baseline should be scored as 0.
    - pred is detached here so D updates do not backpropagate into G.
    """

    real = D(_condition(gt_bmode, baseline_bmode))
    fake = D(_condition(pred_bmode.detach(), baseline_bmode))
    real_target = torch.ones_like(real)
    fake_target = torch.zeros_like(fake)
    d_real_loss = F.mse_loss(real, real_target)
    d_fake_loss = F.mse_loss(fake, fake_target)
    d_loss = 0.5 * (d_real_loss + d_fake_loss)
    return {
        "d_loss": d_loss,
        "d_real_loss": d_real_loss,
        "d_fake_loss": d_fake_loss,
        "d_real_score_raw": real.mean(),
        "d_fake_score_raw": fake.mean(),
        "d_real_score_sigmoid": torch.sigmoid(real).mean(),
        "d_fake_score_sigmoid": torch.sigmoid(fake).mean(),
    }


def g_adv_loss(D: nn.Module, pred_bmode: torch.Tensor, baseline_bmode: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Generator adversarial objective for LSGAN.

    Sign convention:
    - G should make D score fake pred conditioned on baseline as 1.
    - pred is not detached here so gradients flow into G.
    """

    fake = D(_condition(pred_bmode, baseline_bmode))
    adv = F.mse_loss(fake, torch.ones_like(fake))
    return {
        "g_adv": adv,
        "g_fake_score_raw": fake.mean(),
        "g_fake_score_sigmoid": torch.sigmoid(fake).mean(),
    }
