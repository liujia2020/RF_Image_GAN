from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rf_models import MODEL_NAME_TO_CLASS


# ============================================================
# Reproducibility / device helpers
# ============================================================

def seed_everything(seed: int = 20260522) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Loss helpers
# ============================================================

def complex_abs_2ch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute magnitude from a two-channel complex tensor.

    Parameters
    ----------
    x:
        Tensor with shape [B, 2, Z, X, Y]. Channel 0 is real and channel 1 is imaginary.
    eps:
        Small value to avoid zero gradient around zero.

    Returns
    -------
    Tensor with shape [B, 1, Z, X, Y].
    """
    if x.ndim != 5 or x.shape[1] != 2:
        raise ValueError(f"Expected x shape [B,2,Z,X,Y], got {tuple(x.shape)}")
    return torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + eps)


def compute_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    abs_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss used for RF residual training.

    loss = complex real/imag L1 + abs_weight * envelope L1

    Returns
    -------
    loss, loss_l1, loss_abs
    """
    loss_l1 = F.l1_loss(pred, label)

    pred_abs = complex_abs_2ch(pred)
    label_abs = complex_abs_2ch(label)
    loss_abs = F.l1_loss(pred_abs, label_abs)

    loss = loss_l1 + abs_weight * loss_abs
    return loss, loss_l1, loss_abs


# ============================================================
# Evaluation
# ============================================================

def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["input"].to(device, non_blocking=True)
    y = batch["label"].to(device, non_blocking=True)
    b = batch["baseline"].to(device, non_blocking=True)
    return x, y, b


@torch.no_grad()
def evaluate_normalized(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    abs_weight: float = 0.1,
) -> Dict[str, float]:
    """
    Evaluate model in normalized space.

    This matches training loss when RFLearningDataset(normalize=True) is used.
    It does not denormalize by sample scale.
    """
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_abs = 0.0
    total_baseline_l1 = 0.0
    total_baseline_abs = 0.0
    n_batches = 0

    for batch in loader:
        x, y, b = _move_batch_to_device(batch, device)

        pred = model(x, b)
        loss, loss_l1, loss_abs = compute_loss(pred, y, abs_weight=abs_weight)

        baseline_l1 = F.l1_loss(b, y)
        baseline_abs = F.l1_loss(complex_abs_2ch(b), complex_abs_2ch(y))

        total_loss += float(loss.item())
        total_l1 += float(loss_l1.item())
        total_abs += float(loss_abs.item())
        total_baseline_l1 += float(baseline_l1.item())
        total_baseline_abs += float(baseline_abs.item())
        n_batches += 1

    n = max(n_batches, 1)
    total_loss /= n
    total_l1 /= n
    total_abs /= n
    total_baseline_l1 /= n
    total_baseline_abs /= n

    improvement = 1.0 - total_l1 / (total_baseline_l1 + 1e-12)
    abs_improvement = 1.0 - total_abs / (total_baseline_abs + 1e-12)

    return {
        "loss": total_loss,
        "l1": total_l1,
        "abs": total_abs,
        "baseline_l1": total_baseline_l1,
        "baseline_abs": total_baseline_abs,
        "improvement": improvement,
        "abs_improvement": abs_improvement,
    }


# ============================================================
# Training
# ============================================================

def train_model_jupyter(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    ckpt_dir: str | Path,
    experiment_name: str = "experiment",
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    abs_weight: float = 0.1,
    grad_clip: Optional[float] = 1.0,
    print_every: int = 5,
    eta_min: float = 1e-5,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    patience: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generic training loop for RF residual models.

    The model forward signature must be:
        pred = model(input, baseline)

    Files saved under ckpt_dir:
        best_model.pth
        best_state_dict.pth
        final_state_dict.pth
        training_history.csv
        run_config.json
    """
    if seed is not None:
        seed_everything(seed)

    ckpt_dir = Path(ckpt_dir).expanduser()
    ckpt_dir_full = ckpt_dir.resolve()
    model_class = type(model).__name__
    trainable_params = count_trainable_parameters(model)

    if config is None:
        config = {}

    config = dict(config)
    config.update({
        "experiment_name": experiment_name,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "abs_weight": abs_weight,
        "grad_clip": grad_clip,
        "eta_min": eta_min,
        "model_class": model_class,
    })
    if patience is not None:
        if patience <= 0:
            raise ValueError("patience must be a positive integer or None.")
        config["patience"] = patience

    separator = "=" * 72
    print(f"\n{separator}")
    print("RF TRAINING STARTUP CHECK")
    print(f"experiment_name      : {experiment_name}")
    print(f"ckpt_dir             : {ckpt_dir_full}")
    print(f"model_class          : {model_class}")
    print(f"trainable_parameters : {trainable_params:,}")

    assert experiment_name in str(ckpt_dir_full), (
        "Checkpoint directory does not match experiment_name: "
        f"experiment_name={experiment_name!r}, ckpt_dir={str(ckpt_dir_full)!r}"
    )

    if "model_name" in config:
        model_name = str(config["model_name"]).lower()
        expected_model_class = MODEL_NAME_TO_CLASS.get(model_name)
        if expected_model_class is None:
            print("!" * 72)
            print(
                "WARNING: config['model_name'] is not registered in "
                "MODEL_NAME_TO_CLASS; cannot verify the model class "
                f"(model_name={config['model_name']!r}, actual_class={model_class!r})."
            )
            print("!" * 72)
        else:
            assert expected_model_class == model_class, (
                "Model config mismatch: "
                f"model_name={config['model_name']!r}, "
                f"expected_class={expected_model_class!r}, "
                f"actual_class={model_class!r}"
            )

    print(separator)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=eta_min,
    )

    best_val_l1 = float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    history = []

    init_val = evaluate_normalized(model, val_loader, device, abs_weight=abs_weight)
    print("\nInitial validation:")
    print(
        f"val_L1={init_val['l1']:.6e} | "
        f"baseline_L1={init_val['baseline_l1']:.6e} | "
        f"improvement={init_val['improvement']*100:.2f}%"
    )

    for epoch in range(1, num_epochs + 1):
        model.train()

        train_loss = 0.0
        train_l1 = 0.0
        train_abs = 0.0
        train_base_l1 = 0.0
        train_base_abs = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y, b = _move_batch_to_device(batch, device)

            pred = model(x, b)
            loss, loss_l1, loss_abs = compute_loss(pred, y, abs_weight=abs_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            with torch.no_grad():
                base_l1 = F.l1_loss(b, y)
                base_abs = F.l1_loss(complex_abs_2ch(b), complex_abs_2ch(y))

            train_loss += float(loss.item())
            train_l1 += float(loss_l1.item())
            train_abs += float(loss_abs.item())
            train_base_l1 += float(base_l1.item())
            train_base_abs += float(base_abs.item())
            n_batches += 1

        scheduler.step()

        n = max(n_batches, 1)
        train_loss /= n
        train_l1 /= n
        train_abs /= n
        train_base_l1 /= n
        train_base_abs /= n

        train_impr = 1.0 - train_l1 / (train_base_l1 + 1e-12)
        train_abs_impr = 1.0 - train_abs / (train_base_abs + 1e-12)

        val_metrics = evaluate_normalized(model, val_loader, device, abs_weight=abs_weight)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l1": train_l1,
            "train_abs": train_abs,
            "train_base_l1": train_base_l1,
            "train_base_abs": train_base_abs,
            "train_impr": train_impr,
            "train_abs_impr": train_abs_impr,
            "val_loss": val_metrics["loss"],
            "val_l1": val_metrics["l1"],
            "val_abs": val_metrics["abs"],
            "val_base_l1": val_metrics["baseline_l1"],
            "val_base_abs": val_metrics["baseline_abs"],
            "val_impr": val_metrics["improvement"],
            "val_abs_impr": val_metrics["abs_improvement"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        history_df = pd.DataFrame(history)
        history_df.to_csv(ckpt_dir / "training_history.csv", index=False)

        if val_metrics["l1"] < best_val_l1:
            best_val_l1 = val_metrics["l1"]
            best_epoch = epoch
            no_improve_epochs = 0

            best_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "model_class": model_class,
                "optimizer": optimizer.state_dict(),
                "best_val_l1": best_val_l1,
                "history": history,
                "config": config,
            }

            torch.save(best_ckpt, ckpt_dir / "best_model.pth")
            torch.save(model.state_dict(), ckpt_dir / "best_state_dict.pth")
        else:
            no_improve_epochs += 1

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train_L1={train_l1:.6e} | "
                f"train_base={train_base_l1:.6e} | "
                f"train_impr={train_impr*100:6.2f}% | "
                f"val_L1={val_metrics['l1']:.6e} | "
                f"val_base={val_metrics['baseline_l1']:.6e} | "
                f"val_impr={val_metrics['improvement']*100:6.2f}% | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if patience is not None and no_improve_epochs >= patience:
            print(
                "Early stopping triggered: "
                f"val L1 did not improve for {patience} consecutive epoch(s). "
                f"best_epoch={best_epoch}"
            )
            break

    torch.save(model.state_dict(), ckpt_dir / "final_state_dict.pth")

    history_df = pd.DataFrame(history)
    history_df.to_csv(ckpt_dir / "training_history.csv", index=False)

    print("\nTraining finished.")
    print(f"Best val L1 = {best_val_l1:.6e} at epoch {best_epoch}")
    print(f"Best checkpoint: {ckpt_dir / 'best_model.pth'}")
    print(f"History CSV    : {ckpt_dir / 'training_history.csv'}")

    return history_df


# ============================================================
# Checkpoint / plot helpers
# ============================================================

def load_checkpoint_model(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    device: torch.device,
    key: str = "model",
) -> Dict[str, Any]:
    """Load a checkpoint into a model and return the checkpoint object."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and key in ckpt:
        model.load_state_dict(ckpt[key])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return ckpt if isinstance(ckpt, dict) else {"model": ckpt}


def plot_training_curve(history_df: pd.DataFrame, title: str = "Training curve") -> None:
    """Simple Jupyter-friendly training curve plot."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["train_l1"], label="train L1")
    plt.plot(history_df["epoch"], history_df["val_l1"], label="val L1")

    if "val_base_l1" in history_df.columns:
        plt.axhline(history_df["val_base_l1"].iloc[0], linestyle="--", label="val baseline")

    plt.xlabel("Epoch")
    plt.ylabel("Normalized L1")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    best_idx = history_df["val_l1"].idxmin()
    print("Best row:")
    print(history_df.loc[best_idx])
