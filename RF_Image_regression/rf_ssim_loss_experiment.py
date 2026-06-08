from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rf_cached_dataset import RFCachedDataset
from rf_models import build_model
from rf_train_utils import complex_abs_2ch, count_trainable_parameters, get_device, seed_everything
from rf_visualization import complex_abs_numpy


TASK_NAME = "RF envelope-SSIM loss experiment + realism eval"
BASE_EXPERIMENT = "tiny_bn_random64_full1500"
EXPERIMENT = "tiny_bn_ssim10_random64_full1500"
ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "checkpoint" / BASE_EXPERIMENT / "run_config.json"
CACHE_ROOT = ROOT / "Data_cache_random64_full1500"
CKPT_DIR = ROOT / "checkpoint" / EXPERIMENT
OUT_DIR = ROOT / "test_metrics" / "ssim_exp"
BASE_CKPT = ROOT / "checkpoint" / BASE_EXPERIMENT / "best_model.pth"

LAMBDA_SSIM = 1.0
SSIM_WIN_SIZE = 7
SSIM_SIGMA = 1.5
ASPECT_XZ = 0.0362 / 0.2
DB_MIN = -60.0
EPS = 1e-8
FIXED_TEST_INDEX = 37
FIXED_TEST_BASENAME = "RF000493_carotid_test_patch003.h5"


def print_header() -> None:
    print("=" * 96)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"task: {TASK_NAME}")
    print(f"experiment: {EXPERIMENT}")
    print("=" * 96)


def report_missing(path: Path) -> bool:
    if path.exists():
        return False
    print(f"MISSING: {path}")
    parent = path.parent
    if parent.exists():
        print(f"ls {parent}:")
        for item in sorted(parent.iterdir(), key=lambda p: p.name):
            suffix = "/" if item.is_dir() else ""
            size = "" if item.is_dir() else f" {item.stat().st_size}"
            print(f"  {item.name}{suffix}{size}")
    else:
        print(f"MISSING DIR: {parent}")
    return True


def load_base_config() -> dict[str, Any]:
    if report_missing(BASE_CONFIG_PATH):
        raise FileNotFoundError(BASE_CONFIG_PATH)
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_experiment_config(base_config: dict[str, Any]) -> dict[str, Any]:
    base_compare = dict(base_config)
    base_compare["ckpt_dir"] = str(ROOT / "checkpoint" / BASE_EXPERIMENT)

    config = dict(base_config)
    config["experiment_name"] = EXPERIMENT
    config["ckpt_dir"] = str(CKPT_DIR)
    config["lambda_ssim"] = LAMBDA_SSIM
    config["ssim_win_size"] = SSIM_WIN_SIZE
    config["ssim_impl"] = "custom_gaussian_3d"

    diff_keys = sorted(
        key
        for key in set(base_compare) | set(config)
        if base_compare.get(key) != config.get(key)
    )
    print(
        "config_diff_vs_baseline: "
        + ", ".join(f"{key}: {base_compare.get(key)!r} -> {config.get(key)!r}" for key in diff_keys)
    )
    assert set(diff_keys) == {"experiment_name", "ckpt_dir", "lambda_ssim", "ssim_win_size", "ssim_impl"}, (
        f"Unexpected config differences vs baseline: {diff_keys}"
    )
    return config


def first_norm_name(model: torch.nn.Module) -> str:
    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    return type(first_norm).__name__ if first_norm is not None else "MISSING"


def assert_batchnorm_model(model: torch.nn.Module) -> None:
    norm_name = first_norm_name(model)
    print(f"startup model_class={type(model).__name__}")
    print(f"startup first_norm_class={norm_name}")
    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    assert isinstance(first_norm, nn.BatchNorm3d), f"Expected BatchNorm3d, got {norm_name}"


def make_tiny_bn_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_model(
        config["model_name"],
        in_channels=1536,
        hidden=int(config["hidden"]),
        out_channels=2,
        use_batch_norm=True,
    ).to(device)
    assert_batchnorm_model(model)
    print(f"trainable_parameters={count_trainable_parameters(model):,}")
    return model


def load_model_from_checkpoint(ckpt_path: Path, config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if report_missing(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"]
    use_bn = any("running_mean" in key for key in state_dict)
    model = build_model(
        config["model_name"],
        in_channels=1536,
        hidden=int(config["hidden"]),
        out_channels=2,
        use_batch_norm=use_bn,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    assert_batchnorm_model(model)
    print(f"loaded_checkpoint={ckpt_path}")
    print(f"loaded_epoch={ckpt.get('epoch', 'NA')}")
    return model, ckpt


def make_datasets_and_loaders(config: dict[str, Any]) -> tuple[RFCachedDataset, RFCachedDataset, RFCachedDataset, DataLoader, DataLoader]:
    for split in ("train", "val", "test"):
        if report_missing(CACHE_ROOT / split / "meta.npz"):
            raise FileNotFoundError(CACHE_ROOT / split / "meta.npz")

    train_set = RFCachedDataset(CACHE_ROOT / "train")
    val_set = RFCachedDataset(CACHE_ROOT / "val")
    test_set = RFCachedDataset(CACHE_ROOT / "test")

    expected_categories = set(config["include_categories"])
    for split_name, dataset in (("train", train_set), ("val", val_set), ("test", test_set)):
        actual_categories = set(dataset.categories)
        assert actual_categories <= expected_categories, (
            f"Unexpected category in {split_name}: {sorted(actual_categories)}"
        )

    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": 0,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    print(
        f"datasets: train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"batch_size={loader_kwargs['batch_size']} num_workers={loader_kwargs['num_workers']} "
        f"pin_memory={loader_kwargs['pin_memory']}"
    )
    return train_set, val_set, test_set, train_loader, val_loader


def gaussian_1d(win_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2
    kernel = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_3d_kernel(win_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g = gaussian_1d(win_size, sigma, device, dtype)
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, win_size, win_size, win_size)


def conv_gaussian_3d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = kernel.shape[-1] // 2
    x_pad = F.pad(x, (pad, pad, pad, pad, pad, pad), mode="reflect")
    return F.conv3d(x_pad, kernel)


def ssim_3d_per_sample(
    x: torch.Tensor,
    y: torch.Tensor,
    win_size: int = SSIM_WIN_SIZE,
    sigma: float = SSIM_SIGMA,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != y.shape:
        raise ValueError(f"SSIM inputs must have same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.ndim != 5 or x.shape[1] != 1:
        raise ValueError(f"Expected [B,1,Z,X,Y], got {tuple(x.shape)}")
    spatial_min = min(x.shape[2:])
    if spatial_min < win_size:
        win_size = 5 if spatial_min >= 5 else 3
        print(f"WARNING: reduced SSIM win_size to {win_size} for spatial shape {tuple(x.shape[2:])}")

    kernel = gaussian_3d_kernel(win_size, sigma, x.device, x.dtype)
    data_range = y.detach().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = conv_gaussian_3d(x, kernel)
    mu_y = conv_gaussian_3d(y, kernel)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = conv_gaussian_3d(x * x, kernel) - mu_x2
    sigma_y2 = conv_gaussian_3d(y * y, kernel) - mu_y2
    sigma_xy = conv_gaussian_3d(x * y, kernel) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    ssim_map = torch.relu(ssim_map)
    return ssim_map.mean(dim=(1, 2, 3, 4)), data_range.flatten()


def compute_loss_ssim(
    pred: torch.Tensor,
    label: torch.Tensor,
    abs_weight: float,
    lambda_ssim: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    complex_l1 = F.l1_loss(pred, label)
    env_pred = complex_abs_2ch(pred).clamp_min(0.0)
    env_label = complex_abs_2ch(label).clamp_min(0.0)
    abs_l1 = F.l1_loss(env_pred, env_label)
    ssim_values, data_range = ssim_3d_per_sample(env_pred, env_label)
    ssim_score = ssim_values.mean()
    ssim_loss = 1.0 - ssim_score
    loss = complex_l1 + abs_weight * abs_l1 + lambda_ssim * ssim_loss
    return loss, complex_l1, abs_l1, ssim_loss, ssim_score, data_range


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["input"].to(device, non_blocking=True)
    y = batch["label"].to(device, non_blocking=True)
    b = batch["baseline"].to(device, non_blocking=True)
    return x, y, b


@torch.no_grad()
def evaluate_normalized_ssim(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    abs_weight: float,
    lambda_ssim: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "abs": 0.0,
        "ssim_loss": 0.0,
        "ssim_score": 0.0,
        "baseline_l1": 0.0,
        "baseline_abs": 0.0,
    }
    data_ranges: list[np.ndarray] = []
    n_batches = 0
    for batch in loader:
        x, y, baseline = move_batch(batch, device)
        pred = model(x, baseline)
        loss, l1, abs_l1, ssim_loss, ssim_score, data_range = compute_loss_ssim(
            pred, y, abs_weight=abs_weight, lambda_ssim=lambda_ssim
        )
        base_l1 = F.l1_loss(baseline, y)
        base_abs = F.l1_loss(complex_abs_2ch(baseline), complex_abs_2ch(y))
        totals["loss"] += float(loss.item())
        totals["l1"] += float(l1.item())
        totals["abs"] += float(abs_l1.item())
        totals["ssim_loss"] += float(ssim_loss.item())
        totals["ssim_score"] += float(ssim_score.item())
        totals["baseline_l1"] += float(base_l1.item())
        totals["baseline_abs"] += float(base_abs.item())
        data_ranges.append(data_range.detach().cpu().numpy())
        n_batches += 1

    n = max(n_batches, 1)
    out = {key: value / n for key, value in totals.items()}
    out["improvement"] = 1.0 - out["l1"] / (out["baseline_l1"] + 1e-12)
    out["abs_improvement"] = 1.0 - out["abs"] / (out["baseline_abs"] + 1e-12)
    dr = np.concatenate(data_ranges) if data_ranges else np.array([np.nan], dtype=np.float32)
    out["data_range_p5"] = float(np.percentile(dr, 5))
    out["data_range_p50"] = float(np.percentile(dr, 50))
    out["data_range_p95"] = float(np.percentile(dr, 95))
    out["data_range_mean"] = float(np.mean(dr))
    return out


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric_name: str,
    metric_value: float,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_class": type(model).__name__,
            "optimizer": optimizer.state_dict(),
            metric_name: metric_value,
            "history": history,
            "config": config,
        },
        path,
    )


def checkpoint_complete() -> bool:
    required = [
        CKPT_DIR / "best_by_val_l1.pth",
        CKPT_DIR / "best_by_val_ssim.pth",
        CKPT_DIR / "final.pth",
        CKPT_DIR / "training_history.csv",
        CKPT_DIR / "run_config.json",
    ]
    return all(path.exists() for path in required)


def train_ssim_experiment(
    config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> None:
    if checkpoint_complete():
        print(f"SKIP complete SSIM checkpoint directory: {CKPT_DIR}")
        return

    if CKPT_DIR.exists():
        print(f"RETRAIN incomplete SSIM checkpoint directory in place: {CKPT_DIR}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with (CKPT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    seed_everything(int(config["seed"]))
    model = make_tiny_bn_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["num_epochs"]),
        eta_min=float(config["eta_min"]),
    )

    print(f"ssim_impl={config['ssim_impl']}")
    print(f"ssim_win_size={config['ssim_win_size']} lambda_ssim={config['lambda_ssim']}")

    init_val = evaluate_normalized_ssim(
        model,
        val_loader,
        device,
        abs_weight=float(config["abs_weight"]),
        lambda_ssim=float(config["lambda_ssim"]),
    )
    print("\nInitial validation:")
    print(
        f"val_L1={init_val['l1']:.6e} | "
        f"val_abs={init_val['abs']:.6e} | "
        f"val_ssim={init_val['ssim_score']:.6f} | "
        f"data_range(p5/p50/p95)={init_val['data_range_p5']:.4f}/"
        f"{init_val['data_range_p50']:.4f}/{init_val['data_range_p95']:.4f}"
    )

    best_val_l1 = float("inf")
    best_val_ssim = -float("inf")
    best_l1_epoch = -1
    best_ssim_epoch = -1
    no_l1_improve = 0
    history: list[dict[str, float]] = []
    patience = int(config["patience"]) if config.get("patience") is not None else None

    for epoch in range(1, int(config["num_epochs"]) + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "l1": 0.0,
            "abs": 0.0,
            "ssim_loss": 0.0,
            "ssim_score": 0.0,
            "base_l1": 0.0,
            "base_abs": 0.0,
        }
        n_batches = 0
        for batch in train_loader:
            x, y, baseline = move_batch(batch, device)
            pred = model(x, baseline)
            loss, l1, abs_l1, ssim_loss, ssim_score, _data_range = compute_loss_ssim(
                pred,
                y,
                abs_weight=float(config["abs_weight"]),
                lambda_ssim=float(config["lambda_ssim"]),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = config.get("grad_clip")
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            with torch.no_grad():
                base_l1 = F.l1_loss(baseline, y)
                base_abs = F.l1_loss(complex_abs_2ch(baseline), complex_abs_2ch(y))

            totals["loss"] += float(loss.item())
            totals["l1"] += float(l1.item())
            totals["abs"] += float(abs_l1.item())
            totals["ssim_loss"] += float(ssim_loss.item())
            totals["ssim_score"] += float(ssim_score.item())
            totals["base_l1"] += float(base_l1.item())
            totals["base_abs"] += float(base_abs.item())
            n_batches += 1

        scheduler.step()
        n = max(n_batches, 1)
        train_loss = totals["loss"] / n
        train_l1 = totals["l1"] / n
        train_abs = totals["abs"] / n
        train_ssim_loss = totals["ssim_loss"] / n
        train_ssim_score = totals["ssim_score"] / n
        train_base_l1 = totals["base_l1"] / n
        train_base_abs = totals["base_abs"] / n
        train_impr = 1.0 - train_l1 / (train_base_l1 + 1e-12)
        train_abs_impr = 1.0 - train_abs / (train_base_abs + 1e-12)

        val = evaluate_normalized_ssim(
            model,
            val_loader,
            device,
            abs_weight=float(config["abs_weight"]),
            lambda_ssim=float(config["lambda_ssim"]),
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_l1": train_l1,
            "train_abs": train_abs,
            "train_ssim_loss": train_ssim_loss,
            "train_ssim": train_ssim_score,
            "train_base_l1": train_base_l1,
            "train_base_abs": train_base_abs,
            "train_impr": train_impr,
            "train_abs_impr": train_abs_impr,
            "val_loss": val["loss"],
            "val_l1": val["l1"],
            "val_abs": val["abs"],
            "val_ssim_loss": val["ssim_loss"],
            "val_ssim": val["ssim_score"],
            "val_base_l1": val["baseline_l1"],
            "val_base_abs": val["baseline_abs"],
            "val_impr": val["improvement"],
            "val_abs_impr": val["abs_improvement"],
            "val_data_range_p5": val["data_range_p5"],
            "val_data_range_p50": val["data_range_p50"],
            "val_data_range_p95": val["data_range_p95"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(CKPT_DIR / "training_history.csv", index=False)

        if val["l1"] < best_val_l1:
            best_val_l1 = val["l1"]
            best_l1_epoch = epoch
            no_l1_improve = 0
            save_checkpoint(
                CKPT_DIR / "best_by_val_l1.pth",
                epoch,
                model,
                optimizer,
                "best_val_l1",
                best_val_l1,
                history,
                config,
            )
        else:
            no_l1_improve += 1

        if val["ssim_score"] > best_val_ssim:
            best_val_ssim = val["ssim_score"]
            best_ssim_epoch = epoch
            save_checkpoint(
                CKPT_DIR / "best_by_val_ssim.pth",
                epoch,
                model,
                optimizer,
                "best_val_ssim",
                best_val_ssim,
                history,
                config,
            )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train_L1={train_l1:.6e} train_abs={train_abs:.6e} train_ssim={train_ssim_score:.6f} | "
                f"val_L1={val['l1']:.6e} val_abs={val['abs']:.6e} val_ssim={val['ssim_score']:.6f} | "
                f"val_impr={val['improvement']*100:6.2f}% val_abs_impr={val['abs_improvement']*100:6.2f}% | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if patience is not None and no_l1_improve >= patience:
            print(
                "Early stopping triggered: "
                f"val L1 did not improve for {patience} consecutive epoch(s). "
                f"best_l1_epoch={best_l1_epoch} best_ssim_epoch={best_ssim_epoch}"
            )
            break

    save_checkpoint(
        CKPT_DIR / "final.pth",
        int(history[-1]["epoch"]),
        model,
        optimizer,
        "final_val_l1",
        float(history[-1]["val_l1"]),
        history,
        config,
    )
    torch.save(model.state_dict(), CKPT_DIR / "final_state_dict.pth")

    print("\nTraining finished.")
    print(f"Best val L1  = {best_val_l1:.6e} at epoch {best_l1_epoch}")
    print(f"Best val SSIM = {best_val_ssim:.6f} at epoch {best_ssim_epoch}")
    print(f"Final checkpoint: {CKPT_DIR / 'final.pth'}")


def complex_abs(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(arr[0] ** 2 + arr[1] ** 2)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(af) & np.isfinite(bf)
    af = af[mask]
    bf = bf[mask]
    if af.size == 0:
        return float("nan")
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = float(np.sqrt(np.mean(af * af)) * np.sqrt(np.mean(bf * bf)))
    if denom <= 0:
        return float("nan")
    return float(np.mean(af * bf) / denom)


def rms(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals)))


def regression_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xf = np.asarray(x, dtype=np.float64).reshape(-1)
    yf = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xf) & np.isfinite(yf)
    xf = xf[mask]
    yf = yf[mask]
    if xf.size == 0:
        return float("nan"), float("nan")
    x_mean = float(xf.mean())
    y_mean = float(yf.mean())
    var_x = float(np.mean((xf - x_mean) ** 2))
    if var_x <= 0:
        return float("nan"), float("nan")
    slope = float(np.mean((xf - x_mean) * (yf - y_mean)) / var_x)
    intercept = y_mean - slope * x_mean
    return slope, float(intercept)


def highpass7(mag: np.ndarray) -> np.ndarray:
    center = mag[1:-1, 1:-1, 1:-1]
    local_mean = (
        center
        + mag[:-2, 1:-1, 1:-1]
        + mag[2:, 1:-1, 1:-1]
        + mag[1:-1, :-2, 1:-1]
        + mag[1:-1, 2:, 1:-1]
        + mag[1:-1, 1:-1, :-2]
        + mag[1:-1, 1:-1, 2:]
    ) / 7.0
    return center - local_mean


def compute_realism_metrics(pred_raw: np.ndarray, label_raw: np.ndarray, baseline_raw: np.ndarray) -> dict[str, float]:
    label_mag = complex_abs(label_raw).astype(np.float64, copy=False)
    pred_mag = complex_abs(pred_raw).astype(np.float64, copy=False)
    threshold = float(np.median(label_mag))
    high_signal_mask = label_mag >= threshold

    er = (pred_raw[0] - label_raw[0]).astype(np.float64, copy=False)
    ei = (pred_raw[1] - label_raw[1]).astype(np.float64, copy=False)
    lhat_r = label_raw[0].astype(np.float64, copy=False) / (label_mag + EPS)
    lhat_i = label_raw[1].astype(np.float64, copy=False) / (label_mag + EPS)
    e_par = er * lhat_r + ei * lhat_i
    e_sq = er * er + ei * ei
    e_perp = np.sqrt(np.maximum(e_sq - e_par * e_par, 0.0))

    e_par_m = e_par[high_signal_mask]
    e_perp_m = e_perp[high_signal_mask]
    label_mag_m = label_mag[high_signal_mask]
    pred_mag_m = pred_mag[high_signal_mask]
    slope, intercept = regression_slope_intercept(label_mag_m, pred_mag_m)

    label_hp = highpass7(label_mag)
    pred_hp = highpass7(pred_mag)
    baseline_hp = highpass7(complex_abs(baseline_raw))
    eps = 1e-12
    label_hf_rms = rms(label_hp)
    pred_hf_rms = rms(pred_hp)

    label_detail = np.abs(label_hp)
    top_threshold = float(np.percentile(label_detail, 90.0))
    top_mask = label_detail >= top_threshold
    detail_retention_top10 = float(
        np.median(np.abs(pred_hp[top_mask]) / (np.abs(label_hp[top_mask]) + eps))
    )

    pred_hf_l1 = float(np.mean(np.abs(pred_hp - label_hp)))
    baseline_hf_l1 = float(np.mean(np.abs(baseline_hp - label_hp)))

    return {
        "rms_e_perp_over_rms_e_par": rms(e_perp_m) / max(rms(e_par_m), EPS),
        "mean_e_par_over_RMS_L": float(np.mean(e_par_m)) / max(rms(label_mag_m), EPS),
        "amp_regression_slope": slope,
        "amp_regression_intercept": intercept,
        "corr_e_par_abs_label": pearson(e_par_m, label_mag_m),
        "corr_e_perp_abs_label": pearson(e_perp_m, label_mag_m),
        "pred_hf_rms_over_label": pred_hf_rms / (label_hf_rms + eps),
        "pred_hf_improvement_vs_baseline": 1.0 - pred_hf_l1 / (baseline_hf_l1 + eps),
        "detail_retention_top10": detail_retention_top10,
        "pred_nonboundary_abs_std_over_label": float(np.std(pred_mag) / (np.std(label_mag) + eps)),
    }


def scale_to_broadcast(scale: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while scale.ndim < reference.ndim:
        scale = scale.view(*scale.shape, *([1] * (reference.ndim - scale.ndim)))
    return scale


@torch.no_grad()
def infer_one_raw(model: torch.nn.Module, sample: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
    label = sample["label"].unsqueeze(0).to(device=device, dtype=torch.float32)
    baseline = sample["baseline"].unsqueeze(0).to(device=device, dtype=torch.float32)
    scale = sample["scale"].to(device=device, dtype=torch.float32)
    scale = scale_to_broadcast(scale, label)
    pred = model(x, baseline)
    pred_raw = (pred * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    label_raw = (label * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_raw = (baseline * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return pred_raw, label_raw, baseline_raw


def raw_ssim_score(pred_raw: np.ndarray, label_raw: np.ndarray, device: torch.device) -> float:
    pred_env = torch.from_numpy(complex_abs(pred_raw)).view(1, 1, *pred_raw.shape[1:]).to(device=device, dtype=torch.float32)
    label_env = torch.from_numpy(complex_abs(label_raw)).view(1, 1, *label_raw.shape[1:]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        ssim_values, _dr = ssim_3d_per_sample(pred_env, label_env)
    return float(ssim_values.mean().detach().cpu())


def evaluate_checkpoint(
    row_name: str,
    ckpt_path: Path,
    config: dict[str, Any],
    test_set: RFCachedDataset,
    device: torch.device,
) -> dict[str, Any]:
    print("\n" + "#" * 96)
    print(f"EVAL {row_name}")
    model, ckpt = load_model_from_checkpoint(ckpt_path, config, device)
    per_sample_rows: list[dict[str, Any]] = []
    realism_rows: list[dict[str, float]] = []
    ssim_scores: list[float] = []
    for idx in range(len(test_set)):
        sample = test_set[idx]
        pred_raw, label_raw, baseline_raw = infer_one_raw(model, sample, device)
        pred_complex_l1 = float(np.mean(np.abs(pred_raw - label_raw)))
        base_complex_l1 = float(np.mean(np.abs(baseline_raw - label_raw)))
        pred_abs_l1 = float(np.mean(np.abs(complex_abs(pred_raw) - complex_abs(label_raw))))
        base_abs_l1 = float(np.mean(np.abs(complex_abs(baseline_raw) - complex_abs(label_raw))))
        per_sample_rows.append(
            {
                "path": sample["path"],
                "category": sample["category"],
                "pred_complex_l1": pred_complex_l1,
                "base_complex_l1": base_complex_l1,
                "complex_improvement": 1.0 - pred_complex_l1 / (base_complex_l1 + 1e-12),
                "complex_better": pred_complex_l1 < base_complex_l1,
                "pred_abs_l1": pred_abs_l1,
                "base_abs_l1": base_abs_l1,
                "abs_improvement": 1.0 - pred_abs_l1 / (base_abs_l1 + 1e-12),
                "abs_better": pred_abs_l1 < base_abs_l1,
            }
        )
        realism_rows.append(compute_realism_metrics(pred_raw, label_raw, baseline_raw))
        ssim_scores.append(raw_ssim_score(pred_raw, label_raw, device))
        if (idx + 1) % 25 == 0 or idx == len(test_set) - 1:
            print(f"  eval_inference={idx + 1}/{len(test_set)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_sample = pd.DataFrame(per_sample_rows)
    per_sample.to_csv(OUT_DIR / f"{row_name}_per_sample_metrics.csv", index=False)
    cat = per_sample.groupby("category").agg(
        n=("path", "count"),
        complex_improvement_mean=("complex_improvement", "mean"),
        abs_improvement_mean=("abs_improvement", "mean"),
    )
    cat.to_csv(OUT_DIR / f"{row_name}_per_category_summary.csv")
    realism = pd.DataFrame(realism_rows).mean(numeric_only=True).to_dict()
    row = {
        "row": row_name,
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch", "NA"),
        "complex_impr": float(per_sample["complex_improvement"].mean()),
        "abs_impr": float(per_sample["abs_improvement"].mean()),
        "mean_e_par_over_RMS_L": float(realism["mean_e_par_over_RMS_L"]),
        "perp_par": float(realism["rms_e_perp_over_rms_e_par"]),
        "slope_a": float(realism["amp_regression_slope"]),
        "intercept_b": float(realism["amp_regression_intercept"]),
        "HF_ratio": float(realism["pred_hf_rms_over_label"]),
        "detail_top10": float(realism["detail_retention_top10"]),
        "abs_std_ratio": float(realism["pred_nonboundary_abs_std_over_label"]),
        "test_ssim": float(np.mean(ssim_scores)),
    }
    return row


def to_db(mag: np.ndarray, ref: float, db_min: float = DB_MIN) -> np.ndarray:
    db = 20.0 * np.log10(np.maximum(mag.astype(np.float32), 1e-12) / max(float(ref), 1e-12))
    return np.maximum(db, db_min)


def save_visual_comparison(config: dict[str, Any], test_set: RFCachedDataset, device: torch.device) -> Path:
    sample = test_set[FIXED_TEST_INDEX]
    actual_name = Path(str(sample["path"])).name
    assert actual_name == FIXED_TEST_BASENAME, (
        f"Fixed visual sample mismatch: idx={FIXED_TEST_INDEX}, actual={actual_name}, expected={FIXED_TEST_BASENAME}"
    )

    models = []
    baseline_model, _baseline_ckpt = load_model_from_checkpoint(BASE_CKPT, config, device)
    ssim_model, _ssim_ckpt = load_model_from_checkpoint(CKPT_DIR / "best_by_val_ssim.pth", config, device)
    models.append(("baseline aw=0.1", baseline_model))
    models.append(("SSIM best_by_val_ssim", ssim_model))

    panels_by_row = []
    ref = 0.0
    y_idx = 16
    for label, model in models:
        pred_raw, label_raw, _baseline_raw = infer_one_raw(model, sample, device)
        label_mag = complex_abs(label_raw)
        pred_mag = complex_abs(pred_raw)
        err_mag = complex_abs(pred_raw - label_raw)
        ref = max(ref, float(label_mag.max()), float(pred_mag.max()), float(err_mag.max()))
        panels_by_row.append((label, label_mag, pred_mag, err_mag))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
    image = None
    for row_idx, (row_label, label_mag, pred_mag, err_mag) in enumerate(panels_by_row):
        panels = [("label", label_mag[:, :, y_idx]), ("pred", pred_mag[:, :, y_idx]), ("error", err_mag[:, :, y_idx])]
        for col_idx, (title, panel) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                to_db(panel, ref),
                cmap="gray",
                vmin=DB_MIN,
                vmax=0.0,
                origin="upper",
                aspect=ASPECT_XZ,
            )
            ax.set_title(f"{row_label} | {title} | xz y={y_idx}")
            ax.set_xlabel("x index")
            ax.set_ylabel("z index")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="dB")
    fig.suptitle(f"Fixed worst carotid patch idx={FIXED_TEST_INDEX} | {actual_name}", fontsize=10)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / "ssim_vs_baseline_fixed_carotid_worst_xz_y16.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"saved_visual={save_path}")
    return save_path


def main() -> None:
    print_header()
    base_config = load_base_config()
    config = build_experiment_config(base_config)
    seed_everything(int(config["seed"]))
    device = get_device()
    print(f"device={device}")
    print("pytorch_msssim_status=MISSING; using custom Gaussian 3D SSIM")

    _train_set, _val_set, test_set, train_loader, val_loader = make_datasets_and_loaders(config)
    train_ssim_experiment(config, train_loader, val_loader, device)

    eval_rows = [
        evaluate_checkpoint("baseline_aw01", BASE_CKPT, config, test_set, device),
        evaluate_checkpoint("ssim_best_by_val_ssim", CKPT_DIR / "best_by_val_ssim.pth", config, test_set, device),
        evaluate_checkpoint("ssim_final", CKPT_DIR / "final.pth", config, test_set, device),
    ]
    summary = pd.DataFrame(eval_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "ssim_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 96)
    print("ssim_eval_summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"saved_summary={summary_path}")
    visual_path = save_visual_comparison(config, test_set, device)
    print(f"visual_path={visual_path}")


if __name__ == "__main__":
    main()
