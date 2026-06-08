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
from rf_ssim_loss_experiment import (
    DB_MIN,
    EPS,
    SSIM_WIN_SIZE,
    compute_realism_metrics,
    compute_loss_ssim,
    pearson,
    raw_ssim_score,
    regression_slope_intercept,
    scale_to_broadcast,
    ssim_3d_per_sample,
)
from rf_train_utils import complex_abs_2ch, count_trainable_parameters, get_device, seed_everything


TASK_NAME = "Wide/deeper BN model under envelope-SSIM loss + realism eval"
BASE_EXPERIMENT = "tiny_bn_ssim10_random64_full1500"
EXPERIMENT = "wide_ssim_random64_full1500"
ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "checkpoint" / BASE_EXPERIMENT / "run_config.json"
CACHE_ROOT = ROOT / "Data_cache_random64_full1500"
CKPT_DIR = ROOT / "checkpoint" / EXPERIMENT
OUT_DIR = ROOT / "test_metrics" / "wide_ssim"

BASELINE_CKPT = ROOT / "checkpoint" / "tiny_bn_random64_full1500" / "best_model.pth"
SSIM_TINY_CKPT = ROOT / "checkpoint" / BASE_EXPERIMENT / "best_by_val_ssim.pth"

ASPECT_XZ = 0.0362 / 0.2
FIXED_TEST_INDEX = 37
FIXED_TEST_BASENAME = "RF000493_carotid_test_patch003.h5"


def print_header() -> None:
    print("=" * 104)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"task: {TASK_NAME}")
    print(f"experiment: {EXPERIMENT}")
    print("=" * 104)


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


def load_json(path: Path) -> dict[str, Any]:
    if report_missing(path):
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_wide_config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(base_config)
    config["experiment_name"] = EXPERIMENT
    config["model_name"] = "wide_deep"
    config["model_class"] = "WideDeepResidualRFNet"
    config["hidden"] = 128
    config["ckpt_dir"] = str(CKPT_DIR)
    config["amp"] = True
    config["requested_batch_size"] = int(base_config["batch_size"])

    diff_keys = sorted(
        key
        for key in set(base_config) | set(config)
        if base_config.get(key) != config.get(key)
    )
    print(
        "config_diff_vs_ssim_tiny: "
        + ", ".join(f"{key}: {base_config.get(key)!r} -> {config.get(key)!r}" for key in diff_keys)
    )
    expected = {"experiment_name", "model_name", "model_class", "hidden", "ckpt_dir", "amp", "requested_batch_size"}
    assert set(diff_keys) == expected, f"Unexpected initial config differences: {diff_keys}"
    return config


def first_norm(model: torch.nn.Module) -> nn.Module | None:
    return next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)


def assert_batchnorm_model(model: torch.nn.Module) -> None:
    norm = first_norm(model)
    norm_name = type(norm).__name__ if norm is not None else "MISSING"
    print(f"startup model_class={type(model).__name__}")
    print(f"startup first_norm_class={norm_name}")
    print(f"startup trainable_parameters={count_trainable_parameters(model):,}")
    assert isinstance(norm, nn.BatchNorm3d), f"Expected BatchNorm3d, got {norm_name}"


def make_model_from_config(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_model(
        config["model_name"],
        in_channels=1536,
        hidden=128,
        head_channels=32,
        out_channels=2,
        num_blocks=4,
    ).to(device)
    assert_batchnorm_model(model)
    return model


def load_checkpoint_model(ckpt_path: Path, fallback_config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if report_missing(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        config = fallback_config
    model_name = config.get("model_name", fallback_config["model_name"])
    hidden = int(config.get("hidden", fallback_config.get("hidden", 64)))
    use_bn = any("running_mean" in key for key in ckpt["model"])
    kwargs = {
        "in_channels": 1536,
        "hidden": hidden,
        "head_channels": 32,
        "out_channels": 2,
        "num_blocks": 4,
        "use_batch_norm": use_bn,
    }
    model = build_model(model_name, **kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    assert_batchnorm_model(model)
    print(f"loaded_checkpoint={ckpt_path}")
    print(f"loaded_epoch={ckpt.get('epoch', 'NA')}")
    return model, ckpt


def checkpoint_complete() -> bool:
    required = [
        CKPT_DIR / "best_by_val_l1.pth",
        CKPT_DIR / "best_by_val_ssim.pth",
        CKPT_DIR / "final.pth",
        CKPT_DIR / "training_history.csv",
        CKPT_DIR / "run_config.json",
    ]
    return all(path.exists() for path in required)


def make_dataset(split: str) -> RFCachedDataset:
    meta = CACHE_ROOT / split / "meta.npz"
    if report_missing(meta):
        raise FileNotFoundError(meta)
    return RFCachedDataset(CACHE_ROOT / split)


def make_loader(dataset: RFCachedDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["input"].to(device, non_blocking=True)
    y = batch["label"].to(device, non_blocking=True)
    baseline = batch["baseline"].to(device, non_blocking=True)
    return x, y, baseline


def amp_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def probe_batch_size(config: dict[str, Any], device: torch.device) -> int:
    requested = int(config["batch_size"])
    if device.type != "cuda":
        print(f"batch_probe: cpu device, using requested batch_size={requested}")
        return requested

    candidates: list[int] = []
    value = requested
    while value >= 1:
        candidates.append(value)
        value //= 2
    candidates = sorted(set(candidates), reverse=True)
    print(f"batch_probe_candidates={candidates}")

    for candidate in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = make_model_from_config(config, device)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            x = torch.zeros(candidate, 1536, 64, 32, 32, device=device, dtype=torch.float32)
            y = torch.zeros(candidate, 2, 64, 32, 32, device=device, dtype=torch.float32)
            baseline = torch.zeros_like(y)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, True):
                pred = model(x, baseline)
                loss, *_ = compute_loss_ssim(
                    pred,
                    y,
                    abs_weight=float(config["abs_weight"]),
                    lambda_ssim=float(config["lambda_ssim"]),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 0.0) or 0.0))
            scaler.step(optimizer)
            scaler.update()
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(
                f"batch_probe OK batch_size={candidate} "
                f"peak_alloc_GB={peak_alloc:.2f} peak_reserved_GB={peak_reserved:.2f}"
            )
            del model, optimizer, scaler, x, y, baseline, pred, loss
            torch.cuda.empty_cache()
            return candidate
        except torch.cuda.OutOfMemoryError as exc:
            print(f"batch_probe OOM batch_size={candidate}: {exc}")
            try:
                del model, optimizer, scaler, x, y, baseline
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()

    raise RuntimeError("No viable CUDA batch size found")


@torch.no_grad()
def evaluate_normalized_ssim_amp(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    abs_weight: float,
    lambda_ssim: float,
    use_amp: bool,
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
        with amp_context(device, use_amp):
            pred = model(x, baseline)
            loss, l1, abs_l1, ssim_loss, ssim_score, data_range = compute_loss_ssim(
                pred,
                y,
                abs_weight=abs_weight,
                lambda_ssim=lambda_ssim,
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
    scaler: torch.amp.GradScaler,
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
            "scaler": scaler.state_dict(),
            metric_name: metric_value,
            "history": history,
            "config": config,
        },
        path,
    )


def train_wide_ssim(config: dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> None:
    if checkpoint_complete():
        print(f"SKIP complete checkpoint directory: {CKPT_DIR}")
        return

    if CKPT_DIR.exists():
        print(f"RETRAIN incomplete checkpoint directory in place: {CKPT_DIR}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with (CKPT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    seed_everything(int(config["seed"]))
    model = make_model_from_config(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["num_epochs"]),
        eta_min=float(config["eta_min"]),
    )
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"amp_enabled={use_amp}")
    print(f"ssim_impl={config['ssim_impl']} win_size={config['ssim_win_size']} lambda_ssim={config['lambda_ssim']}")

    init_val = evaluate_normalized_ssim_amp(
        model,
        val_loader,
        device,
        abs_weight=float(config["abs_weight"]),
        lambda_ssim=float(config["lambda_ssim"]),
        use_amp=use_amp,
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
    patience = int(config["patience"]) if config.get("patience") is not None else None
    history: list[dict[str, float]] = []

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
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, use_amp):
                pred = model(x, baseline)
                loss, l1, abs_l1, ssim_loss, ssim_score, _data_range = compute_loss_ssim(
                    pred,
                    y,
                    abs_weight=float(config["abs_weight"]),
                    lambda_ssim=float(config["lambda_ssim"]),
                )

            scaler.scale(loss).backward()
            grad_clip = config.get("grad_clip")
            if grad_clip is not None and float(grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()

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

        val = evaluate_normalized_ssim_amp(
            model,
            val_loader,
            device,
            abs_weight=float(config["abs_weight"]),
            lambda_ssim=float(config["lambda_ssim"]),
            use_amp=use_amp,
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
                scaler,
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
                scaler,
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
        scaler,
        "final_val_l1",
        float(history[-1]["val_l1"]),
        history,
        config,
    )
    print("\nTraining finished.")
    print(f"Best val L1   = {best_val_l1:.6e} at epoch {best_l1_epoch}")
    print(f"Best val SSIM = {best_val_ssim:.6f} at epoch {best_ssim_epoch}")
    print(f"Final checkpoint: {CKPT_DIR / 'final.pth'}")


def complex_abs(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(arr[0] ** 2 + arr[1] ** 2)


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


def pooled_amplitude_stats(pooled_label: list[np.ndarray], pooled_pred: list[np.ndarray]) -> dict[str, float]:
    x = np.concatenate(pooled_label).astype(np.float64, copy=False)
    y = np.concatenate(pooled_pred).astype(np.float64, copy=False)
    slope, intercept = regression_slope_intercept(x, y)
    return {
        "slope": slope,
        "intercept": intercept,
        "pearson_r": pearson(x, y),
        "label_p99": float(np.percentile(x, 99.0)),
        "pred_p99": float(np.percentile(y, 99.0)),
        "n_high_signal_voxels": int(x.size),
    }


def evaluate_checkpoint(
    row_name: str,
    ckpt_path: Path,
    fallback_config: dict[str, Any],
    test_set: RFCachedDataset,
    device: torch.device,
) -> dict[str, Any]:
    print("\n" + "#" * 104)
    print(f"EVAL {row_name}")
    model, ckpt = load_checkpoint_model(ckpt_path, fallback_config, device)
    per_sample_rows: list[dict[str, Any]] = []
    realism_rows: list[dict[str, float]] = []
    ssim_scores: list[float] = []
    pooled_label: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []

    for idx in range(len(test_set)):
        sample = test_set[idx]
        pred_raw, label_raw, baseline_raw = infer_one_raw(model, sample, device)
        label_mag = complex_abs(label_raw)
        pred_mag = complex_abs(pred_raw)
        high_mask = label_mag >= float(np.median(label_mag))
        pooled_label.append(label_mag[high_mask].astype(np.float32, copy=True))
        pooled_pred.append(pred_mag[high_mask].astype(np.float32, copy=True))

        pred_complex_l1 = float(np.mean(np.abs(pred_raw - label_raw)))
        base_complex_l1 = float(np.mean(np.abs(baseline_raw - label_raw)))
        pred_abs_l1 = float(np.mean(np.abs(pred_mag - label_mag)))
        base_abs_l1 = float(np.mean(np.abs(complex_abs(baseline_raw) - label_mag)))
        per_sample_rows.append(
            {
                "path": sample["path"],
                "category": sample["category"],
                "pred_complex_l1": pred_complex_l1,
                "base_complex_l1": base_complex_l1,
                "complex_improvement": 1.0 - pred_complex_l1 / (base_complex_l1 + 1e-12),
                "pred_abs_l1": pred_abs_l1,
                "base_abs_l1": base_abs_l1,
                "abs_improvement": 1.0 - pred_abs_l1 / (base_abs_l1 + 1e-12),
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
    pooled = pooled_amplitude_stats(pooled_label, pooled_pred)

    return {
        "row": row_name,
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch", "NA"),
        "complex_impr": float(per_sample["complex_improvement"].mean()),
        "abs_impr": float(per_sample["abs_improvement"].mean()),
        "mean_e_par_over_RMS_L": float(realism["mean_e_par_over_RMS_L"]),
        "perp_par": float(realism["rms_e_perp_over_rms_e_par"]),
        "slope": float(pooled["slope"]),
        "intercept": float(pooled["intercept"]),
        "pearson_r": float(pooled["pearson_r"]),
        "label_p99": float(pooled["label_p99"]),
        "pred_p99": float(pooled["pred_p99"]),
        "n_high_signal_voxels": int(pooled["n_high_signal_voxels"]),
        "abs_std_ratio": float(realism["pred_nonboundary_abs_std_over_label"]),
        "HF_ratio": float(realism["pred_hf_rms_over_label"]),
        "detail_top10": float(realism["detail_retention_top10"]),
        "test_ssim": float(np.mean(ssim_scores)),
    }


def save_linear_visual(fallback_config: dict[str, Any], test_set: RFCachedDataset, device: torch.device) -> Path:
    sample = test_set[FIXED_TEST_INDEX]
    actual_name = Path(str(sample["path"])).name
    assert actual_name == FIXED_TEST_BASENAME, (
        f"Fixed visual sample mismatch: idx={FIXED_TEST_INDEX}, actual={actual_name}, expected={FIXED_TEST_BASENAME}"
    )

    label_panels: list[tuple[str, np.ndarray]] = []
    baseline_model, _ = load_checkpoint_model(BASELINE_CKPT, fallback_config, device)
    tiny_ssim_model, _ = load_checkpoint_model(SSIM_TINY_CKPT, fallback_config, device)
    wide_model, _ = load_checkpoint_model(CKPT_DIR / "best_by_val_ssim.pth", fallback_config, device)

    pred_base, label_raw, _baseline_raw = infer_one_raw(baseline_model, sample, device)
    pred_tiny, _label2, _base2 = infer_one_raw(tiny_ssim_model, sample, device)
    pred_wide, _label3, _base3 = infer_one_raw(wide_model, sample, device)

    y_idx = 16
    label_mag = complex_abs(label_raw)
    panels = [
        ("label", label_mag[:, :, y_idx]),
        ("baseline pred", complex_abs(pred_base)[:, :, y_idx]),
        ("SSIM-Tiny pred", complex_abs(pred_tiny)[:, :, y_idx]),
        ("wide pred", complex_abs(pred_wide)[:, :, y_idx]),
    ]
    stacked = np.concatenate([panel.reshape(-1) for _title, panel in panels])
    vmax = float(np.percentile(stacked, 99.5))
    vmin = 0.0

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    image = None
    for ax, (title, panel) in zip(axes, panels):
        image = ax.imshow(
            panel,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            aspect=ASPECT_XZ,
        )
        ax.set_title(f"{title} | xz y={y_idx}")
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="amplitude")
    fig.suptitle(f"Fixed worst carotid patch idx={FIXED_TEST_INDEX} | {actual_name}", fontsize=10)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / "linear_amp_idx037_label_baseline_tinyssim_wide.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"saved_visual={save_path}")
    return save_path


def main() -> None:
    print_header()
    base_config = load_json(BASE_CONFIG_PATH)
    config = build_wide_config(base_config)
    seed_everything(int(config["seed"]))
    device = get_device()
    print(f"device={device}")
    print("pytorch_msssim_status=MISSING; using custom Gaussian 3D SSIM")

    if not checkpoint_complete():
        selected_batch = probe_batch_size(config, device)
        if selected_batch != int(config["batch_size"]):
            print(f"batch_size_adjusted_for_memory: {config['batch_size']} -> {selected_batch}")
            config["batch_size"] = selected_batch
    else:
        existing = load_json(CKPT_DIR / "run_config.json")
        config.update(existing)
        print(f"using_existing_run_config batch_size={config['batch_size']}")

    train_set = make_dataset("train")
    val_set = make_dataset("val")
    test_set = make_dataset("test")
    train_loader = make_loader(train_set, int(config["batch_size"]), shuffle=True)
    val_loader = make_loader(val_set, int(config["batch_size"]), shuffle=False)
    print(
        f"datasets: train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"batch_size={config['batch_size']} num_workers=0 pin_memory=True"
    )

    train_wide_ssim(config, train_loader, val_loader, device)

    eval_rows = [
        evaluate_checkpoint("baseline_aw01", BASELINE_CKPT, config, test_set, device),
        evaluate_checkpoint("ssim_tiny_best_by_val_ssim", SSIM_TINY_CKPT, config, test_set, device),
        evaluate_checkpoint("wide_best_by_val_ssim", CKPT_DIR / "best_by_val_ssim.pth", config, test_set, device),
        evaluate_checkpoint("wide_final", CKPT_DIR / "final.pth", config, test_set, device),
    ]
    summary = pd.DataFrame(eval_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "wide_ssim_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 104)
    print("wide_ssim_eval_summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"saved_summary={summary_path}")
    visual_path = save_linear_visual(config, test_set, device)
    print(f"visual_path={visual_path}")


if __name__ == "__main__":
    main()
