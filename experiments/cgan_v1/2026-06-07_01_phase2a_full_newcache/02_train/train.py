from __future__ import annotations

import atexit
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset


RUN_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_DIR = RUN_DIR / "01_config"
TRAIN_DIR = RUN_DIR / "02_train"
LOG_DIR = TRAIN_DIR / "logs"
METRICS_DIR = TRAIN_DIR / "metrics"
FIG_DIR = TRAIN_DIR / "figures"
CKPT_DIR = TRAIN_DIR / "checkpoints"
LOCK_PATH = LOG_DIR / "train.lock"


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or CONFIG_DIR / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_import_paths(config: dict[str, Any]) -> None:
    legacy_code_dir = Path(config["data"]["legacy_code_dir"])
    for path in [PROJECT_ROOT, legacy_code_dir]:
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
    sys.path.insert(0, str(legacy_code_dir))
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_dirs() -> None:
    for path in [LOG_DIR, METRICS_DIR, FIG_DIR, CKPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def acquire_lock() -> None:
    ensure_dirs()
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        try:
            existing_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            existing_pid = -1
        if existing_pid > 0 and Path(f"/proc/{existing_pid}").exists():
            raise RuntimeError(f"已有训练进程仍在运行: pid={existing_pid}")
        LOCK_PATH.unlink(missing_ok=True)
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    def cleanup_lock() -> None:
        try:
            if LOCK_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
                LOCK_PATH.unlink(missing_ok=True)
        except FileNotFoundError:
            pass

    atexit.register(cleanup_lock)


def read_available_memory_gb() -> float:
    meminfo = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as f:
        for line in f:
            key, value = line.split(":", 1)
            meminfo[key] = float(value.strip().split()[0]) / (1024**2)
    return float(meminfo.get("MemAvailable", 0.0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(config: dict[str, Any]) -> None:
    seed = int(config["sampler"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device_float(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=True).float()


def batch_tensor_stats(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats = {}
    for key in ["input", "label", "baseline"]:
        tensor = batch[key].float()
        stats[key] = {
            "shape": list(tensor.shape),
            "dtype": str(batch[key].dtype),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "mean_abs": float(tensor.abs().mean().item()),
            "has_nan": bool(torch.isnan(tensor).any().item()),
            "has_inf": bool(torch.isinf(tensor).any().item()),
        }
    return stats


def make_datasets(config: dict[str, Any]):
    from rf_cached_dataset import RFCachedDataset

    cache_root = Path(config["data"]["cache_dir"])
    return {
        "train": RFCachedDataset(
            cache_root / config["data"]["train_split"],
            return_fp16=bool(config["training"]["cache_return_fp16"]),
            restore_scale=bool(config["training"]["restore_scale"]),
        ),
        "val": RFCachedDataset(
            cache_root / config["data"]["val_split"],
            return_fp16=bool(config["training"]["cache_return_fp16"]),
            restore_scale=bool(config["training"]["restore_scale"]),
        ),
    }


def make_train_loader(dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    from rf_cgan_data import StratifiedCategoryBatchSampler

    sampler = StratifiedCategoryBatchSampler(
        dataset.categories,
        required_categories=list(config["data"]["categories"]),
        samples_per_category=int(config["sampler"]["samples_per_category"]),
        shuffle=shuffle,
        seed=int(config["sampler"]["seed"]),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )


def pick_fixed_val_indices(val_dataset, config: dict[str, Any]) -> dict[str, list[int]]:
    per_category = int(config["diagnostics"]["fixed_val_per_category"])
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, category in enumerate(val_dataset.categories):
        grouped[str(category)].append(idx)
    fixed = {}
    for category in config["data"]["categories"]:
        if len(grouped[category]) < per_category:
            raise RuntimeError(f"val split 中 {category} 数量不足: {len(grouped[category])} < {per_category}")
        fixed[category] = grouped[category][:per_category]
    return fixed


def make_fixed_loader(val_dataset, fixed_indices_by_category: dict[str, list[int]]) -> DataLoader:
    fixed_indices = [idx for indices in fixed_indices_by_category.values() for idx in indices]
    return DataLoader(Subset(val_dataset, fixed_indices), batch_size=3, shuffle=False, num_workers=0)


def make_models(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from rf_cgan_losses import AnisotropicGaussianLowpass3D
    from rf_cgan_models import Envelope2DPatchDiscriminator
    from rf_models import TinyResidualRFNet

    g = TinyResidualRFNet(use_batch_norm=bool(config["generator"]["use_bn"])).to(device)
    d = Envelope2DPatchDiscriminator(ndf=int(config["discriminator"]["ndf"])).to(device)
    lowpass_cfg = config["loss"]["lowpass"]["sigma_voxels"]
    lowpass = AnisotropicGaussianLowpass3D(
        (float(lowpass_cfg["z"]), float(lowpass_cfg["x"]), float(lowpass_cfg["y"])),
        truncate=float(config["loss"]["lowpass"]["truncate"]),
    ).to(device)
    betas = (float(config["training"]["beta1"]), float(config["training"]["beta2"]))
    opt_g = torch.optim.Adam(g.parameters(), lr=float(config["training"]["lr_G"]), betas=betas)
    opt_d = torch.optim.Adam(d.parameters(), lr=float(config["training"]["lr_D"]), betas=betas)
    return {"G": g, "D": d, "lowpass": lowpass, "opt_G": opt_g, "opt_D": opt_d}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def first_norm_class(model: nn.Module) -> str:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d)):
            return type(module).__name__
    return "MISSING"


def cell1_self_check(config: dict[str, Any]) -> dict[str, Any]:
    add_import_paths(config)
    set_seed(config)
    ensure_dirs()
    cache_dir = str(config["data"]["cache_dir"])
    assert cache_dir == str(config["safety"]["expected_cache_dir"]), f"cache 路径不是新 cache: {cache_dir}"
    assert str(config["safety"]["forbidden_cache_token"]) not in cache_dir, f"cache 路径含旧坏 cache token: {cache_dir}"
    assert bool(config["training"]["use_amp"]) is False, "use_amp 必须为 false"
    expected_workers = int(config["safety"]["expected_num_workers"])
    assert int(config["training"]["num_workers"]) == expected_workers, f"num_workers 必须为 {expected_workers}"

    available_gb = read_available_memory_gb()
    min_required_gb = float(config["diagnostics"]["min_available_memory_gb"])
    assert available_gb > min_required_gb, f"系统可用内存不足: {available_gb:.2f}GB <= {min_required_gb:.2f}GB"

    datasets = make_datasets(config)
    loader = make_train_loader(datasets["train"], config, shuffle=False)
    batch = next(iter(loader))
    categories = sorted(str(x) for x in batch["category"])
    expected = sorted(str(x) for x in config["data"]["categories"])
    assert categories == expected, f"batch 三类别不齐: {categories} != {expected}"
    stats = batch_tensor_stats(batch)
    for field in ["input", "label", "baseline"]:
        assert not stats[field]["has_nan"], f"{field} 含 NaN"
        assert not stats[field]["has_inf"], f"{field} 含 Inf"
    assert stats["label"]["max_abs"] > 1.0e4, f"label 幅值不像真实大幅值: {stats['label']['max_abs']}"
    assert stats["baseline"]["max_abs"] > 1.0e4, f"baseline 幅值不像真实大幅值: {stats['baseline']['max_abs']}"

    report = {
        "timestamp": timestamp(),
        "cache_dir": cache_dir,
        "use_amp": bool(config["training"]["use_amp"]),
        "num_workers": int(config["training"]["num_workers"]),
        "available_memory_gb": available_gb,
        "batch_categories": list(batch["category"]),
        "batch_stats": stats,
        "status": "self_check_passed",
    }
    (LOG_DIR / "cell1_自检结果.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def cell2_config(config: dict[str, Any]) -> dict[str, Any]:
    report = {
        "run_name": config["run_name"],
        "cache_dir": config["data"]["cache_dir"],
        "epochs": int(config["training"]["epochs"]),
        "use_amp": bool(config["training"]["use_amp"]),
        "num_workers": int(config["training"]["num_workers"]),
        "loss": config["loss"],
        "sampler": config["sampler"],
    }
    (LOG_DIR / "cell2_配置摘要.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def cell3_data(config: dict[str, Any]) -> dict[str, Any]:
    datasets = make_datasets(config)
    train_loader = make_train_loader(datasets["train"], config, shuffle=True)
    fixed_indices = pick_fixed_val_indices(datasets["val"], config)
    fixed_loader = make_fixed_loader(datasets["val"], fixed_indices)
    (LOG_DIR / "固定val探针索引.json").write_text(json.dumps(fixed_indices, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "datasets": datasets,
        "train_loader": train_loader,
        "fixed_loader": fixed_loader,
        "fixed_indices": fixed_indices,
        "summary": {"train_samples": len(datasets["train"]), "val_samples": len(datasets["val"])},
    }


def cell4_models(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    models = make_models(config, device)
    norm_class = first_norm_class(models["G"])
    assert norm_class == "BatchNorm3d", f"生成器归一化层不是 BatchNorm3d: {norm_class}"
    report = {
        "device": str(device),
        "G_params": count_params(models["G"]),
        "D_params": count_params(models["D"]),
        "G_first_norm": norm_class,
    }
    (LOG_DIR / "cell4_模型摘要.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    models["device"] = device
    return {"models": models, "summary": report}


def finite_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def train_one_batch(
    batch: dict[str, Any],
    config: dict[str, Any],
    models: dict[str, Any],
    rng: random.Random,
) -> dict[str, float]:
    from rf_cgan_losses import discriminator_lsgan_loss, generator_lsgan_struct_carrier_loss, random_y_index

    device = models["device"]
    g = models["G"]
    d = models["D"]
    x = to_device_float(batch["input"], device)
    label = to_device_float(batch["label"], device)
    baseline = to_device_float(batch["baseline"], device)
    y_idx = random_y_index(label, rng=rng)

    models["opt_D"].zero_grad(set_to_none=True)
    pred_for_d = g(x, baseline)
    d_loss, d_terms = discriminator_lsgan_loss(d, pred_for_d, label, baseline, y_idx=y_idx)
    d_loss.backward()
    models["opt_D"].step()

    models["opt_G"].zero_grad(set_to_none=True)
    pred = g(x, baseline)
    g_loss, g_terms = generator_lsgan_struct_carrier_loss(
        d,
        pred,
        label,
        baseline,
        models["lowpass"],
        y_idx=y_idx,
        lambda_adv=float(config["loss"]["lambda_adv"]),
        lambda_struct=float(config["loss"]["lambda_struct"]),
        lambda_carrier=float(config["loss"]["lambda_carrier"]),
    )
    g_loss.backward()
    models["opt_G"].step()

    adv_raw = finite_float(g_terms["g_adv_raw"])
    struct_raw = finite_float(g_terms["g_struct_raw"])
    carrier_raw = finite_float(g_terms["g_carrier_raw"])
    metrics = {
        "D_real": finite_float(d_terms["d_real_loss"]),
        "D_fake": finite_float(d_terms["d_fake_loss"]),
        "D_real_score": finite_float(d_terms["d_real_score_sigmoid"]),
        "D_fake_score": finite_float(d_terms["d_fake_score_sigmoid"]),
        "D_loss": finite_float(d_loss),
        "G_adv_raw": adv_raw,
        "G_struct_raw": struct_raw,
        "G_carrier_raw": carrier_raw,
        "G_adv_weighted": float(config["loss"]["lambda_adv"]) * adv_raw,
        "G_struct_weighted": float(config["loss"]["lambda_struct"]) * struct_raw,
        "G_carrier_weighted": float(config["loss"]["lambda_carrier"]) * carrier_raw,
        "G_total": finite_float(g_loss),
        "pred_max_abs": float(pred.detach().float().abs().max().item()),
        "has_nan": float(any(torch.isnan(t.detach().float()).any().item() for t in [pred, d_loss, g_loss])),
        "has_inf": float(any(torch.isinf(t.detach().float()).any().item() for t in [pred, d_loss, g_loss])),
        "y_idx": float(y_idx),
    }
    return metrics


def aggregate_epoch(epoch: int, batch_metrics: list[dict[str, float]], seconds: float, peak_gb: float) -> dict[str, Any]:
    mean_keys = [
        "D_real",
        "D_fake",
        "D_real_score",
        "D_fake_score",
        "D_loss",
        "G_adv_raw",
        "G_struct_raw",
        "G_carrier_raw",
        "G_adv_weighted",
        "G_struct_weighted",
        "G_carrier_weighted",
        "G_total",
    ]
    row: dict[str, Any] = {
        "epoch": epoch,
        "batches": len(batch_metrics),
        "seconds": seconds,
        "peak_cuda_mem_gb": peak_gb,
        "pred_max_abs": max(float(x["pred_max_abs"]) for x in batch_metrics),
        "has_nan": int(any(bool(x["has_nan"]) for x in batch_metrics)),
        "has_inf": int(any(bool(x["has_inf"]) for x in batch_metrics)),
        "available_memory_gb": read_available_memory_gb(),
    }
    for key in mean_keys:
        row[key] = float(np.mean([float(x[key]) for x in batch_metrics]))
    return row


def complex_envelope_np(volume: torch.Tensor) -> np.ndarray:
    v = volume.detach().float().cpu()
    env = torch.sqrt(v[:, 0].square() + v[:, 1].square() + 1e-12)
    return env.numpy()


def ks_statistic(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    hi = float(max(np.percentile(a, 99.5), np.percentile(b, 99.5), 1e-12))
    hist_a, edges = np.histogram(np.clip(a, 0, hi), bins=bins, range=(0, hi), density=False)
    hist_b, _ = np.histogram(np.clip(b, 0, hi), bins=edges, density=False)
    cdf_a = np.cumsum(hist_a) / max(np.sum(hist_a), 1)
    cdf_b = np.cumsum(hist_b) / max(np.sum(hist_b), 1)
    return float(np.max(np.abs(cdf_a - cdf_b)))


@torch.no_grad()
def evaluate_speckle(
    epoch: int,
    config: dict[str, Any],
    models: dict[str, Any],
    fixed_loader: DataLoader,
) -> list[dict[str, Any]]:
    models["G"].eval()
    pred_by_category: dict[str, list[np.ndarray]] = defaultdict(list)
    label_by_category: dict[str, list[np.ndarray]] = defaultdict(list)
    for batch in fixed_loader:
        x = to_device_float(batch["input"], models["device"])
        label = to_device_float(batch["label"], models["device"])
        baseline = to_device_float(batch["baseline"], models["device"])
        pred = models["G"](x, baseline)
        pred_env = complex_envelope_np(pred)
        label_env = complex_envelope_np(label)
        for idx, category in enumerate(batch["category"]):
            pred_by_category[str(category)].append(pred_env[idx])
            label_by_category[str(category)].append(label_env[idx])

    rows = []
    bins = int(config["diagnostics"]["speckle_hist_bins"])
    for category in config["data"]["categories"]:
        pred_env = np.concatenate([x.ravel() for x in pred_by_category[category]])
        label_env = np.concatenate([x.ravel() for x in label_by_category[category]])
        rows.append(
            {
                "epoch": epoch,
                "category": category,
                "pred_SNR": float(np.mean(pred_env) / (np.std(pred_env) + 1e-12)),
                "label_SNR": float(np.mean(label_env) / (np.std(label_env) + 1e-12)),
                "env_hist_KS": ks_statistic(pred_env, label_env, bins),
            }
        )
    return rows


@torch.no_grad()
def save_triplet_figures(
    epoch: int,
    config: dict[str, Any],
    models: dict[str, Any],
    val_dataset,
    fixed_indices: dict[str, list[int]],
) -> None:
    models["G"].eval()
    y_idx = int(config["diagnostics"]["probe_y_idx"])
    z_spacing = float(config["diagnostics"]["voxel_spacing_mm"]["z"])
    x_spacing = float(config["diagnostics"]["voxel_spacing_mm"]["x"])
    aspect = z_spacing / x_spacing

    for category in config["data"]["categories"]:
        sample = val_dataset[fixed_indices[category][0]]
        x = to_device_float(sample["input"].unsqueeze(0), models["device"])
        label = to_device_float(sample["label"].unsqueeze(0), models["device"])
        baseline = to_device_float(sample["baseline"].unsqueeze(0), models["device"])
        pred = models["G"](x, baseline)

        label_env = complex_envelope_np(label)[0, :, :, y_idx]
        baseline_env = complex_envelope_np(baseline)[0, :, :, y_idx]
        pred_env = complex_envelope_np(pred)[0, :, :, y_idx]
        vmax = float(max(np.percentile(label_env, 99.5), 1e-12))

        fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
        for ax, arr, title in zip(axes, [label_env, baseline_env, pred_env], ["label", "baseline", "pred"]):
            image = ax.imshow(arr, cmap="gray", vmin=0, vmax=vmax, aspect=aspect)
            ax.set_title(f"{category} {title} epoch {epoch}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(FIG_DIR / f"{category}_epoch{epoch:03d}_triplet_y{y_idx}.png", dpi=180)
        plt.close(fig)


def plot_loss_curves(stability_rows: list[dict[str, Any]]) -> None:
    if not stability_rows:
        return
    epochs = [int(row["epoch"]) for row in stability_rows]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for key in ["D_real", "D_fake", "G_adv_weighted", "G_struct_weighted", "G_carrier_weighted", "G_total"]:
        ax.plot(epochs, [float(row[key]) for row in stability_rows], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / weighted loss")
    ax.set_title("Phase2a loss curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(FIG_DIR / "phase2a_loss_curves.png", dpi=180)
    plt.close(fig)


def save_checkpoint(epoch: int, config: dict[str, Any], models: dict[str, Any]) -> str:
    path = CKPT_DIR / f"checkpoint_epoch{epoch:03d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "G_state_dict": models["G"].state_dict(),
            "D_state_dict": models["D"].state_dict(),
            "opt_G_state_dict": models["opt_G"].state_dict(),
            "opt_D_state_dict": models["opt_D"].state_dict(),
            "config": config,
        },
        path,
    )
    return str(path.relative_to(RUN_DIR))


def is_nonfinite(row: dict[str, Any]) -> bool:
    for value in row.values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return True
    return bool(row.get("has_nan")) or bool(row.get("has_inf"))


def cell5_train_loop(
    config: dict[str, Any],
    data_state: dict[str, Any],
    model_state: dict[str, Any],
) -> dict[str, Any]:
    acquire_lock()
    train_loader = data_state["train_loader"]
    fixed_loader = data_state["fixed_loader"]
    val_dataset = data_state["datasets"]["val"]
    fixed_indices = data_state["fixed_indices"]
    models = model_state["models"]
    rng = random.Random(int(config["sampler"]["seed"]))

    total_epochs = int(config["training"]["epochs"])
    snapshot_epochs = set(int(x) for x in config["training"]["snapshot_epochs"])
    checkpoint_epochs = set(int(x) for x in config["training"]["checkpoint_epochs"])
    png_epochs = set(int(x) for x in config["diagnostics"]["png_epochs"])
    stability_rows: list[dict[str, Any]] = []
    speckle_rows: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    abort_reason = None
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        models["G"].train()
        models["D"].train()
        if models["device"].type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()
        batch_metrics: list[dict[str, float]] = []

        for batch_idx, batch in enumerate(train_loader, start=1):
            metrics = train_one_batch(batch, config, models, rng)
            batch_metrics.append(metrics)
            if batch_idx == 1 or batch_idx % 50 == 0:
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d} "
                    f"D_real={metrics['D_real']:.6g} D_fake={metrics['D_fake']:.6g} "
                    f"G_adv_w={metrics['G_adv_weighted']:.6g} "
                    f"G_struct_w={metrics['G_struct_weighted']:.6g} "
                    f"G_carrier_w={metrics['G_carrier_weighted']:.6g} "
                    f"G_total={metrics['G_total']:.6g} pred_max_abs={metrics['pred_max_abs']:.6g}",
                    flush=True,
                )
            if any(not math.isfinite(float(value)) for value in metrics.values()) or metrics["has_nan"] or metrics["has_inf"]:
                abort_reason = f"非有限值: epoch={epoch}, batch={batch_idx}, metrics={metrics}"
                break

        peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if models["device"].type == "cuda" else 0.0
        epoch_row = aggregate_epoch(epoch, batch_metrics, time.time() - epoch_start, peak_gb)
        stability_rows.append(epoch_row)
        write_csv(METRICS_DIR / "stability.csv", stability_rows)
        plot_loss_curves(stability_rows)

        speckle_epoch_rows = evaluate_speckle(epoch, config, models, fixed_loader)
        speckle_rows.extend(speckle_epoch_rows)
        write_csv(METRICS_DIR / "speckle_check.csv", speckle_rows)
        if epoch in png_epochs or epoch in snapshot_epochs:
            save_triplet_figures(epoch, config, models, val_dataset, fixed_indices)

        if epoch in checkpoint_epochs or epoch == total_epochs:
            checkpoints.append(save_checkpoint(epoch, config, models))
            (TRAIN_DIR / "checkpoints.txt").write_text("\n".join(checkpoints) + "\n", encoding="utf-8")

        print(
            f"EPOCH_SUMMARY epoch={epoch:03d} batches={epoch_row['batches']} "
            f"D_real={epoch_row['D_real']:.6g} D_fake={epoch_row['D_fake']:.6g} "
            f"D_real_score={epoch_row['D_real_score']:.6g} D_fake_score={epoch_row['D_fake_score']:.6g} "
            f"G_adv_w={epoch_row['G_adv_weighted']:.6g} "
            f"G_struct_w={epoch_row['G_struct_weighted']:.6g} "
            f"G_carrier_w={epoch_row['G_carrier_weighted']:.6g} "
            f"G_total={epoch_row['G_total']:.6g} pred_max_abs={epoch_row['pred_max_abs']:.6g} "
            f"peakGB={epoch_row['peak_cuda_mem_gb']:.3f} memGB={epoch_row['available_memory_gb']:.3f}",
            flush=True,
        )

        if abort_reason or is_nonfinite(epoch_row):
            abort_reason = abort_reason or f"epoch 汇总出现非有限值: {epoch_row}"
            break

    elapsed_sec = time.time() - start_time
    status = "completed" if abort_reason is None and len(stability_rows) == total_epochs else "stopped"
    result = {
        "timestamp": timestamp(),
        "status": status,
        "abort_reason": abort_reason,
        "epochs_completed": len(stability_rows),
        "epochs_expected": total_epochs,
        "elapsed_sec": elapsed_sec,
        "last_stability": stability_rows[-1] if stability_rows else None,
        "checkpoints": checkpoints,
    }
    (LOG_DIR / "cell5_训练结果.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"result": result, "stability_rows": stability_rows, "speckle_rows": speckle_rows}


def cell6_loss_curves(stability_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if stability_rows is None:
        path = METRICS_DIR / "stability.csv"
        if not path.exists():
            return {"status": "missing_stability_csv"}
        with path.open("r", encoding="utf-8") as f:
            stability_rows = list(csv.DictReader(f))
    plot_loss_curves(stability_rows)
    return {"loss_curve": str((FIG_DIR / "phase2a_loss_curves.png").relative_to(RUN_DIR))}


def cell7_cleanup() -> dict[str, Any]:
    if LOCK_PATH.exists():
        try:
            if LOCK_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
                LOCK_PATH.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "cleanup_done", "timestamp": timestamp()}


def cell8_auto_summary(config: dict[str, Any]) -> dict[str, Any]:
    stability_path = METRICS_DIR / "stability.csv"
    speckle_path = METRICS_DIR / "speckle_check.csv"
    stability_rows = list(csv.DictReader(stability_path.open("r", encoding="utf-8"))) if stability_path.exists() else []
    speckle_rows = list(csv.DictReader(speckle_path.open("r", encoding="utf-8"))) if speckle_path.exists() else []
    last = stability_rows[-1] if stability_rows else {}
    final_epoch = max([int(row["epoch"]) for row in speckle_rows], default=0)
    final_speckle = [row for row in speckle_rows if int(row["epoch"]) == final_epoch]

    lines = [
        "# Phase2a 正式训练自动摘要",
        "",
        f"生成时间：{timestamp()}",
        "",
        "## 配置事实",
        "",
        f"- cache：`{config['data']['cache_dir']}`",
        f"- use_amp：`{config['training']['use_amp']}`",
        f"- num_workers：`{config['training']['num_workers']}`",
        f"- epochs 配置：`{config['training']['epochs']}`",
        "",
        "## 训练健康事实",
        "",
        f"- 已记录 epoch 数：{len(stability_rows)}",
    ]
    for key in [
        "D_real",
        "D_fake",
        "D_real_score",
        "D_fake_score",
        "G_adv_weighted",
        "G_struct_weighted",
        "G_carrier_weighted",
        "G_total",
        "pred_max_abs",
        "has_nan",
        "has_inf",
        "available_memory_gb",
    ]:
        if key in last:
            lines.append(f"- 最后 epoch {key}：{last[key]}")
    lines.extend(["", "## 最后一次 speckle 固定探针", ""])
    for row in final_speckle:
        lines.append(
            f"- epoch {row['epoch']} {row['category']}: pred_SNR={row['pred_SNR']}, "
            f"label_SNR={row['label_SNR']}, env_hist_KS={row['env_hist_KS']}"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本摘要只报损失设计健康事实，不做图像质量结论；质量结论需后续 NIfTI 与用户 3D Slicer 签收。",
        ]
    )
    summary_path = TRAIN_DIR / "phase2a_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {"summary_path": str(summary_path.relative_to(RUN_DIR))}
