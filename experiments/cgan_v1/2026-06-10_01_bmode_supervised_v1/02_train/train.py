from __future__ import annotations

import atexit
import csv
import json
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
import torch.nn.functional as F
import yaml
from pytorch_msssim import SSIM
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


RUN_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "rf_cgan_data.py").exists())
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
    with (config_path or CONFIG_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_import_paths() -> None:
    path = str(PROJECT_ROOT)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


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
    with Path("/proc/meminfo").open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024**2)
    return 0.0


def set_seed(config: dict[str, Any]) -> None:
    seed = int(config["sampler"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bmode_from_complex_np(volume: np.ndarray, ref: float, eps: float, db_clip: tuple[float, float]) -> np.ndarray:
    real = volume[0].astype(np.float32, copy=False)
    imag = volume[1].astype(np.float32, copy=False)
    env = np.sqrt(real * real + imag * imag, dtype=np.float32)
    db = 20.0 * np.log10(env / np.float32(ref) + np.float32(eps))
    db = np.clip(db, np.float32(db_clip[0]), np.float32(db_clip[1]))
    return ((db - np.float32(db_clip[0])) / np.float32(db_clip[1] - db_clip[0]))[None].astype(np.float32)


class BModeCachedDataset(Dataset):
    def __init__(self, rf_split_dir: Path, bmode_split_dir: Path, config: dict[str, Any]):
        self.rf_split_dir = Path(rf_split_dir)
        self.bmode_split_dir = Path(bmode_split_dir)
        self.restore_input_scale = bool(config["training"]["restore_input_scale"])
        self.ref = float(config["data"]["bmode_reference_value"])
        self.eps = float(config["data"]["bmode_eps"])
        self.db_clip = tuple(float(x) for x in config["data"]["bmode_db_clip"])

        with np.load(self.rf_split_dir / "meta.npz", allow_pickle=False) as meta:
            self.input_shape = tuple(int(x) for x in meta["input_shape"])
            self.baseline_shape = tuple(int(x) for x in meta["baseline_shape"])
            self.paths = meta["path"].astype(str).tolist()
            self.categories = meta["category"].astype(str).tolist()
            self.z_idx = meta["z_idx"].astype(np.int32, copy=False)
            self.x_idx = meta["x_idx"].astype(np.int32, copy=False)
            self.y_idx = meta["y_idx"].astype(np.int32, copy=False)
            self.normalized = bool(meta["normalize"]) if "normalize" in meta.files else False

        with np.load(self.bmode_split_dir / "meta.npz", allow_pickle=False) as meta:
            self.gt_shape = tuple(int(x) for x in meta["label_bmode_shape"])
            gt_paths = meta["path"].astype(str).tolist()
            gt_categories = meta["category"].astype(str).tolist()
            gt_ref = float(meta["reference_value"])
        if gt_paths != self.paths:
            raise ValueError(f"path 对齐失败: {self.rf_split_dir} vs {self.bmode_split_dir}")
        if gt_categories != self.categories:
            raise ValueError(f"category 对齐失败: {self.rf_split_dir} vs {self.bmode_split_dir}")
        if abs(gt_ref - np.float32(self.ref)) > 1e-2:
            raise ValueError(f"B-mode REF 不一致: meta={gt_ref} config={self.ref}")

        self.n_samples = self.input_shape[0]
        self.input = np.memmap(self.rf_split_dir / "input.dat", dtype=np.float16, mode="r", shape=self.input_shape)
        self.baseline = np.memmap(self.rf_split_dir / "baseline.dat", dtype=np.float16, mode="r", shape=self.baseline_shape)
        self.scale = np.memmap(self.rf_split_dir / "scale.dat", dtype=np.float32, mode="r", shape=(self.n_samples,))
        self.gt = np.memmap(self.bmode_split_dir / "label_bmode.dat", dtype=np.float16, mode="r", shape=self.gt_shape)

    def __len__(self) -> int:
        return self.n_samples

    def _restore(self, mmap: np.memmap, idx: int, scale: float) -> np.ndarray:
        arr = np.array(mmap[idx], dtype=np.float32, copy=True)
        if self.restore_input_scale and self.normalized:
            arr *= np.float32(scale)
        return arr

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scale = float(self.scale[idx])
        baseline_complex = self._restore(self.baseline, idx, scale)
        return {
            "input": torch.from_numpy(self._restore(self.input, idx, scale)),
            "gt_bmode": torch.from_numpy(np.array(self.gt[idx], dtype=np.float32, copy=True)),
            "baseline_bmode": torch.from_numpy(bmode_from_complex_np(baseline_complex, self.ref, self.eps, self.db_clip)),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "path": self.paths[idx],
            "category": self.categories[idx],
            "z_idx": torch.from_numpy(np.array(self.z_idx[idx], dtype=np.int32, copy=True)),
            "x_idx": torch.from_numpy(np.array(self.x_idx[idx], dtype=np.int32, copy=True)),
            "y_idx": torch.from_numpy(np.array(self.y_idx[idx], dtype=np.int32, copy=True)),
        }


class TinyBModeRFNet(nn.Module):
    """
    TinyResidualRFNet 的 3D Conv/BN 层级，改为单通道 B-mode 直接输出。
    baseline 不进入网络；末端 sigmoid 将预测约束在 [0,1]。
    """

    def __init__(self, in_channels: int = 1536, hidden: int = 64, use_batch_norm: bool = True):
        super().__init__()
        norm = nn.BatchNorm3d if use_batch_norm else nn.InstanceNorm3d
        norm_kwargs = {"affine": True, "track_running_stats": True} if use_batch_norm else {"affine": True}
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, padding=0),
            norm(hidden, **norm_kwargs),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            norm(hidden, **norm_kwargs),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden, 32, kernel_size=3, padding=1),
            norm(32, **norm_kwargs),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 1, kernel_size=1, padding=0),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def to_device_float(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=True).float()


def make_datasets(config: dict[str, Any]) -> dict[str, BModeCachedDataset]:
    rf_root = Path(config["data"]["rf_cache_dir"])
    gt_root = Path(config["data"]["bmode_gt_dir"])
    return {
        split: BModeCachedDataset(rf_root / config["data"][f"{split}_split"], gt_root / config["data"][f"{split}_split"], config)
        for split in ["train", "val"]
    }


def make_train_loader(dataset: BModeCachedDataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    add_import_paths()
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


def pick_fixed_val_indices(val_dataset: BModeCachedDataset, config: dict[str, Any]) -> dict[str, list[int]]:
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


def make_fixed_loader(val_dataset: BModeCachedDataset, fixed_indices: dict[str, list[int]]) -> DataLoader:
    indices = [idx for category in fixed_indices.values() for idx in category]
    return DataLoader(Subset(val_dataset, indices), batch_size=3, shuffle=False, num_workers=0)


def make_loss_fn(config: dict[str, Any], device: torch.device) -> SSIM:
    ssim_cfg = config["loss"]["ssim"]
    return SSIM(
        data_range=float(ssim_cfg["data_range"]),
        size_average=True,
        win_size=int(ssim_cfg["win_size"]),
        win_sigma=float(ssim_cfg["win_sigma"]),
        channel=int(ssim_cfg["channel"]),
        spatial_dims=int(ssim_cfg["spatial_dims"]),
        nonnegative_ssim=bool(ssim_cfg["nonnegative_ssim"]),
    ).to(device)


def make_model_state(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = TinyBModeRFNet(
        in_channels=int(config["generator"]["input_channels"]),
        hidden=int(config["generator"]["hidden_channels"]),
        use_batch_norm=bool(config["generator"]["use_bn"]),
    ).to(device)
    betas = (float(config["training"]["beta1"]), float(config["training"]["beta2"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]), betas=betas)
    return {"device": device, "G": model, "opt": optimizer, "ssim_fn": make_loss_fn(config, device)}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_losses(pred: torch.Tensor, gt: torch.Tensor, ssim_fn: SSIM, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    ssim_value = ssim_fn(pred, gt)
    l1_value = F.l1_loss(pred, gt)
    loss = float(config["loss"]["lambda_ssim"]) * (1.0 - ssim_value) + float(config["loss"]["lambda_l1"]) * l1_value
    return {"loss": loss, "ssim": ssim_value, "l1": l1_value}


def batch_stats(batch: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in ["input", "gt_bmode", "baseline_bmode"]:
        tensor = batch[key].float()
        result[key] = {
            "shape": list(tensor.shape),
            "dtype": str(batch[key].dtype),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "mean": float(tensor.mean().item()),
            "has_nan": bool(torch.isnan(tensor).any().item()),
            "has_inf": bool(torch.isinf(tensor).any().item()),
        }
    return result


def cell1_self_check(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(config)
    ensure_dirs()
    assert str(config["data"]["rf_cache_dir"]) == str(config["safety"]["expected_rf_cache_dir"])
    assert str(config["data"]["bmode_gt_dir"]) == str(config["safety"]["expected_bmode_gt_dir"])
    assert str(config["safety"]["forbidden_cache_token"]) not in str(config["data"]["rf_cache_dir"])
    assert bool(config["training"]["use_amp"]) is bool(config["safety"]["expected_use_amp"])
    assert int(config["training"]["num_workers"]) == int(config["safety"]["expected_num_workers"])
    assert float(config["loss"]["lambda_ssim"]) == 0.84
    assert float(config["loss"]["lambda_l1"]) == 0.16
    assert config["loss"]["ssim"]["library"] == "pytorch-msssim"
    assert int(config["loss"]["ssim"]["spatial_dims"]) == 3
    assert int(config["loss"]["ssim"]["win_size"]) in (5, 7)
    available_gb = read_available_memory_gb()
    assert available_gb > float(config["diagnostics"]["min_available_memory_gb"])

    datasets = make_datasets(config)
    loader = make_train_loader(datasets["train"], config, shuffle=False)
    batch = next(iter(loader))
    categories = sorted(str(x) for x in batch["category"])
    assert categories == sorted(str(x) for x in config["data"]["categories"])
    stats = batch_stats(batch)
    for field in ["input", "gt_bmode", "baseline_bmode"]:
        assert not stats[field]["has_nan"], f"{field} 含 NaN"
        assert not stats[field]["has_inf"], f"{field} 含 Inf"
    assert stats["gt_bmode"]["shape"][1:] == [1, 64, 32, 32]
    assert 0.0 <= stats["gt_bmode"]["min"] <= stats["gt_bmode"]["max"] <= 1.0
    assert 0.0 <= stats["baseline_bmode"]["min"] <= stats["baseline_bmode"]["max"] <= 1.0

    state = make_model_state(config)
    state["G"].eval()
    with torch.no_grad():
        x = to_device_float(batch["input"], state["device"])
        gt = to_device_float(batch["gt_bmode"], state["device"])
        pred = state["G"](x)
        losses = compute_losses(pred, gt, state["ssim_fn"], config)
    pred_stats = {
        "shape": list(pred.shape),
        "min": float(pred.min().item()),
        "max": float(pred.max().item()),
        "mean": float(pred.mean().item()),
        "has_nan": bool(torch.isnan(pred).any().item()),
        "has_inf": bool(torch.isinf(pred).any().item()),
    }
    report = {
        "timestamp": timestamp(),
        "rf_cache_dir": config["data"]["rf_cache_dir"],
        "bmode_gt_dir": config["data"]["bmode_gt_dir"],
        "available_memory_gb": available_gb,
        "use_amp": bool(config["training"]["use_amp"]),
        "num_workers": int(config["training"]["num_workers"]),
        "batch_categories": list(batch["category"]),
        "batch_stats": stats,
        "pred_stats": pred_stats,
        "loss_probe": {
            "loss": float(losses["loss"].item()),
            "ssim": float(losses["ssim"].item()),
            "l1": float(losses["l1"].item()),
        },
        "ssim_library": "pytorch-msssim",
        "ssim_parameters": config["loss"]["ssim"],
        "status": "self_check_passed",
    }
    assert not pred_stats["has_nan"] and not pred_stats["has_inf"]
    assert 0.0 <= pred_stats["min"] <= pred_stats["max"] <= 1.0
    assert np.isfinite(report["loss_probe"]["loss"])
    write_json(LOG_DIR / "cell1_自检结果.json", report)
    return report


def cell2_config(config: dict[str, Any]) -> dict[str, Any]:
    report = {
        "run_name": config["run_name"],
        "rf_cache_dir": config["data"]["rf_cache_dir"],
        "bmode_gt_dir": config["data"]["bmode_gt_dir"],
        "epochs": int(config["training"]["epochs"]),
        "loss": config["loss"],
        "sampler": config["sampler"],
        "generator": config["generator"],
    }
    write_json(LOG_DIR / "cell2_配置摘要.json", report)
    return report


def cell3_data(config: dict[str, Any]) -> dict[str, Any]:
    datasets = make_datasets(config)
    train_loader = make_train_loader(datasets["train"], config, shuffle=True)
    fixed_indices = pick_fixed_val_indices(datasets["val"], config)
    fixed_loader = make_fixed_loader(datasets["val"], fixed_indices)
    write_json(LOG_DIR / "固定val探针索引.json", fixed_indices)
    return {
        "datasets": datasets,
        "train_loader": train_loader,
        "fixed_loader": fixed_loader,
        "fixed_indices": fixed_indices,
        "summary": {"train_samples": len(datasets["train"]), "val_samples": len(datasets["val"])},
    }


def cell4_model(config: dict[str, Any]) -> dict[str, Any]:
    state = make_model_state(config)
    report = {
        "device": str(state["device"]),
        "G_params": count_params(state["G"]),
        "model_class": type(state["G"]).__name__,
        "output_activation": config["generator"]["output_activation"],
        "baseline_used_by_model": bool(config["generator"]["baseline_used_by_model"]),
        "ssim_library": "pytorch-msssim",
        "ssim_parameters": config["loss"]["ssim"],
    }
    write_json(LOG_DIR / "cell4_模型摘要.json", report)
    state["summary"] = report
    return {"models": state, "summary": report}


def train_one_batch(batch: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    model = state["G"]
    model.train()
    x = to_device_float(batch["input"], state["device"])
    gt = to_device_float(batch["gt_bmode"], state["device"])
    baseline = to_device_float(batch["baseline_bmode"], state["device"])
    state["opt"].zero_grad(set_to_none=True)
    pred = model(x)
    losses = compute_losses(pred, gt, state["ssim_fn"], config)
    losses["loss"].backward()
    state["opt"].step()
    return {
        "loss": float(losses["loss"].detach().cpu().item()),
        "ssim": float(losses["ssim"].detach().cpu().item()),
        "l1": float(losses["l1"].detach().cpu().item()),
        "pred_min": float(pred.detach().min().cpu().item()),
        "pred_max": float(pred.detach().max().cpu().item()),
        "pred_mean": float(pred.detach().mean().cpu().item()),
        "pred_baseline_l1": float(F.l1_loss(pred.detach(), baseline).cpu().item()),
        "baseline_gt_l1": float(F.l1_loss(baseline, gt).cpu().item()),
        "pred_gt_l1": float(F.l1_loss(pred.detach(), gt).cpu().item()),
        "has_nan": float(any(torch.isnan(t.detach()).any().item() for t in [pred, losses["loss"]])),
        "has_inf": float(any(torch.isinf(t.detach()).any().item() for t in [pred, losses["loss"]])),
    }


@torch.no_grad()
def evaluate_fixed(config: dict[str, Any], state: dict[str, Any], fixed_loader: DataLoader) -> dict[str, float]:
    state["G"].eval()
    rows = []
    for batch in fixed_loader:
        x = to_device_float(batch["input"], state["device"])
        gt = to_device_float(batch["gt_bmode"], state["device"])
        baseline = to_device_float(batch["baseline_bmode"], state["device"])
        pred = state["G"](x)
        losses = compute_losses(pred, gt, state["ssim_fn"], config)
        rows.append(
            {
                "val_loss": float(losses["loss"].cpu().item()),
                "val_ssim": float(losses["ssim"].cpu().item()),
                "val_l1": float(losses["l1"].cpu().item()),
                "val_pred_baseline_l1": float(F.l1_loss(pred, baseline).cpu().item()),
                "val_baseline_gt_l1": float(F.l1_loss(baseline, gt).cpu().item()),
            }
        )
    return {key: float(np.mean([x[key] for x in rows])) for key in rows[0]}


def aggregate_epoch(epoch: int, batch_metrics: list[dict[str, float]], val_metrics: dict[str, float], seconds: float, peak_gb: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "batches": len(batch_metrics),
        "seconds": seconds,
        "peak_cuda_mem_gb": peak_gb,
        "available_memory_gb": read_available_memory_gb(),
    }
    for key in [
        "loss",
        "ssim",
        "l1",
        "pred_min",
        "pred_max",
        "pred_mean",
        "pred_baseline_l1",
        "baseline_gt_l1",
        "pred_gt_l1",
        "has_nan",
        "has_inf",
    ]:
        values = [float(x[key]) for x in batch_metrics]
        row[key] = float(np.max(values)) if key in ["pred_max", "has_nan", "has_inf"] else float(np.mean(values))
    row.update(val_metrics)
    return row


@torch.no_grad()
def save_triplet_figures(epoch: int, config: dict[str, Any], state: dict[str, Any], val_dataset: BModeCachedDataset, fixed_indices: dict[str, list[int]]) -> None:
    state["G"].eval()
    y_idx = int(config["diagnostics"]["probe_y_idx"])
    aspect = float(config["diagnostics"]["voxel_spacing_mm"]["z"]) / float(config["diagnostics"]["voxel_spacing_mm"]["x"])
    for category in config["data"]["categories"]:
        sample = val_dataset[fixed_indices[category][0]]
        x = to_device_float(sample["input"].unsqueeze(0), state["device"])
        pred = state["G"](x)[0, 0].detach().cpu().numpy()
        gt = sample["gt_bmode"][0].numpy()
        baseline = sample["baseline_bmode"][0].numpy()
        images = [
            ("baseline-bmode", baseline[:, :, y_idx]),
            ("pred", pred[:, :, y_idx]),
            ("gt-bmode", gt[:, :, y_idx]),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
        for ax, (title, image) in zip(axes, images):
            im = ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, aspect=aspect, origin="lower")
            ax.set_title(title)
            ax.set_axis_off()
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
        fig.suptitle(f"{category} epoch {epoch} y={y_idx}")
        fig.savefig(FIG_DIR / f"{category}_epoch{epoch:03d}_bmode_triplet_y{y_idx}.png", dpi=160)
        plt.close(fig)


def save_loss_curves(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    epochs = [int(x["epoch"]) for x in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [float(x["loss"]) for x in rows], label="train loss")
    axes[0].plot(epochs, [float(x["val_loss"]) for x in rows], label="fixed val loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [float(x["ssim"]) for x in rows], label="train SSIM")
    axes[1].plot(epochs, [float(x["val_ssim"]) for x in rows], label="fixed val SSIM")
    axes[1].plot(epochs, [float(x["l1"]) for x in rows], label="train L1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.savefig(FIG_DIR / "bmode_supervised_loss_curves.png", dpi=160)
    plt.close(fig)


def save_checkpoint(epoch: int, config: dict[str, Any], state: dict[str, Any]) -> None:
    path = CKPT_DIR / f"epoch{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state["G"].state_dict(),
            "optimizer_state_dict": state["opt"].state_dict(),
            "config": config,
        },
        path,
    )
    with (TRAIN_DIR / "checkpoints.txt").open("a", encoding="utf-8") as f:
        f.write(f"epoch {epoch}: {path}\n")


def cell5_train(config: dict[str, Any], data_state: dict[str, Any], model_state: dict[str, Any]) -> dict[str, Any]:
    acquire_lock()
    rows: list[dict[str, Any]] = []
    total_start = time.time()
    epochs = int(config["training"]["epochs"])
    snapshot_epochs = set(int(x) for x in config["training"]["snapshot_epochs"])
    checkpoint_epochs = set(int(x) for x in config["training"]["checkpoint_epochs"])
    rng = random.Random(int(config["sampler"]["seed"]))
    for epoch in range(1, epochs + 1):
        if model_state["device"].type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        batch_metrics = []
        for batch in data_state["train_loader"]:
            metrics = train_one_batch(batch, config, model_state)
            if bool(config["diagnostics"]["stop_on_nonfinite_loss"]) and (metrics["has_nan"] or metrics["has_inf"]):
                raise RuntimeError(f"非有限数值，epoch={epoch}, metrics={metrics}")
            batch_metrics.append(metrics)
        val_metrics = evaluate_fixed(config, model_state, data_state["fixed_loader"])
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if model_state["device"].type == "cuda" else 0.0
        row = aggregate_epoch(epoch, batch_metrics, val_metrics, time.time() - start, peak_gb)
        rows.append(row)
        write_csv(METRICS_DIR / "stability.csv", rows)
        if epoch in snapshot_epochs:
            save_triplet_figures(epoch, config, model_state, data_state["datasets"]["val"], data_state["fixed_indices"])
            save_loss_curves(rows)
        if epoch in checkpoint_epochs:
            save_checkpoint(epoch, config, model_state)
        print(
            f"EPOCH {epoch:03d}/{epochs} loss={row['loss']:.6f} ssim={row['ssim']:.6f} "
            f"l1={row['l1']:.6f} val_loss={row['val_loss']:.6f} seconds={row['seconds']:.1f}",
            flush=True,
        )
        rng.random()
    save_loss_curves(rows)
    summary = {
        "timestamp": timestamp(),
        "status": "completed_50_epochs",
        "epochs": epochs,
        "total_seconds": time.time() - total_start,
        "first_epoch": rows[0],
        "last_epoch": rows[-1],
        "outputs": {
            "stability_csv": str(METRICS_DIR / "stability.csv"),
            "figures_dir": str(FIG_DIR),
            "checkpoints_dir": str(CKPT_DIR),
        },
    }
    write_json(LOG_DIR / "cell5_训练结果.json", summary)
    return summary


def cell6_loss_curves() -> dict[str, Any]:
    path = METRICS_DIR / "stability.csv"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k not in ["epoch"] else int(v) for k, v in row.items()})
    save_loss_curves(rows)
    return {"loss_curve": str(FIG_DIR / "bmode_supervised_loss_curves.png"), "rows": len(rows)}


def cell7_cleanup() -> dict[str, Any]:
    LOCK_PATH.unlink(missing_ok=True)
    return {"lock_removed": not LOCK_PATH.exists(), "timestamp": timestamp()}


def cell8_summary(config: dict[str, Any]) -> dict[str, Any]:
    metrics_path = METRICS_DIR / "stability.csv"
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    summary = {
        "run_name": config["run_name"],
        "timestamp": timestamp(),
        "metrics_rows": len(rows),
        "ssim_library": "pytorch-msssim",
        "ssim_parameters": config["loss"]["ssim"],
        "first_epoch": first,
        "last_epoch": last,
        "quality_conclusion": "none; requires later 3D Slicer review",
    }
    lines = [
        f"# {config['run_name']} summary",
        "",
        "本 run 为 B-mode 单通道纯监督基线，不做质量结论。",
        "",
        f"- metrics rows: {len(rows)}",
        f"- SSIM library: pytorch-msssim, 3D single-scale, win_size={config['loss']['ssim']['win_size']}",
    ]
    if rows:
        lines.extend(
            [
                f"- first loss/ssim/l1: {first['loss']} / {first['ssim']} / {first['l1']}",
                f"- last loss/ssim/l1: {last['loss']} / {last['ssim']} / {last['l1']}",
                f"- first pred_baseline_l1: {first['pred_baseline_l1']}",
                f"- last pred_baseline_l1: {last['pred_baseline_l1']}",
            ]
        )
    (TRAIN_DIR / "phase2a_bmode_supervised_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(LOG_DIR / "cell8_自动摘要.json", summary)
    return summary


def run_all() -> dict[str, Any]:
    config = load_config()
    ensure_dirs()
    set_seed(config)
    c1 = cell1_self_check(config)
    c2 = cell2_config(config)
    data = cell3_data(config)
    model = cell4_model(config)
    c5 = cell5_train(config, data, model["models"])
    c6 = cell6_loss_curves()
    c7 = cell7_cleanup()
    c8 = cell8_summary(config)
    return {"cell1": c1, "cell2": c2, "cell5": c5, "cell6": c6, "cell7": c7, "cell8": c8}


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, ensure_ascii=False))
