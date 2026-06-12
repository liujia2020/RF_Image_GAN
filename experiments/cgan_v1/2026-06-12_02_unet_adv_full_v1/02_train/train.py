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
PROBE_TRAIN_DIR = TRAIN_DIR / "probes" / "train"
PROBE_VAL_DIR = TRAIN_DIR / "probes" / "val"
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
    for path in [LOG_DIR, METRICS_DIR, FIG_DIR, CKPT_DIR, PROBE_TRAIN_DIR, PROBE_VAL_DIR]:
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


class ProbeDatasetView(Dataset):
    """Adapter for probe_utils without changing the training dataset contract."""

    def __init__(self, dataset: BModeCachedDataset):
        self.dataset = dataset
        self.categories = dataset.categories

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.dataset[idx])
        if sample["baseline_bmode"].ndim == 4 and sample["baseline_bmode"].shape[0] == 1:
            sample["baseline_bmode"] = sample["baseline_bmode"][0]
        return sample


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


def make_fixed_loaders(datasets: dict[str, BModeCachedDataset], config: dict[str, Any]) -> dict[str, Any]:
    train_indices = pick_fixed_val_indices(datasets["train"], config)
    val_indices = pick_fixed_val_indices(datasets["val"], config)
    write_json(LOG_DIR / "固定train探针索引.json", train_indices)
    write_json(LOG_DIR / "固定val探针索引.json", val_indices)
    return {
        "train": {"indices": train_indices, "loader": make_fixed_loader(datasets["train"], train_indices)},
        "val": {"indices": val_indices, "loader": make_fixed_loader(datasets["val"], val_indices)},
    }


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
    add_import_paths()
    from rf_cgan_bmode_adv import d_lsgan_loss, g_adv_loss
    from rf_cgan_models import BMode3DPatchDiscriminator, Light3DUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = Light3DUNet().to(device)
    discriminator = BMode3DPatchDiscriminator().to(device)
    betas = (float(config["training"]["beta1"]), float(config["training"]["beta2"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]), betas=betas)
    d_cfg = config["adversarial"]["opt_D"]
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(d_cfg["lr"]),
        betas=(float(d_cfg["beta1"]), float(d_cfg["beta2"])),
    )
    return {
        "device": device,
        "G": model,
        "D": discriminator,
        "opt": optimizer,
        "opt_D": opt_d,
        "ssim_fn": make_loss_fn(config, device),
        "d_lsgan_loss": d_lsgan_loss,
        "g_adv_loss": g_adv_loss,
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batchnorm_stats(model: nn.Module) -> dict[str, Any]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            layers.append(
                {
                    "name": name,
                    "running_mean_mean": float(module.running_mean.detach().mean().cpu().item()),
                    "running_var_mean": float(module.running_var.detach().mean().cpu().item()),
                }
            )
    if not layers:
        return {
            "bn_layers": 0,
            "bn_running_mean_mean": 0.0,
            "bn_running_var_mean": 0.0,
            "bn_layers_detail": [],
        }
    return {
        "bn_layers": len(layers),
        "bn_running_mean_mean": float(np.mean([x["running_mean_mean"] for x in layers])),
        "bn_running_var_mean": float(np.mean([x["running_var_mean"] for x in layers])),
        "bn_layers_detail": layers,
    }


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
    assert bool(config["adversarial"]["enabled"]) is True
    assert config["adversarial"]["discriminator_class"] == "BMode3DPatchDiscriminator"
    assert abs(float(config["adversarial"]["lambda_adv"]) - 0.1) < 1e-12
    assert float(config["adversarial"]["opt_D"]["lr"]) == float(config["training"]["lr"])
    assert float(config["adversarial"]["opt_D"]["beta1"]) == float(config["training"]["beta1"])
    assert float(config["adversarial"]["opt_D"]["beta2"]) == float(config["training"]["beta2"])
    available_gb = read_available_memory_gb()
    assert available_gb > float(config["diagnostics"]["min_available_memory_gb"])

    datasets = make_datasets(config)
    loader = make_train_loader(datasets["train"], config, shuffle=False)
    batch = next(iter(loader))
    category_counts = {str(category): list(batch["category"]).count(category) for category in config["data"]["categories"]}
    assert all(count == int(config["sampler"]["samples_per_category"]) for count in category_counts.values()), category_counts
    stats = batch_stats(batch)
    for field in ["input", "gt_bmode", "baseline_bmode"]:
        assert not stats[field]["has_nan"], f"{field} 含 NaN"
        assert not stats[field]["has_inf"], f"{field} 含 Inf"
    assert stats["gt_bmode"]["shape"][1:] == [1, 64, 32, 32]
    assert 0.0 <= stats["gt_bmode"]["min"] <= stats["gt_bmode"]["max"] <= 1.0
    assert 0.0 <= stats["baseline_bmode"]["min"] <= stats["baseline_bmode"]["max"] <= 1.0

    state = make_model_state(config)
    param_count = count_params(state["G"])
    d_param_count = count_params(state["D"])
    assert 1_900_000 <= param_count <= 2_300_000, f"PARAM COUNT OFF: {param_count:,}"
    assert 1_000_000 <= d_param_count <= 3_000_000, f"D PARAM COUNT OFF: {d_param_count:,}"
    bn_report = batchnorm_stats(state["G"])
    assert bn_report["bn_layers"] > 0, "Light3DUNet should contain BatchNorm3d layers"

    state["G"].train()
    with torch.no_grad():
        random_x = torch.randn(1, 1536, 64, 32, 32, device=state["device"])
        random_y = state["G"](random_x)
    random_pred_stats = {
        "shape": list(random_y.shape),
        "min": float(random_y.min().item()),
        "max": float(random_y.max().item()),
        "mean": float(random_y.mean().item()),
        "std": float(random_y.std().item()),
    }
    assert random_y.shape == (1, 1, 64, 32, 32), f"BAD SHAPE: {tuple(random_y.shape)}"
    assert 0.0 <= random_pred_stats["min"] <= random_pred_stats["max"] <= 1.0
    assert random_pred_stats["std"] > 0.01, f"NEARLY CONSTANT OUTPUT, std={random_pred_stats['std']:.6f}"
    del random_x, random_y
    if state["device"].type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    state["G"].train()
    x6 = torch.randn(6, 1536, 64, 32, 32, device=state["device"])
    y6 = state["G"](x6)
    baseline6 = torch.rand_like(y6)
    gt6 = torch.rand_like(y6)
    d_out6 = state["D"](torch.cat([y6.detach(), baseline6], dim=1))
    assert d_out6.shape[1] == 1, f"BAD D SHAPE: {tuple(d_out6.shape)}"
    assert torch.isfinite(d_out6).all().item(), "D output is not finite"
    d_probe = state["d_lsgan_loss"](state["D"], y6.detach(), gt6, baseline6)
    g_probe = state["g_adv_loss"](state["D"], y6, baseline6)
    adv_probe_report = {
        "d_loss": float(d_probe["d_loss"].detach().cpu().item()),
        "d_real_score_sigmoid": float(d_probe["d_real_score_sigmoid"].detach().cpu().item()),
        "d_fake_score_sigmoid": float(d_probe["d_fake_score_sigmoid"].detach().cpu().item()),
        "g_adv": float(g_probe["g_adv"].detach().cpu().item()),
        "g_fake_score_sigmoid": float(g_probe["g_fake_score_sigmoid"].detach().cpu().item()),
    }
    batch6_loss = y6.mean() + d_probe["d_loss"] + g_probe["g_adv"]
    assert torch.isfinite(batch6_loss).item(), "batch=6 adversarial loss is not finite"
    state["opt"].zero_grad(set_to_none=True)
    state["opt_D"].zero_grad(set_to_none=True)
    batch6_loss.backward()
    batch6_peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if state["device"].type == "cuda" else 0.0
    assert batch6_peak_gb < 6.0, f"VRAM SPIKE: {batch6_peak_gb:.2f} GB"
    del x6, y6, baseline6, gt6, d_out6, d_probe, g_probe, batch6_loss
    state["opt"].zero_grad(set_to_none=True)
    state["opt_D"].zero_grad(set_to_none=True)
    if state["device"].type == "cuda":
        torch.cuda.empty_cache()

    state["G"].train()
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
        "std": float(pred.std().item()),
        "has_nan": bool(torch.isnan(pred).any().item()),
        "has_inf": bool(torch.isinf(pred).any().item()),
    }
    assert pred_stats["std"] > 0.01, f"NEARLY CONSTANT OUTPUT, std={pred_stats['std']:.6f}"
    report = {
        "timestamp": timestamp(),
        "rf_cache_dir": config["data"]["rf_cache_dir"],
        "bmode_gt_dir": config["data"]["bmode_gt_dir"],
        "available_memory_gb": available_gb,
        "use_amp": bool(config["training"]["use_amp"]),
        "num_workers": int(config["training"]["num_workers"]),
        "batch_categories": list(batch["category"]),
        "batch_category_counts": category_counts,
        "batch_stats": stats,
        "model_class": type(state["G"]).__name__,
        "discriminator_class": type(state["D"]).__name__,
        "param_count": param_count,
        "d_param_count": d_param_count,
        "bn_report": bn_report,
        "random_pred_stats": random_pred_stats,
        "batch6_peak_cuda_mem_gb": batch6_peak_gb,
        "batch6_adversarial_probe": adv_probe_report,
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
    add_import_paths()
    from probe_utils import pick_first_per_category

    datasets = make_datasets(config)
    train_loader = make_train_loader(datasets["train"], config, shuffle=True)
    fixed = make_fixed_loaders(datasets, config)
    probe_indices = {
        "train": pick_first_per_category(datasets["train"], list(config["data"]["categories"])),
        "val": pick_first_per_category(datasets["val"], list(config["data"]["categories"])),
    }
    write_json(LOG_DIR / "probe_indices.json", probe_indices)
    return {
        "datasets": datasets,
        "probe_datasets": {"train": ProbeDatasetView(datasets["train"]), "val": ProbeDatasetView(datasets["val"])},
        "train_loader": train_loader,
        "fixed": fixed,
        "probe_indices": probe_indices,
        "summary": {"train_samples": len(datasets["train"]), "val_samples": len(datasets["val"])},
    }


def cell4_model(config: dict[str, Any]) -> dict[str, Any]:
    state = make_model_state(config)
    bn_report = batchnorm_stats(state["G"])
    report = {
        "device": str(state["device"]),
        "G_params": count_params(state["G"]),
        "D_params": count_params(state["D"]),
        "model_class": type(state["G"]).__name__,
        "discriminator_class": type(state["D"]).__name__,
        "output_activation": config["generator"]["output_activation"],
        "baseline_used_by_model": bool(config["generator"]["baseline_used_by_model"]),
        "bn_layers": bn_report["bn_layers"],
        "bn_running_mean_mean": bn_report["bn_running_mean_mean"],
        "bn_running_var_mean": bn_report["bn_running_var_mean"],
        "ssim_library": "pytorch-msssim",
        "ssim_parameters": config["loss"]["ssim"],
    }
    assert report["model_class"] == "Light3DUNet", report["model_class"]
    assert 1_900_000 <= int(report["G_params"]) <= 2_300_000, report["G_params"]
    assert report["discriminator_class"] == "BMode3DPatchDiscriminator", report["discriminator_class"]
    assert 1_000_000 <= int(report["D_params"]) <= 3_000_000, report["D_params"]
    write_json(LOG_DIR / "cell4_模型摘要.json", report)
    state["summary"] = report
    return {"models": state, "summary": report}


def train_one_batch(batch: dict[str, Any], config: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    model = state["G"]
    discriminator = state["D"]
    model.train()
    discriminator.train()
    x = to_device_float(batch["input"], state["device"])
    gt = to_device_float(batch["gt_bmode"], state["device"])
    baseline = to_device_float(batch["baseline_bmode"], state["device"])
    pred = model(x)

    state["opt_D"].zero_grad(set_to_none=True)
    d_losses = state["d_lsgan_loss"](discriminator, pred.detach(), gt, baseline)
    d_losses["d_loss"].backward()
    state["opt_D"].step()

    for param in discriminator.parameters():
        param.requires_grad_(False)
    try:
        state["opt"].zero_grad(set_to_none=True)
        sup_losses = compute_losses(pred, gt, state["ssim_fn"], config)
        adv_losses = state["g_adv_loss"](discriminator, pred, baseline)
        lambda_adv = float(config["adversarial"]["lambda_adv"])
        g_total = sup_losses["loss"] + lambda_adv * adv_losses["g_adv"]
        g_total.backward()
        state["opt"].step()
    finally:
        for param in discriminator.parameters():
            param.requires_grad_(True)
    return {
        "loss": float(g_total.detach().cpu().item()),
        "sup_loss": float(sup_losses["loss"].detach().cpu().item()),
        "ssim": float(sup_losses["ssim"].detach().cpu().item()),
        "l1": float(sup_losses["l1"].detach().cpu().item()),
        "d_loss": float(d_losses["d_loss"].detach().cpu().item()),
        "d_real_loss": float(d_losses["d_real_loss"].detach().cpu().item()),
        "d_fake_loss": float(d_losses["d_fake_loss"].detach().cpu().item()),
        "d_real_score_raw": float(d_losses["d_real_score_raw"].detach().cpu().item()),
        "d_fake_score_raw": float(d_losses["d_fake_score_raw"].detach().cpu().item()),
        "d_real_score_sigmoid": float(d_losses["d_real_score_sigmoid"].detach().cpu().item()),
        "d_fake_score_sigmoid": float(d_losses["d_fake_score_sigmoid"].detach().cpu().item()),
        "g_adv": float(adv_losses["g_adv"].detach().cpu().item()),
        "g_fake_score_raw": float(adv_losses["g_fake_score_raw"].detach().cpu().item()),
        "g_fake_score_sigmoid": float(adv_losses["g_fake_score_sigmoid"].detach().cpu().item()),
        "pred_min": float(pred.detach().min().cpu().item()),
        "pred_max": float(pred.detach().max().cpu().item()),
        "pred_mean": float(pred.detach().mean().cpu().item()),
        "pred_std": float(pred.detach().std().cpu().item()),
        "pred_baseline_l1": float(F.l1_loss(pred.detach(), baseline).cpu().item()),
        "baseline_gt_l1": float(F.l1_loss(baseline, gt).cpu().item()),
        "pred_gt_l1": float(F.l1_loss(pred.detach(), gt).cpu().item()),
        "has_nan": float(
            any(torch.isnan(t.detach()).any().item() for t in [pred, g_total, d_losses["d_loss"], adv_losses["g_adv"]])
        ),
        "has_inf": float(
            any(torch.isinf(t.detach()).any().item() for t in [pred, g_total, d_losses["d_loss"], adv_losses["g_adv"]])
        ),
    }


@torch.no_grad()
def evaluate_fixed(
    config: dict[str, Any],
    state: dict[str, Any],
    fixed_loader: DataLoader,
    prefix: str,
    adversarial_health: bool = False,
) -> dict[str, float | str]:
    was_g_training = state["G"].training
    was_d_training = state["D"].training
    state["G"].eval()
    state["D"].eval()
    rows = []
    adv_rows = []
    category_rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"loss": [], "ssim": [], "l1": [], "pred_std": [], "gt_std": []}
    )
    for batch in fixed_loader:
        x = to_device_float(batch["input"], state["device"])
        gt = to_device_float(batch["gt_bmode"], state["device"])
        baseline = to_device_float(batch["baseline_bmode"], state["device"])
        pred = state["G"](x)
        losses = compute_losses(pred, gt, state["ssim_fn"], config)
        rows.append(
            {
                f"{prefix}_loss": float(losses["loss"].cpu().item()),
                f"{prefix}_ssim": float(losses["ssim"].cpu().item()),
                f"{prefix}_l1": float(losses["l1"].cpu().item()),
                f"{prefix}_pred_mean": float(pred.mean().cpu().item()),
                f"{prefix}_pred_std": float(pred.std().cpu().item()),
                f"{prefix}_pred_baseline_l1": float(F.l1_loss(pred, baseline).cpu().item()),
                f"{prefix}_baseline_gt_l1": float(F.l1_loss(baseline, gt).cpu().item()),
            }
        )
        if adversarial_health:
            d_losses = state["d_lsgan_loss"](state["D"], pred.detach(), gt, baseline)
            g_losses = state["g_adv_loss"](state["D"], pred, baseline)
            adv_rows.append(
                {
                    "d_loss_evalfixed": float(d_losses["d_loss"].cpu().item()),
                    "d_real_loss_evalfixed": float(d_losses["d_real_loss"].cpu().item()),
                    "d_fake_loss_evalfixed": float(d_losses["d_fake_loss"].cpu().item()),
                    "d_real_score_raw_evalfixed": float(d_losses["d_real_score_raw"].cpu().item()),
                    "d_fake_score_raw_evalfixed": float(d_losses["d_fake_score_raw"].cpu().item()),
                    "d_real_score_sigmoid_evalfixed": float(d_losses["d_real_score_sigmoid"].cpu().item()),
                    "d_fake_score_sigmoid_evalfixed": float(d_losses["d_fake_score_sigmoid"].cpu().item()),
                    "g_adv_evalfixed": float(g_losses["g_adv"].cpu().item()),
                    "g_fake_score_raw_evalfixed": float(g_losses["g_fake_score_raw"].cpu().item()),
                    "g_fake_score_sigmoid_evalfixed": float(g_losses["g_fake_score_sigmoid"].cpu().item()),
                }
            )
        for i, category in enumerate(batch["category"]):
            cat = str(category)
            sample_pred = pred[i : i + 1]
            sample_gt = gt[i : i + 1]
            sample_losses = compute_losses(sample_pred, sample_gt, state["ssim_fn"], config)
            category_rows[cat]["loss"].append(float(sample_losses["loss"].cpu().item()))
            category_rows[cat]["ssim"].append(float(sample_losses["ssim"].cpu().item()))
            category_rows[cat]["l1"].append(float(sample_losses["l1"].cpu().item()))
            category_rows[cat]["pred_std"].append(float(pred[i].std().cpu().item()))
            category_rows[cat]["gt_std"].append(float(gt[i].std().cpu().item()))
    result = {key: float(np.mean([x[key] for x in rows])) for key in rows[0]}
    if adversarial_health and adv_rows:
        result.update({key: float(np.mean([x[key] for x in adv_rows])) for key in adv_rows[0]})
    for category in config["data"]["categories"]:
        values = category_rows[str(category)]
        for metric in ["loss", "ssim", "l1", "pred_std", "gt_std"]:
            result[f"{prefix}_{category}_{metric}"] = float(np.mean(values[metric])) if values[metric] else 0.0
    if prefix == "val":
        result["gcnr_carotid"] = "MISSING_ROI"
        result["gcnr_muscle"] = "MISSING_ROI"
    if was_g_training:
        state["G"].train()
    if was_d_training:
        state["D"].train()
    return result


def aggregate_epoch(
    epoch: int,
    batch_metrics: list[dict[str, float]],
    val_metrics: dict[str, float],
    bn_metrics: dict[str, Any],
    seconds: float,
    peak_gb: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "batches": len(batch_metrics),
        "seconds": seconds,
        "peak_cuda_mem_gb": peak_gb,
        "available_memory_gb": read_available_memory_gb(),
    }
    for key in [
        "loss",
        "sup_loss",
        "ssim",
        "l1",
        "d_loss",
        "d_real_loss",
        "d_fake_loss",
        "d_real_score_raw",
        "d_fake_score_raw",
        "d_real_score_sigmoid",
        "d_fake_score_sigmoid",
        "g_adv",
        "g_fake_score_raw",
        "g_fake_score_sigmoid",
        "pred_min",
        "pred_max",
        "pred_mean",
        "pred_std",
        "pred_baseline_l1",
        "baseline_gt_l1",
        "pred_gt_l1",
        "has_nan",
        "has_inf",
    ]:
        values = [float(x[key]) for x in batch_metrics]
        row[key] = float(np.max(values)) if key in ["pred_max", "has_nan", "has_inf"] else float(np.mean(values))
    for key in [
        "d_loss",
        "d_real_loss",
        "d_fake_loss",
        "d_real_score_raw",
        "d_fake_score_raw",
        "d_real_score_sigmoid",
        "d_fake_score_sigmoid",
        "g_adv",
        "g_fake_score_raw",
        "g_fake_score_sigmoid",
    ]:
        row[f"{key}_train"] = row[key]
    row.update(val_metrics)
    row["bn_layers"] = int(bn_metrics["bn_layers"])
    row["bn_running_mean_mean"] = float(bn_metrics["bn_running_mean_mean"])
    row["bn_running_var_mean"] = float(bn_metrics["bn_running_var_mean"])
    return row


def check_redlines(row: dict[str, Any], config: dict[str, Any]) -> None:
    if float(row["has_nan"]) or float(row["has_inf"]):
        raise RuntimeError(f"REDLINE: NaN/Inf detected at epoch {row['epoch']}")
    if float(row["peak_cuda_mem_gb"]) >= 6.0:
        raise RuntimeError(f"REDLINE: VRAM >= 6GB at epoch {row['epoch']}: {row['peak_cuda_mem_gb']:.2f}GB")

    adv_cfg = config["adversarial"]["redlines"]
    epoch = int(row["epoch"])
    if epoch >= int(adv_cfg["min_epoch_for_flat_d"]):
        eps = float(adv_cfg["flat_sigmoid_eps"])
        real_key = "d_real_score_sigmoid_evalfixed" if "d_real_score_sigmoid_evalfixed" in row else "d_real_score_sigmoid"
        fake_key = "d_fake_score_sigmoid_evalfixed" if "d_fake_score_sigmoid_evalfixed" in row else "d_fake_score_sigmoid"
        real_flat = abs(float(row[real_key]) - 0.5) <= eps
        fake_flat = abs(float(row[fake_key]) - 0.5) <= eps
        if real_flat and fake_flat:
            raise RuntimeError(
                f"REDLINE: evalfixed D_real/D_fake sigmoid both near 0.5 at epoch {epoch}: "
                f"real={float(row[real_key]):.4f}, fake={float(row[fake_key]):.4f}"
            )

    if float(row["d_loss_train"]) <= float(adv_cfg["d_loss_collapse_threshold"]) and float(row["g_adv_train"]) >= float(
        adv_cfg["g_adv_explosion_threshold"]
    ):
        raise RuntimeError(
            f"REDLINE: train D_loss collapsed and G_adv exploded at epoch {epoch}: "
            f"d_loss={float(row['d_loss_train']):.6g}, g_adv={float(row['g_adv_train']):.6g}"
        )


@torch.no_grad()
def save_triplet_figures(
    epoch: int,
    split: str,
    config: dict[str, Any],
    state: dict[str, Any],
    dataset: BModeCachedDataset,
    fixed_indices: dict[str, list[int]],
) -> None:
    state["G"].eval()
    y_idx = int(config["diagnostics"]["probe_y_idx"])
    aspect = float(config["diagnostics"]["voxel_spacing_mm"]["z"]) / float(config["diagnostics"]["voxel_spacing_mm"]["x"])
    for category in config["data"]["categories"]:
        sample = dataset[fixed_indices[category][0]]
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
        fig.suptitle(f"{split} {category} epoch {epoch} y={y_idx}")
        fig.savefig(FIG_DIR / f"{split}_{category}_epoch{epoch:03d}_bmode_triplet_y{y_idx}.png", dpi=160)
        plt.close(fig)


def save_loss_curves(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    epochs = [int(x["epoch"]) for x in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [float(x["loss"]) for x in rows], label="train loss")
    axes[0].plot(epochs, [float(x["val_loss"]) for x in rows], label="fixed val loss")
    if "trainfixed_loss" in rows[0]:
        axes[0].plot(epochs, [float(x["trainfixed_loss"]) for x in rows], label="fixed train loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [float(x["ssim"]) for x in rows], label="train SSIM")
    axes[1].plot(epochs, [float(x["val_ssim"]) for x in rows], label="fixed val SSIM")
    if "trainfixed_ssim" in rows[0]:
        axes[1].plot(epochs, [float(x["trainfixed_ssim"]) for x in rows], label="fixed train SSIM")
    axes[1].plot(epochs, [float(x["l1"]) for x in rows], label="train L1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.savefig(FIG_DIR / "light3dunet_bmode_supervised_loss_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [float(x["d_real_score_sigmoid_train"]) for x in rows], label="D real train")
    axes[0].plot(epochs, [float(x["d_fake_score_sigmoid_train"]) for x in rows], label="D fake train")
    if "d_real_score_sigmoid_evalfixed" in rows[0]:
        axes[0].plot(epochs, [float(x["d_real_score_sigmoid_evalfixed"]) for x in rows], label="D real evalfixed")
        axes[0].plot(epochs, [float(x["d_fake_score_sigmoid_evalfixed"]) for x in rows], label="D fake evalfixed")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    axes[1].plot(epochs, [float(x["d_loss_train"]) for x in rows], label="D loss train")
    axes[1].plot(epochs, [float(x["g_adv_train"]) for x in rows], label="G adv train")
    if "d_loss_evalfixed" in rows[0]:
        axes[1].plot(epochs, [float(x["d_loss_evalfixed"]) for x in rows], label="D loss evalfixed")
        axes[1].plot(epochs, [float(x["g_adv_evalfixed"]) for x in rows], label="G adv evalfixed")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.savefig(FIG_DIR / "adversarial_health_curves.png", dpi=160)
    plt.close(fig)

    categories = ["carotid", "muscle", "phantom"]
    metrics = ["loss", "ssim", "l1", "pred_std"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        for category in categories:
            train_key = f"trainfixed_{category}_{metric}"
            val_key = f"val_{category}_{metric}"
            if train_key in rows[0]:
                ax.plot(epochs, [float(x[train_key]) for x in rows], label=f"train {category}")
            if val_key in rows[0]:
                ax.plot(epochs, [float(x[val_key]) for x in rows], linestyle="--", label=f"val {category}")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(FIG_DIR / "category_train_val_curves.png", dpi=160)
    plt.close(fig)


def save_checkpoint(epoch: int, config: dict[str, Any], state: dict[str, Any]) -> None:
    path = CKPT_DIR / f"epoch{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state["G"].state_dict(),
            "discriminator_state_dict": state["D"].state_dict(),
            "optimizer_state_dict": state["opt"].state_dict(),
            "optimizer_D_state_dict": state["opt_D"].state_dict(),
            "config": config,
        },
        path,
    )
    with (TRAIN_DIR / "checkpoints.txt").open("a", encoding="utf-8") as f:
        f.write(f"epoch {epoch}: {path}\n")


@torch.no_grad()
def save_snapshot_diagnostics(epoch: int, config: dict[str, Any], state: dict[str, Any], data_state: dict[str, Any]) -> None:
    y_idx = int(config["diagnostics"]["probe_y_idx"])
    aspect = float(config["diagnostics"]["voxel_spacing_mm"]["z"]) / float(config["diagnostics"]["voxel_spacing_mm"]["x"])
    state["G"].eval()
    val_dataset = data_state["datasets"]["val"]
    val_indices = data_state["fixed"]["val"]["indices"]

    phantom_idx = val_indices["phantom"][0]
    phantom_sample = val_dataset[phantom_idx]
    phantom_x = to_device_float(phantom_sample["input"].unsqueeze(0), state["device"])
    phantom_pred = state["G"](phantom_x)[0, 0].detach().cpu().numpy()
    phantom_gt = phantom_sample["gt_bmode"][0].numpy()
    snr_row = {
        "epoch": epoch,
        "space": "B-mode log-compressed [0,1]; compare pred vs gt, not Rayleigh linear-envelope SNR",
        "category": "phantom",
        "pred_mean": float(np.mean(phantom_pred)),
        "pred_std": float(np.std(phantom_pred)),
        "pred_mean_over_std": float(np.mean(phantom_pred) / (np.std(phantom_pred) + 1e-12)),
        "gt_mean": float(np.mean(phantom_gt)),
        "gt_std": float(np.std(phantom_gt)),
        "gt_mean_over_std": float(np.mean(phantom_gt) / (np.std(phantom_gt) + 1e-12)),
    }
    snr_path = METRICS_DIR / "phantom_bmode_snr.csv"
    old_rows = []
    if snr_path.exists():
        with snr_path.open("r", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
    old_rows = [r for r in old_rows if int(r["epoch"]) != epoch]
    old_rows.append(snr_row)
    write_csv(snr_path, old_rows)

    if epoch not in {1, int(config["training"]["epochs"])}:
        return

    fft_records = []
    for category in ["phantom", "muscle"]:
        sample = val_dataset[val_indices[category][0]]
        x = to_device_float(sample["input"].unsqueeze(0), state["device"])
        pred = state["G"](x)[0, 0].detach().cpu().numpy()
        image = pred[:, :, y_idx]
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
        center = (spectrum.shape[0] // 2, spectrum.shape[1] // 2)
        mask = np.ones_like(spectrum, dtype=bool)
        z0, x0 = center
        mask[max(0, z0 - 2) : z0 + 3, max(0, x0 - 2) : x0 + 3] = False
        non_dc = spectrum[mask]
        peak = float(non_dc.max())
        median = float(np.median(non_dc))
        ratio = float(peak / (median + 1e-12))
        fft_records.append(
            {
                "epoch": epoch,
                "category": category,
                "non_dc_peak_over_median": ratio,
                "note": "Review spectrum image for discrete non-DC periodic peaks; numeric ratio is only a screening aid.",
            }
        )
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
        im = ax.imshow(spectrum, cmap="magma", aspect=aspect, origin="lower")
        ax.set_title(f"{category} pred FFT epoch {epoch:03d} y={y_idx}")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(FIG_DIR / f"fft_{category}_pred_epoch{epoch:03d}_y{y_idx}.png", dpi=160)
        plt.close(fig)

    fft_path = METRICS_DIR / "fft_artifact_screen.json"
    existing = []
    if fft_path.exists():
        existing = json.loads(fft_path.read_text(encoding="utf-8"))
        existing = [r for r in existing if int(r["epoch"]) != epoch]
    existing.extend(fft_records)
    fft_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def cell5_train(config: dict[str, Any], data_state: dict[str, Any], model_state: dict[str, Any]) -> dict[str, Any]:
    add_import_paths()
    from probe_utils import generate_epoch_probes

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
        trainfixed_metrics = evaluate_fixed(
            config, model_state, data_state["fixed"]["train"]["loader"], prefix="trainfixed", adversarial_health=False
        )
        val_metrics = evaluate_fixed(
            config, model_state, data_state["fixed"]["val"]["loader"], prefix="val", adversarial_health=True
        )
        val_metrics.update(trainfixed_metrics)
        bn_metrics = batchnorm_stats(model_state["G"])
        write_json(LOG_DIR / f"bn_stats_epoch{epoch:03d}.json", bn_metrics)
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if model_state["device"].type == "cuda" else 0.0
        row = aggregate_epoch(epoch, batch_metrics, val_metrics, bn_metrics, time.time() - start, peak_gb)
        rows.append(row)
        write_csv(METRICS_DIR / "stability.csv", rows)
        check_redlines(row, config)
        generate_epoch_probes(
            epoch=epoch,
            model=model_state["G"],
            train_dataset=data_state["probe_datasets"]["train"],
            val_dataset=data_state["probe_datasets"]["val"],
            train_indices=data_state["probe_indices"]["train"],
            val_indices=data_state["probe_indices"]["val"],
            categories=config["data"]["categories"],
            y_idx=int(config["diagnostics"]["probe_y_idx"]),
            aspect=float(config["diagnostics"]["voxel_spacing_mm"]["z"]) / float(config["diagnostics"]["voxel_spacing_mm"]["x"]),
            train_out_dir=PROBE_TRAIN_DIR,
            val_out_dir=PROBE_VAL_DIR,
        )
        if epoch in snapshot_epochs:
            save_snapshot_diagnostics(epoch, config, model_state, data_state)
            save_loss_curves(rows)
        if epoch in checkpoint_epochs:
            save_checkpoint(epoch, config, model_state)
        print(
            f"EPOCH {epoch:03d}/{epochs} loss={row['loss']:.6f} ssim={row['ssim']:.6f} "
            f"l1={row['l1']:.6f} pred_std={row['pred_std']:.6f} "
            f"d_loss={row['d_loss']:.6f} d_real_sig={row['d_real_score_sigmoid']:.4f} "
            f"d_fake_sig={row['d_fake_score_sigmoid']:.4f} g_adv={row['g_adv']:.6f} "
            f"val_loss={row['val_loss']:.6f} bn_var={row['bn_running_var_mean']:.6f} "
            f"seconds={row['seconds']:.1f}",
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
            parsed = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    save_loss_curves(rows)
    return {
        "loss_curve": str(FIG_DIR / "light3dunet_bmode_supervised_loss_curves.png"),
        "adversarial_health_curve": str(FIG_DIR / "adversarial_health_curves.png"),
        "category_train_val_curve": str(FIG_DIR / "category_train_val_curves.png"),
        "rows": len(rows),
    }


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
        "lambda_adv": float(config["adversarial"]["lambda_adv"]),
        "ssim_library": "pytorch-msssim",
        "ssim_parameters": config["loss"]["ssim"],
        "first_epoch": first,
        "last_epoch": last,
        "quality_conclusion": "none; requires later 3D Slicer review",
    }
    lines = [
        f"# {config['run_name']} summary",
        "",
        "Formal B-mode adversarial 50-epoch run. This summary reports numbers only; quality judgment is reserved for later 3D Slicer review.",
        "",
        f"- metrics rows: {len(rows)}",
        f"- lambda_adv: {config['adversarial']['lambda_adv']}",
        f"- SSIM library: pytorch-msssim, 3D single-scale, win_size={config['loss']['ssim']['win_size']}",
    ]
    if rows:
        lines.extend(
            [
                f"- first total/sup/ssim/l1: {first['loss']} / {first.get('sup_loss', 'NA')} / {first['ssim']} / {first['l1']}",
                f"- last total/sup/ssim/l1: {last['loss']} / {last.get('sup_loss', 'NA')} / {last['ssim']} / {last['l1']}",
                f"- train D real/fake sigmoid first: {first.get('d_real_score_sigmoid_train', first.get('d_real_score_sigmoid', 'NA'))} / {first.get('d_fake_score_sigmoid_train', first.get('d_fake_score_sigmoid', 'NA'))}",
                f"- train D real/fake sigmoid last: {last.get('d_real_score_sigmoid_train', last.get('d_real_score_sigmoid', 'NA'))} / {last.get('d_fake_score_sigmoid_train', last.get('d_fake_score_sigmoid', 'NA'))}",
                f"- evalfixed D real/fake sigmoid last: {last.get('d_real_score_sigmoid_evalfixed', 'NA')} / {last.get('d_fake_score_sigmoid_evalfixed', 'NA')}",
                f"- G_adv train first/last: {first.get('g_adv_train', first.get('g_adv', 'NA'))} / {last.get('g_adv_train', last.get('g_adv', 'NA'))}",
                f"- G_adv evalfixed last: {last.get('g_adv_evalfixed', 'NA')}",
                f"- pred_baseline_l1 first/last: {first['pred_baseline_l1']} / {last['pred_baseline_l1']}",
                f"- pred_std first/last: {first.get('pred_std', 'NA')} / {last.get('pred_std', 'NA')}",
                f"- fixed-val pred_std last: {last.get('val_pred_std', 'NA')}",
                f"- BN running mean/var mean last: {last.get('bn_running_mean_mean', 'NA')} / {last.get('bn_running_var_mean', 'NA')}",
            ]
        )
        for category in config["data"]["categories"]:
            lines.append(
                f"- last val {category} pred_std/gt_std: "
                f"{last.get(f'val_{category}_pred_std', 'NA')} / {last.get(f'val_{category}_gt_std', 'NA')}"
            )
    lines.extend(
        [
            "",
            "Additional diagnostics:",
            "- phantom B-mode mean/std table: metrics/phantom_bmode_snr.csv",
            "- FFT artifact screening records: metrics/fft_artifact_screen.json",
            "- FFT spectrum figures: figures/fft_{phantom|muscle}_pred_epoch{001|050}_y16.png",
            "- category train/val curves: figures/category_train_val_curves.png",
            "- gCNR: carotid/muscle marked MISSING_ROI in stability.csv because ROI is not defined",
            "",
            "Redline status: if this file exists after Run All, no scripted redline exception stopped training.",
            "Quality conclusion: none; requires later 3D Slicer human review.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (TRAIN_DIR / "summary.md").write_text(text, encoding="utf-8")
    write_json(LOG_DIR / "cell8_summary.json", summary)
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
