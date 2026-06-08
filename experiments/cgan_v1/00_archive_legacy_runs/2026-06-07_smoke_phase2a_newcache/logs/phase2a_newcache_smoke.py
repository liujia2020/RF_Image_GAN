from __future__ import annotations

import importlib
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[5]
RUN_DIR = PROJECT_ROOT / "experiments/cgan_v1/runs/2026-06-07_smoke_phase2a_newcache"
CONFIG_PATH = RUN_DIR / "config.yaml"
LOG_PATH = RUN_DIR / "logs/phase2a_newcache_smoke_result.json"


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")


def add_paths(config: dict) -> None:
    legacy_code_dir = Path(config["data"]["legacy_code_dir"])
    for path in [PROJECT_ROOT, legacy_code_dir]:
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
    sys.path.insert(0, str(legacy_code_dir))
    sys.path.insert(0, str(PROJECT_ROOT))


def tensor_stats(t: torch.Tensor) -> dict[str, object]:
    tf = t.detach().float()
    finite = torch.isfinite(tf)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "max_abs": float(tf.abs().max().item()),
        "mean": float(tf.mean().item()),
        "mean_abs": float(tf.abs().mean().item()),
        "has_nan": bool(torch.isnan(tf).any().item()),
        "has_inf": bool(torch.isinf(tf).any().item()),
        "finite_fraction": float(finite.float().mean().item()),
    }


def nonfinite(values: dict[str, float]) -> bool:
    return any(not math.isfinite(float(v)) for v in values.values())


def make_loader(config: dict):
    from rf_cached_dataset import RFCachedDataset
    from rf_cgan_data import StratifiedCategoryBatchSampler

    cache_root = Path(config["data"]["cache_dir"])
    split_dir = cache_root / str(config["data"]["train_split"])
    dataset = RFCachedDataset(
        split_dir,
        return_fp16=bool(config["training"]["cache_return_fp16"]),
        restore_scale=True,
    )
    sampler = StratifiedCategoryBatchSampler(
        dataset.categories,
        required_categories=config["data"]["categories"],
        samples_per_category=int(config["sampler"]["samples_per_category"]),
        shuffle=bool(config["sampler"]["shuffle"]),
        seed=int(config["sampler"]["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader, split_dir


def import_generator(config: dict):
    class_name = str(config["generator"]["model_class"])
    try:
        module = importlib.import_module("rf_models")
    except Exception as exc:
        return None, f"cannot import rf_models: {exc}"
    if not hasattr(module, class_name):
        return None, f"rf_models has no {class_name}"
    return getattr(module, class_name), ""


def build_models(config: dict, device: torch.device):
    from rf_cgan_models import Envelope2DPatchDiscriminator

    generator_cls, import_error = import_generator(config)
    if generator_cls is None:
        raise RuntimeError(import_error)
    g = generator_cls(use_batch_norm=bool(config["generator"]["use_bn"])).to(device)
    d = Envelope2DPatchDiscriminator(ndf=int(config["discriminator"]["ndf"])).to(device)
    return g, d


def move_batch(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["input"].to(device, non_blocking=True).float()
    label = batch["label"].to(device, non_blocking=True).float()
    baseline = batch["baseline"].to(device, non_blocking=True).float()
    return x, label, baseline


def run_training_smoke(config: dict, loader: DataLoader, first_batch: dict[str, object]) -> dict[str, object]:
    from rf_cgan_losses import (
        AnisotropicGaussianLowpass3D,
        discriminator_lsgan_loss,
        generator_lsgan_struct_carrier_loss,
        random_y_index,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    g, d = build_models(config, device)
    lowpass_cfg = config["loss"]["lowpass"]["sigma_voxels"]
    lowpass = AnisotropicGaussianLowpass3D(
        (
            float(lowpass_cfg["z"]),
            float(lowpass_cfg["x"]),
            float(lowpass_cfg["y"]),
        ),
        truncate=float(config["loss"]["lowpass"]["truncate"]),
    ).to(device)

    opt_g = torch.optim.Adam(
        g.parameters(),
        lr=float(config["training"]["lr_G"]),
        betas=(float(config["training"]["beta1"]), float(config["training"]["beta2"])),
    )
    opt_d = torch.optim.Adam(
        d.parameters(),
        lr=float(config["training"]["lr_D"]),
        betas=(float(config["training"]["beta1"]), float(config["training"]["beta2"])),
    )

    use_amp = bool(config["training"]["use_amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = random.Random(int(config["sampler"]["seed"]))
    rows = []
    start = time.time()
    batch_iter = iter(loader)
    max_iters = int(config["training"]["smoke_iterations"])

    for iteration in range(1, max_iters + 1):
        batch = first_batch if iteration == 1 else next(batch_iter)
        x, label, baseline = move_batch(batch, device)
        y_idx = random_y_index(label, rng=rng)

        opt_d.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_for_d = g(x, baseline)
            d_loss, d_metrics = discriminator_lsgan_loss(d, pred_for_d, label, baseline, y_idx=y_idx)
        scaler.scale(d_loss).backward()
        scaler.step(opt_d)

        opt_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = g(x, baseline)
            g_loss, g_metrics = generator_lsgan_struct_carrier_loss(
                d,
                pred,
                label,
                baseline,
                lowpass,
                y_idx=y_idx,
                lambda_adv=float(config["loss"]["lambda_adv"]),
                lambda_struct=float(config["loss"]["lambda_struct"]),
                lambda_carrier=float(config["loss"]["lambda_carrier"]),
            )
        scaler.scale(g_loss).backward()
        scaler.step(opt_g)
        scaler.update()

        row = {
            "iteration": iteration,
            "y_idx": int(y_idx),
            "D_real": float(d_metrics["d_real_loss"].float().item()),
            "D_fake": float(d_metrics["d_fake_loss"].float().item()),
            "G_adv": float(g_metrics["g_adv_raw"].float().item()),
            "G_struct": float(g_metrics["g_struct_raw"].float().item()),
            "G_carrier": float(g_metrics["g_carrier_raw"].float().item()),
            "G_total": float(g_metrics["g_total"].float().item()),
            "pred_max_abs": float(pred.detach().float().abs().max().item()),
        }
        row["has_nonfinite_loss"] = nonfinite(
            {k: row[k] for k in ["D_real", "D_fake", "G_adv", "G_struct", "G_carrier", "G_total"]}
        )
        rows.append(row)
        if row["has_nonfinite_loss"]:
            break

    peak_mem_gb = None
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "device": str(device),
        "use_amp": use_amp,
        "iterations": rows,
        "elapsed_sec": time.time() - start,
        "peak_cuda_mem_gb": peak_mem_gb,
    }


def main() -> int:
    result: dict[str, object] = {
        "timestamp": timestamp(),
        "status": "started",
        "config_path": str(CONFIG_PATH),
    }
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    add_paths(config)

    cache_root = Path(config["data"]["cache_dir"])
    result["effective_cache_dir"] = str(cache_root)
    result["old_bad_cache_path_present"] = str(cache_root) == "/home/liujia/RF_Image/Data_cache_random64_full1500"

    dataset, loader, split_dir = make_loader(config)
    first_batch = next(iter(loader))
    result["train_split_dir"] = str(split_dir)
    result["train_dataset_len"] = len(dataset)
    result["batch_categories"] = list(first_batch["category"])
    result["batch_stats"] = {
        "input": tensor_stats(first_batch["input"]),
        "label": tensor_stats(first_batch["label"]),
        "baseline": tensor_stats(first_batch["baseline"]),
    }

    generator_cls, generator_error = import_generator(config)
    if generator_cls is None:
        result["status"] = "blocked_missing_generator"
        result["error"] = generator_error
        LOG_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2

    try:
        result["training_smoke"] = run_training_smoke(config, loader, first_batch)
        iterations = result["training_smoke"]["iterations"]
        has_nonfinite = any(row["has_nonfinite_loss"] for row in iterations)
        result["status"] = "failed_nonfinite_loss" if has_nonfinite else "completed"
        LOG_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if has_nonfinite else 0
    except RuntimeError as exc:
        result["status"] = "failed_runtime_error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        if torch.cuda.is_available():
            result["peak_cuda_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        LOG_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
