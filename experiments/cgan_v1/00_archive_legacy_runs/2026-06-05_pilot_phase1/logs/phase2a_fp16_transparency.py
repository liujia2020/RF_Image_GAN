from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn


TASK_NAME = "cGAN Phase 2a fp16-read transparency check"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
RUN_DIR = PROJECT_ROOT / "experiments/cgan_v1/runs/2026-06-05_pilot_phase1"
LOG_DIR = RUN_DIR / "logs"
LEGACY_CODE_DIR = Path("/home/liujia/RF_Image")


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")


def add_paths() -> None:
    for p in [PROJECT_ROOT, LEGACY_CODE_DIR]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def first_by_category(dataset, categories: list[str]) -> list[int]:
    out = []
    for cat in categories:
        for idx, item_cat in enumerate(dataset.categories):
            if item_cat == cat:
                out.append(idx)
                break
        else:
            raise RuntimeError(f"Missing category in dataset: {cat}")
    return out


def stack_batch(dataset, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    samples = [dataset[i] for i in indices]
    return {
        "input": torch.stack([s["input"] for s in samples], dim=0).to(device, non_blocking=True).float(),
        "label": torch.stack([s["label"] for s in samples], dim=0).to(device, non_blocking=True).float(),
        "baseline": torch.stack([s["baseline"] for s in samples], dim=0).to(device, non_blocking=True).float(),
    }


@torch.no_grad()
def compute_terms(config: dict, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    from rf_cgan_losses import AnisotropicGaussianLowpass3D, generator_lsgan_struct_carrier_loss
    from rf_cgan_models import Envelope2DPatchDiscriminator
    from rf_models import TinyResidualRFNet

    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)

    g = TinyResidualRFNet(use_batch_norm=bool(config["generator"]["use_bn"])).to(device).eval()
    d = Envelope2DPatchDiscriminator(ndf=int(config["discriminator"]["ndf"])).to(device).eval()
    lowpass = AnisotropicGaussianLowpass3D((6.0, 3.0, 3.0)).to(device).eval()
    criterion = nn.MSELoss()
    y_idx = 16

    pred = g(batch["input"], batch["baseline"])
    loss, terms = generator_lsgan_struct_carrier_loss(
        d,
        pred,
        batch["label"],
        batch["baseline"],
        lowpass,
        y_idx=y_idx,
        lambda_adv=float(config["loss"]["lambda_adv"]),
        lambda_struct=float(config["loss"]["lambda_struct"]),
        lambda_carrier=float(config["loss"]["lambda_carrier"]),
        criterion=criterion,
    )
    out = {key: float(value.detach().float().cpu()) for key, value in terms.items() if key.startswith("g_")}
    out["loss_return_value"] = float(loss.detach().float().cpu())
    out["y_idx"] = float(y_idx)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print(f"{timestamp()} | TASK: {TASK_NAME}", flush=True)
    add_paths()
    from rf_cached_dataset import RFCachedDataset

    with (RUN_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config["sampler"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(config["data"]["cache_dir"]) / config["data"]["train_split"]
    ds_fp32 = RFCachedDataset(train_dir, return_fp16=False)
    ds_fp16 = RFCachedDataset(train_dir, return_fp16=True)
    categories = list(config["data"]["categories"])
    indices = first_by_category(ds_fp32, categories)
    batch_fp32 = stack_batch(ds_fp32, indices, device)
    batch_fp16 = stack_batch(ds_fp16, indices, device)

    input_max_abs_diff = float((batch_fp32["input"] - batch_fp16["input"]).abs().max().cpu())
    label_max_abs_diff = float((batch_fp32["label"] - batch_fp16["label"]).abs().max().cpu())
    baseline_max_abs_diff = float((batch_fp32["baseline"] - batch_fp16["baseline"]).abs().max().cpu())

    terms_fp32 = compute_terms(config, batch_fp32, device)
    terms_fp16 = compute_terms(config, batch_fp16, device)
    rows = []
    max_term_abs_diff = 0.0
    for key in sorted(terms_fp32):
        diff = abs(terms_fp16[key] - terms_fp32[key])
        max_term_abs_diff = max(max_term_abs_diff, diff)
        rows.append(
            {
                "term": key,
                "fp32_read": terms_fp32[key],
                "fp16_read_then_gpu_float": terms_fp16[key],
                "abs_diff": diff,
            }
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(LOG_DIR / "phase2a_fp16_transparency.csv", rows)
    summary = {
        "indices": indices,
        "categories": [ds_fp32.categories[i] for i in indices],
        "input_max_abs_diff_after_gpu_float": input_max_abs_diff,
        "label_max_abs_diff_after_gpu_float": label_max_abs_diff,
        "baseline_max_abs_diff_after_gpu_float": baseline_max_abs_diff,
        "max_term_abs_diff": max_term_abs_diff,
        "pass_threshold_1e-3": max_term_abs_diff < 1e-3,
    }
    (LOG_DIR / "phase2a_fp16_transparency_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    for row in rows:
        print(
            f"{row['term']}: fp32={row['fp32_read']:.9f} "
            f"fp16_read={row['fp16_read_then_gpu_float']:.9f} diff={row['abs_diff']:.9g}",
            flush=True,
        )
    if max_term_abs_diff >= 1e-3:
        raise SystemExit("FAILED: max G_loss term difference >= 1e-3")
    print("FP16_READ_TRANSPARENCY: PASS", flush=True)


if __name__ == "__main__":
    main()
