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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset


TASK_NAME = "cGAN pilot Phase 2a training"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
RUN_DIR = PROJECT_ROOT / "experiments/cgan_v1/runs/2026-06-05_pilot_phase1"
LOG_DIR = RUN_DIR / "logs"
FIG_DIR = RUN_DIR / "figures/phase2a"
CKPT_DIR = RUN_DIR / "checkpoints"
LEGACY_CODE_DIR = Path("/home/liujia/RF_Image")
LOCK_PATH = LOG_DIR / "phase2a_train.lock"


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")


def acquire_run_lock() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
            break
        except FileExistsError:
            try:
                existing_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
            except Exception:
                existing_pid = -1
            if existing_pid > 0 and Path(f"/proc/{existing_pid}").exists():
                raise RuntimeError(f"Another Phase2a training process is running: pid={existing_pid}")
            LOCK_PATH.unlink(missing_ok=True)

    def cleanup_lock() -> None:
        try:
            if LOCK_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
                LOCK_PATH.unlink(missing_ok=True)
        except FileNotFoundError:
            pass

    atexit.register(cleanup_lock)


def add_paths() -> None:
    for p in [PROJECT_ROOT, LEGACY_CODE_DIR]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ks_statistic(a: np.ndarray, b: np.ndarray, bins: int = 128) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    hi = float(max(np.percentile(a, 99.5), np.percentile(b, 99.5), 1e-12))
    hist_a, edges = np.histogram(np.clip(a, 0, hi), bins=bins, range=(0, hi), density=False)
    hist_b, _ = np.histogram(np.clip(b, 0, hi), bins=edges, density=False)
    cdf_a = np.cumsum(hist_a) / max(np.sum(hist_a), 1)
    cdf_b = np.cumsum(hist_b) / max(np.sum(hist_b), 1)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def env_np(volume: torch.Tensor) -> np.ndarray:
    v = volume.detach().float().cpu()
    env = torch.sqrt(v[:, 0].square() + v[:, 1].square() + 1e-12)
    return env.numpy()


def pick_fixed_val_indices(dataset, per_category: int, categories: list[str]) -> dict[str, list[int]]:
    grouped = defaultdict(list)
    for idx, cat in enumerate(dataset.categories):
        if cat in categories:
            grouped[cat].append(idx)
    out = {}
    for cat in categories:
        idxs = grouped[cat]
        if len(idxs) < per_category:
            raise ValueError(f"Not enough val samples for {cat}: {len(idxs)} < {per_category}")
        out[cat] = idxs[:per_category]
    return out


def first_norm_class(model: nn.Module) -> str:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
            return type(m).__name__
    return "MISSING"


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int):
    from rf_cgan_data import StratifiedCategoryBatchSampler

    categories = ["carotid", "muscle", "phantom"]
    if batch_size % len(categories) != 0:
        raise ValueError(f"batch_size must be multiple of {len(categories)}, got {batch_size}")
    sampler = StratifiedCategoryBatchSampler(
        dataset.categories,
        required_categories=categories,
        samples_per_category=batch_size // len(categories),
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())


def to_device_float(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move cached fp16 tensors cheaply, then compute model/loss in fp32/autocast."""
    return tensor.to(device, non_blocking=True).float()


def make_models(config: dict, device: torch.device):
    from rf_cgan_models import Envelope2DPatchDiscriminator
    from rf_models import TinyResidualRFNet

    g = TinyResidualRFNet(use_batch_norm=bool(config["generator"]["use_bn"])).to(device)
    d = Envelope2DPatchDiscriminator(ndf=int(config["discriminator"]["ndf"])).to(device)
    return g, d


def choose_batch_size(config: dict, train_dataset, device: torch.device) -> tuple[int, float]:
    from rf_cgan_losses import (
        AnisotropicGaussianLowpass3D,
        discriminator_lsgan_loss,
        generator_lsgan_struct_carrier_loss,
        random_y_index,
    )

    if device.type != "cuda":
        return int(config["training"]["min_batch_size"]), 0.0

    candidates = [int(x) for x in config["training"]["batch_candidates"]]
    for batch_size in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            g, d = make_models(config, device)
            lowpass = AnisotropicGaussianLowpass3D((6.0, 3.0, 3.0)).to(device)
            loader = make_loader(train_dataset, batch_size=batch_size, shuffle=False, seed=123)
            batch = next(iter(loader))
            x = to_device_float(batch["input"], device)
            label = to_device_float(batch["label"], device)
            baseline = to_device_float(batch["baseline"], device)
            pred = g(x, baseline)
            y_idx = random_y_index(label, rng=random.Random(123))
            d_loss, _ = discriminator_lsgan_loss(d, pred, label, baseline, y_idx=y_idx)
            g_loss, _ = generator_lsgan_struct_carrier_loss(
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
            (d_loss + g_loss).backward()
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            del g, d, lowpass, loader, batch, x, label, baseline, pred, d_loss, g_loss
            torch.cuda.empty_cache()
            return batch_size, peak_gb
        except RuntimeError as exc:
            msg = str(exc).lower()
            del_vars = list(locals().keys())
            if "out of memory" not in msg:
                raise
            print(f"OOM during batch_size={batch_size}; falling back if possible")
            torch.cuda.empty_cache()
    raise RuntimeError(f"No configured batch size fits: {candidates}")


@torch.no_grad()
def evaluate_snapshot(
    epoch: int,
    g: nn.Module,
    d: nn.Module,
    lowpass: nn.Module,
    fixed_loader,
    fixed_indices_by_category: dict[str, list[int]],
    val_dataset,
    config: dict,
    device: torch.device,
    save_png: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    from rf_cgan_losses import discriminator_lsgan_loss, generator_lsgan_struct_carrier_loss

    g.eval()
    d.eval()
    y_idx = 16
    stability_acc = defaultdict(float)
    n_batches = 0
    pred_by_cat = defaultdict(list)
    label_by_cat = defaultdict(list)

    for batch in fixed_loader:
        x = to_device_float(batch["input"], device)
        label = to_device_float(batch["label"], device)
        baseline = to_device_float(batch["baseline"], device)
        cats = list(batch["category"])
        pred = g(x, baseline)
        d_loss, d_terms = discriminator_lsgan_loss(d, pred, label, baseline, y_idx=y_idx)
        g_loss, g_terms = generator_lsgan_struct_carrier_loss(
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
        stability_acc["D_real"] += float(d_terms["d_real_score_sigmoid"].cpu())
        stability_acc["D_fake"] += float(d_terms["d_fake_score_sigmoid"].cpu())
        stability_acc["D_loss"] += float(d_loss.cpu())
        stability_acc["G_adv"] += float(g_terms["g_adv_raw"].cpu())
        stability_acc["G_struct"] += float(g_terms["g_struct_raw"].cpu())
        stability_acc["G_carrier"] += float(g_terms["g_carrier_raw"].cpu())
        stability_acc["G_total"] += float(g_loss.cpu())
        n_batches += 1

        pred_env = env_np(pred)
        label_env = env_np(label)
        for i, cat in enumerate(cats):
            pred_by_cat[cat].append(pred_env[i])
            label_by_cat[cat].append(label_env[i])

    stability = {"epoch": epoch}
    for key, value in stability_acc.items():
        stability[key] = value / max(n_batches, 1)

    speckle_rows = []
    bins = int(config["diagnostics"]["speckle_hist_bins"])
    for cat in ["carotid", "muscle", "phantom"]:
        pred_env = np.concatenate([x.reshape(1, -1) for x in pred_by_cat[cat]], axis=1).ravel()
        label_env = np.concatenate([x.reshape(1, -1) for x in label_by_cat[cat]], axis=1).ravel()
        speckle_rows.append(
            {
                "epoch": epoch,
                "category": cat,
                "pred_SNR": float(np.mean(pred_env) / (np.std(pred_env) + 1e-12)),
                "label_SNR": float(np.mean(label_env) / (np.std(label_env) + 1e-12)),
                "env_hist_KS": ks_statistic(pred_env, label_env, bins=bins),
            }
        )

    if save_png:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for cat, idxs in fixed_indices_by_category.items():
            sample = val_dataset[idxs[0]]
            x = to_device_float(sample["input"].unsqueeze(0), device)
            label = to_device_float(sample["label"].unsqueeze(0), device)
            baseline = to_device_float(sample["baseline"].unsqueeze(0), device)
            pred = g(x, baseline)
            label_env = env_np(label)[0, :, :, y_idx]
            base_env = env_np(baseline)[0, :, :, y_idx]
            pred_env = env_np(pred)[0, :, :, y_idx]
            vmax = float(np.percentile(label_env, 99.5))
            fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
            for ax, arr, title in zip(axes, [label_env, base_env, pred_env], ["label env", "baseline env", "pred env"]):
                im = ax.imshow(arr, cmap="gray", vmin=0, vmax=vmax, aspect=0.0362 / 0.2)
                ax.set_title(f"{cat} {title} ep{epoch}")
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            out = FIG_DIR / f"{cat}_epoch{epoch:03d}_label_baseline_pred_xz_y{y_idx}.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)

    return stability, speckle_rows


def plot_loss_curves(train_rows: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not train_rows:
        return
    epochs = [int(r["epoch"]) for r in train_rows]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for key in ["D_loss", "G_adv", "G_struct", "G_carrier"]:
        ax.plot(epochs, [float(r[key]) for r in train_rows], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Phase 2a training losses")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(FIG_DIR / "phase2a_loss_curves.png", dpi=180)
    plt.close(fig)


def stitch_report() -> str:
    return (
        "Full-volume stitch needs dense coverage patches, not random64 training patches. "
        "Random64_full1500 samples are sparse random positions and cannot tile a whole volume. "
        "Old RF_Image has rf_stitch.py, rf_stitch_vis.py, rf_export_stitched_to_nii.py, "
        "and dense64 generation/stitch artifacts. For cGAN full-volume validation, migrate or wrap "
        "the old dense64 extraction + rf_stitch.py path; do not attempt stitch from random64 cache."
    )


def main() -> None:
    print(f"{timestamp()} | TASK: {TASK_NAME}", flush=True)
    acquire_run_lock()
    add_paths()
    from rf_cached_dataset import RFCachedDataset
    from rf_cgan_losses import (
        AnisotropicGaussianLowpass3D,
        discriminator_lsgan_loss,
        generator_lsgan_struct_carrier_loss,
        random_y_index,
    )

    with (RUN_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["training"]["enabled"] is True

    seed = int(config["sampler"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = RFCachedDataset(Path(config["data"]["cache_dir"]) / config["data"]["train_split"], return_fp16=True)
    val_dataset = RFCachedDataset(Path(config["data"]["cache_dir"]) / config["data"]["val_split"], return_fp16=True)

    batch_size, dry_peak_gb = choose_batch_size(config, train_dataset, device)
    print(f"ACTUAL_BATCH_SIZE: {batch_size} | dry_run_peak_mem_GB: {dry_peak_gb:.3f}", flush=True)

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    fixed_indices_by_category = pick_fixed_val_indices(
        val_dataset,
        per_category=int(config["diagnostics"]["fixed_val_per_category"]),
        categories=list(config["data"]["categories"]),
    )
    fixed_indices = [idx for cat in config["data"]["categories"] for idx in fixed_indices_by_category[cat]]
    fixed_loader = DataLoader(Subset(val_dataset, fixed_indices), batch_size=3, shuffle=False, num_workers=0)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "fixed_val_indices.json").write_text(json.dumps(fixed_indices_by_category, indent=2), encoding="utf-8")

    g, d = make_models(config, device)
    first_norm = first_norm_class(g)
    assert first_norm == "BatchNorm3d", f"Expected BatchNorm3d, got {first_norm}"
    lowpass = AnisotropicGaussianLowpass3D((6.0, 3.0, 3.0)).to(device)
    criterion = nn.MSELoss()
    betas = (float(config["training"]["beta1"]), float(config["training"]["beta2"]))
    opt_g = torch.optim.Adam(g.parameters(), lr=float(config["training"]["lr_G"]), betas=betas)
    opt_d = torch.optim.Adam(d.parameters(), lr=float(config["training"]["lr_D"]), betas=betas)
    use_amp = bool(config["training"]["use_amp"]) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    snapshot_epochs = set(int(x) for x in config["training"]["snapshot_epochs"])
    checkpoint_epochs = set(int(x) for x in config["training"]["checkpoint_epochs"])
    png_epochs = set(int(x) for x in config["diagnostics"]["png_epochs"])
    total_epochs = int(config["training"]["epochs"])
    epoch_override = os.environ.get("PHASE2A_EPOCH_OVERRIDE")
    if epoch_override:
        total_epochs = int(epoch_override)
        print(f"PHASE2A_EPOCH_OVERRIDE: {total_epochs}", flush=True)

    train_rows = []
    stability_rows = []
    speckle_rows = []
    first_epoch_seconds = None
    first_epoch_peak_gb = None
    aborted = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_train_start = time.time()
    for epoch in range(1, total_epochs + 1):
        g.train()
        d.train()
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sums = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            x = to_device_float(batch["input"], device)
            label = to_device_float(batch["label"], device)
            baseline = to_device_float(batch["baseline"], device)
            y_idx = random_y_index(label)

            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = g(x, baseline)
                d_loss, d_terms = discriminator_lsgan_loss(d, pred, label, baseline, y_idx=y_idx, criterion=criterion)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = g(x, baseline)
                g_loss, g_terms = generator_lsgan_struct_carrier_loss(
                    d,
                    pred,
                    label,
                    baseline,
                    lowpass,
                    y_idx=y_idx,
                    lambda_adv=float(config["loss"]["lambda_adv"]),
                    lambda_struct=float(config["loss"]["lambda_struct"]),
                    lambda_carrier=float(config["loss"]["lambda_carrier"]),
                    criterion=criterion,
                )
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            values = {
                "D_loss": float(d_loss.detach().float().cpu()),
                "D_real": float(d_terms["d_real_score_sigmoid"].cpu()),
                "D_fake": float(d_terms["d_fake_score_sigmoid"].cpu()),
                "G_adv": float(g_terms["g_adv_raw"].cpu()),
                "G_struct": float(g_terms["g_struct_raw"].cpu()),
                "G_carrier": float(g_terms["g_carrier_raw"].cpu()),
                "G_total": float(g_loss.detach().float().cpu()),
            }
            if any(not math.isfinite(v) for v in values.values()):
                aborted = f"NaN/Inf at epoch {epoch}, batch {n_batches + 1}: {values}"
                break
            for key, value in values.items():
                sums[key] += value
            n_batches += 1

        seconds = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        row = {"epoch": epoch, "seconds": seconds, "peak_mem_GB": peak_gb, "batches": n_batches}
        for key in ["D_loss", "D_real", "D_fake", "G_adv", "G_struct", "G_carrier", "G_total"]:
            row[key] = sums[key] / max(n_batches, 1)
        train_rows.append(row)

        if epoch == 1:
            first_epoch_seconds = seconds
            first_epoch_peak_gb = peak_gb
            print(f"FIRST_EPOCH_SECONDS: {seconds:.3f}", flush=True)
            print(f"FIRST_EPOCH_PEAK_MEM_GB: {peak_gb:.3f}", flush=True)
            print(f"ESTIMATED_50_EPOCH_HOURS: {seconds * total_epochs / 3600:.3f}", flush=True)

        print(
            f"epoch {epoch:03d}/{total_epochs} D_loss={row['D_loss']:.4f} "
            f"D_real={row['D_real']:.4f} D_fake={row['D_fake']:.4f} "
            f"G_adv={row['G_adv']:.4f} G_struct={row['G_struct']:.5f} "
            f"G_carrier={row['G_carrier']:.5f} sec={seconds:.1f} peakGB={peak_gb:.2f}",
            flush=True,
        )

        if epoch in snapshot_epochs or epoch == total_epochs:
            stability, sp_rows = evaluate_snapshot(
                epoch,
                g,
                d,
                lowpass,
                fixed_loader,
                fixed_indices_by_category,
                val_dataset,
                config,
                device,
                save_png=epoch in png_epochs,
            )
            stability_rows.append(stability)
            speckle_rows.extend(sp_rows)
            write_csv(LOG_DIR / "stability.csv", stability_rows)
            write_csv(LOG_DIR / "speckle_check.csv", speckle_rows)

        if epoch in checkpoint_epochs or epoch == total_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_class": type(g).__name__,
                    "discriminator_class": type(d).__name__,
                    "G_state_dict": g.state_dict(),
                    "D_state_dict": d.state_dict(),
                    "config": config,
                },
                CKPT_DIR / f"checkpoint_epoch{epoch:03d}.pth",
            )

        write_csv(LOG_DIR / "train_losses.csv", train_rows)
        plot_loss_curves(train_rows)
        if aborted:
            print(f"EARLY_ABORT: {aborted}", flush=True)
            break

    total_seconds = time.time() - t_train_start
    plot_loss_curves(train_rows)

    final_speckle_by_cat = {}
    if speckle_rows:
        final_epoch = max(int(r["epoch"]) for r in speckle_rows)
        for r in speckle_rows:
            if int(r["epoch"]) == final_epoch:
                final_speckle_by_cat[r["category"]] = r

    summary_lines = [
        "# Phase 2a summary",
        "",
        f"Generated: {timestamp()}",
        "",
        "## 1. NaN / collapse",
        "",
        f"- aborted: {aborted}",
        f"- completed_epochs: {len(train_rows)} / {total_epochs}",
        "",
        "## 2. Batch / memory / timing",
        "",
        f"- actual_batch_size: {batch_size}",
        f"- dry_run_peak_mem_GB: {dry_peak_gb:.3f}",
        f"- first_epoch_seconds: {first_epoch_seconds:.3f}",
        f"- first_epoch_peak_mem_GB: {first_epoch_peak_gb:.3f}",
        f"- total_seconds: {total_seconds:.3f}",
        "",
        "## 3. Final fixed-val speckle check",
        "",
    ]
    for cat in ["carotid", "muscle", "phantom"]:
        r = final_speckle_by_cat.get(cat)
        if r:
            summary_lines.append(
                f"- {cat}: pred_SNR={float(r['pred_SNR']):.4f}, "
                f"label_SNR={float(r['label_SNR']):.4f}, KS={float(r['env_hist_KS']):.4f}"
            )
    summary_lines.extend(
        [
            "",
            "## 4. Objective observation only",
            "",
            "This run is not a quality conclusion. Inspect the saved PNGs only as a loss-design health check.",
            "If pred_SNR stays far above label_SNR, the output is still smoother than label speckle. If it approaches label_SNR while KS decreases, the adversarial term is moving in the intended direction.",
            "",
            "## 5. Stitch capability report",
            "",
            stitch_report(),
            "",
        ]
    )
    (LOG_DIR / "phase2a_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines), flush=True)


if __name__ == "__main__":
    main()
