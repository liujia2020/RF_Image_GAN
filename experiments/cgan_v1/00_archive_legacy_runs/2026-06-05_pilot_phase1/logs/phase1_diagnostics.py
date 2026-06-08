from __future__ import annotations

import csv
import json
import math
import random
import sys
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
from torch.utils.data import DataLoader


TASK_NAME = "cGAN pilot Phase 1 diagnostics"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
RUN_DIR = PROJECT_ROOT / "experiments/cgan_v1/runs/2026-06-05_pilot_phase1"
LOG_DIR = RUN_DIR / "logs"
FIG_DIR = RUN_DIR / "figures/lowpass"
NII_DIR = RUN_DIR / "nii/lowpass"
LEGACY_CODE_DIR = Path("/home/liujia/RF_Image")
CACHE_ROOT = Path("/home/liujia/RF_Image/Data_cache_random64_full1500")
CATEGORIES = ["carotid", "muscle", "phantom"]
SPACING = {"z": 0.0362, "x": 0.2, "y": 0.2}


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")


def add_paths() -> None:
    for p in [PROJECT_ROOT, LEGACY_CODE_DIR]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_meta(split: str = "train") -> dict[str, np.ndarray]:
    return dict(np.load(CACHE_ROOT / split / "meta.npz", allow_pickle=False))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def envelope_numpy(label: np.ndarray, scale: float) -> np.ndarray:
    real = label[0].astype(np.float32)
    imag = label[1].astype(np.float32)
    return np.sqrt(real * real + imag * imag + 1e-12).astype(np.float32) * float(scale)


def central_axis_fwhm(axis_values: np.ndarray) -> float:
    center = len(axis_values) // 2
    vals = axis_values.astype(np.float64)
    if not np.isfinite(vals[center]) or vals[center] <= 0:
        return float("nan")
    vals = vals / vals[center]
    for i in range(center + 1, len(vals)):
        if vals[i] <= 0.5:
            v0, v1 = vals[i - 1], vals[i]
            frac = 0.0 if v0 == v1 else (0.5 - v0) / (v1 - v0)
            return 2.0 * max((i - 1 - center) + frac, 0.0)
    return float("nan")


def normalized_autocorr_2d(roi: np.ndarray) -> np.ndarray:
    x = roi.astype(np.float64)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom <= 1e-20:
        return np.full((2 * x.shape[0] - 1, 2 * x.shape[1] - 1), np.nan)
    shape = (2 * x.shape[0] - 1, 2 * x.shape[1] - 1)
    fft_shape = (int(2 ** math.ceil(math.log2(shape[0]))), int(2 ** math.ceil(math.log2(shape[1]))))
    axes = (0, 1)
    f = np.fft.rfftn(x, fft_shape, axes=axes)
    ac = np.fft.irfftn(f * np.conj(f), fft_shape, axes=axes)
    ac = ac[: shape[0], : shape[1]]
    return np.fft.fftshift(ac) / denom


def roi_score(window: np.ndarray, global_p95: float) -> float:
    mean = float(np.mean(window))
    dz = np.diff(window, axis=0)
    dy = np.diff(window, axis=1)
    grad_norm = (float(np.mean(np.abs(dz))) + float(np.mean(np.abs(dy)))) / (mean + 1e-12)
    high_frac = float(np.mean(window > global_p95))
    p99_over_mean = float(np.percentile(window, 99)) / (mean + 1e-12)
    return grad_norm + 3.0 * high_frac + 0.05 * max(0.0, p99_over_mean - 5.0)


def y_direction_fwhm(max_rois: int = 12, roi_z: int = 32, roi_y: int = 16) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    meta = load_meta("train")
    label_shape = tuple(int(x) for x in meta["label_shape"])
    label = np.memmap(CACHE_ROOT / "train/label.dat", dtype=np.float16, mode="r", shape=label_shape)
    scale = np.memmap(CACHE_ROOT / "train/scale.dat", dtype=np.float32, mode="r", shape=(label_shape[0],))
    cats = meta["category"].astype(str)
    paths = meta["path"].astype(str)

    rows = []
    for cat in CATEGORIES:
        idxs = np.where(cats == cat)[0]
        sample_idxs = idxs[np.linspace(0, len(idxs) - 1, min(35, len(idxs)), dtype=int)]
        candidates = []
        for idx in sample_idxs:
            env = envelope_numpy(np.asarray(label[idx], dtype=np.float32), float(scale[idx]))
            x_idx = env.shape[1] // 2
            zy = env[:, x_idx, :]
            global_p95 = float(np.percentile(zy, 95))
            for z0 in range(0, zy.shape[0] - roi_z + 1, 8):
                for y0 in range(0, zy.shape[1] - roi_y + 1, 4):
                    win = zy[z0 : z0 + roi_z, y0 : y0 + roi_y]
                    candidates.append((roi_score(win, global_p95), int(idx), z0, y0, x_idx))
        candidates.sort(key=lambda x: x[0])

        selected_paths = set()
        selected = 0
        for score, idx, z0, y0, x_idx in candidates:
            if paths[idx] in selected_paths and len(selected_paths) < max_rois:
                continue
            env = envelope_numpy(np.asarray(label[idx], dtype=np.float32), float(scale[idx]))
            roi = env[z0 : z0 + roi_z, x_idx, y0 : y0 + roi_y]
            ac = normalized_autocorr_2d(roi)
            cz, cy = ac.shape[0] // 2, ac.shape[1] // 2
            fwhm_z = central_axis_fwhm(ac[:, cy])
            fwhm_y = central_axis_fwhm(ac[cz, :])
            mean = float(np.mean(roi))
            std = float(np.std(roi))
            rows.append(
                {
                    "category": cat,
                    "split": "train",
                    "sample_idx": int(idx),
                    "path": paths[idx],
                    "x_idx": int(x_idx),
                    "roi_z0": int(z0),
                    "roi_z1": int(z0 + roi_z - 1),
                    "roi_y0": int(y0),
                    "roi_y1": int(y0 + roi_y - 1),
                    "roi_shape_zy": f"{roi_z}x{roi_y}",
                    "mean_env": mean,
                    "std_env": std,
                    "speckle_snr_mean_over_std": mean / (std + 1e-12),
                    "autocorr_fwhm_z_vox": fwhm_z,
                    "autocorr_fwhm_y_vox": fwhm_y,
                    "autocorr_fwhm_z_mm": fwhm_z * SPACING["z"] if np.isfinite(fwhm_z) else float("nan"),
                    "autocorr_fwhm_y_mm": fwhm_y * SPACING["y"] if np.isfinite(fwhm_y) else float("nan"),
                    "selection_score": float(score),
                }
            )
            selected_paths.add(paths[idx])
            selected += 1
            if selected >= max_rois:
                break

    summary = []
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        out = {"category": cat, "n_rois": len(cat_rows)}
        for key in [
            "speckle_snr_mean_over_std",
            "autocorr_fwhm_z_vox",
            "autocorr_fwhm_y_vox",
            "autocorr_fwhm_z_mm",
            "autocorr_fwhm_y_mm",
        ]:
            vals = np.array([float(r[key]) for r in cat_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out[f"{key}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        summary.append(out)
    return rows, summary


def one_forward_loss_probe() -> list[dict[str, object]]:
    add_paths()
    from rf_cached_dataset import RFCachedDataset
    from rf_models import TinyResidualRFNet
    from rf_cgan_data import StratifiedCategoryBatchSampler
    from rf_cgan_losses import (
        AnisotropicGaussianLowpass3D,
        discriminator_lsgan_loss,
        generator_lsgan_struct_carrier_loss,
        random_y_index,
    )
    from rf_cgan_models import Envelope2DPatchDiscriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RFCachedDataset(CACHE_ROOT / "train")
    sampler = StratifiedCategoryBatchSampler(dataset.categories, samples_per_category=1, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    batch = next(iter(loader))
    batch_categories = list(batch["category"])

    x = batch["input"].to(device)
    label = batch["label"].to(device)
    baseline = batch["baseline"].to(device)

    g = TinyResidualRFNet(use_batch_norm=True).to(device).eval()
    d = Envelope2DPatchDiscriminator(ndf=64).to(device).eval()
    lowpass = AnisotropicGaussianLowpass3D((6.0, 3.0, 3.0)).to(device).eval()
    criterion = nn.MSELoss()
    y_idx = random_y_index(label, rng=random.Random(20260605))

    with torch.no_grad():
        pred = g(x, baseline)
        g_loss, g_terms = generator_lsgan_struct_carrier_loss(
            d,
            pred,
            label,
            baseline,
            lowpass,
            y_idx=y_idx,
            lambda_adv=1.0,
            lambda_struct=10.0,
            lambda_carrier=1.0,
            criterion=criterion,
        )
        d_loss, d_terms = discriminator_lsgan_loss(d, pred, label, baseline, y_idx=y_idx, criterion=criterion)

    row = {
        "device": str(device),
        "batch_size": int(label.shape[0]),
        "batch_categories": ",".join(batch_categories),
        "y_idx": int(y_idx),
        "g_adv_raw": float(g_terms["g_adv_raw"].cpu()),
        "g_struct_raw": float(g_terms["g_struct_raw"].cpu()),
        "g_carrier_raw": float(g_terms["g_carrier_raw"].cpu()),
        "g_adv_weighted": float(g_terms["g_adv_weighted"].cpu()),
        "g_struct_weighted": float(g_terms["g_struct_weighted"].cpu()),
        "g_carrier_weighted": float(g_terms["g_carrier_weighted"].cpu()),
        "g_total": float(g_loss.cpu()),
        "d_loss": float(d_loss.cpu()),
        "d_real_score_sigmoid": float(d_terms["d_real_score_sigmoid"].cpu()),
        "d_fake_score_sigmoid": float(d_terms["d_fake_score_sigmoid"].cpu()),
        "first_norm_class": next(type(m).__name__ for m in g.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))),
    }
    return [row]


def save_nifti(path: Path, volume_zyx: np.ndarray) -> bool:
    try:
        import nibabel as nib
    except Exception:
        return False
    affine = np.diag([SPACING["z"], SPACING["x"], SPACING["y"], 1.0]).astype(np.float32)
    img = nib.Nifti1Image(volume_zyx.astype(np.float32), affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return True


def lowpass_visualization() -> list[dict[str, object]]:
    add_paths()
    from rf_cgan_losses import AnisotropicGaussianLowpass3D

    meta = load_meta("train")
    label_shape = tuple(int(x) for x in meta["label_shape"])
    label = np.memmap(CACHE_ROOT / "train/label.dat", dtype=np.float16, mode="r", shape=label_shape)
    scale = np.memmap(CACHE_ROOT / "train/scale.dat", dtype=np.float32, mode="r", shape=(label_shape[0],))
    cats = meta["category"].astype(str)
    paths = meta["path"].astype(str)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NII_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lowpass = AnisotropicGaussianLowpass3D((6.0, 3.0, 3.0)).to(device).eval()

    rows = []
    for cat in CATEGORIES:
        idx = int(np.where(cats == cat)[0][0])
        env = envelope_numpy(np.asarray(label[idx], dtype=np.float32), float(scale[idx]))
        with torch.no_grad():
            env_t = torch.from_numpy(env).unsqueeze(0).unsqueeze(0).to(device)
            lp = lowpass(env_t).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        residual = env - lp
        y_idx = env.shape[-1] // 2

        slices = [env[:, :, y_idx], lp[:, :, y_idx], residual[:, :, y_idx]]
        titles = ["env(label)", "LP(env(label))", "env - LP"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        vmax = float(np.percentile(env[:, :, y_idx], 99.5))
        rlim = float(np.percentile(np.abs(residual[:, :, y_idx]), 99.5))
        for ax, arr, title in zip(axes, slices, titles):
            if title == "env - LP":
                im = ax.imshow(arr, cmap="seismic", vmin=-rlim, vmax=rlim, aspect=SPACING["z"] / SPACING["x"])
            else:
                im = ax.imshow(arr, cmap="gray", vmin=0, vmax=vmax, aspect=SPACING["z"] / SPACING["x"])
            ax.set_title(f"{cat} {title}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        png_path = FIG_DIR / f"{cat}_env_lowpass_residual_xz_y{y_idx}.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

        nii_saved = {
            "env": save_nifti(NII_DIR / f"{cat}_env_label.nii", env),
            "lp": save_nifti(NII_DIR / f"{cat}_lp_env_label.nii", lp),
            "residual": save_nifti(NII_DIR / f"{cat}_env_minus_lp.nii", residual),
        }

        rows.append(
            {
                "category": cat,
                "sample_idx": idx,
                "path": paths[idx],
                "png": str(png_path),
                "nifti_written": all(nii_saved.values()),
                "env_mean": float(np.mean(env)),
                "lp_mean": float(np.mean(lp)),
                "residual_abs_mean": float(np.mean(np.abs(residual))),
                "residual_abs_over_env_mean": float(np.mean(np.abs(residual)) / (np.mean(env) + 1e-12)),
            }
        )
    return rows


def stitch_capability_report() -> list[dict[str, object]]:
    old = LEGACY_CODE_DIR
    new = PROJECT_ROOT
    rows = []
    for name, root in [("RF_Image_GAN", new), ("RF_Image", old)]:
        files = sorted(
            [
                p
                for p in root.glob("**/*")
                if p.is_file()
                and (
                    "stitch" in p.name.lower()
                    or "nii" in p.name.lower()
                    or "nifti" in p.name.lower()
                )
            ]
        )
        rows.append(
            {
                "repo": name,
                "root": str(root),
                "matching_file_count": len(files),
                "files": "; ".join(str(p.relative_to(root)) for p in files[:30]),
            }
        )
    return rows


def main() -> None:
    print(f"{timestamp()} | TASK: {TASK_NAME}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with (RUN_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["training"]["enabled"] is False, "Phase 1 diagnostics must not train"

    loss_rows = one_forward_loss_probe()
    y_rows, y_summary = y_direction_fwhm()
    lowpass_rows = lowpass_visualization()
    stitch_rows = stitch_capability_report()

    write_csv(LOG_DIR / "phase1_loss_probe.csv", loss_rows)
    write_csv(LOG_DIR / "phase1_y_fwhm_rois.csv", y_rows)
    write_csv(LOG_DIR / "phase1_y_fwhm_summary.csv", y_summary)
    write_csv(LOG_DIR / "phase1_lowpass_visuals.csv", lowpass_rows)
    write_csv(LOG_DIR / "phase1_stitch_capability.csv", stitch_rows)

    summary = "\n".join(
        [
            "# Phase 1 diagnostics summary",
            "",
            f"Generated: {timestamp()}",
            "",
            "## Loss probe",
            markdown_table(
                loss_rows,
                [
                    "batch_categories",
                    "y_idx",
                    "g_adv_raw",
                    "g_struct_raw",
                    "g_carrier_raw",
                    "g_adv_weighted",
                    "g_struct_weighted",
                    "g_carrier_weighted",
                    "g_total",
                    "first_norm_class",
                ],
            ),
            "",
            "## Y-direction speckle FWHM",
            markdown_table(
                y_summary,
                [
                    "category",
                    "n_rois",
                    "speckle_snr_mean_over_std_mean",
                    "autocorr_fwhm_y_vox_mean",
                    "autocorr_fwhm_y_mm_mean",
                ],
            ),
            "",
            "## Low-pass visualizations",
            markdown_table(
                lowpass_rows,
                ["category", "sample_idx", "png", "nifti_written", "residual_abs_over_env_mean"],
            ),
            "",
            "## Stitch capability",
            markdown_table(stitch_rows, ["repo", "matching_file_count", "files"]),
            "",
        ]
    )
    (LOG_DIR / "phase1_summary.md").write_text(summary, encoding="utf-8")

    print("\nLOSS_PROBE")
    print(summary.split("## Y-direction")[0])
    print("Y_FWHM")
    print(markdown_table(y_summary, ["category", "n_rois", "speckle_snr_mean_over_std_mean", "autocorr_fwhm_y_vox_mean", "autocorr_fwhm_y_mm_mean"]))
    print("\nLOWPASS_VISUALS")
    print(markdown_table(lowpass_rows, ["category", "sample_idx", "png", "nifti_written", "residual_abs_over_env_mean"]))
    print("\nSTITCH_CAPABILITY")
    print(markdown_table(stitch_rows, ["repo", "matching_file_count", "files"]))


if __name__ == "__main__":
    main()
