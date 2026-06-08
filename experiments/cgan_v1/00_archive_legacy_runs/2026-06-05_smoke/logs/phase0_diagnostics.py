from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np


TASK_NAME = "cGAN pilot Phase 0 diagnostics"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
RUN_DIR = PROJECT_ROOT / "experiments/cgan_v1/runs/2026-06-05_smoke"
LOG_DIR = RUN_DIR / "logs"
CACHE_ROOT = Path("/home/liujia/RF_Image/Data_cache_random64_full1500")
LEGACY_CODE_DIR = Path("/home/liujia/RF_Image")

SPACING = {"z": 0.0362, "x": 0.2, "y": 0.2}
CATEGORIES = ["carotid", "muscle", "phantom"]


def timestamp() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %z")


def load_meta(split: str) -> dict[str, np.ndarray]:
    path = CACHE_ROOT / split / "meta.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=False))


def category_counts() -> list[dict[str, object]]:
    rows = []
    for split in ["train", "val", "test"]:
        meta = load_meta(split)
        cats = meta["category"].astype(str)
        counts = Counter(cats)
        total = len(cats)
        for cat in CATEGORIES:
            n = counts.get(cat, 0)
            rows.append(
                {
                    "split": split,
                    "category": cat,
                    "patch_count": n,
                    "fraction": n / total if total else math.nan,
                    "total_split_patches": total,
                }
            )
        other = total - sum(counts.get(cat, 0) for cat in CATEGORIES)
        rows.append(
            {
                "split": split,
                "category": "other",
                "patch_count": other,
                "fraction": other / total if total else math.nan,
                "total_split_patches": total,
            }
        )
    return rows


def load_cached_arrays(split: str):
    meta = load_meta(split)
    label_shape = tuple(int(x) for x in meta["label_shape"])
    label = np.memmap(
        CACHE_ROOT / split / "label.dat",
        dtype=np.float16,
        mode="r",
        shape=label_shape,
    )
    scale = np.memmap(
        CACHE_ROOT / split / "scale.dat",
        dtype=np.float32,
        mode="r",
        shape=(label_shape[0],),
    )
    return meta, label, scale


def envelope_xz(label_sample: np.ndarray, scale: float, y_idx: int | None = None) -> np.ndarray:
    if y_idx is None:
        y_idx = label_sample.shape[-1] // 2
    real = label_sample[0].astype(np.float32)
    imag = label_sample[1].astype(np.float32)
    env = np.sqrt(real * real + imag * imag + 1e-12) * float(scale)
    return env[:, :, y_idx].astype(np.float32, copy=False)


def central_axis_fwhm(axis_values: np.ndarray) -> float:
    center = len(axis_values) // 2
    vals = axis_values.astype(np.float64)
    if not np.isfinite(vals[center]) or vals[center] <= 0:
        return float("nan")
    vals = vals / vals[center]

    # Search positive side for first crossing below half maximum.
    dist = None
    for i in range(center + 1, len(vals)):
        if vals[i] <= 0.5:
            v0 = vals[i - 1]
            v1 = vals[i]
            if v0 == v1:
                d = i - center
            else:
                frac = (0.5 - v0) / (v1 - v0)
                d = (i - 1 - center) + frac
            dist = max(float(d), 0.0)
            break
    if dist is None:
        return float("nan")
    return 2.0 * dist


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
    ac = np.fft.fftshift(ac)
    return ac / denom


def roi_score(window: np.ndarray, global_p95: float) -> tuple[float, dict[str, float]]:
    mean = float(np.mean(window))
    std = float(np.std(window))
    cv = std / (mean + 1e-12)
    dz = np.diff(window, axis=0)
    dx = np.diff(window, axis=1)
    grad = float(np.mean(np.abs(dz))) + float(np.mean(np.abs(dx)))
    high_frac = float(np.mean(window > global_p95))
    p99 = float(np.percentile(window, 99))
    max_over_mean = p99 / (mean + 1e-12)
    # Prefer low structural gradient and low strong-reflector occupancy.
    # Do not minimize CV directly to zero; fully developed speckle has real variance.
    score = grad / (mean + 1e-12) + 3.0 * high_frac + 0.05 * max(0.0, max_over_mean - 5.0)
    return score, {
        "mean": mean,
        "std": std,
        "cv": cv,
        "grad_norm": grad / (mean + 1e-12),
        "high_frac": high_frac,
        "p99_over_mean": max_over_mean,
    }


def pick_rois_for_category(
    category: str,
    max_rois: int = 12,
    split: str = "train",
    roi_z: int = 32,
    roi_x: int = 16,
) -> list[dict[str, object]]:
    meta, label, scale = load_cached_arrays(split)
    cats = meta["category"].astype(str)
    paths = meta["path"].astype(str)
    indices = np.where(cats == category)[0]
    if len(indices) == 0:
        return []

    # Deterministic spread across the category, not random.
    chosen_samples = indices[np.linspace(0, len(indices) - 1, min(40, len(indices)), dtype=int)]
    candidates = []
    for idx in chosen_samples:
        sample = np.asarray(label[idx], dtype=np.float32)
        env = envelope_xz(sample, float(scale[idx]), y_idx=None)
        global_p95 = float(np.percentile(env, 95))
        for z0 in range(0, env.shape[0] - roi_z + 1, 8):
            for x0 in range(0, env.shape[1] - roi_x + 1, 4):
                win = env[z0 : z0 + roi_z, x0 : x0 + roi_x]
                score, stats = roi_score(win, global_p95)
                if not np.isfinite(score):
                    continue
                candidates.append((score, int(idx), z0, x0, stats))

    candidates.sort(key=lambda x: x[0])
    selected = []
    used_paths = set()
    for score, idx, z0, x0, stats in candidates:
        # Spread ROIs across source files where possible.
        path = paths[idx]
        if path in used_paths and len(used_paths) < max_rois:
            continue
        sample = np.asarray(label[idx], dtype=np.float32)
        env = envelope_xz(sample, float(scale[idx]), y_idx=None)
        roi = env[z0 : z0 + roi_z, x0 : x0 + roi_x]
        ac = normalized_autocorr_2d(roi)
        cz, cx = ac.shape[0] // 2, ac.shape[1] // 2
        fwhm_z = central_axis_fwhm(ac[:, cx])
        fwhm_x = central_axis_fwhm(ac[cz, :])
        mean = float(np.mean(roi))
        std = float(np.std(roi))
        row = {
            "category": category,
            "split": split,
            "sample_idx": int(idx),
            "path": path,
            "y_idx": int(sample.shape[-1] // 2),
            "roi_z0": int(z0),
            "roi_z1": int(z0 + roi_z - 1),
            "roi_x0": int(x0),
            "roi_x1": int(x0 + roi_x - 1),
            "roi_shape_zx": f"{roi_z}x{roi_x}",
            "mean_env": mean,
            "std_env": std,
            "speckle_snr_mean_over_std": mean / (std + 1e-12),
            "autocorr_fwhm_z_vox": fwhm_z,
            "autocorr_fwhm_x_vox": fwhm_x,
            "autocorr_fwhm_z_mm": fwhm_z * SPACING["z"] if np.isfinite(fwhm_z) else float("nan"),
            "autocorr_fwhm_x_mm": fwhm_x * SPACING["x"] if np.isfinite(fwhm_x) else float("nan"),
            "selection_score": float(score),
            **{f"selection_{k}": float(v) for k, v in stats.items()},
        }
        selected.append(row)
        used_paths.add(path)
        if len(selected) >= max_rois:
            break
    return selected


def summarize_rois(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    summary = []
    for cat in CATEGORIES:
        rs = grouped.get(cat, [])
        out = {"category": cat, "n_rois": len(rs)}
        for key in [
            "mean_env",
            "std_env",
            "speckle_snr_mean_over_std",
            "autocorr_fwhm_z_vox",
            "autocorr_fwhm_x_vox",
            "autocorr_fwhm_z_mm",
            "autocorr_fwhm_x_mm",
        ]:
            vals = np.array([float(r[key]) for r in rs], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out[f"{key}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        summary.append(out)
    return summary


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


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def implementation_facts() -> dict[str, object]:
    # These facts are intentionally derived from current source/notebook text.
    nb_path = RUN_DIR / "train.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"] if cell.get("cell_type") == "code")
    model_text = (PROJECT_ROOT / "rf_cgan_models.py").read_text(encoding="utf-8")
    legacy_model_text = (LEGACY_CODE_DIR / "rf_models.py").read_text(encoding="utf-8")

    if str(LEGACY_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_CODE_DIR))
    import torch.nn as nn  # noqa: WPS433 - local import keeps diagnostics self-contained
    from rf_models import TinyResidualRFNet  # noqa: WPS433

    g = TinyResidualRFNet(use_batch_norm=True)
    first_norm = next((m for m in g.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    first_norm_class = type(first_norm).__name__ if first_norm is not None else "MISSING"

    facts = {
        "g_loss_formula": "g_loss = g_adv + lambda_fid * g_fid",
        "g_adv": "MSELoss(D(cat([pred_env, baseline_env])), ones)",
        "g_fid": "torch.mean(torch.abs(pred - label))",
        "lambda_fidelity": 10.0,
        "d_loss_formula": "0.5 * (MSE(D(label_env+baseline_env), 1) + MSE(D(pred_env_detached+baseline_env), 0))",
        "d_condition": "baseline envelope is concatenated with candidate/label envelope as channel 1",
        "d_slice": "extract_envelope_slice(..., y_idx=None), therefore middle y slice only in current notebook",
        "extract_envelope_slice_y_support": "supports arbitrary single y_idx; no multi-slice helper yet",
        "g_norm": "TinyResidualRFNet(use_batch_norm=True); rf_models.py uses BatchNorm3d when use_batch_norm=True",
        "first_norm_class_runtime": first_norm_class,
        "batchnorm_confirmed_runtime": first_norm_class == "BatchNorm3d",
        "batchnorm_available_in_source": "nn.BatchNorm3d" in legacy_model_text,
        "instance_norm_in_g_current_path": "current TinyResidualRFNet path uses BatchNorm3d, not InstanceNorm3d",
        "complex_phase_constraint": "no explicit carrier/phase loss or phase consistency term; only complex L1 in smoke G fidelity",
        "no_training_or_loss_change": True,
    }
    # Small consistency flags for reviewers.
    facts["notebook_contains_complex_l1"] = "torch.mean(torch.abs(pred - label))" in code
    facts["notebook_concats_baseline_to_d"] = "torch.cat([pred_env, baseline_env], dim=1)" in code
    facts["extractor_has_y_idx_arg"] = "y_idx: int | None = None" in model_text
    return facts


def main() -> None:
    print(f"{timestamp()} | TASK: {TASK_NAME}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    count_rows = category_counts()
    count_csv = LOG_DIR / "phase0_cache_counts.csv"
    write_csv(count_csv, count_rows)

    roi_rows = []
    for cat in CATEGORIES:
        roi_rows.extend(pick_rois_for_category(cat))
    roi_csv = LOG_DIR / "phase0_speckle_rois.csv"
    write_csv(roi_csv, roi_rows)

    roi_summary = summarize_rois(roi_rows)
    roi_summary_csv = LOG_DIR / "phase0_speckle_summary.csv"
    write_csv(roi_summary_csv, roi_summary)

    facts = implementation_facts()

    summary_md = LOG_DIR / "phase0_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Phase 0 diagnostics summary",
                "",
                f"Generated: {timestamp()}",
                "",
                "## A. Cache category counts",
                "",
                "Category source: `meta.npz['category']` in each cache split. The path field is retained for traceability, but category counts are not inferred from filename or index ranges.",
                "",
                markdown_table(count_rows, ["split", "category", "patch_count", "fraction", "total_split_patches"]),
                "",
                "## B. Automated homogeneous ROI speckle diagnostics",
                "",
                "ROI method: deterministic automatic proxy ROIs on the label middle-y XZ slice. Candidate windows are ranked by low normalized gradient, low strong-reflector occupancy, and low p99/mean excess. This is not a human-frozen Slicer ROI set.",
                "",
                markdown_table(
                    roi_summary,
                    [
                        "category",
                        "n_rois",
                        "speckle_snr_mean_over_std_mean",
                        "autocorr_fwhm_z_vox_mean",
                        "autocorr_fwhm_x_vox_mean",
                        "autocorr_fwhm_z_mm_mean",
                        "autocorr_fwhm_x_mm_mean",
                    ],
                ),
                "",
                "Detailed ROI rows: `phase0_speckle_rois.csv`.",
                "",
                "## C. Current implementation facts",
                "",
                "```json",
                json.dumps(facts, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"WROTE {count_csv}")
    print(f"WROTE {roi_csv}")
    print(f"WROTE {roi_summary_csv}")
    print(f"WROTE {summary_md}")

    print("\nCACHE_COUNTS")
    print(markdown_table(count_rows, ["split", "category", "patch_count", "fraction", "total_split_patches"]))

    print("\nSPECKLE_SUMMARY")
    print(
        markdown_table(
            roi_summary,
            [
                "category",
                "n_rois",
                "speckle_snr_mean_over_std_mean",
                "autocorr_fwhm_z_vox_mean",
                "autocorr_fwhm_x_vox_mean",
                "autocorr_fwhm_z_mm_mean",
                "autocorr_fwhm_x_mm_mean",
            ],
        )
    )

    print("\nIMPLEMENTATION_FACTS")
    print(json.dumps(facts, indent=2, sort_keys=True))


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()
