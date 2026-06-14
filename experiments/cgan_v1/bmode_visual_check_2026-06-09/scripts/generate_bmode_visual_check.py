from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "rf_cached_dataset.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_cached_dataset import RFCachedDataset


CACHE_ROOT = Path("/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607")
OUT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = OUT_DIR / "figures"
RESULTS_DIR = OUT_DIR / "results"
CATEGORIES = ["carotid", "muscle", "phantom"]
SAMPLES_PER_CATEGORY = 5
Z_SPACING_MM = 0.0362
X_SPACING_MM = 0.2
Y_SPACING_MM = 0.2
EPS = 1e-12


def envelope_db(volume: np.ndarray) -> np.ndarray:
    env = np.sqrt(volume[0].astype(np.float32) ** 2 + volume[1].astype(np.float32) ** 2)
    return 20.0 * np.log10(env + EPS)


def shared_window(label_db: np.ndarray, baseline_db: np.ndarray) -> tuple[float, float]:
    merged = np.concatenate([label_db.ravel(), baseline_db.ravel()])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return -80.0, 0.0
    vmax = float(np.percentile(finite, 99.5))
    vmin = vmax - 60.0
    return vmin, vmax


def orient_slice(volume_db: np.ndarray, plane: str) -> tuple[np.ndarray, str, float]:
    z_size, x_size, y_size = volume_db.shape
    z_idx = z_size // 2
    x_idx = x_size // 2
    y_idx = 16 if y_size > 16 else y_size // 2

    if plane == "XZ":
        return volume_db[:, :, y_idx], f"XZ y={y_idx}", Z_SPACING_MM / X_SPACING_MM
    if plane == "XY":
        return volume_db[z_idx, :, :], f"XY z={z_idx}", X_SPACING_MM / Y_SPACING_MM
    if plane == "ZY":
        return volume_db[:, x_idx, :], f"ZY x={x_idx}", Z_SPACING_MM / Y_SPACING_MM
    raise ValueError(f"Unknown plane: {plane}")


def save_pair(
    label_db: np.ndarray,
    baseline_db: np.ndarray,
    category: str,
    sample_idx: int,
    path: str,
    plane: str,
) -> Path:
    label_slice, plane_title, aspect = orient_slice(label_db, plane)
    baseline_slice, _, _ = orient_slice(baseline_db, plane)
    vmin, vmax = shared_window(label_slice, baseline_slice)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), constrained_layout=True)
    for ax, arr, title in zip(axes, [label_slice, baseline_slice], ["label 33-angle", "baseline 3-angle"]):
        image = ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
        ax.set_title(title)
        ax.set_xlabel(plane[-1].lower())
        ax.set_ylabel(plane[0].lower())
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="dB")

    source_name = Path(path).stem
    fig.suptitle(f"{category} sample={sample_idx} {plane_title} shared [{vmin:.1f}, {vmax:.1f}] dB\n{source_name}")
    out = FIG_DIR / f"{category}_idx{sample_idx:03d}_{plane.lower()}_label_vs_baseline.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = RFCachedDataset(CACHE_ROOT / "test", return_fp16=False, restore_scale=True)
    by_category: dict[str, list[int]] = defaultdict(list)
    for idx, category in enumerate(dataset.categories):
        if category in CATEGORIES and len(by_category[category]) < SAMPLES_PER_CATEGORY:
            by_category[category].append(idx)

    rows = []
    for category in CATEGORIES:
        indices = by_category[category]
        if len(indices) != SAMPLES_PER_CATEGORY:
            raise RuntimeError(f"{category} only has {len(indices)} selected samples")
        for idx in indices:
            sample = dataset[idx]
            label = sample["label"].numpy()
            baseline = sample["baseline"].numpy()
            label_db = envelope_db(label)
            baseline_db = envelope_db(baseline)
            for plane in ["XZ", "XY", "ZY"]:
                out = save_pair(label_db, baseline_db, category, idx, sample["path"], plane)
                label_slice, _, _ = orient_slice(label_db, plane)
                baseline_slice, _, _ = orient_slice(baseline_db, plane)
                vmin, vmax = shared_window(label_slice, baseline_slice)
                rows.append(
                    {
                        "category": category,
                        "sample_idx": idx,
                        "plane": plane,
                        "path": sample["path"],
                        "scale": float(sample["scale"]),
                        "label_db_min": float(np.nanmin(label_slice)),
                        "label_db_max": float(np.nanmax(label_slice)),
                        "baseline_db_min": float(np.nanmin(baseline_slice)),
                        "baseline_db_max": float(np.nanmax(baseline_slice)),
                        "shared_vmin": vmin,
                        "shared_vmax": vmax,
                        "figure": str(out.relative_to(OUT_DIR)),
                    }
                )

    manifest = RESULTS_DIR / "bmode_visual_check_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} figures")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
