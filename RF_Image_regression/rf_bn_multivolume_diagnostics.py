from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from rf_visualization import complex_abs_numpy, extract_slice, volume_to_db


FIELDS = ("baseline", "pred", "label")
DEFAULT_X_BOUNDARIES = (32, 64, 96)
DEFAULT_PATCH_SIZE = (64, 32, 32)
DEFAULT_DB_MIN = -60.0
DEFAULT_STRESS_DB_MIN = -40.0
DZ_MM = 0.0362
DX_MM = 0.2
ASPECT_XZ = DZ_MM / DX_MM


def load_volumes(volume_dir: Path) -> dict[str, np.ndarray]:
    volumes = {}
    for field in FIELDS:
        path = volume_dir / f"{field}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        volume = np.load(path, mmap_mode="r")
        if volume.ndim != 4 or volume.shape[0] != 2:
            raise ValueError(f"{path} must be [2,Z,X,Y], got {volume.shape}")
        volumes[field] = volume
    return volumes


def complex_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def abs_l1(a: np.ndarray, b: np.ndarray) -> float:
    a_abs = complex_abs_numpy(a)
    b_abs = complex_abs_numpy(b)
    return float(np.mean(np.abs(a_abs - b_abs)))


def x_boundary_abs_jumps(mag: np.ndarray, boundaries: Iterable[int]) -> np.ndarray:
    chunks = []
    for x in boundaries:
        if x <= 0 or x >= mag.shape[1]:
            raise ValueError(f"x boundary {x} out of bounds for magnitude shape {mag.shape}")
        chunks.append(np.abs(mag[:, x, :] - mag[:, x - 1, :]).reshape(-1))
    return np.concatenate(chunks)


def xz_y_boundary_abs_jumps(mag: np.ndarray, boundaries: Iterable[int], y_index: int) -> np.ndarray:
    chunks = []
    for x in boundaries:
        chunks.append(np.abs(mag[:, x, y_index] - mag[:, x - 1, y_index]).reshape(-1))
    return np.concatenate(chunks)


def nonboundary_mask(
    shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    margin: int,
) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    for axis, step in enumerate(patch_size):
        dim = shape[axis]
        for boundary in range(step, dim, step):
            lo = max(0, boundary - margin)
            hi = min(dim, boundary + margin)
            slices = [slice(None), slice(None), slice(None)]
            slices[axis] = slice(lo, hi)
            mask[tuple(slices)] = False
    return mask


def highpass7(mag: np.ndarray) -> np.ndarray:
    """Seven-point high-pass residual on the interior voxels."""
    center = mag[1:-1, 1:-1, 1:-1]
    local_mean = (
        center
        + mag[:-2, 1:-1, 1:-1]
        + mag[2:, 1:-1, 1:-1]
        + mag[1:-1, :-2, 1:-1]
        + mag[1:-1, 2:, 1:-1]
        + mag[1:-1, 1:-1, :-2]
        + mag[1:-1, 1:-1, 2:]
    ) / 7.0
    return center - local_mean


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def high_frequency_metrics(
    pred: np.ndarray,
    label: np.ndarray,
    baseline: np.ndarray,
    patch_size: tuple[int, int, int],
    boundary_margin: int,
) -> dict[str, float]:
    pred_abs = complex_abs_numpy(pred)
    label_abs = complex_abs_numpy(label)
    baseline_abs = complex_abs_numpy(baseline)

    mask = nonboundary_mask(label_abs.shape, patch_size=patch_size, margin=boundary_margin)
    mask_i = mask[1:-1, 1:-1, 1:-1]

    pred_hp = highpass7(pred_abs)[mask_i]
    label_hp = highpass7(label_abs)[mask_i]
    baseline_hp = highpass7(baseline_abs)[mask_i]

    eps = 1e-12
    label_hf_rms = rms(label_hp)
    pred_hf_rms = rms(pred_hp)
    baseline_hf_rms = rms(baseline_hp)
    pred_hf_l1 = float(np.mean(np.abs(pred_hp - label_hp)))
    baseline_hf_l1 = float(np.mean(np.abs(baseline_hp - label_hp)))

    label_detail = np.abs(label_hp)
    top_threshold = float(np.percentile(label_detail, 90.0))
    top_mask = label_detail >= top_threshold
    detail_retention_top10 = float(
        np.median(np.abs(pred_hp[top_mask]) / (np.abs(label_hp[top_mask]) + eps))
    )

    pred_nonboundary = pred_abs[mask]
    label_nonboundary = label_abs[mask]
    baseline_nonboundary = baseline_abs[mask]

    return {
        "nonboundary_voxels": int(np.count_nonzero(mask)),
        "label_hf_rms": label_hf_rms,
        "pred_hf_rms": pred_hf_rms,
        "baseline_hf_rms": baseline_hf_rms,
        "pred_hf_rms_over_label": pred_hf_rms / (label_hf_rms + eps),
        "baseline_hf_rms_over_label": baseline_hf_rms / (label_hf_rms + eps),
        "pred_hf_corr_label": corrcoef(pred_hp, label_hp),
        "baseline_hf_corr_label": corrcoef(baseline_hp, label_hp),
        "pred_hf_l1": pred_hf_l1,
        "baseline_hf_l1": baseline_hf_l1,
        "pred_hf_improvement_vs_baseline": 1.0 - pred_hf_l1 / (baseline_hf_l1 + eps),
        "detail_retention_top10": detail_retention_top10,
        "pred_nonboundary_abs_std_over_label": float(np.std(pred_nonboundary) / (np.std(label_nonboundary) + eps)),
        "baseline_nonboundary_abs_std_over_label": float(
            np.std(baseline_nonboundary) / (np.std(label_nonboundary) + eps)
        ),
    }


def masked_l1_metrics(
    pred: np.ndarray,
    label: np.ndarray,
    baseline: np.ndarray,
    patch_size: tuple[int, int, int],
    boundary_margin: int,
) -> dict[str, float]:
    mask = nonboundary_mask(label.shape[1:], patch_size=patch_size, margin=boundary_margin)
    pred_arr = np.asarray(pred, dtype=np.float32)
    label_arr = np.asarray(label, dtype=np.float32)
    baseline_arr = np.asarray(baseline, dtype=np.float32)

    pred_complex_l1 = float(np.mean(np.abs(pred_arr[:, mask] - label_arr[:, mask])))
    baseline_complex_l1 = float(np.mean(np.abs(baseline_arr[:, mask] - label_arr[:, mask])))

    pred_abs = complex_abs_numpy(pred)
    label_abs = complex_abs_numpy(label)
    baseline_abs = complex_abs_numpy(baseline)
    pred_abs_l1 = float(np.mean(np.abs(pred_abs[mask] - label_abs[mask])))
    baseline_abs_l1 = float(np.mean(np.abs(baseline_abs[mask] - label_abs[mask])))

    eps = 1e-12
    return {
        "nonboundary_complex_l1_pred_label": pred_complex_l1,
        "nonboundary_complex_l1_baseline_label": baseline_complex_l1,
        "nonboundary_complex_improvement": 1.0 - pred_complex_l1 / (baseline_complex_l1 + eps),
        "nonboundary_abs_l1_pred_label": pred_abs_l1,
        "nonboundary_abs_l1_baseline_label": baseline_abs_l1,
        "nonboundary_abs_improvement": 1.0 - pred_abs_l1 / (baseline_abs_l1 + eps),
    }


def metric_row(
    case: str,
    model: str,
    vols: dict[str, np.ndarray],
    x_boundaries: tuple[int, ...],
    y_index: int,
    patch_size: tuple[int, int, int],
    boundary_margin: int,
) -> dict[str, object]:
    label_mag = complex_abs_numpy(vols["label"])
    pred_mag = complex_abs_numpy(vols["pred"])
    baseline_mag = complex_abs_numpy(vols["baseline"])

    label_jump = x_boundary_abs_jumps(label_mag, x_boundaries)
    pred_jump = x_boundary_abs_jumps(pred_mag, x_boundaries)
    baseline_jump = x_boundary_abs_jumps(baseline_mag, x_boundaries)

    label_xz_jump = xz_y_boundary_abs_jumps(label_mag, x_boundaries, y_index=y_index)
    pred_xz_jump = xz_y_boundary_abs_jumps(pred_mag, x_boundaries, y_index=y_index)

    row: dict[str, object] = {
        "case": case,
        "model": model,
        "complex_l1_pred_label": complex_l1(vols["pred"], vols["label"]),
        "abs_l1_pred_label": abs_l1(vols["pred"], vols["label"]),
        "pred_abs_mean": float(np.mean(pred_mag)),
        "label_abs_mean": float(np.mean(label_mag)),
        "baseline_abs_mean": float(np.mean(baseline_mag)),
        "x_boundary_pred_abs_jump_mean": float(np.mean(pred_jump)),
        "x_boundary_label_abs_jump_mean": float(np.mean(label_jump)),
        "x_boundary_baseline_abs_jump_mean": float(np.mean(baseline_jump)),
        "x_boundary_pred_over_label": float(np.mean(pred_jump) / (np.mean(label_jump) + 1e-12)),
        "x_boundary_baseline_over_label": float(np.mean(baseline_jump) / (np.mean(label_jump) + 1e-12)),
        "xz_y_boundary_pred_abs_jump_mean": float(np.mean(pred_xz_jump)),
        "xz_y_boundary_label_abs_jump_mean": float(np.mean(label_xz_jump)),
        "xz_y_boundary_pred_over_label": float(np.mean(pred_xz_jump) / (np.mean(label_xz_jump) + 1e-12)),
    }
    row.update(masked_l1_metrics(vols["pred"], vols["label"], vols["baseline"], patch_size, boundary_margin))
    row.update(high_frequency_metrics(vols["pred"], vols["label"], vols["baseline"], patch_size, boundary_margin))
    return row


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to save.")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_full_xz(
    case: str,
    in_vols: dict[str, np.ndarray],
    bn_vols: dict[str, np.ndarray],
    out_dir: Path,
    x_boundaries: tuple[int, ...],
    y_index: int,
    db_min: float,
) -> Path:
    all_vols = [in_vols[field] for field in FIELDS] + [bn_vols[field] for field in FIELDS]
    ref = max(float(np.max(complex_abs_numpy(v))) for v in all_vols)
    rows = [("IN", in_vols), ("BN", bn_vols)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    im = None
    for r, (row_name, vols) in enumerate(rows):
        for c, field in enumerate(FIELDS):
            db = volume_to_db(complex_abs_numpy(vols[field]), ref=ref, db_min=db_min)
            img, used, xlabel, ylabel = extract_slice(db, view="xz", slice_index=y_index)
            ax = axes[r, c]
            im = ax.imshow(img, cmap="gray", vmin=db_min, vmax=0.0, origin="upper", aspect=ASPECT_XZ)
            for x in x_boundaries:
                ax.axvline(x - 0.5, color="tab:red", linewidth=0.7, alpha=0.65)
            ax.set_title(f"{case} | {row_name} {field} | xz y={used}", fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="dB")
    fig.suptitle(
        f"{case} | IN vs BN | shared dB ref | red x-boundaries | db_min={db_min:g}",
        fontsize=11,
    )
    path = out_dir / f"{case}_IN_vs_BN_full_xz_y{y_index}_boundaries.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_stress_pred(
    case: str,
    in_vols: dict[str, np.ndarray],
    bn_vols: dict[str, np.ndarray],
    out_dir: Path,
    x_boundaries: tuple[int, ...],
    y_index: int,
    stress_db_min: float,
) -> Path:
    items = [("IN pred self ref", in_vols["pred"]), ("BN pred self ref", bn_vols["pred"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.5), constrained_layout=True)
    im = None
    for ax, (title, vol) in zip(axes, items):
        mag = complex_abs_numpy(vol)
        ref = float(np.max(mag))
        db = volume_to_db(mag, ref=ref, db_min=stress_db_min)
        img, used, xlabel, ylabel = extract_slice(db, view="xz", slice_index=y_index)
        im = ax.imshow(img, cmap="gray", vmin=stress_db_min, vmax=0.0, origin="upper", aspect=ASPECT_XZ)
        for x in x_boundaries:
            ax.axvline(x - 0.5, color="tab:red", linewidth=0.7, alpha=0.75)
        ax.set_title(f"{case} | {title} | ref={ref:.2e} | xz y={used}", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="dB")
    fig.suptitle(
        f"{case} | pred-only stress display | per-model ref | db_min={stress_db_min:g}",
        fontsize=11,
    )
    path = out_dir / f"{case}_IN_vs_BN_pred_selfref_dbmin{abs(int(stress_db_min))}_xz_y{y_index}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_boundary_profiles(
    case: str,
    in_vols: dict[str, np.ndarray],
    bn_vols: dict[str, np.ndarray],
    out_dir: Path,
    x_boundaries: tuple[int, ...],
    y_index: int,
    db_min: float,
) -> Path:
    fields = [
        ("baseline", in_vols["baseline"], "0.35"),
        ("label", in_vols["label"], "black"),
        ("IN pred", in_vols["pred"], "tab:orange"),
        ("BN pred", bn_vols["pred"], "tab:blue"),
    ]
    ref = max(float(np.max(complex_abs_numpy(volume))) for _, volume, _ in fields)
    z_band = slice(150, 650)

    fig, axes = plt.subplots(1, len(x_boundaries), figsize=(4.8 * len(x_boundaries), 3.8), constrained_layout=True)
    if len(x_boundaries) == 1:
        axes = [axes]
    for ax, x in zip(axes, x_boundaries):
        xs = np.arange(max(0, x - 16), min(128, x + 17))
        for name, volume, color in fields:
            mag = complex_abs_numpy(volume)
            profile = np.mean(mag[z_band, xs, y_index], axis=0)
            db = 20.0 * np.log10(np.maximum(profile, 1e-12) / max(ref, 1e-12))
            ax.plot(xs, np.maximum(db, db_min), label=name, color=color, linewidth=1.5)
        ax.axvline(x - 0.5, color="tab:red", linewidth=0.9, alpha=0.8)
        ax.set_title(f"x={x}, z=150..649, y={y_index}")
        ax.set_xlabel("x index")
        ax.set_ylabel("dB")
        ax.set_ylim(db_min, 0)
        ax.grid(True, alpha=0.25)
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{case} | boundary-crossing magnitude profiles | shared ref", fontsize=11)
    path = out_dir / f"{case}_IN_vs_BN_x_boundary_profiles_y{y_index}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_case(args: argparse.Namespace) -> None:
    in_dir = Path(args.in_dir)
    bn_dir = Path(args.bn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_boundaries = tuple(int(v) for v in args.x_boundaries)
    patch_size = tuple(int(v) for v in args.patch_size)

    in_vols = load_volumes(in_dir)
    bn_vols = load_volumes(bn_dir)
    if in_vols["pred"].shape != bn_vols["pred"].shape:
        raise ValueError(f"IN/BN shape mismatch: {in_vols['pred'].shape} vs {bn_vols['pred'].shape}")

    rows = [
        metric_row(args.case, "IN", in_vols, x_boundaries, args.y_index, patch_size, args.boundary_margin),
        metric_row(args.case, "BN", bn_vols, x_boundaries, args.y_index, patch_size, args.boundary_margin),
    ]
    metrics_path = out_dir / f"{args.case}_bn_multivolume_diagnostics.csv"
    save_csv(metrics_path, rows)

    figures = [
        make_full_xz(args.case, in_vols, bn_vols, out_dir, x_boundaries, args.y_index, args.db_min),
        make_stress_pred(args.case, in_vols, bn_vols, out_dir, x_boundaries, args.y_index, args.stress_db_min),
        make_boundary_profiles(args.case, in_vols, bn_vols, out_dir, x_boundaries, args.y_index, args.db_min),
    ]

    print(f"metrics_csv,{metrics_path}")
    for path in figures:
        print(f"figure,{path}")
    print("case,model,x_boundary_pred_over_label,nonboundary_complex_improvement,nonboundary_abs_improvement,pred_hf_rms_over_label,pred_hf_corr_label,detail_retention_top10,pred_nonboundary_abs_std_over_label,pred_hf_improvement_vs_baseline")
    for row in rows:
        print(
            f"{row['case']},{row['model']},"
            f"{row['x_boundary_pred_over_label']:.6f},"
            f"{row['nonboundary_complex_improvement']:.6f},"
            f"{row['nonboundary_abs_improvement']:.6f},"
            f"{row['pred_hf_rms_over_label']:.6f},"
            f"{row['pred_hf_corr_label']:.6f},"
            f"{row['detail_retention_top10']:.6f},"
            f"{row['pred_nonboundary_abs_std_over_label']:.6f},"
            f"{row['pred_hf_improvement_vs_baseline']:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose IN-vs-BN seam removal and over-smoothing per dense volume.")
    parser.add_argument("--case", required=True, help="Case id used in CSV rows and figure names.")
    parser.add_argument("--in-dir", required=True, help="IN/Tiny full-volume directory with pred/label/baseline.npy.")
    parser.add_argument("--bn-dir", required=True, help="BN full-volume directory with pred/label/baseline.npy.")
    parser.add_argument("--out-dir", required=True, help="Output directory for metrics CSV and figures.")
    parser.add_argument("--x-boundaries", nargs="+", type=int, default=list(DEFAULT_X_BOUNDARIES))
    parser.add_argument("--patch-size", nargs=3, type=int, default=list(DEFAULT_PATCH_SIZE))
    parser.add_argument("--boundary-margin", type=int, default=3)
    parser.add_argument("--y-index", type=int, default=64)
    parser.add_argument("--db-min", type=float, default=DEFAULT_DB_MIN)
    parser.add_argument("--stress-db-min", type=float, default=DEFAULT_STRESS_DB_MIN)
    return parser.parse_args()


if __name__ == "__main__":
    run_case(parse_args())
