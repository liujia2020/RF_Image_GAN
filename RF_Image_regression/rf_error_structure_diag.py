from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rf_cached_dataset import RFCachedDataset
from rf_models import build_model


TASK_NAME = "RF error-structure diagnostic (read-only)"
EXPERIMENT = "tiny_bn_random64_full1500"
ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoint" / EXPERIMENT / "best_model.pth"
CACHE_DIR = ROOT / "Data_cache_random64_full1500" / "test"
METRICS_CSV = ROOT / "test_metrics" / EXPERIMENT / "test_per_sample_metrics.csv"
OUT_DIR = ROOT / "test_metrics" / EXPERIMENT / "error_structure"

ASPECT_XZ = 0.0362 / 0.2
DB_MIN = -60.0
LAGS = range(1, 6)


def print_header() -> None:
    print("=" * 96)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"task: {TASK_NAME}")
    print(f"experiment: {EXPERIMENT}")
    print("=" * 96)


def report_missing(path: Path) -> bool:
    if path.exists():
        return False
    print(f"MISSING: {path}")
    parent = path.parent
    if parent.exists():
        print(f"ls {parent}:")
        for item in sorted(parent.iterdir(), key=lambda p: p.name):
            suffix = "/" if item.is_dir() else ""
            size = "" if item.is_dir() else f" {item.stat().st_size}"
            print(f"  {item.name}{suffix}{size}")
    else:
        print(f"MISSING DIR: {parent}")
    return True


def load_bn_model(device: torch.device) -> torch.nn.Module | None:
    if report_missing(CKPT_PATH):
        return None

    ckpt = torch.load(CKPT_PATH, map_location=device)
    state_dict = ckpt["model"]
    use_bn = any("running_mean" in key for key in state_dict)
    model = build_model(
        "tiny",
        in_channels=1536,
        hidden=64,
        out_channels=2,
        use_batch_norm=use_bn,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    first_norm = next((module for module in model.modules() if isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    first_norm_name = type(first_norm).__name__ if first_norm is not None else "MISSING"
    print(f"startup model_class={type(model).__name__}")
    print(f"startup first_norm_class={first_norm_name}")
    assert isinstance(first_norm, nn.BatchNorm3d), f"Expected first norm to be BatchNorm3d, got {first_norm_name}"
    return model


def select_samples(metrics: pd.DataFrame) -> pd.DataFrame:
    worst = metrics.sort_values("complex_improvement", ascending=True).head(3).copy()
    best = metrics.sort_values("complex_improvement", ascending=False).head(3).copy()
    median_value = float(metrics["complex_improvement"].median())
    used_paths = set(worst["path"]) | set(best["path"])
    median_candidates = metrics[~metrics["path"].isin(used_paths)].copy()
    median = (
        median_candidates.assign(_median_distance=(median_candidates["complex_improvement"] - median_value).abs())
        .sort_values(["_median_distance", "complex_improvement"], ascending=[True, True])
        .head(3)
        .drop(columns=["_median_distance"])
        .copy()
    )
    worst["group"] = "worst"
    median["group"] = "median"
    best["group"] = "best"
    selected = pd.concat([worst, median, best], ignore_index=True)
    return selected


def dataset_index_by_path(dataset: RFCachedDataset, path: str) -> int:
    if not hasattr(dataset, "_path_to_idx"):
        exact = {p: i for i, p in enumerate(dataset.paths)}
        by_name = {Path(p).name: i for i, p in enumerate(dataset.paths)}
        dataset._path_to_idx = (exact, by_name)  # type: ignore[attr-defined]
    exact, by_name = dataset._path_to_idx  # type: ignore[attr-defined]
    if path in exact:
        return int(exact[path])
    name = Path(path).name
    if name in by_name:
        return int(by_name[name])
    raise KeyError(f"Sample path not found in cache metadata: {path}")


def complex_abs(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(arr[0] ** 2 + arr[1] ** 2)


def to_db(mag: np.ndarray, ref: float, db_min: float = DB_MIN) -> np.ndarray:
    db = 20.0 * np.log10(np.maximum(mag.astype(np.float32), 1e-12) / max(float(ref), 1e-12))
    return np.maximum(db, db_min)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(af) & np.isfinite(bf)
    af = af[mask]
    bf = bf[mask]
    if af.size == 0:
        return float("nan")
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = float(np.sqrt(np.mean(af * af)) * np.sqrt(np.mean(bf * bf)))
    if denom <= 0:
        return float("nan")
    return float(np.mean(af * bf) / denom)


def autocorr_axis(real_error: np.ndarray, axis: int, lag: int) -> float:
    slices_a = [slice(None)] * real_error.ndim
    slices_b = [slice(None)] * real_error.ndim
    slices_a[axis] = slice(0, -lag)
    slices_b[axis] = slice(lag, None)
    return pearson(real_error[tuple(slices_a)], real_error[tuple(slices_b)])


def gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    grads = np.gradient(volume.astype(np.float64), edge_order=1)
    out = np.zeros_like(volume, dtype=np.float64)
    for grad in grads:
        out += grad * grad
    return np.sqrt(out)


def rms_complex(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr[0] ** 2 + arr[1] ** 2)))


def save_triptych(
    group: str,
    rank: int,
    sample_index: int,
    path: str,
    category: str,
    label_raw: np.ndarray,
    pred_raw: np.ndarray,
    error: np.ndarray,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_mid = label_raw.shape[-1] // 2
    label_mag = complex_abs(label_raw)
    pred_mag = complex_abs(pred_raw)
    error_real_mag = np.abs(error[0])
    ref = max(float(label_mag.max()), float(pred_mag.max()), float(error_real_mag.max()))

    panels = [
        ("label", to_db(label_mag[:, :, y_mid], ref)),
        ("BN pred", to_db(pred_mag[:, :, y_mid], ref)),
        ("e_real", to_db(error_real_mag[:, :, y_mid], ref)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    image = None
    for ax, (title, img) in zip(axes, panels):
        image = ax.imshow(img, cmap="gray", vmin=DB_MIN, vmax=0.0, origin="upper", aspect=ASPECT_XZ)
        ax.set_title(f"{title} | xz y={y_mid}")
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="dB")
    fig.suptitle(f"{group} rank={rank} idx={sample_index} category={category} | {Path(path).name}", fontsize=10)
    save_path = OUT_DIR / f"{group}_{rank:02d}_idx{sample_index:03d}_{category}_{Path(path).stem}_xz_y{y_mid}.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def infer_patch(model: torch.nn.Module, sample: dict[str, object], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)  # type: ignore[index]
    label = sample["label"].unsqueeze(0).to(device=device, dtype=torch.float32)  # type: ignore[index]
    baseline = sample["baseline"].unsqueeze(0).to(device=device, dtype=torch.float32)  # type: ignore[index]
    scale = sample["scale"].to(device=device, dtype=torch.float32)  # type: ignore[index]
    while scale.ndim < label.ndim:
        scale = scale.view(*scale.shape, *([1] * (label.ndim - scale.ndim)))

    with torch.no_grad():
        pred = model(x, baseline)

    pred_raw = (pred * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    label_raw = (label * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_raw = (baseline * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return pred_raw, label_raw, baseline_raw


def compute_metrics(label_raw: np.ndarray, pred_raw: np.ndarray, baseline_raw: np.ndarray) -> dict[str, object]:
    error = pred_raw - label_raw
    real_error = error[0]
    error_mag = complex_abs(error)
    label_mag = complex_abs(label_raw)
    baseline_delta = label_raw - baseline_raw
    mag_error = complex_abs(pred_raw) - label_mag
    label_grad = gradient_magnitude(label_mag)

    z_autocorr = [autocorr_axis(real_error, axis=0, lag=lag) for lag in LAGS]
    x_autocorr = [autocorr_axis(real_error, axis=1, lag=lag) for lag in LAGS]
    rms_e = rms_complex(error)
    rms_label_minus_baseline = rms_complex(baseline_delta)
    rms_label = rms_complex(label_raw)

    return {
        "z_autocorr": z_autocorr,
        "x_autocorr": x_autocorr,
        "corr_abs_e_abs_label": pearson(error_mag, label_mag),
        "corr_abs_e_label_grad": pearson(error_mag, label_grad),
        "rms_e_over_rms_label_minus_baseline": rms_e / max(rms_label_minus_baseline, 1e-12),
        "rms_e_over_rms_label": rms_e / max(rms_label, 1e-12),
        "rms_d": float(np.sqrt(np.mean(mag_error * mag_error))),
    }


def fmt_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def main() -> None:
    print_header()
    if report_missing(METRICS_CSV) or report_missing(CACHE_DIR / "meta.npz"):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = load_bn_model(device)
    if model is None:
        return

    metrics = pd.read_csv(METRICS_CSV)
    selected = select_samples(metrics)
    print("selected_samples:")
    print(selected[["group", "path", "category", "complex_improvement"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    dataset = RFCachedDataset(CACHE_DIR)
    rows: list[dict[str, object]] = []

    for local_rank, (_, row) in enumerate(selected.iterrows(), start=1):
        sample_index = dataset_index_by_path(dataset, str(row["path"]))
        sample = dataset[sample_index]
        pred_raw, label_raw, baseline_raw = infer_patch(model, sample, device)
        error = pred_raw - label_raw
        m = compute_metrics(label_raw, pred_raw, baseline_raw)
        save_path = save_triptych(
            group=str(row["group"]),
            rank=local_rank,
            sample_index=sample_index,
            path=str(row["path"]),
            category=str(row["category"]),
            label_raw=label_raw,
            pred_raw=pred_raw,
            error=error,
        )
        out_row = {
            "group": str(row["group"]),
            "sample_index": sample_index,
            "path": str(row["path"]),
            "category": str(row["category"]),
            "complex_improvement": float(row["complex_improvement"]),
            **m,
            "figure": str(save_path),
        }
        rows.append(out_row)

        print("-" * 96)
        print(
            f"patch group={out_row['group']} sample_index={sample_index} "
            f"category={out_row['category']} complex_improvement={out_row['complex_improvement']:.6f}"
        )
        print(f"path={out_row['path']}")
        print(f"z_autocorr_lag1_5={fmt_list(out_row['z_autocorr'])}")  # type: ignore[arg-type]
        print(f"x_autocorr_lag1_5={fmt_list(out_row['x_autocorr'])}")  # type: ignore[arg-type]
        print(f"corr_abs_e_abs_label={out_row['corr_abs_e_abs_label']:.6f}")  # type: ignore[operator]
        print(f"corr_abs_e_label_grad={out_row['corr_abs_e_label_grad']:.6f}")  # type: ignore[operator]
        print(f"rms_e_over_rms_label_minus_baseline={out_row['rms_e_over_rms_label_minus_baseline']:.6f}")  # type: ignore[operator]
        print(f"rms_e_over_rms_label={out_row['rms_e_over_rms_label']:.6f}")  # type: ignore[operator]
        print(f"rms_d={out_row['rms_d']:.6f}")  # type: ignore[operator]
        print(f"figure={out_row['figure']}")

    print("\n" + "=" * 96)
    print("group_means")
    print("=" * 96)
    for group in ("worst", "median", "best"):
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            print(f"group={group} n=0")
            continue
        z_mean = np.mean(np.array([r["z_autocorr"] for r in group_rows], dtype=np.float64), axis=0)
        x_mean = np.mean(np.array([r["x_autocorr"] for r in group_rows], dtype=np.float64), axis=0)
        print(f"group={group} n={len(group_rows)}")
        print(f"  z_autocorr_lag1_5_mean={fmt_list(z_mean.tolist())}")
        print(f"  x_autocorr_lag1_5_mean={fmt_list(x_mean.tolist())}")
        for key in (
            "corr_abs_e_abs_label",
            "corr_abs_e_label_grad",
            "rms_e_over_rms_label_minus_baseline",
            "rms_e_over_rms_label",
            "rms_d",
        ):
            value = float(np.mean([float(r[key]) for r in group_rows]))
            print(f"  {key}_mean={value:.6f}")


if __name__ == "__main__":
    main()
