from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rf_cached_dataset import RFCachedDataset
from rf_models import build_model


TASK_NAME = "RF SSIM-vs-baseline reality check (2d-hist + error autocorr)"
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "Data_cache_random64_full1500" / "test"
METRICS_CSV = ROOT / "test_metrics" / "tiny_bn_random64_full1500" / "test_per_sample_metrics.csv"
BASELINE_CKPT = ROOT / "checkpoint" / "tiny_bn_random64_full1500" / "best_model.pth"
SSIM_CKPT = ROOT / "checkpoint" / "tiny_bn_ssim10_random64_full1500" / "best_by_val_ssim.pth"
OUT_DIR = ROOT / "test_metrics" / "ssim_check"
LAGS = range(1, 6)
FIXED_TEST_INDEX = 37
FIXED_TEST_BASENAME = "RF000493_carotid_test_patch003.h5"
ASPECT_XZ = 0.0362 / 0.2


def print_header() -> None:
    print("=" * 96)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"task: {TASK_NAME}")
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


def load_bn_model(name: str, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    if report_missing(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
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

    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    first_norm_name = type(first_norm).__name__ if first_norm is not None else "MISSING"
    print(f"startup {name} model_class={type(model).__name__}")
    print(f"startup {name} first_norm_class={first_norm_name}")
    print(f"startup {name} ckpt_path={ckpt_path}")
    print(f"startup {name} epoch={ckpt.get('epoch', 'NA')}")
    assert isinstance(first_norm, nn.BatchNorm3d), f"{name}: Expected BatchNorm3d, got {first_norm_name}"
    return model


def complex_abs(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(arr[0] ** 2 + arr[1] ** 2)


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


def regression_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xf = np.asarray(x, dtype=np.float64).reshape(-1)
    yf = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xf) & np.isfinite(yf)
    xf = xf[mask]
    yf = yf[mask]
    x_mean = float(xf.mean())
    y_mean = float(yf.mean())
    var_x = float(np.mean((xf - x_mean) ** 2))
    if var_x <= 0:
        return float("nan"), float("nan")
    slope = float(np.mean((xf - x_mean) * (yf - y_mean)) / var_x)
    intercept = y_mean - slope * x_mean
    return slope, float(intercept)


def autocorr_axis(real_error: np.ndarray, axis: int, lag: int) -> float:
    slices_a = [slice(None)] * real_error.ndim
    slices_b = [slice(None)] * real_error.ndim
    slices_a[axis] = slice(0, -lag)
    slices_b[axis] = slice(lag, None)
    return pearson(real_error[tuple(slices_a)], real_error[tuple(slices_b)])


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


@torch.no_grad()
def infer_one_raw(model: torch.nn.Module, sample: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
    label = sample["label"].unsqueeze(0).to(device=device, dtype=torch.float32)
    baseline = sample["baseline"].unsqueeze(0).to(device=device, dtype=torch.float32)
    scale = sample["scale"].to(device=device, dtype=torch.float32)
    while scale.ndim < label.ndim:
        scale = scale.view(*scale.shape, *([1] * (label.ndim - scale.ndim)))
    pred = model(x, baseline)
    pred_raw = (pred * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    label_raw = (label * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_raw = (baseline * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return pred_raw, label_raw, baseline_raw


def select_fixed_nine(metrics: pd.DataFrame) -> pd.DataFrame:
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
    selected["rank"] = np.arange(1, len(selected) + 1)
    return selected


def collect_hist_data(
    model: torch.nn.Module,
    dataset: RFCachedDataset,
    device: torch.device,
    label_reference: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    labels: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        pred_raw, label_raw, _baseline_raw = infer_one_raw(model, sample, device)
        label_mag = complex_abs(label_raw)
        pred_mag = complex_abs(pred_raw)
        mask = label_mag >= float(np.median(label_mag))
        labels.append(label_mag[mask].astype(np.float32, copy=False))
        preds.append(pred_mag[mask].astype(np.float32, copy=False))
        if (idx + 1) % 25 == 0 or idx == len(dataset) - 1:
            print(f"  hist_inference={idx + 1}/{len(dataset)}")

    x = np.concatenate(labels)
    y = np.concatenate(preds)
    slope, intercept = regression_slope_intercept(x, y)
    stats = {
        "slope": slope,
        "intercept": intercept,
        "pearson_r": pearson(x, y),
        "n_voxels": float(x.size),
        "label_p99": float(np.percentile(x, 99)),
        "pred_p99": float(np.percentile(y, 99)),
    }
    return x, y, stats


def save_hist2d(
    baseline_xy: tuple[np.ndarray, np.ndarray],
    baseline_stats: dict[str, float],
    ssim_xy: tuple[np.ndarray, np.ndarray],
    ssim_stats: dict[str, float],
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bx, by = baseline_xy
    sx, sy = ssim_xy
    vmax = float(np.percentile(np.concatenate([bx, by, sx, sy]), 99.5))
    bins = 220
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True, sharex=True, sharey=True)
    for ax, title, x, y, stats in [
        (axes[0], "baseline aw=0.1", bx, by, baseline_stats),
        (axes[1], "SSIM best_by_val_ssim", sx, sy, ssim_stats),
    ]:
        image = ax.hist2d(x, y, bins=bins, range=[[0, vmax], [0, vmax]], norm=matplotlib.colors.LogNorm(), cmap="magma")
        line_x = np.linspace(0, vmax, 300)
        line_y = stats["slope"] * line_x + stats["intercept"]
        ax.plot(line_x, line_x, color="white", linewidth=1.0, linestyle="--", label="y=x")
        ax.plot(line_x, line_y, color="cyan", linewidth=1.2, label="fit")
        ax.set_title(
            f"{title}\n"
            f"slope={stats['slope']:.4f}, intercept={stats['intercept']:.1f}, r={stats['pearson_r']:.4f}"
        )
        ax.set_xlabel("|label|")
        ax.set_ylabel("|pred|")
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.legend(loc="upper left")
    fig.colorbar(image[3], ax=axes.ravel().tolist(), shrink=0.85, label="voxel count")
    path = OUT_DIR / "hist2d.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_linear_triptych(
    baseline_model: torch.nn.Module,
    ssim_model: torch.nn.Module,
    dataset: RFCachedDataset,
    device: torch.device,
) -> Path:
    sample = dataset[FIXED_TEST_INDEX]
    actual_name = Path(str(sample["path"])).name
    assert actual_name == FIXED_TEST_BASENAME, (
        f"Fixed sample mismatch: idx={FIXED_TEST_INDEX}, actual={actual_name}, expected={FIXED_TEST_BASENAME}"
    )
    baseline_pred, label_raw, _baseline_raw = infer_one_raw(baseline_model, sample, device)
    ssim_pred, _label_raw_2, _baseline_raw_2 = infer_one_raw(ssim_model, sample, device)
    label_mag = complex_abs(label_raw)
    baseline_mag = complex_abs(baseline_pred)
    ssim_mag = complex_abs(ssim_pred)
    y_idx = 16
    vmax = float(np.percentile(np.concatenate([label_mag.reshape(-1), baseline_mag.reshape(-1), ssim_mag.reshape(-1)]), 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    image = None
    panels = [
        ("label", label_mag[:, :, y_idx]),
        ("baseline pred", baseline_mag[:, :, y_idx]),
        ("SSIM pred", ssim_mag[:, :, y_idx]),
    ]
    for ax, (title, panel) in zip(axes, panels):
        image = ax.imshow(panel, cmap="gray", vmin=0, vmax=vmax, origin="upper", aspect=ASPECT_XZ)
        ax.set_title(f"{title} | linear | xz y={y_idx}")
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="amplitude")
    fig.suptitle(f"Fixed worst carotid patch idx={FIXED_TEST_INDEX} | {actual_name}", fontsize=10)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "linear_triptych_idx037.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def print_autocorr_table(
    selected: pd.DataFrame,
    baseline_model: torch.nn.Module,
    ssim_model: torch.nn.Module,
    dataset: RFCachedDataset,
    device: torch.device,
) -> None:
    rows = []
    for _, row in selected.iterrows():
        idx = dataset_index_by_path(dataset, str(row["path"]))
        sample = dataset[idx]
        b_pred, label_raw, _base = infer_one_raw(baseline_model, sample, device)
        s_pred, _label2, _base2 = infer_one_raw(ssim_model, sample, device)
        for name, pred in [("baseline", b_pred), ("ssim", s_pred)]:
            real_error = (pred - label_raw)[0]
            z_autocorr = [autocorr_axis(real_error, axis=0, lag=lag) for lag in LAGS]
            rows.append(
                {
                    "group": row["group"],
                    "rank": int(row["rank"]),
                    "model": name,
                    "cache_index": idx,
                    "category": str(row["category"]),
                    "complex_improvement": float(row["complex_improvement"]),
                    "path": str(row["path"]),
                    **{f"z_lag{lag}": z_autocorr[i] for i, lag in enumerate(LAGS)},
                }
            )

    table = pd.DataFrame(rows)
    print("\nerror_z_autocorr_fixed9:")
    cols = ["group", "rank", "model", "cache_index", "category", "complex_improvement", "z_lag1", "z_lag2", "z_lag3", "z_lag4", "z_lag5", "path"]
    print(table[cols].to_string(index=False, float_format="{:.6f}".format))
    print("\nerror_z_autocorr_group_means:")
    mean_cols = [f"z_lag{lag}" for lag in LAGS]
    means = table.groupby(["group", "model"], sort=True)[mean_cols].mean()
    print(means.to_string(float_format="{:.6f}".format))
    table.to_csv(OUT_DIR / "error_z_autocorr_fixed9.csv", index=False)


def main() -> None:
    print_header()
    missing = report_missing(CACHE_DIR / "meta.npz") | report_missing(METRICS_CSV)
    if missing:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    dataset = RFCachedDataset(CACHE_DIR)
    baseline_model = load_bn_model("baseline", BASELINE_CKPT, device)
    ssim_model = load_bn_model("ssim", SSIM_CKPT, device)

    print("\ncollect_hist baseline")
    bx, by, bstats = collect_hist_data(baseline_model, dataset, device)
    print("baseline_hist_stats:", {k: round(v, 6) for k, v in bstats.items()})
    print("\ncollect_hist ssim")
    sx, sy, sstats = collect_hist_data(ssim_model, dataset, device)
    print("ssim_hist_stats:", {k: round(v, 6) for k, v in sstats.items()})
    hist_path = save_hist2d((bx, by), bstats, (sx, sy), sstats)
    print(f"hist2d_path={hist_path}")

    metrics = pd.read_csv(METRICS_CSV)
    selected = select_fixed_nine(metrics)
    print("\nselected_fixed9:")
    print(selected[["group", "rank", "path", "category", "complex_improvement"]].to_string(index=False))
    print_autocorr_table(selected, baseline_model, ssim_model, dataset, device)

    triptych_path = save_linear_triptych(baseline_model, ssim_model, dataset, device)
    print(f"linear_triptych_path={triptych_path}")


if __name__ == "__main__":
    main()
