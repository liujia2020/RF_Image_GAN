from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rf_cached_dataset import RFCachedDataset
from rf_models import build_model


TASK_NAME = "RF residual gain/phase decomposition diagnostic (read-only)"
EXPERIMENT = "tiny_bn_random64_full1500"
ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoint" / EXPERIMENT / "best_model.pth"
CACHE_DIR = ROOT / "Data_cache_random64_full1500" / "test"
METRICS_CSV = ROOT / "test_metrics" / EXPERIMENT / "test_per_sample_metrics.csv"
EPS = 1e-8


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

    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    first_norm_name = type(first_norm).__name__ if first_norm is not None else "MISSING"
    print(f"startup model_class={type(model).__name__}")
    print(f"startup first_norm_class={first_norm_name}")
    assert isinstance(first_norm, nn.BatchNorm3d), f"Expected first norm to be BatchNorm3d, got {first_norm_name}"
    return model


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


def assign_thirds(metrics: pd.DataFrame) -> pd.DataFrame:
    sorted_df = metrics.sort_values("complex_improvement", ascending=True).reset_index(drop=True).copy()
    n = len(sorted_df)
    if n % 3 != 0:
        print(f"WARNING: sample count {n} is not divisible by 3; using integer thirds")
    first = n // 3
    second = (2 * n) // 3
    groups = np.empty(n, dtype=object)
    groups[:first] = "worst"
    groups[first:second] = "median"
    groups[second:] = "best"
    sorted_df["group"] = groups
    sorted_df["rank_in_sorted"] = np.arange(n, dtype=np.int32)
    return sorted_df


def infer_patch(
    model: torch.nn.Module,
    sample: dict[str, object],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
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
    return pred_raw, label_raw


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
    if xf.size == 0:
        return float("nan"), float("nan")
    x_mean = float(xf.mean())
    y_mean = float(yf.mean())
    var_x = float(np.mean((xf - x_mean) ** 2))
    if var_x <= 0:
        return float("nan"), float("nan")
    slope = float(np.mean((xf - x_mean) * (yf - y_mean)) / var_x)
    intercept = y_mean - slope * x_mean
    return slope, float(intercept)


def rms(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals)))


def compute_decomp_metrics(pred_raw: np.ndarray, label_raw: np.ndarray) -> dict[str, float]:
    label_mag = np.sqrt(label_raw[0] ** 2 + label_raw[1] ** 2).astype(np.float64, copy=False)
    pred_mag = np.sqrt(pred_raw[0] ** 2 + pred_raw[1] ** 2).astype(np.float64, copy=False)
    threshold = float(np.median(label_mag))
    mask = label_mag >= threshold

    er = (pred_raw[0] - label_raw[0]).astype(np.float64, copy=False)
    ei = (pred_raw[1] - label_raw[1]).astype(np.float64, copy=False)
    lhat_r = label_raw[0].astype(np.float64, copy=False) / (label_mag + EPS)
    lhat_i = label_raw[1].astype(np.float64, copy=False) / (label_mag + EPS)

    e_par = er * lhat_r + ei * lhat_i
    e_sq = er * er + ei * ei
    e_perp = np.sqrt(np.maximum(e_sq - e_par * e_par, 0.0))

    e_par_m = e_par[mask]
    e_perp_m = e_perp[mask]
    label_mag_m = label_mag[mask]
    pred_mag_m = pred_mag[mask]

    e_par_rms = rms(e_par_m)
    e_perp_rms = rms(e_perp_m)
    label_mag_rms = rms(label_mag_m)
    slope, intercept = regression_slope_intercept(label_mag_m, pred_mag_m)

    return {
        "n_high_signal_voxels": float(np.count_nonzero(mask)),
        "label_mag_median_threshold": threshold,
        "rms_e_perp_over_rms_e_par": e_perp_rms / max(e_par_rms, EPS),
        "mean_e_par_over_rms_label_mag": float(np.mean(e_par_m)) / max(label_mag_rms, EPS),
        "amp_regression_slope": slope,
        "amp_regression_intercept": intercept,
        "corr_e_par_abs_label": pearson(e_par_m, label_mag_m),
        "corr_e_perp_abs_label": pearson(e_perp_m, label_mag_m),
    }


def print_group_ranges(grouped_metrics: pd.DataFrame) -> None:
    print("group_definition:")
    for group in ["worst", "median", "best"]:
        part = grouped_metrics[grouped_metrics["group"] == group]
        print(
            f"  {group}: n={len(part)} "
            f"complex_improvement_min={part['complex_improvement'].min():.6f} "
            f"p50={part['complex_improvement'].median():.6f} "
            f"max={part['complex_improvement'].max():.6f}"
        )


def print_patch_rows(rows: pd.DataFrame) -> None:
    columns = [
        "group",
        "category",
        "cache_index",
        "complex_improvement",
        "rms_e_perp_over_rms_e_par",
        "mean_e_par_over_rms_label_mag",
        "amp_regression_slope",
        "amp_regression_intercept",
        "corr_e_par_abs_label",
        "corr_e_perp_abs_label",
        "path",
    ]
    print("per_patch_metrics:")
    print(rows[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_aggregates(rows: pd.DataFrame, by: str) -> None:
    metric_cols = [
        "rms_e_perp_over_rms_e_par",
        "mean_e_par_over_rms_label_mag",
        "amp_regression_slope",
        "amp_regression_intercept",
        "corr_e_par_abs_label",
        "corr_e_perp_abs_label",
        "n_high_signal_voxels",
        "label_mag_median_threshold",
    ]
    grouped = rows.groupby(by, sort=True)[metric_cols].mean()
    counts = rows.groupby(by, sort=True).size().rename("n")
    out = pd.concat([counts, grouped], axis=1)
    print(f"{by}_means:")
    print(out.to_string(float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    print_header()
    missing = report_missing(METRICS_CSV) | report_missing(CACHE_DIR / "meta.npz")
    if missing:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = load_bn_model(device)
    if model is None:
        return

    metrics = pd.read_csv(METRICS_CSV)
    required = {"path", "category", "complex_improvement"}
    missing_cols = sorted(required - set(metrics.columns))
    if missing_cols:
        raise KeyError(f"Missing metrics CSV columns: {missing_cols}")

    grouped_metrics = assign_thirds(metrics)
    print_group_ranges(grouped_metrics)

    dataset = RFCachedDataset(CACHE_DIR)
    rows: list[dict[str, object]] = []
    print("inference_progress:")
    for i, metric_row in grouped_metrics.iterrows():
        cache_index = dataset_index_by_path(dataset, str(metric_row["path"]))
        sample = dataset[cache_index]
        pred_raw, label_raw = infer_patch(model, sample, device)
        decomp = compute_decomp_metrics(pred_raw, label_raw)
        row = {
            "group": metric_row["group"],
            "rank_in_sorted": int(metric_row["rank_in_sorted"]),
            "cache_index": cache_index,
            "path": str(metric_row["path"]),
            "category": str(metric_row["category"]),
            "complex_improvement": float(metric_row["complex_improvement"]),
            **decomp,
        }
        rows.append(row)
        if (i + 1) % 25 == 0 or i == len(grouped_metrics) - 1:
            print(f"  processed={i + 1}/{len(grouped_metrics)}")

    results = pd.DataFrame(rows)
    print("-" * 96)
    print_patch_rows(results)
    print("-" * 96)
    print_aggregates(results, "group")
    print("-" * 96)
    print_aggregates(results, "category")


if __name__ == "__main__":
    main()
