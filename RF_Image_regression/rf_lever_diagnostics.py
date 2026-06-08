from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "tiny_bn_random64_full1500"
TASK_NAME = "RF lever-selection diagnostics (read-only)"
ROOT = Path(__file__).resolve().parent

HISTORY_CSV = ROOT / "checkpoint" / EXPERIMENT / "training_history.csv"
METRIC_DIR = ROOT / "test_metrics" / EXPERIMENT
PER_SAMPLE_CSV = METRIC_DIR / "test_per_sample_metrics.csv"
PER_CATEGORY_CSV = METRIC_DIR / "test_per_category_summary.csv"
MULTIVOLUME_CSV = METRIC_DIR / "bn_multivolume_diagnostics_13.csv"
CACHE_ROOT = ROOT / "Data_cache_random64_full1500"


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


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for candidate in candidates:
        needle = candidate.lower()
        matches = [c for c in df.columns if needle in c.lower()]
        if matches:
            return matches[0]
    return None


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "MISSING"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def block1_history() -> None:
    print("\n" + "=" * 96)
    print("Block 1 -- model selection / early-stopping audit")
    print("=" * 96)
    if report_missing(HISTORY_CSV):
        return

    df = pd.read_csv(HISTORY_CSV)
    cols = {
        "epoch": find_col(df, ["epoch"]),
        "val_l1": find_col(df, ["val_l1"]),
        "val_abs": find_col(df, ["val_abs"]),
        "val_loss": find_col(df, ["val_loss"]),
        "val_impr": find_col(df, ["val_impr", "val_improvement"]),
        "val_abs_impr": find_col(df, ["val_abs_impr", "val_abs_improvement"]),
    }
    print(f"source: {HISTORY_CSV}")
    print(f"shape: rows={len(df)}, cols={len(df.columns)}")
    print("selected columns:", ", ".join(f"{k}={v or 'MISSING'}" for k, v in cols.items()))

    present = [c for c in cols.values() if c is not None]
    if present:
        print(df[present].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    for key in ("val_l1", "val_abs", "val_loss"):
        col = cols[key]
        if col is None:
            print(f"{key}_min_epoch: MISSING")
            continue
        idx = df[col].idxmin()
        row = df.loc[idx]
        epoch = row[cols["epoch"]] if cols["epoch"] is not None else idx
        triple = tuple(fmt(row[cols[name]]) if cols[name] is not None else "MISSING" for name in ("val_l1", "val_abs", "val_loss"))
        print(f"{key}_min_epoch: epoch={epoch}, (val_l1,val_abs,val_loss)=({triple[0]}, {triple[1]}, {triple[2]})")


def metric_col(df: pd.DataFrame, label: str) -> str | None:
    candidates = {
        "seam": ["x_boundary_pred_over_label", "seam"],
        "nonboundary_complex_improvement": ["nonboundary_complex_improvement"],
        "nonboundary_abs_improvement": ["nonboundary_abs_improvement"],
        "pred_hf_rms_over_label": ["pred_hf_rms_over_label"],
        "pred_hf_improvement_vs_baseline": ["pred_hf_improvement_vs_baseline"],
        "detail_retention_top10": ["detail_retention_top10"],
        "pred_nonboundary_abs_std_over_label": ["pred_nonboundary_abs_std_over_label"],
    }
    return find_col(df, candidates[label])


def sign_label(value: float) -> str:
    if value > 1e-6:
        return "positive"
    if value < -1e-6:
        return "negative"
    return "near_zero"


def block2_multivolume() -> None:
    print("\n" + "=" * 96)
    print("Block 2 -- HF detail vs noise audit")
    print("=" * 96)
    if report_missing(MULTIVOLUME_CSV):
        return

    df = pd.read_csv(MULTIVOLUME_CSV)
    print(f"source: {MULTIVOLUME_CSV}")
    print(f"shape: rows={len(df)}, cols={len(df.columns)}")
    print("header:")
    print(", ".join(df.columns))

    view = df.copy()
    if "model" in view.columns:
        bn = view[view["model"].astype(str).str.upper() == "BN"].copy()
        print(f"actual_rows={len(view)}, bn_rows={len(bn)}")
        if len(bn) > 0:
            view = bn
    else:
        print(f"actual_rows={len(view)}, model_column=MISSING")

    cols = {
        "seam": metric_col(df, "seam"),
        "nonboundary_complex_improvement": metric_col(df, "nonboundary_complex_improvement"),
        "nonboundary_abs_improvement": metric_col(df, "nonboundary_abs_improvement"),
        "pred_hf_rms_over_label": metric_col(df, "pred_hf_rms_over_label"),
        "pred_hf_improvement_vs_baseline": metric_col(df, "pred_hf_improvement_vs_baseline"),
        "detail_retention_top10": metric_col(df, "detail_retention_top10"),
        "pred_nonboundary_abs_std_over_label": metric_col(df, "pred_nonboundary_abs_std_over_label"),
    }
    print("matched metric columns:", ", ".join(f"{k}={v or 'MISSING'}" for k, v in cols.items()))

    display_cols = [c for c in ["case", "category", "source_file", "file_id", "model"] if c in view.columns]
    display_cols += [c for c in cols.values() if c is not None and c not in display_cols]
    print("per-volume rows:")
    print(view[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("category means:")
    if "category" not in view.columns:
        print("category: MISSING")
    else:
        for category, group in view.groupby("category", sort=True):
            parts = [f"category={category}", f"n={len(group)}"]
            for label, col in cols.items():
                if col is None:
                    parts.append(f"{label}=MISSING")
                else:
                    parts.append(f"{label}_mean={group[col].astype(float).mean():.6f}")
            print(" | ".join(parts))

    hf_col = cols["pred_hf_improvement_vs_baseline"]
    print("pred_hf_improvement_vs_baseline sign check:")
    if hf_col is None:
        print("pred_hf_improvement_vs_baseline: MISSING")
    else:
        id_cols = [c for c in ["case", "category", "source_file", "file_id"] if c in view.columns]
        for _, row in view.iterrows():
            ident = " | ".join(f"{c}={row[c]}" for c in id_cols)
            value = float(row[hf_col])
            print(f"{ident} | pred_hf_improvement_vs_baseline={value:.6f} | sign={sign_label(value)}")


def quantiles(series: pd.Series) -> dict[str, float]:
    probs = [0.05, 0.25, 0.50, 0.75, 0.95]
    values = series.astype(float).quantile(probs)
    out = {f"p{int(p * 100)}": float(values.loc[p]) for p in probs}
    out["min"] = float(series.astype(float).min())
    out["max"] = float(series.astype(float).max())
    return out


def print_quantile_row(label: str, values: dict[str, float]) -> None:
    keys = ["p5", "p25", "p50", "p75", "p95", "min", "max"]
    print(label + " | " + " | ".join(f"{k}={values[k]:.6f}" for k in keys))


def block3_error_concentration() -> None:
    print("\n" + "=" * 96)
    print("Block 3 -- error concentration")
    print("=" * 96)
    if report_missing(PER_SAMPLE_CSV):
        return

    df = pd.read_csv(PER_SAMPLE_CSV)
    print(f"source: {PER_SAMPLE_CSV}")
    print(f"shape: rows={len(df)}, cols={len(df.columns)}")

    required = [
        "complex_improvement",
        "category",
        "path",
        "pred_complex_l1",
        "base_complex_l1",
        "abs_improvement",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("MISSING columns:", ", ".join(missing))
        return

    print_quantile_row("overall complex_improvement", quantiles(df["complex_improvement"]))
    print("by-category complex_improvement:")
    for category, group in df.groupby("category", sort=True):
        print_quantile_row(f"category={category} n={len(group)}", quantiles(group["complex_improvement"]))

    worst_cols = ["path", "category", "pred_complex_l1", "base_complex_l1", "complex_improvement", "abs_improvement"]
    worst = df.sort_values("complex_improvement", ascending=True).head(15)
    print("worst 15 by complex_improvement:")
    print(worst[worst_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if "abs_improvement" not in df.columns:
        print("pearson_corr_complex_abs_improvement: MISSING")
    else:
        corr = df["complex_improvement"].astype(float).corr(df["abs_improvement"].astype(float), method="pearson")
        print(f"pearson_corr_complex_abs_improvement={corr:.6f}")


def block4_split_balance() -> None:
    print("\n" + "=" * 96)
    print("Block 4 -- train/val/test category balance")
    print("=" * 96)

    for split in ("train", "val", "test"):
        path = CACHE_ROOT / split / "meta.npz"
        print(f"split={split}")
        if report_missing(path):
            continue
        data = np.load(path, allow_pickle=True)
        if "category" not in data.files:
            print("category: MISSING")
            continue
        cats = np.asarray(data["category"]).astype(str)
        values, counts = np.unique(cats, return_counts=True)
        total = int(counts.sum())
        print(f"total={total}")
        for value, count in zip(values, counts):
            print(f"  category={value} count={int(count)} pct={int(count) / total:.6f}")


def main() -> None:
    print_header()
    block1_history()
    block2_multivolume()
    block3_error_concentration()
    block4_split_balance()


if __name__ == "__main__":
    main()
