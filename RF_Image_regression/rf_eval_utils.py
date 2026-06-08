"""
rf_eval_utils.py

Evaluation utilities for RF learning experiments.

Expected batch format from RFLearningDataset:
    batch["input"]    : [B, 1536, Z, X, Y]
    batch["label"]    : [B, 2,    Z, X, Y]
    batch["baseline"] : [B, 2,    Z, X, Y]
    batch["scale"]    : scalar or [B]
    batch["path"]     : list[str]
    batch["category"] : list[str]  (optional, fallback from path)

The Dataset is usually created with normalize=True. Therefore, evaluation
restores the original amplitude by multiplying pred/label/baseline by scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None

from rf_train_utils import complex_abs_2ch


DEFAULT_CATEGORIES = ("carotid", "muscle", "phantom", "simu_point")


def infer_category_from_path(path: str | Path, categories: Iterable[str] = DEFAULT_CATEGORIES) -> str:
    """Infer category from a file path.

    It first checks path parts and then checks substring fallback.
    """
    p = Path(path)
    lower_parts = [part.lower() for part in p.parts]

    for cat in categories:
        if cat.lower() in lower_parts:
            return cat

    p_lower = str(path).lower()
    for cat in categories:
        if cat.lower() in p_lower:
            return cat

    return "unknown"


def _scale_to_broadcast(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert sample scale to [B,1,1,1,1] for broadcasting."""
    scale = scale.to(device)

    if scale.ndim == 0:
        scale = scale.view(1, 1, 1, 1, 1)
    elif scale.ndim == 1:
        scale = scale.view(-1, 1, 1, 1, 1)
    else:
        # Already broadcast-like. Keep it but make sure it has 5 dims.
        while scale.ndim < 5:
            scale = scale.unsqueeze(-1)

    return scale


def _get_batch_categories(batch: Dict, paths: Iterable[str]) -> list[str]:
    """Read category from batch if present, otherwise infer from path."""
    if "category" in batch:
        cats = batch["category"]
        if isinstance(cats, (list, tuple)):
            return [str(c) for c in cats]
        # DataLoader may keep strings as list, but keep fallback robust.
        try:
            return [str(c) for c in list(cats)]
        except Exception:
            pass

    return [infer_category_from_path(p) for p in paths]


@torch.no_grad()
def evaluate_full_test_set(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 4,
    save_csv_path: Optional[str | Path] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> pd.DataFrame:
    """Evaluate a model on a full dataset and return per-sample metrics.

    Metrics are computed in original amplitude scale by undoing per-sample
    normalization with batch["scale"].

    Returns a DataFrame with columns:
        path, category,
        pred_complex_l1, base_complex_l1, complex_improvement, complex_better,
        pred_abs_l1,     base_abs_l1,     abs_improvement,     abs_better
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model.eval()
    rows = []

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        b = batch["baseline"].to(device, non_blocking=True)

        pred = model(x, b)

        scale = batch.get("scale", torch.tensor(1.0))
        if not torch.is_tensor(scale):
            scale = torch.as_tensor(scale, dtype=pred.dtype)
        scale = _scale_to_broadcast(scale, device=device)

        pred_raw = pred * scale
        y_raw = y * scale
        b_raw = b * scale

        # Per-sample complex real/imag L1.
        pred_l1 = torch.mean(torch.abs(pred_raw - y_raw), dim=(1, 2, 3, 4))
        base_l1 = torch.mean(torch.abs(b_raw - y_raw), dim=(1, 2, 3, 4))

        # Per-sample envelope/magnitude L1.
        pred_abs = complex_abs_2ch(pred_raw)
        y_abs = complex_abs_2ch(y_raw)
        b_abs = complex_abs_2ch(b_raw)

        pred_abs_l1 = torch.mean(torch.abs(pred_abs - y_abs), dim=(1, 2, 3, 4))
        base_abs_l1 = torch.mean(torch.abs(b_abs - y_abs), dim=(1, 2, 3, 4))

        paths = [str(p) for p in batch["path"]]
        categories = _get_batch_categories(batch, paths)

        for i, path in enumerate(paths):
            pl1 = float(pred_l1[i].detach().cpu())
            bl1 = float(base_l1[i].detach().cpu())
            pa1 = float(pred_abs_l1[i].detach().cpu())
            ba1 = float(base_abs_l1[i].detach().cpu())

            rows.append(
                {
                    "path": path,
                    "category": categories[i],
                    "pred_complex_l1": pl1,
                    "base_complex_l1": bl1,
                    "complex_improvement": 1.0 - pl1 / (bl1 + 1e-12),
                    "complex_better": bool(pl1 < bl1),
                    "pred_abs_l1": pa1,
                    "base_abs_l1": ba1,
                    "abs_improvement": 1.0 - pa1 / (ba1 + 1e-12),
                    "abs_better": bool(pa1 < ba1),
                }
            )

    df = pd.DataFrame(rows)

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        print(f"Saved per-sample metrics to: {save_csv_path}")

    return df


def summarize_test_metrics(df: pd.DataFrame, show_table: bool = True) -> Tuple[Dict, pd.DataFrame]:
    """Print and return overall and per-category summaries."""
    if len(df) == 0:
        raise ValueError("Empty metrics DataFrame.")

    overall = {
        "n": int(len(df)),
        "complex_pred_mean": float(df["pred_complex_l1"].mean()),
        "complex_base_mean": float(df["base_complex_l1"].mean()),
        "complex_improvement_mean": float(df["complex_improvement"].mean()),
        "complex_better_rate": float(df["complex_better"].mean()),
        "abs_pred_mean": float(df["pred_abs_l1"].mean()),
        "abs_base_mean": float(df["base_abs_l1"].mean()),
        "abs_improvement_mean": float(df["abs_improvement"].mean()),
        "abs_better_rate": float(df["abs_better"].mean()),
    }

    print("\n================ Overall test summary ================")
    print(f"Samples: {overall['n']}")

    print("\n[Complex L1]")
    print(f"pred mean       : {overall['complex_pred_mean']:.4e}")
    print(f"baseline mean   : {overall['complex_base_mean']:.4e}")
    print(f"mean improvement: {overall['complex_improvement_mean'] * 100:.2f}%")
    print(f"better rate     : {overall['complex_better_rate'] * 100:.2f}%")

    print("\n[Envelope abs L1]")
    print(f"pred mean       : {overall['abs_pred_mean']:.4e}")
    print(f"baseline mean   : {overall['abs_base_mean']:.4e}")
    print(f"mean improvement: {overall['abs_improvement_mean'] * 100:.2f}%")
    print(f"better rate     : {overall['abs_better_rate'] * 100:.2f}%")

    cat_summary = df.groupby("category").agg(
        n=("path", "count"),
        complex_pred_mean=("pred_complex_l1", "mean"),
        complex_base_mean=("base_complex_l1", "mean"),
        complex_improvement_mean=("complex_improvement", "mean"),
        complex_better_rate=("complex_better", "mean"),
        abs_pred_mean=("pred_abs_l1", "mean"),
        abs_base_mean=("base_abs_l1", "mean"),
        abs_improvement_mean=("abs_improvement", "mean"),
        abs_better_rate=("abs_better", "mean"),
    )

    print("\n================ Per-category summary ================")
    if show_table and display is not None:
        display(cat_summary)
    else:
        print(cat_summary)

    return overall, cat_summary


def save_test_summaries(
    df: pd.DataFrame,
    metric_dir: str | Path,
    overall: Optional[Dict] = None,
    cat_summary: Optional[pd.DataFrame] = None,
    prefix: str = "test",
) -> None:
    """Save per-sample, overall, and category test summaries."""
    metric_dir = Path(metric_dir)
    metric_dir.mkdir(parents=True, exist_ok=True)

    if overall is None or cat_summary is None:
        overall, cat_summary = summarize_test_metrics(df, show_table=False)

    df.to_csv(metric_dir / f"{prefix}_per_sample_metrics.csv", index=False)
    pd.DataFrame([overall]).to_csv(metric_dir / f"{prefix}_overall_summary.csv", index=False)
    cat_summary.to_csv(metric_dir / f"{prefix}_per_category_summary.csv")

    print(f"Saved test summaries to: {metric_dir}")


def find_worse_samples(
    df: pd.DataFrame,
    metric: str = "complex",
    top_k: int = 20,
) -> pd.DataFrame:
    """Return samples where prediction is worse than baseline.

    metric:
        "complex" or "abs"
    """
    if metric not in {"complex", "abs"}:
        raise ValueError("metric must be 'complex' or 'abs'")

    better_col = f"{metric}_better"
    improvement_col = f"{metric}_improvement"

    worse = df[df[better_col] == False].copy()  # noqa: E712
    worse = worse.sort_values(improvement_col, ascending=True)

    print(f"{metric} worse samples: {len(worse)}")
    if len(worse) > 0:
        if display is not None:
            display(worse.head(top_k))
        else:
            print(worse.head(top_k))

    return worse


def compare_experiment_summaries(summary_paths: Dict[str, str | Path]) -> pd.DataFrame:
    """Load multiple overall summary CSVs and compare experiments.

    Example:
        compare_experiment_summaries({
            "tiny_nonpoint": ".../test_overall_summary.csv",
            "earlymix": ".../test_overall_summary.csv",
        })
    """
    rows = []
    for name, path in summary_paths.items():
        path = Path(path)
        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"Expected one-row summary at {path}, got {len(df)} rows")
        row = df.iloc[0].to_dict()
        row["experiment"] = name
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["experiment"] + [c for c in out.columns if c != "experiment"]
    out = out[cols]
    return out
