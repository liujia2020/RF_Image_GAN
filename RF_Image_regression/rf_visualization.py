"""
Visualization utilities for RF learning experiments.

This module is designed to be used from Jupyter notebooks.
It does not define or load network architectures. Instead, pass an already
constructed and loaded model to the visualization functions.

Expected dataset item keys:
    input     : [C, Z, X, Y]
    label     : [2, Z, X, Y]
    baseline  : [2, Z, X, Y]
    scale     : scalar tensor or float
    path      : str
    z_idx     : tensor/list
    x_idx     : tensor/list
    y_idx     : tensor/list
    category  : optional str
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from rf_eval_utils import infer_category_from_path
except Exception:
    def infer_category_from_path(path: Union[str, Path]) -> str:
        p = str(path).lower()
        for cat in ["carotid", "muscle", "phantom", "simu_point"]:
            if cat in p:
                return cat
        return "unknown"


# ============================================================
# Basic helpers
# ============================================================

DZ_MM = 0.0362
DX_MM = 0.2
DY_MM = 0.2


def physical_aspect_for_view(view: str) -> float:
    """
    Return imshow aspect for the RCA-OPW physical voxel spacing.

    Matplotlib's aspect is the displayed vertical-unit / horizontal-unit ratio.
    """
    view = view.lower()
    if view == "xz":
        return DZ_MM / DX_MM
    if view == "xy":
        return DX_MM / DY_MM
    if view in ["zy", "yz"]:
        return DZ_MM / DY_MM
    raise ValueError(f"Unsupported view: {view}. Use 'xz', 'xy', or 'zy'.")


def physical_spacing_label(view: str) -> str:
    aspect = physical_aspect_for_view(view)
    return f"spacing dz={DZ_MM:g} mm, dx={DX_MM:g} mm, dy={DY_MM:g} mm | aspect={aspect:.3f}"


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_scalar_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().reshape(-1)[0])
    arr = np.asarray(x).reshape(-1)
    return float(arr[0])


def _idx_range_to_string(x) -> str:
    if x is None:
        return ""
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return ""
    return f"{int(arr[0])}~{int(arr[-1])}"


def complex_abs_numpy(x_2ch: np.ndarray) -> np.ndarray:
    """
    Compute magnitude for a 2-channel complex tensor.

    Parameters
    ----------
    x_2ch:
        Array with shape [2, Z, X, Y]. Channel 0 is real and channel 1 is imag.

    Returns
    -------
    mag:
        Magnitude with shape [Z, X, Y].
    """
    x_2ch = np.asarray(x_2ch)
    if x_2ch.ndim != 4 or x_2ch.shape[0] != 2:
        raise ValueError(f"Expected [2,Z,X,Y], got {x_2ch.shape}")
    return np.sqrt(x_2ch[0] ** 2 + x_2ch[1] ** 2 + 1e-12)


def volume_to_db(img: np.ndarray, ref: Optional[float] = None, db_min: float = -60.0) -> np.ndarray:
    """
    Convert magnitude image to normalized dB image.
    """
    img = np.asarray(img, dtype=np.float32)
    if ref is None:
        ref = float(np.max(img))
    if ref <= 0 or not np.isfinite(ref):
        ref = 1.0

    out = 20.0 * np.log10(img / ref + 1e-12)
    out = np.clip(out, db_min, 0.0)
    return out


def extract_slice(volume: np.ndarray, view: str = "xz", slice_index: Optional[int] = None) -> tuple[np.ndarray, int, str, str]:
    """
    Extract a 2D slice from a [Z, X, Y] volume.

    Parameters
    ----------
    volume:
        [Z, X, Y] array.
    view:
        'xz', 'xy', or 'zy'.
    slice_index:
        Slice index along the omitted dimension. If None, use the center.

    Returns
    -------
    img2d, used_index, xlabel, ylabel
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected [Z,X,Y], got {volume.shape}")

    Z, X, Y = volume.shape
    view = view.lower()

    if view == "xz":
        iy = Y // 2 if slice_index is None else int(slice_index)
        return volume[:, :, iy], iy, "x index", "z index"

    if view == "xy":
        iz = Z // 2 if slice_index is None else int(slice_index)
        return volume[iz, :, :], iz, "y index", "x index"

    if view in ["zy", "yz"]:
        ix = X // 2 if slice_index is None else int(slice_index)
        return volume[:, ix, :], ix, "y index", "z index"

    raise ValueError(f"Unsupported view: {view}. Use 'xz', 'xy', or 'zy'.")


# ============================================================
# Prediction helpers
# ============================================================


@torch.no_grad()
def predict_one_sample(model: torch.nn.Module, sample: Dict[str, Any], device: torch.device, denormalize: bool = True):
    """
    Run prediction for one dataset sample.

    Returns
    -------
    baseline, pred, label:
        torch tensors with shape [2, Z, X, Y]. If denormalize=True, restored to
        original amplitude scale using sample['scale'].
    """
    model.eval()

    x = sample["input"].unsqueeze(0).to(device)       # [1,C,Z,X,Y]
    y = sample["label"].unsqueeze(0).to(device)       # [1,2,Z,X,Y]
    b = sample["baseline"].unsqueeze(0).to(device)    # [1,2,Z,X,Y]

    pred = model(x, b)

    if denormalize:
        scale = sample.get("scale", torch.tensor(1.0))
        if torch.is_tensor(scale):
            scale = scale.to(device).reshape(-1)[0]
        else:
            scale = torch.tensor(float(scale), device=device)
        scale = scale.view(1, 1, 1, 1, 1)
        pred = pred * scale
        y = y * scale
        b = b * scale

    return b.squeeze(0).detach().cpu(), pred.squeeze(0).detach().cpu(), y.squeeze(0).detach().cpu()


# ============================================================
# Sample selection
# ============================================================


def select_indices_by_category(dataset, n_per_category: int = 3, categories: Optional[Sequence[str]] = None, seed: Optional[int] = 20260522, shuffle: bool = False) -> list[int]:
    """
    Select sample indices from each category.

    If shuffle=False, take the first n_per_category samples per category.
    If shuffle=True, randomize within each category using the provided seed.
    """
    category_to_indices: Dict[str, List[int]] = {}

    for i, path in enumerate(dataset.files):
        cat = infer_category_from_path(path)
        if categories is not None and cat not in categories:
            continue
        category_to_indices.setdefault(cat, []).append(i)

    rng = np.random.default_rng(seed)
    selected: list[int] = []

    for cat in sorted(category_to_indices.keys()):
        idxs = category_to_indices[cat]
        if shuffle:
            idxs = list(rng.permutation(idxs))
        selected.extend(idxs[:n_per_category])

    return selected


def select_indices_from_metrics_df(dataset, df, top_k: int = 12, metric: str = "complex_improvement", ascending: bool = True) -> list[int]:
    """
    Select dataset indices according to a metrics DataFrame saved by rf_eval_utils.

    This is useful for visualizing worst samples.

    Parameters
    ----------
    dataset:
        RFLearningDataset instance.
    df:
        DataFrame with a 'path' column.
    top_k:
        Number of samples to select.
    metric:
        Column used for sorting. For worse samples, use 'complex_improvement'
        or 'abs_improvement' with ascending=True.
    ascending:
        Sort direction.
    """
    if "path" not in df.columns:
        raise ValueError("metrics df must contain a 'path' column")
    if metric not in df.columns:
        raise ValueError(f"metrics df does not contain metric column: {metric}")

    path_to_index = {str(Path(p)): i for i, p in enumerate(dataset.files)}
    selected = []

    df_sorted = df.sort_values(metric, ascending=ascending)

    for _, row in df_sorted.iterrows():
        p = str(Path(row["path"]))
        if p in path_to_index:
            selected.append(path_to_index[p])
        if len(selected) >= top_k:
            break

    return selected


# ============================================================
# Plotting
# ============================================================


def plot_prediction_panel(
    sample: Dict[str, Any],
    baseline,
    pred,
    label,
    save_path: Optional[Union[str, Path]] = None,
    view: str = "xz",
    slice_index: Optional[int] = None,
    db_min: float = -60.0,
    cmap: str = "gray",
    show: bool = False,
    dpi: int = 200,
):
    """
    Plot baseline, prediction, label, and error maps for one sample.

    baseline, pred, label:
        Tensors or arrays with shape [2, Z, X, Y].
    """
    baseline_np = _to_numpy(baseline)
    pred_np = _to_numpy(pred)
    label_np = _to_numpy(label)

    baseline_abs = complex_abs_numpy(baseline_np)
    pred_abs = complex_abs_numpy(pred_np)
    label_abs = complex_abs_numpy(label_np)

    err_pred = np.abs(pred_abs - label_abs)
    err_base = np.abs(baseline_abs - label_abs)

    ref = float(max(baseline_abs.max(), pred_abs.max(), label_abs.max(), 1.0))

    base_slice, used_idx, xlabel, ylabel = extract_slice(baseline_abs, view=view, slice_index=slice_index)
    pred_slice, _, _, _ = extract_slice(pred_abs, view=view, slice_index=slice_index)
    label_slice, _, _, _ = extract_slice(label_abs, view=view, slice_index=slice_index)
    err_pred_slice, _, _, _ = extract_slice(err_pred, view=view, slice_index=slice_index)
    err_base_slice, _, _, _ = extract_slice(err_base, view=view, slice_index=slice_index)

    imgs = [
        volume_to_db(base_slice, ref=ref, db_min=db_min),
        volume_to_db(pred_slice, ref=ref, db_min=db_min),
        volume_to_db(label_slice, ref=ref, db_min=db_min),
        volume_to_db(err_pred_slice, ref=ref, db_min=db_min),
        volume_to_db(err_base_slice, ref=ref, db_min=db_min),
    ]

    titles = [
        "3-angle baseline",
        "network prediction",
        "33-angle label",
        "|pred-label|",
        "|baseline-label|",
    ]

    pred_l1 = float(np.mean(np.abs(pred_np - label_np)))
    base_l1 = float(np.mean(np.abs(baseline_np - label_np)))
    pred_abs_l1 = float(np.mean(np.abs(pred_abs - label_abs)))
    base_abs_l1 = float(np.mean(np.abs(baseline_abs - label_abs)))

    path = Path(sample.get("path", "unknown.h5"))
    category = sample.get("category", infer_category_from_path(path))
    if isinstance(category, (list, tuple)):
        category = category[0]

    z_str = _idx_range_to_string(sample.get("z_idx", None))
    x_str = _idx_range_to_string(sample.get("x_idx", None))
    y_str = _idx_range_to_string(sample.get("y_idx", None))

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    aspect = physical_aspect_for_view(view)

    for ax, img, title in zip(axes, imgs, titles):
        im = ax.imshow(img, cmap=cmap, vmin=db_min, vmax=0, aspect=aspect)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{category} | {path.name} | {view}-slice={used_idx}\n"
        f"z_idx={z_str}, x_idx={x_str}, y_idx={y_str} | "
        f"complex L1 pred/base: {pred_l1:.3e} / {base_l1:.3e} | "
        f"abs L1 pred/base: {pred_abs_l1:.3e} / {base_abs_l1:.3e}\n"
        f"{physical_spacing_label(view)}",
        fontsize=10,
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    metrics = {
        "pred_complex_l1": pred_l1,
        "base_complex_l1": base_l1,
        "complex_improvement": 1.0 - pred_l1 / (base_l1 + 1e-12),
        "pred_abs_l1": pred_abs_l1,
        "base_abs_l1": base_abs_l1,
        "abs_improvement": 1.0 - pred_abs_l1 / (base_abs_l1 + 1e-12),
        "save_path": str(save_path) if save_path is not None else None,
        "path": str(path),
        "category": str(category),
        "view": view,
        "slice_index": used_idx,
    }

    return metrics


@torch.no_grad()
def visualize_model_samples(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    save_dir: Union[str, Path],
    indices: Optional[Sequence[int]] = None,
    n_per_category: int = 3,
    categories: Optional[Sequence[str]] = None,
    view: str = "xz",
    slice_index: Optional[int] = None,
    db_min: float = -60.0,
    show: bool = False,
    seed: int = 20260522,
    shuffle: bool = False,
    prefix: str = "sample",
):
    """
    Visualize model outputs for selected samples.

    If indices is None, samples are selected by category.
    Returns a list of per-sample metric dictionaries.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    if indices is None:
        indices = select_indices_by_category(
            dataset,
            n_per_category=n_per_category,
            categories=categories,
            seed=seed,
            shuffle=shuffle,
        )

    print("Selected indices:", list(indices))

    all_metrics = []

    for count, idx in enumerate(indices, start=1):
        sample = dataset[idx]
        baseline, pred, label = predict_one_sample(model, sample, device, denormalize=True)

        path = Path(sample["path"])
        category = sample.get("category", infer_category_from_path(path))
        if isinstance(category, (list, tuple)):
            category = category[0]

        save_path = save_dir / f"{count:03d}_{prefix}_{category}_{path.stem}.png"

        metrics = plot_prediction_panel(
            sample=sample,
            baseline=baseline,
            pred=pred,
            label=label,
            save_path=save_path,
            view=view,
            slice_index=slice_index,
            db_min=db_min,
            show=show,
        )

        all_metrics.append(metrics)

        print(f"Saved: {save_path}")
        print(f"  complex L1 pred/base: {metrics['pred_complex_l1']:.4e} / {metrics['base_complex_l1']:.4e}")
        print(f"  abs     L1 pred/base: {metrics['pred_abs_l1']:.4e} / {metrics['base_abs_l1']:.4e}")

    return all_metrics
