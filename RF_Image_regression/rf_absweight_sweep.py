from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader

from rf_cached_dataset import RFCachedDataset
from rf_eval_utils import evaluate_full_test_set, save_test_summaries, summarize_test_metrics
from rf_models import build_model
from rf_train_utils import (
    count_trainable_parameters,
    get_device,
    seed_everything,
    train_model_jupyter,
)
from rf_visualization import complex_abs_numpy


TASK_NAME = "RF abs_weight sweep + realism eval"
BASE_EXPERIMENT = "tiny_bn_random64_full1500"
ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "checkpoint" / BASE_EXPERIMENT / "run_config.json"
CACHE_ROOT = ROOT / "Data_cache_random64_full1500"
SWEEP_OUT_DIR = ROOT / "test_metrics" / "absweight_sweep"
ASPECT_XZ = 0.0362 / 0.2
DB_MIN = -60.0
EPS = 1e-8

SWEEP_SPECS = [
    (0.1, BASE_EXPERIMENT),
    (0.5, "tiny_bn_aw05_random64_full1500"),
    (1.0, "tiny_bn_aw10_random64_full1500"),
    (2.0, "tiny_bn_aw20_random64_full1500"),
]
TRAIN_ABS_WEIGHTS = {0.5, 1.0, 2.0}
ALLOWED_CONFIG_DIFFS = {"abs_weight", "experiment_name", "ckpt_dir"}


def print_header() -> None:
    print("=" * 96)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"task: {TASK_NAME}")
    print(f"base_experiment: {BASE_EXPERIMENT}")
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


def load_base_config() -> dict[str, Any]:
    if report_missing(BASE_CONFIG_PATH):
        raise FileNotFoundError(BASE_CONFIG_PATH)
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_dir(experiment_name: str) -> Path:
    return ROOT / "checkpoint" / experiment_name


def metric_dir(experiment_name: str) -> Path:
    return ROOT / "test_metrics" / experiment_name


def build_candidate_config(base_config: dict[str, Any], abs_weight: float, experiment_name: str) -> dict[str, Any]:
    base_compare = dict(base_config)
    base_compare["ckpt_dir"] = str(checkpoint_dir(BASE_EXPERIMENT))

    config = dict(base_config)
    config["experiment_name"] = experiment_name
    config["abs_weight"] = abs_weight
    config["ckpt_dir"] = str(checkpoint_dir(experiment_name))

    diff_keys = sorted(
        key
        for key in set(base_compare) | set(config)
        if base_compare.get(key) != config.get(key)
    )
    print(
        "config_diff_vs_baseline: "
        + ", ".join(f"{key}: {base_compare.get(key)!r} -> {config.get(key)!r}" for key in diff_keys)
    )
    assert set(diff_keys) == ALLOWED_CONFIG_DIFFS, (
        "Unexpected config differences vs baseline: "
        f"{diff_keys}; allowed={sorted(ALLOWED_CONFIG_DIFFS)}"
    )
    return config


def first_norm_name(model: torch.nn.Module) -> str:
    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    return type(first_norm).__name__ if first_norm is not None else "MISSING"


def assert_batchnorm_model(model: torch.nn.Module) -> None:
    norm_name = first_norm_name(model)
    print(f"startup model_class={type(model).__name__}")
    print(f"startup first_norm_class={norm_name}")
    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    assert isinstance(first_norm, nn.BatchNorm3d), f"Expected BatchNorm3d, got {norm_name}"


def make_tiny_bn_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_model(
        config["model_name"],
        in_channels=1536,
        hidden=int(config["hidden"]),
        out_channels=2,
        use_batch_norm=True,
    ).to(device)
    assert_batchnorm_model(model)
    print(f"trainable_parameters={count_trainable_parameters(model):,}")
    return model


def checkpoint_complete(ckpt_dir: Path) -> bool:
    required = [
        ckpt_dir / "best_model.pth",
        ckpt_dir / "best_state_dict.pth",
        ckpt_dir / "final_state_dict.pth",
        ckpt_dir / "training_history.csv",
        ckpt_dir / "run_config.json",
    ]
    return all(path.exists() for path in required)


def load_checkpoint_model(experiment_name: str, base_config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    ckpt_path = checkpoint_dir(experiment_name) / "best_model.pth"
    if report_missing(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"]
    use_bn = any("running_mean" in key for key in state_dict)
    model = build_model(
        base_config["model_name"],
        in_channels=1536,
        hidden=int(base_config["hidden"]),
        out_channels=2,
        use_batch_norm=use_bn,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    assert_batchnorm_model(model)
    print(f"loaded_checkpoint={ckpt_path}")
    print(f"loaded_epoch={ckpt.get('epoch', 'NA')} best_val_l1={ckpt.get('best_val_l1', 'NA')}")
    return model


def make_loaders(config: dict[str, Any]) -> tuple[RFCachedDataset, RFCachedDataset, RFCachedDataset, DataLoader, DataLoader]:
    for split in ("train", "val", "test"):
        if report_missing(CACHE_ROOT / split / "meta.npz"):
            raise FileNotFoundError(CACHE_ROOT / split / "meta.npz")

    train_set = RFCachedDataset(CACHE_ROOT / "train")
    val_set = RFCachedDataset(CACHE_ROOT / "val")
    test_set = RFCachedDataset(CACHE_ROOT / "test")

    expected_categories = set(config["include_categories"])
    for split_name, dataset in (("train", train_set), ("val", val_set), ("test", test_set)):
        actual_categories = set(dataset.categories)
        assert actual_categories <= expected_categories, (
            f"Unexpected category in {split_name}: {sorted(actual_categories)}"
        )

    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": 0,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    print(
        f"datasets: train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"batch_size={loader_kwargs['batch_size']} num_workers={loader_kwargs['num_workers']} "
        f"pin_memory={loader_kwargs['pin_memory']}"
    )
    return train_set, val_set, test_set, train_loader, val_loader


def train_missing_models(
    base_config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    abs_weights: set[float],
) -> None:
    for abs_weight, experiment_name in SWEEP_SPECS:
        if abs_weight not in abs_weights:
            continue
        ckpt_dir = checkpoint_dir(experiment_name)
        if checkpoint_complete(ckpt_dir):
            print(f"SKIP complete checkpoint: {ckpt_dir / 'best_model.pth'}")
            continue
        if ckpt_dir.exists():
            print(f"RETRAIN incomplete checkpoint directory in place: {ckpt_dir}")

        print("\n" + "#" * 96)
        print(f"TRAIN abs_weight={abs_weight} experiment={experiment_name}")
        config = build_candidate_config(base_config, abs_weight, experiment_name)
        seed_everything(int(config["seed"]))
        model = make_tiny_bn_model(config, device)

        train_model_jupyter(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            ckpt_dir=ckpt_dir,
            experiment_name=experiment_name,
            num_epochs=int(config["num_epochs"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            abs_weight=float(config["abs_weight"]),
            grad_clip=config.get("grad_clip"),
            print_every=5,
            eta_min=float(config["eta_min"]),
            seed=int(config["seed"]),
            config=config,
            patience=int(config["patience"]) if config.get("patience") is not None else None,
        )


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


def rms(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(vals * vals)))


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


def highpass7(mag: np.ndarray) -> np.ndarray:
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


def compute_realism_metrics(pred_raw: np.ndarray, label_raw: np.ndarray, baseline_raw: np.ndarray) -> dict[str, float]:
    label_mag = complex_abs(label_raw).astype(np.float64, copy=False)
    pred_mag = complex_abs(pred_raw).astype(np.float64, copy=False)
    threshold = float(np.median(label_mag))
    high_signal_mask = label_mag >= threshold

    er = (pred_raw[0] - label_raw[0]).astype(np.float64, copy=False)
    ei = (pred_raw[1] - label_raw[1]).astype(np.float64, copy=False)
    lhat_r = label_raw[0].astype(np.float64, copy=False) / (label_mag + EPS)
    lhat_i = label_raw[1].astype(np.float64, copy=False) / (label_mag + EPS)
    e_par = er * lhat_r + ei * lhat_i
    e_sq = er * er + ei * ei
    e_perp = np.sqrt(np.maximum(e_sq - e_par * e_par, 0.0))

    e_par_m = e_par[high_signal_mask]
    e_perp_m = e_perp[high_signal_mask]
    label_mag_m = label_mag[high_signal_mask]
    pred_mag_m = pred_mag[high_signal_mask]
    slope, intercept = regression_slope_intercept(label_mag_m, pred_mag_m)

    label_hp = highpass7(label_mag)
    pred_hp = highpass7(pred_mag)
    baseline_hp = highpass7(complex_abs(baseline_raw))
    eps = 1e-12
    label_hf_rms = rms(label_hp)
    pred_hf_rms = rms(pred_hp)

    label_detail = np.abs(label_hp)
    top_threshold = float(np.percentile(label_detail, 90.0))
    top_mask = label_detail >= top_threshold
    detail_retention_top10 = float(
        np.median(np.abs(pred_hp[top_mask]) / (np.abs(label_hp[top_mask]) + eps))
    )

    pred_hf_l1 = float(np.mean(np.abs(pred_hp - label_hp)))
    baseline_hf_l1 = float(np.mean(np.abs(baseline_hp - label_hp)))

    return {
        "rms_e_perp_over_rms_e_par": rms(e_perp_m) / max(rms(e_par_m), EPS),
        "mean_e_par_over_rms_L": float(np.mean(e_par_m)) / max(rms(label_mag_m), EPS),
        "amp_regression_slope": slope,
        "amp_regression_intercept": intercept,
        "corr_e_par_abs_label": pearson(e_par_m, label_mag_m),
        "corr_e_perp_abs_label": pearson(e_perp_m, label_mag_m),
        "pred_hf_rms_over_label": pred_hf_rms / (label_hf_rms + eps),
        "pred_hf_improvement_vs_baseline": 1.0 - pred_hf_l1 / (baseline_hf_l1 + eps),
        "detail_retention_top10": detail_retention_top10,
        "pred_nonboundary_abs_std_over_label": float(np.std(pred_mag) / (np.std(label_mag) + eps)),
    }


def scale_to_broadcast(scale: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while scale.ndim < reference.ndim:
        scale = scale.view(*scale.shape, *([1] * (reference.ndim - scale.ndim)))
    return scale


@torch.no_grad()
def infer_one_raw(model: torch.nn.Module, sample: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sample["input"].unsqueeze(0).to(device=device, dtype=torch.float32)
    label = sample["label"].unsqueeze(0).to(device=device, dtype=torch.float32)
    baseline = sample["baseline"].unsqueeze(0).to(device=device, dtype=torch.float32)
    scale = sample["scale"].to(device=device, dtype=torch.float32)
    scale = scale_to_broadcast(scale, label)

    pred = model(x, baseline)
    pred_raw = (pred * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    label_raw = (label * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_raw = (baseline * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return pred_raw, label_raw, baseline_raw


def evaluate_realism(
    model: torch.nn.Module,
    dataset: RFCachedDataset,
    device: torch.device,
) -> dict[str, float]:
    rows = []
    for idx in range(len(dataset)):
        pred_raw, label_raw, baseline_raw = infer_one_raw(model, dataset[idx], device)
        rows.append(compute_realism_metrics(pred_raw, label_raw, baseline_raw))
        if (idx + 1) % 25 == 0 or idx == len(dataset) - 1:
            print(f"  realism_inference={idx + 1}/{len(dataset)}")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def evaluate_checkpoint(
    abs_weight: float,
    experiment_name: str,
    base_config: dict[str, Any],
    test_set: RFCachedDataset,
    device: torch.device,
) -> dict[str, float | str]:
    print("\n" + "#" * 96)
    print(f"EVAL abs_weight={abs_weight} experiment={experiment_name}")
    model = load_checkpoint_model(experiment_name, base_config, device)

    df_test = evaluate_full_test_set(
        model=model,
        dataset=test_set,
        device=device,
        batch_size=int(base_config["batch_size"]),
        save_csv_path=metric_dir(experiment_name) / "test_per_sample_metrics.csv",
        num_workers=0,
        pin_memory=True,
    )
    overall, cat_summary = summarize_test_metrics(df_test, show_table=False)
    save_test_summaries(df_test, metric_dir(experiment_name), overall=overall, cat_summary=cat_summary, prefix="test")
    realism = evaluate_realism(model, test_set, device)

    row: dict[str, float | str] = {
        "abs_weight": abs_weight,
        "experiment_name": experiment_name,
        "complex_improvement_mean": float(overall["complex_improvement_mean"]),
        "abs_improvement_mean": float(overall["abs_improvement_mean"]),
        "mean_e_par/RMS_L": realism["mean_e_par_over_rms_L"],
        "perp/par": realism["rms_e_perp_over_rms_e_par"],
        "slope_a": realism["amp_regression_slope"],
        "intercept_b": realism["amp_regression_intercept"],
        "pred_hf_rms_over_label": realism["pred_hf_rms_over_label"],
        "detail_retention_top10": realism["detail_retention_top10"],
        "pred_nonboundary_abs_std_over_label": realism["pred_nonboundary_abs_std_over_label"],
    }
    return row


def dataset_index_by_path(dataset: RFCachedDataset, path: str) -> int:
    exact = {p: i for i, p in enumerate(dataset.paths)}
    by_name = {Path(p).name: i for i, p in enumerate(dataset.paths)}
    if path in exact:
        return int(exact[path])
    name = Path(path).name
    if name in by_name:
        return int(by_name[name])
    raise KeyError(f"Sample path not found in cache metadata: {path}")


def to_db(mag: np.ndarray, ref: float, db_min: float = DB_MIN) -> np.ndarray:
    db = 20.0 * np.log10(np.maximum(mag.astype(np.float32), 1e-12) / max(float(ref), 1e-12))
    return np.maximum(db, db_min)


def save_visual_comparison(
    base_config: dict[str, Any],
    test_set: RFCachedDataset,
    device: torch.device,
) -> Path:
    metrics_path = metric_dir(BASE_EXPERIMENT) / "test_per_sample_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    fixed = (
        metrics[metrics["category"].astype(str) == "carotid"]
        .sort_values("complex_improvement", ascending=True)
        .iloc[0]
    )
    sample_idx = dataset_index_by_path(test_set, str(fixed["path"]))
    sample = test_set[sample_idx]
    y_mid = sample["label"].shape[-1] // 2

    raw_by_aw = []
    ref = 0.0
    for abs_weight, experiment_name in SWEEP_SPECS:
        model = load_checkpoint_model(experiment_name, base_config, device)
        pred_raw, label_raw, _baseline_raw = infer_one_raw(model, sample, device)
        err_mag = complex_abs(pred_raw - label_raw)
        label_mag = complex_abs(label_raw)
        pred_mag = complex_abs(pred_raw)
        ref = max(ref, float(label_mag.max()), float(pred_mag.max()), float(err_mag.max()))
        raw_by_aw.append((abs_weight, experiment_name, label_mag, pred_mag, err_mag))

    fig, axes = plt.subplots(
        len(raw_by_aw),
        3,
        figsize=(12, 3.1 * len(raw_by_aw)),
        constrained_layout=True,
    )
    if len(raw_by_aw) == 1:
        axes = np.expand_dims(axes, axis=0)

    image = None
    for row_idx, (abs_weight, experiment_name, label_mag, pred_mag, err_mag) in enumerate(raw_by_aw):
        panels = [
            ("label", label_mag[:, :, y_mid]),
            ("pred", pred_mag[:, :, y_mid]),
            ("error", err_mag[:, :, y_mid]),
        ]
        for col_idx, (name, panel) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                to_db(panel, ref),
                cmap="gray",
                vmin=DB_MIN,
                vmax=0.0,
                origin="upper",
                aspect=ASPECT_XZ,
            )
            ax.set_title(f"aw={abs_weight:g} {name} | xz y={y_mid}")
            ax.set_xlabel("x index")
            ax.set_ylabel("z index")
        axes[row_idx, 0].set_ylabel(f"aw={abs_weight:g}\nz index")

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="dB")
    fig.suptitle(
        "abs_weight sweep fixed baseline-worst carotid patch | "
        f"idx={sample_idx} | {Path(str(fixed['path'])).name}",
        fontsize=10,
    )

    SWEEP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SWEEP_OUT_DIR / "absweight_sweep_fixed_carotid_worst_xz_triptych.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"saved_visual={save_path}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TASK_NAME)
    parser.add_argument(
        "--train-abs-weights",
        default="0.5,1.0,2.0",
        help="Comma-separated abs_weight values to train. Use empty string for eval-only.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate existing checkpoints.")
    return parser.parse_args()


def main() -> None:
    print_header()
    args = parse_args()
    base_config = load_base_config()
    seed_everything(int(base_config["seed"]))
    device = get_device()
    print(f"device={device}")

    _train_set, _val_set, test_set, train_loader, val_loader = make_loaders(base_config)

    train_abs_weights: set[float] = set()
    if not args.eval_only and args.train_abs_weights.strip():
        train_abs_weights = {float(v) for v in args.train_abs_weights.split(",") if v.strip()}
        unknown = train_abs_weights - TRAIN_ABS_WEIGHTS
        if unknown:
            raise ValueError(f"Unsupported training abs_weight values: {sorted(unknown)}")
    train_missing_models(base_config, train_loader, val_loader, device, train_abs_weights)

    rows = []
    for abs_weight, experiment_name in SWEEP_SPECS:
        rows.append(evaluate_checkpoint(abs_weight, experiment_name, base_config, test_set, device))

    summary = pd.DataFrame(rows)
    SWEEP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SWEEP_OUT_DIR / "absweight_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 96)
    print("abs_weight_sweep_summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"saved_summary={summary_path}")

    visual_path = save_visual_comparison(base_config, test_set, device)
    print(f"visual_path={visual_path}")


if __name__ == "__main__":
    main()
