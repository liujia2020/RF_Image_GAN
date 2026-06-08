from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rf_bn_multivolume_diagnostics import DEFAULT_X_BOUNDARIES, metric_row
from rf_models import build_model
from rf_stitch import (
    DEFAULT_SAMPLE_GROUP,
    RFPatch,
    _format_histogram,
    _read_patch_tensors,
    load_dense_patches,
    make_hann3d_weight,
    stitch_volume,
)
from rf_train_utils import count_trainable_parameters, get_device, seed_everything
from rf_visualization import complex_abs_numpy, extract_slice, physical_aspect_for_view, physical_spacing_label, volume_to_db


TASK_NAME = "wide+SSIM z-band overlap/crop seam fix"
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "test_metrics" / "wide_ssim_zband"
VIS_DIR = ROOT / "vis_best_model" / "wide_ssim_random64_full1500" / "zband_seamfix"
CKPT_PATH = ROOT / "checkpoint" / "wide_ssim_random64_full1500" / "best_by_val_ssim.pth"
CONFIG1_DIR = (
    ROOT
    / "vis_best_model"
    / "wide_ssim_random64_full1500"
    / "full_volume_stitch_validation"
    / "RF000489_Carotid_008_frame1_dense64_hann"
)
HALFSTRIDE_PATCH_DIR = Path("/mnt/f/DAS/RF_LearningSamples_dense_RF000489_zband448-576_halfstride/test/carotid")
CROP8_PATCH_DIR = Path("/mnt/f/DAS/RF_LearningSamples_dense_RF000489_zband448-576_crop8/test/carotid")

CASE_ID = "RF000489_Carotid_008"
PATCH_SIZE = (64, 32, 32)
BAND_Z_RANGE_1BASED = (448, 576)
BAND_SHAPE = (BAND_Z_RANGE_1BASED[1] - BAND_Z_RANGE_1BASED[0] + 1, 128, 128)
FULL_Z_FOR_VIS_0BASED = 512
LOCAL_Z_FOR_VIS = FULL_Z_FOR_VIS_0BASED - (BAND_Z_RANGE_1BASED[0] - 1)
Y_INDEX = 64
BOUNDARY_MARGIN = 3
DB_MIN = -60.0
CROP = (8, 8, 8)


@dataclass(frozen=True)
class StitchConfig:
    name: str
    patch_dir: Path | None
    fusion: str
    crop: tuple[int, int, int] | None = None


def print_header() -> None:
    print("=" * 104)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"task: {TASK_NAME}")
    print("=" * 104)


def first_norm_name(model: torch.nn.Module) -> str:
    norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    return type(norm).__name__ if norm is not None else "MISSING"


def load_wide_model(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(CKPT_PATH, map_location=device)
    config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    model_name = str(config.get("model_name", "wide_deep"))
    hidden = int(config.get("hidden", 64))
    use_bn = any("running_mean" in key for key in ckpt["model"])
    model = build_model(
        model_name,
        in_channels=1536,
        hidden=hidden,
        head_channels=32,
        out_channels=2,
        num_blocks=4,
        use_batch_norm=use_bn,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    model_class = type(model).__name__
    norm_class = first_norm_name(model)
    print(f"startup ckpt={CKPT_PATH}")
    print(f"startup model_class={model_class}")
    print(f"startup first_norm_class={norm_class}")
    print(f"startup trainable_parameters={count_trainable_parameters(model):,}")
    print(f"startup checkpoint_epoch={ckpt.get('epoch', 'NA')}")
    assert model_class == "WideDeepResidualRFNet", model_class
    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    assert isinstance(first_norm, nn.BatchNorm3d), norm_class
    return model


def coverage_summary(patches: Sequence[RFPatch], full_shape: Sequence[int]) -> dict[str, Any]:
    coverage = np.zeros(tuple(full_shape), dtype=np.uint16)
    for patch in patches:
        z0, z1 = int(patch.z_idx[0]) - 1, int(patch.z_idx[-1])
        x0, x1 = int(patch.x_idx[0]) - 1, int(patch.x_idx[-1])
        y0, y1 = int(patch.y_idx[0]) - 1, int(patch.y_idx[-1])
        coverage[z0:z1, x0:x1, y0:y1] += 1
    unique, counts = np.unique(coverage, return_counts=True)
    hist = {int(k): int(v) for k, v in zip(unique, counts)}
    return {
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "missing_voxels": int(np.count_nonzero(coverage == 0)),
        "overlapped_voxels": int(np.count_nonzero(coverage > 1)),
        "coverage_histogram": hist,
    }


def shift_patch_to_band(patch: RFPatch) -> RFPatch:
    return RFPatch(
        path=patch.path,
        z_idx=(np.asarray(patch.z_idx, dtype=np.int32) - BAND_Z_RANGE_1BASED[0] + 1).astype(np.int32),
        x_idx=patch.x_idx,
        y_idx=patch.y_idx,
        source_file=patch.source_file,
        frame_id=patch.frame_id,
        data=patch.data,
    )


def crop_patch_to_valid_center(patch: RFPatch, crop: tuple[int, int, int]) -> RFPatch:
    z_idx_abs = np.asarray(patch.z_idx, dtype=np.int32)
    x_idx = np.asarray(patch.x_idx, dtype=np.int32)
    y_idx = np.asarray(patch.y_idx, dtype=np.int32)
    idxs = [z_idx_abs, x_idx, y_idx]
    bounds = [(BAND_Z_RANGE_1BASED[0], BAND_Z_RANGE_1BASED[1]), (1, 128), (1, 128)]
    data_slices = []
    new_idxs = []
    for idx, (lo, hi), c in zip(idxs, bounds, crop):
        before = 0 if int(idx[0]) == lo else int(c)
        after = 0 if int(idx[-1]) == hi else int(c)
        stop = len(idx) - after
        if before >= stop:
            raise ValueError(f"Invalid crop for {patch.path}: idx={idx[0]}..{idx[-1]} crop={c}")
        data_slices.append(slice(before, stop))
        new_idxs.append(idx[before:stop])

    z_slice, x_slice, y_slice = data_slices
    data: dict[str, np.ndarray] = {}
    for field, arr in patch.data.items():
        data[field] = arr[:, z_slice, x_slice, y_slice].astype(np.float32, copy=False)

    return RFPatch(
        path=patch.path,
        z_idx=(new_idxs[0] - BAND_Z_RANGE_1BASED[0] + 1).astype(np.int32),
        x_idx=new_idxs[1].astype(np.int32),
        y_idx=new_idxs[2].astype(np.int32),
        source_file=patch.source_file,
        frame_id=patch.frame_id,
        data=data,
    )


def scale_to_broadcast(scale: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while scale.ndim < reference.ndim:
        scale = scale.view(*scale.shape, *([1] * (reference.ndim - scale.ndim)))
    return scale


@torch.no_grad()
def infer_patch_dir(model: torch.nn.Module, patch_dir: Path, device: torch.device, config_name: str) -> list[RFPatch]:
    patches = load_dense_patches(patch_dir, fields=(), sample_group=DEFAULT_SAMPLE_GROUP)
    print(f"\n{config_name} patch_dir={patch_dir}")
    print(f"{config_name} patch_count={len(patches)}")
    starts = {
        "z": sorted({int(p.z_idx[0]) for p in patches}),
        "x": sorted({int(p.x_idx[0]) for p in patches}),
        "y": sorted({int(p.y_idx[0]) for p in patches}),
    }
    print(f"{config_name} z_starts={starts['z']}")
    print(f"{config_name} x_starts={starts['x']}")
    print(f"{config_name} y_starts={starts['y']}")

    local_meta = [shift_patch_to_band(p) for p in patches]
    cov = coverage_summary(local_meta, BAND_SHAPE)
    print(
        f"{config_name} raw coverage min/max/missing/overlap="
        f"{cov['coverage_min']}/{cov['coverage_max']}/{cov['missing_voxels']}/{cov['overlapped_voxels']}"
    )
    print(f"{config_name} raw coverage histogram={_format_histogram(cov['coverage_histogram'])}")
    assert cov["coverage_min"] >= 1 and cov["missing_voxels"] == 0

    inferred: list[RFPatch] = []
    use_amp = device.type == "cuda"
    for i, patch in enumerate(patches, start=1):
        x, y, b = _read_patch_tensors(patch.path, DEFAULT_SAMPLE_GROUP, device)
        scale = torch.amax(torch.abs(y)) + 1e-8
        scale = scale_to_broadcast(scale, y)
        x_norm = x / scale
        y_norm = y / scale
        b_norm = b / scale

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(x_norm, b_norm)

        patch.data["pred"] = (pred.float() * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        patch.data["label"] = (y_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        patch.data["baseline"] = (b_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        inferred.append(patch)

        del x, y, b, x_norm, y_norm, b_norm, pred, scale
        if device.type == "cuda" and i % 16 == 0:
            torch.cuda.empty_cache()
        if i == 1 or i % 32 == 0 or i == len(patches):
            print(f"{config_name} inferred {i:04d}/{len(patches)} patches")
    return inferred


def stitch_inferred(
    patches_abs: list[RFPatch],
    config: StitchConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if config.crop is None:
        patches = [shift_patch_to_band(p) for p in patches_abs]
    else:
        patches = [crop_patch_to_valid_center(p, config.crop) for p in patches_abs]

    cov = coverage_summary(patches, BAND_SHAPE)
    print(
        f"{config.name} effective coverage min/max/missing/overlap="
        f"{cov['coverage_min']}/{cov['coverage_max']}/{cov['missing_voxels']}/{cov['overlapped_voxels']}"
    )
    print(f"{config.name} effective coverage histogram={_format_histogram(cov['coverage_histogram'])}")
    assert cov["coverage_min"] >= 1 and cov["missing_voxels"] == 0

    weight = make_hann3d_weight(PATCH_SIZE, floor=0.01) if config.fusion == "hann" else None
    pred, weight_buf = stitch_volume(patches, BAND_SHAPE, "pred", weight=weight, return_weight=True)
    label = stitch_volume(patches, BAND_SHAPE, "label", weight=weight)
    baseline = stitch_volume(patches, BAND_SHAPE, "baseline", weight=weight)
    zero_count = int(np.count_nonzero(weight_buf <= 0))
    assert zero_count == 0, f"{config.name} weight buffer has zeros: {zero_count}"
    print(
        f"{config.name} weight self-check PASS | "
        f"min={float(weight_buf.min()):.6e} max={float(weight_buf.max()):.6e} zero={zero_count}"
    )
    return {"pred": pred, "label": label, "baseline": baseline}, cov


def load_config1_zband() -> dict[str, np.ndarray]:
    z0 = BAND_Z_RANGE_1BASED[0] - 1
    z1 = BAND_Z_RANGE_1BASED[1]
    out = {}
    for field in ("baseline", "pred", "label"):
        path = CONFIG1_DIR / f"{field}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        out[field] = np.load(path)[:, z0:z1, :, :].astype(np.float32, copy=False)
        print(f"config1 loaded {field}: {path} -> {out[field].shape}")
    return out


def dynamic_metrics(vols: dict[str, np.ndarray]) -> dict[str, float]:
    pred_abs = complex_abs_numpy(vols["pred"])
    label_abs = complex_abs_numpy(vols["label"])
    eps = 1e-12
    return {
        "pred_p99": float(np.percentile(pred_abs, 99.0)),
        "label_p99": float(np.percentile(label_abs, 99.0)),
        "pred_p99_over_label_p99": float(np.percentile(pred_abs, 99.0) / (np.percentile(label_abs, 99.0) + eps)),
        "abs_std_ratio": float(np.std(pred_abs) / (np.std(label_abs) + eps)),
    }


def metrics_for(config_name: str, vols: dict[str, np.ndarray], coverage: dict[str, Any] | None) -> dict[str, Any]:
    row = metric_row(
        CASE_ID,
        config_name,
        vols,
        tuple(DEFAULT_X_BOUNDARIES),
        Y_INDEX,
        PATCH_SIZE,
        BOUNDARY_MARGIN,
    )
    row.update(dynamic_metrics(vols))
    row.update(
        {
            "stitch_config": config_name,
            "band_z_range_1based": f"{BAND_Z_RANGE_1BASED[0]}-{BAND_Z_RANGE_1BASED[1]}",
            "local_shape": str(BAND_SHAPE),
        }
    )
    if coverage:
        row.update(
            {
                "effective_coverage_min": coverage["coverage_min"],
                "effective_coverage_max": coverage["coverage_max"],
                "effective_missing_voxels": coverage["missing_voxels"],
                "effective_overlapped_voxels": coverage["overlapped_voxels"],
            }
        )
    return row


def x_jump_rows(vols_by_config: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_name, vols in vols_by_config.items():
        mags = {field: complex_abs_numpy(vol) for field, vol in vols.items()}
        for x in DEFAULT_X_BOUNDARIES:
            label_jump = float(np.mean(np.abs(mags["label"][:, x, :] - mags["label"][:, x - 1, :])))
            pred_jump = float(np.mean(np.abs(mags["pred"][:, x, :] - mags["pred"][:, x - 1, :])))
            baseline_jump = float(np.mean(np.abs(mags["baseline"][:, x, :] - mags["baseline"][:, x - 1, :])))
            rows.append(
                {
                    "config": config_name,
                    "x_boundary": x,
                    "pred_jump": pred_jump,
                    "label_jump": label_jump,
                    "baseline_jump": baseline_jump,
                    "pred_over_label": pred_jump / (label_jump + 1e-12),
                    "baseline_over_label": baseline_jump / (label_jump + 1e-12),
                }
            )
    return rows


def save_volumes(config_name: str, vols: dict[str, np.ndarray], coverage: dict[str, Any] | None) -> Path:
    out_dir = VIS_DIR / f"{CASE_ID}_zband448-576_{config_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for field, vol in vols.items():
        np.save(out_dir / f"{field}.npy", vol)
    meta_lines = [
        f"task: {TASK_NAME}",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"case_id: {CASE_ID}",
        f"checkpoint: {CKPT_PATH}",
        f"band_z_range_1based: {BAND_Z_RANGE_1BASED}",
        f"band_shape: {BAND_SHAPE}",
        f"config: {config_name}",
        f"coverage: {coverage}",
        "",
    ]
    (out_dir / "meta.txt").write_text("\n".join(meta_lines), encoding="utf-8")
    print(f"saved_zband_volume_dir={out_dir}")
    return out_dir


def save_visual(vols_by_config: dict[str, dict[str, np.ndarray]]) -> Path:
    out_dir = OUT_DIR / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = ["config1_current_nooverlap", "config3_crop8_center"]
    ref = max(float(np.max(complex_abs_numpy(vols_by_config[name]["pred"]))) for name in configs)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    for ax, name in zip(axes, configs):
        db = volume_to_db(complex_abs_numpy(vols_by_config[name]["pred"]), ref=ref, db_min=DB_MIN)
        img, used, xlabel, ylabel = extract_slice(db, view="xy", slice_index=LOCAL_Z_FOR_VIS)
        im = ax.imshow(img, cmap="gray", vmin=DB_MIN, vmax=0.0, origin="upper", aspect=physical_aspect_for_view("xy"))
        for boundary in DEFAULT_X_BOUNDARIES:
            ax.axhline(boundary - 0.5, color="tab:red", linewidth=0.7, alpha=0.7)
        ax.set_title(f"{name}\nxy local_z={used} full_z0={FULL_Z_FOR_VIS_0BASED}", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="dB")
    fig.suptitle(f"{CASE_ID} | wide+SSIM | config1 vs config3 | {physical_spacing_label('xy')}", fontsize=10)
    path = out_dir / f"{CASE_ID}_wide_ssim_config1_vs_config3_xy_z{FULL_Z_FOR_VIS_0BASED}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved_visual={path}")
    return path


def save_x_jump_plot(rows: list[dict[str, Any]]) -> Path:
    out_dir = OUT_DIR / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for config_name in df["config"].unique():
        sub = df[df["config"] == config_name]
        ax.plot(sub["x_boundary"], sub["pred_over_label"], marker="o", label=config_name)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(DEFAULT_X_BOUNDARIES))
    ax.set_xlabel("x boundary")
    ax.set_ylabel("pred jump / label jump")
    ax.set_title(f"{CASE_ID} | z-band x-jump ratios")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    path = out_dir / f"{CASE_ID}_wide_ssim_zband_x_jump_ratios.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved_x_jump_plot={path}")
    return path


def main() -> None:
    print_header()
    seed_everything(20260522)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device={device}")
    print(f"band_z_range_1based={BAND_Z_RANGE_1BASED}")
    print(f"band_shape={BAND_SHAPE}")
    print(f"full_z_for_visual_0based={FULL_Z_FOR_VIS_0BASED} local_z={LOCAL_Z_FOR_VIS}")

    model = load_wide_model(device)
    config1_vols = load_config1_zband()
    save_volumes("config1_current_nooverlap", config1_vols, None)

    configs = [
        StitchConfig("config2_halfstride_hann", HALFSTRIDE_PATCH_DIR, "hann", None),
        StitchConfig("config3_crop8_center", CROP8_PATCH_DIR, "uniform", CROP),
    ]

    vols_by_config: dict[str, dict[str, np.ndarray]] = {"config1_current_nooverlap": config1_vols}
    coverage_by_config: dict[str, dict[str, Any] | None] = {"config1_current_nooverlap": None}

    for config in configs:
        assert config.patch_dir is not None
        inferred = infer_patch_dir(model, config.patch_dir, device, config.name)
        vols, cov = stitch_inferred(inferred, config)
        vols_by_config[config.name] = vols
        coverage_by_config[config.name] = cov
        save_volumes(config.name, vols, cov)

    metric_rows = [
        metrics_for(config_name, vols_by_config[config_name], coverage_by_config[config_name])
        for config_name in vols_by_config
    ]
    metrics_path = OUT_DIR / "wide_ssim_zband_seamfix_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f"saved_metrics={metrics_path}")

    jump_rows = x_jump_rows(vols_by_config)
    jump_path = OUT_DIR / "wide_ssim_zband_x_jump_profile.csv"
    pd.DataFrame(jump_rows).to_csv(jump_path, index=False)
    print(f"saved_x_jump_profile={jump_path}")

    visual_path = save_visual(vols_by_config)
    x_jump_plot = save_x_jump_plot(jump_rows)

    summary_cols = [
        "stitch_config",
        "x_boundary_pred_over_label",
        "xz_y_boundary_pred_over_label",
        "pred_p99_over_label_p99",
        "abs_std_ratio",
        "nonboundary_complex_improvement",
        "nonboundary_abs_improvement",
        "effective_coverage_min",
        "effective_coverage_max",
        "effective_missing_voxels",
    ]
    print("\nsummary_metrics")
    print(pd.DataFrame(metric_rows).reindex(columns=summary_cols).to_string(index=False))
    print("\nx_boundary_rows")
    print(
        pd.DataFrame(jump_rows)
        .sort_values(["config", "x_boundary"])
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print("\nsaved_outputs")
    print(f"metrics={metrics_path}")
    print(f"x_jump_profile={jump_path}")
    print(f"visual={visual_path}")
    print(f"x_jump_plot={x_jump_plot}")


if __name__ == "__main__":
    main()
