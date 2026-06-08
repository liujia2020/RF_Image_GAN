from __future__ import annotations

import csv
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
    DEFAULT_FULL_SHAPE,
    DEFAULT_SAMPLE_GROUP,
    RFPatch,
    _format_histogram,
    _read_patch_tensors,
    _source_stem,
    check_coverage,
    load_dense_patches,
    make_hann3d_weight,
    resolve_patch_dir,
    stitch_volume,
)
from rf_train_utils import count_trainable_parameters, get_device, seed_everything
from rf_visualization import complex_abs_numpy, extract_slice, physical_aspect_for_view, physical_spacing_label, volume_to_db


TASK_NAME = "wide+SSIM full-volume stitch validation"
ROOT = Path(__file__).resolve().parent
INDEX_CSV = ROOT / "dense64_index.csv"
OUT_METRIC_DIR = ROOT / "test_metrics" / "wide_ssim_volume"
VIS_ROOT = ROOT / "vis_best_model"

BASELINE_EXPERIMENT = "tiny_bn_random64_full1500"
BASELINE_CKPT = ROOT / "checkpoint" / BASELINE_EXPERIMENT / "best_model.pth"
WIDE_EXPERIMENT = "wide_ssim_random64_full1500"
WIDE_CKPT = ROOT / "checkpoint" / WIDE_EXPERIMENT / "best_by_val_ssim.pth"

PATCH_SIZE = (64, 32, 32)
STRIDE = (64, 32, 32)
FULL_SHAPE = DEFAULT_FULL_SHAPE
Y_INDEX = 64
DB_MIN = -60.0
MAX_CASES = 1


@dataclass(frozen=True)
class Case:
    category: str
    source_file: str
    file_id: str
    patch_dir: Path
    disk: str

    @property
    def source_stem(self) -> str:
        return _source_stem(self.source_file)

    @property
    def case_id(self) -> str:
        return f"{self.file_id}_{self.source_stem}"


@dataclass(frozen=True)
class ModelSpec:
    row_name: str
    experiment: str
    ckpt_path: Path
    expected_model_class: str


def print_header() -> None:
    print("=" * 104)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"task: {TASK_NAME}")
    print("=" * 104)


def first_norm_name(model: torch.nn.Module) -> str:
    norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    return type(norm).__name__ if norm is not None else "MISSING"


def assert_startup(model: torch.nn.Module, expected_model_class: str, ckpt_path: Path) -> None:
    model_class = type(model).__name__
    norm_name = first_norm_name(model)
    n_params = count_trainable_parameters(model)
    print(f"startup ckpt={ckpt_path}")
    print(f"startup model_class={model_class}")
    print(f"startup first_norm_class={norm_name}")
    print(f"startup trainable_parameters={n_params:,}")
    assert model_class == expected_model_class, f"Expected {expected_model_class}, got {model_class}"
    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    assert isinstance(first_norm, nn.BatchNorm3d), f"Expected BatchNorm3d, got {norm_name}"


def load_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    if not spec.ckpt_path.exists():
        raise FileNotFoundError(spec.ckpt_path)
    ckpt = torch.load(spec.ckpt_path, map_location=device)
    config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    model_name = str(config.get("model_name", "tiny"))
    hidden = int(config.get("hidden", 64 if model_name == "tiny" else 128))
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
    assert_startup(model, spec.expected_model_class, spec.ckpt_path)
    print(f"loaded_epoch={ckpt.get('epoch', 'NA')}")
    return model


def read_cases(max_cases: int = MAX_CASES) -> tuple[list[Case], list[Case]]:
    if not INDEX_CSV.exists():
        raise FileNotFoundError(INDEX_CSV)
    all_carotid: list[Case] = []
    readable: list[Case] = []
    with INDEX_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("category") != "carotid":
                continue
            patch_dir = resolve_patch_dir(row["patch_dir"])
            case = Case(
                category=row["category"],
                source_file=row["source_file"],
                file_id=row["file_id"],
                patch_dir=patch_dir,
                disk=row.get("disk", ""),
            )
            all_carotid.append(case)
            h5_count = len(list(patch_dir.glob("*.h5"))) if patch_dir.exists() else 0
            print(
                f"case_probe {case.file_id} {case.source_file} disk={case.disk} "
                f"path={patch_dir} exists={patch_dir.exists()} h5_count={h5_count}"
            )
            if h5_count == 256:
                readable.append(case)

    selected: list[Case] = []
    preferred = [case for case in readable if case.source_file == "Carotid_012.mat"]
    if preferred:
        selected.extend(preferred[:max_cases])
    else:
        selected.extend(readable[:max_cases])

    if not selected:
        raise RuntimeError("No readable carotid dense64 cases were found.")

    if not any(case.source_file == "Carotid_012.mat" for case in selected):
        print("WARNING: preferred Carotid_012.mat is not readable now; using first readable carotid case(s).")
    return selected, all_carotid


def output_dir_for(spec: ModelSpec, case: Case) -> Path:
    return (
        VIS_ROOT
        / spec.experiment
        / "full_volume_stitch_validation"
        / f"{case.file_id}_{case.source_stem}_frame1_dense64_hann"
    )


def scale_to_broadcast(scale: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while scale.ndim < reference.ndim:
        scale = scale.view(*scale.shape, *([1] * (reference.ndim - scale.ndim)))
    return scale


@torch.no_grad()
def infer_and_stitch_case(spec: ModelSpec, case: Case, device: torch.device) -> dict[str, np.ndarray]:
    out_dir = output_dir_for(spec, case)
    model = load_model(spec, device)

    patches = load_dense_patches(case.patch_dir, fields=(), sample_group=DEFAULT_SAMPLE_GROUP)
    print(f"\ncase={case.case_id} model={spec.row_name}")
    print(f"patch_dir={case.patch_dir}")
    print(f"num_patches={len(patches)}")
    coverage = check_coverage(patches, FULL_SHAPE)
    print("coverage check: PASS")
    print(f"coverage min/max: {coverage['coverage_min']} / {coverage['coverage_max']}")
    print(f"coverage histogram: {_format_histogram(coverage['coverage_histogram'])}")
    print(f"missing voxels: {coverage['missing_voxels']}")
    print(f"overlapped voxels: {coverage['overlapped_voxels']}")

    first = patches[0]
    patch_size = (len(first.z_idx), len(first.x_idx), len(first.y_idx))
    assert patch_size == PATCH_SIZE, f"Expected patch_size={PATCH_SIZE}, got {patch_size}"
    hann = make_hann3d_weight(PATCH_SIZE, floor=0.01)
    print(
        f"weighted fusion: Hann 3D | shape={hann.shape} "
        f"min={float(hann.min()):.6e} max={float(hann.max()):.6e}"
    )

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

        del x, y, b, x_norm, y_norm, b_norm, pred, scale
        if device.type == "cuda" and i % 32 == 0:
            torch.cuda.empty_cache()
        if i == 1 or i % 64 == 0 or i == len(patches):
            print(f"inferred {i:04d}/{len(patches)} patches")

    pred, weight = stitch_volume(patches, FULL_SHAPE, "pred", weight=hann, return_weight=True)
    label = stitch_volume(patches, FULL_SHAPE, "label", weight=hann)
    baseline = stitch_volume(patches, FULL_SHAPE, "baseline", weight=hann)

    zero_count = int(np.count_nonzero(weight <= 0))
    assert zero_count == 0, f"Weight buffer has {zero_count} zero voxels"
    print(
        f"weight-buffer self-check: PASS | min={float(weight.min()):.6e} "
        f"max={float(weight.max()):.6e} zero_count={zero_count}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pred.npy", pred)
    np.save(out_dir / "label.npy", label)
    np.save(out_dir / "baseline.npy", baseline)
    meta = "\n".join(
        [
            f"task: {TASK_NAME}",
            f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
            f"case_id: {case.case_id}",
            f"category: {case.category}",
            f"source_file: {case.source_file}",
            f"file_id: {case.file_id}",
            f"patch_dir: {case.patch_dir}",
            f"model_row: {spec.row_name}",
            f"experiment: {spec.experiment}",
            f"ckpt_path: {spec.ckpt_path}",
            f"patch_size: {PATCH_SIZE}",
            f"stride: {STRIDE}",
            f"full_shape: {FULL_SHAPE}",
            "fusion: hann3d_floor0.01",
            f"weight_min: {float(weight.min()):.9e}",
            f"weight_max: {float(weight.max()):.9e}",
            f"weight_zero_count: {zero_count}",
            "",
        ]
    )
    (out_dir / "meta.txt").write_text(meta, encoding="utf-8")
    print(f"saved_volume_dir={out_dir}")
    return {"pred": pred, "label": label, "baseline": baseline}


def volume_dynamic_metrics(vols: dict[str, np.ndarray]) -> dict[str, float]:
    pred_abs = complex_abs_numpy(vols["pred"])
    label_abs = complex_abs_numpy(vols["label"])
    baseline_abs = complex_abs_numpy(vols["baseline"])
    eps = 1e-12
    pred_p99 = float(np.percentile(pred_abs, 99.0))
    label_p99 = float(np.percentile(label_abs, 99.0))
    baseline_p99 = float(np.percentile(baseline_abs, 99.0))
    return {
        "pred_p99": pred_p99,
        "label_p99": label_p99,
        "baseline_p99": baseline_p99,
        "pred_p99_over_label_p99": pred_p99 / (label_p99 + eps),
        "baseline_p99_over_label_p99": baseline_p99 / (label_p99 + eps),
        "abs_std_ratio": float(np.std(pred_abs) / (np.std(label_abs) + eps)),
        "baseline_abs_std_ratio": float(np.std(baseline_abs) / (np.std(label_abs) + eps)),
    }


def save_comparison_figures(case: Case, baseline_vols: dict[str, np.ndarray], wide_vols: dict[str, np.ndarray]) -> list[Path]:
    out_dir = OUT_METRIC_DIR / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_vols = [baseline_vols["label"], baseline_vols["pred"], wide_vols["pred"]]
    ref = max(float(np.max(complex_abs_numpy(volume))) for volume in all_vols)

    panels = [
        ("label", baseline_vols["label"]),
        ("BN-L1 pred", baseline_vols["pred"]),
        ("wide+SSIM pred", wide_vols["pred"]),
    ]
    specs = [
        ("xz", 64, f"{case.case_id}_bmode_xz_y64.png"),
        ("xz", 32, f"{case.case_id}_bmode_xz_y32.png"),
        ("xy", 512, f"{case.case_id}_bmode_xy_z512.png"),
    ]
    saved: list[Path] = []
    for view, index, filename in specs:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
        im = None
        for ax, (title, volume) in zip(axes, panels):
            db = volume_to_db(complex_abs_numpy(volume), ref=ref, db_min=DB_MIN)
            img, used, xlabel, ylabel = extract_slice(db, view=view, slice_index=index)
            im = ax.imshow(
                img,
                cmap="gray",
                vmin=DB_MIN,
                vmax=0.0,
                origin="upper",
                aspect=physical_aspect_for_view(view),
            )
            ax.set_title(f"{title} | {view}={used}", fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="dB")
        fig.suptitle(
            f"{case.case_id} | shared dB ref | {physical_spacing_label(view)}",
            fontsize=10,
        )
        path = out_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)
        print(f"saved_figure={path}")
    return saved


def summarize_case(case: Case, rows: list[dict[str, Any]]) -> None:
    print("\nsummary_case", case.case_id)
    print(
        "model,x_boundary_pred_over_label,pred_p99_over_label_p99,"
        "abs_std_ratio,nonboundary_complex_improvement,nonboundary_abs_improvement"
    )
    for row in rows:
        print(
            f"{row['model']},"
            f"{row['x_boundary_pred_over_label']:.6f},"
            f"{row['pred_p99_over_label_p99']:.6f},"
            f"{row['abs_std_ratio']:.6f},"
            f"{row['nonboundary_complex_improvement']:.6f},"
            f"{row['nonboundary_abs_improvement']:.6f}"
        )


def main() -> None:
    print_header()
    seed_everything(20260522)
    device = get_device()
    print(f"device={device}")
    OUT_METRIC_DIR.mkdir(parents=True, exist_ok=True)

    selected_cases, _all_cases = read_cases(MAX_CASES)
    specs = [
        ModelSpec("BN-L1", BASELINE_EXPERIMENT, BASELINE_CKPT, "TinyResidualRFNet"),
        ModelSpec("wide+SSIM", WIDE_EXPERIMENT, WIDE_CKPT, "WideDeepResidualRFNet"),
    ]

    all_rows: list[dict[str, Any]] = []
    all_figures: list[Path] = []

    for case in selected_cases:
        vols_by_model: dict[str, dict[str, np.ndarray]] = {}
        for spec in specs:
            vols_by_model[spec.row_name] = infer_and_stitch_case(spec, case, device)

        case_rows: list[dict[str, Any]] = []
        for spec in specs:
            vols = vols_by_model[spec.row_name]
            row = metric_row(
                case.case_id,
                spec.row_name,
                vols,
                tuple(DEFAULT_X_BOUNDARIES),
                Y_INDEX,
                PATCH_SIZE,
                boundary_margin=3,
            )
            row.update(
                {
                    "category": case.category,
                    "source_file": case.source_file,
                    "file_id": case.file_id,
                    "patch_dir": str(case.patch_dir),
                    "experiment": spec.experiment,
                    "checkpoint": str(spec.ckpt_path),
                }
            )
            row.update(volume_dynamic_metrics(vols))
            case_rows.append(row)
            all_rows.append(row)

        all_figures.extend(save_comparison_figures(case, vols_by_model["BN-L1"], vols_by_model["wide+SSIM"]))
        summarize_case(case, case_rows)

    metrics_path = OUT_METRIC_DIR / "wide_ssim_volume_stitch_metrics.csv"
    pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
    print(f"\nsaved_metrics={metrics_path}")
    print("saved_figures:")
    for path in all_figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
