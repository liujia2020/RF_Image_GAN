from __future__ import annotations

import argparse
import shutil
import subprocess
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
from rf_stitch import DEFAULT_SAMPLE_GROUP, _format_histogram, _read_patch_tensors, load_dense_patches
from rf_train_utils import count_trainable_parameters, get_device, seed_everything
from rf_visualization import complex_abs_numpy, extract_slice, physical_aspect_for_view, physical_spacing_label, volume_to_db


TASK_NAME = "wide+SSIM full-volume crop8 streamed validation"
ROOT = Path(__file__).resolve().parent
MATLAB_TEMPLATE = ROOT / "MATLAB_dense_generation" / "build_RF_dense_zband_RF000489_crop8.m"
MATLAB_TMP_DIR = ROOT / "MATLAB_dense_generation" / "_stream_tmp"
MATLAB_LOG_DIR = ROOT / "MATLAB_dense_generation" / "_stream_logs"
OUT_DIR = ROOT / "test_metrics" / "wide_ssim_fullvol"
VIS_DIR = (
    ROOT
    / "vis_best_model"
    / "wide_ssim_random64_full1500"
    / "full_volume_stream_crop8"
    / "RF000489_Carotid_008_frame1_dense64_crop8_streamed"
)
WIDE_CKPT = ROOT / "checkpoint" / "wide_ssim_random64_full1500" / "best_by_val_ssim.pth"
BN_DIR = (
    ROOT
    / "vis_best_model"
    / "tiny_bn_random64_full1500"
    / "full_volume_stitch_validation"
    / "RF000489_Carotid_008_frame1_dense64_hann"
)

CASE_ID = "RF000489_Carotid_008"
FULL_SHAPE = (1024, 128, 128)
PATCH_SIZE = (64, 32, 32)
STRIDE = (48, 16, 16)
CROP = (8, 8, 8)
Y_INDEX = 64
BOUNDARY_MARGIN = 3
DB_MIN = -60.0
MATLAB_EXE = r"D:\Program Files\MATLAB\R2024b\bin\matlab.exe"
TEMP_ROOT_PREFIX_WIN = r"F:\DAS\RF_LearningSamples_dense_RF000489_fullvol_stream_crop8_"
TEMP_ROOT_PREFIX_WSL = "/mnt/f/DAS/RF_LearningSamples_dense_RF000489_fullvol_stream_crop8_"


@dataclass(frozen=True)
class Band:
    index: int
    z_starts: tuple[int, ...]

    @property
    def z_min(self) -> int:
        return self.z_starts[0]

    @property
    def z_max(self) -> int:
        return self.z_starts[-1] + PATCH_SIZE[0] - 1

    @property
    def expected_patches(self) -> int:
        return len(self.z_starts) * 7 * 7

    @property
    def name(self) -> str:
        return f"band{self.index:02d}_z{self.z_min:04d}-{self.z_max:04d}"

    @property
    def output_root_win(self) -> str:
        return TEMP_ROOT_PREFIX_WIN + self.name

    @property
    def output_root_wsl(self) -> Path:
        return Path(TEMP_ROOT_PREFIX_WSL + self.name)

    @property
    def patch_dir_wsl(self) -> Path:
        return self.output_root_wsl / "test" / "carotid"


def print_header() -> None:
    print("=" * 104, flush=True)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"task: {TASK_NAME}", flush=True)
    print("=" * 104, flush=True)


def z_start_grid() -> list[int]:
    starts = list(range(1, FULL_SHAPE[0] - PATCH_SIZE[0] + 2, STRIDE[0]))
    assert starts[-1] == FULL_SHAPE[0] - PATCH_SIZE[0] + 1
    return starts


def make_bands(starts_per_band: int = 3) -> list[Band]:
    starts = z_start_grid()
    return [
        Band(band_index, tuple(starts[i : i + starts_per_band]))
        for band_index, i in enumerate(range(0, len(starts), starts_per_band), start=1)
    ]


def first_norm_name(model: torch.nn.Module) -> str:
    norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    return type(norm).__name__ if norm is not None else "MISSING"


def load_wide_model(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(WIDE_CKPT, map_location=device)
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
    print(f"startup ckpt={WIDE_CKPT}", flush=True)
    print(f"startup model_class={model_class}", flush=True)
    print(f"startup first_norm_class={norm_class}", flush=True)
    print(f"startup trainable_parameters={count_trainable_parameters(model):,}", flush=True)
    print(f"startup checkpoint_epoch={ckpt.get('epoch', 'NA')}", flush=True)
    assert model_class == "WideDeepResidualRFNet", model_class
    first_norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    assert isinstance(first_norm, nn.BatchNorm3d), norm_class
    return model


def scale_to_broadcast(scale: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while scale.ndim < reference.ndim:
        scale = scale.view(*scale.shape, *([1] * (reference.ndim - scale.ndim)))
    return scale


def safe_rmtree(path: Path) -> None:
    path = path.resolve()
    expected = Path(TEMP_ROOT_PREFIX_WSL).parent.resolve()
    if expected not in path.parents:
        raise ValueError(f"Refusing to delete outside F DAS temp root: {path}")
    if not path.name.startswith(Path(TEMP_ROOT_PREFIX_WSL).name):
        raise ValueError(f"Refusing to delete unexpected temp directory: {path}")
    if path.exists():
        print(f"delete_temp_dir={path}", flush=True)
        shutil.rmtree(path)


def render_matlab_script(band: Band) -> Path:
    MATLAB_TMP_DIR.mkdir(parents=True, exist_ok=True)
    text = MATLAB_TEMPLATE.read_text(encoding="utf-8-sig")
    text = text.replace(
        "OutputRoot = 'F:\\DAS\\RF_LearningSamples_dense_RF000489_zband448-576_crop8';",
        f"OutputRoot = '{band.output_root_win}';",
    )
    text = text.replace(
        "'RF_manifest_dense_RF000489_zband448-576_crop8.csv'",
        f"'RF_manifest_dense_RF000489_fullvol_stream_crop8_{band.name}.csv'",
    )
    text = text.replace("z_range = [448, 576];", f"z_range = [{band.z_min}, {band.z_max}];")
    matlab_safe_name = band.name.replace("-", "_")
    tmp_path = MATLAB_TMP_DIR / f"build_RF_dense_RF000489_fullvol_stream_crop8_{matlab_safe_name}.m"
    tmp_path.write_text(text, encoding="utf-8")
    return tmp_path


def run_matlab_generation(band: Band) -> None:
    safe_rmtree(band.output_root_wsl)
    tmp_script = render_matlab_script(band)
    MATLAB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout = MATLAB_LOG_DIR / f"{band.name}_stdout.txt"
    stderr = MATLAB_LOG_DIR / f"{band.name}_stderr.txt"
    repo_win = r"\\wsl.localhost\Ubuntu\home\liujia\RF_Image"
    script_win = rf"\\wsl.localhost\Ubuntu\home\liujia\RF_Image\MATLAB_dense_generation\_stream_tmp\{tmp_script.name}"
    ps = (
        "$matlab = 'D:\\Program Files\\MATLAB\\R2024b\\bin\\matlab.exe'; "
        f"$repo = '{repo_win}'; "
        f"$script = '{script_win}'; "
        f"$stdout = '{repo_win}\\MATLAB_dense_generation\\_stream_logs\\{stdout.name}'; "
        f"$stderr = '{repo_win}\\MATLAB_dense_generation\\_stream_logs\\{stderr.name}'; "
        "$cmd = \"cd('$repo'); run('$script');\"; "
        "& $matlab -batch $cmd 1> $stdout 2> $stderr; "
        "exit $LASTEXITCODE"
    )
    print(
        f"\nmatlab_generate {band.name}: z_range=[{band.z_min},{band.z_max}] "
        f"expected_patches={band.expected_patches}",
        flush=True,
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        tail = ""
        if stdout.exists():
            tail += "\n--- MATLAB stdout tail ---\n" + "\n".join(stdout.read_text(errors="replace").splitlines()[-80:])
        if stderr.exists():
            tail += "\n--- MATLAB stderr tail ---\n" + "\n".join(stderr.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"MATLAB generation failed for {band.name}: code={result.returncode}{tail}")

    count = len(list(band.patch_dir_wsl.glob("*.h5")))
    print(f"matlab_done {band.name}: h5_count={count}", flush=True)
    if count != band.expected_patches:
        raise AssertionError(f"{band.name}: expected {band.expected_patches} patches, got {count}")


def crop_slices_for_patch(patch: Any) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    z_idx = np.asarray(patch.z_idx, dtype=np.int32)
    x_idx = np.asarray(patch.x_idx, dtype=np.int32)
    y_idx = np.asarray(patch.y_idx, dtype=np.int32)
    idxs = [z_idx, x_idx, y_idx]
    bounds = [(1, FULL_SHAPE[0]), (1, FULL_SHAPE[1]), (1, FULL_SHAPE[2])]
    crop = CROP
    data_slices = []
    out_slices = []
    for idx, (lo, hi), c in zip(idxs, bounds, crop):
        before = 0 if int(idx[0]) == lo else int(c)
        after = 0 if int(idx[-1]) == hi else int(c)
        stop = len(idx) - after
        if before >= stop:
            raise ValueError(f"Invalid crop idx={idx[0]}..{idx[-1]} crop={c}")
        kept = idx[before:stop]
        data_slices.append(slice(before, stop))
        out_slices.append(slice(int(kept[0]) - 1, int(kept[-1])))
    return tuple(data_slices), tuple(out_slices)


def coverage_summary_from_array(coverage: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(coverage, return_counts=True)
    hist = {int(k): int(v) for k, v in zip(unique, counts)}
    return {
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "missing_voxels": int(np.count_nonzero(coverage == 0)),
        "overlapped_voxels": int(np.count_nonzero(coverage > 1)),
        "coverage_histogram": hist,
    }


@torch.no_grad()
def process_band(
    band: Band,
    model: torch.nn.Module,
    device: torch.device,
    acc: dict[str, np.ndarray],
    weight: np.ndarray,
) -> dict[str, Any]:
    patches = load_dense_patches(band.patch_dir_wsl, fields=(), sample_group=DEFAULT_SAMPLE_GROUP)
    z_starts = sorted({int(p.z_idx[0]) for p in patches})
    x_starts = sorted({int(p.x_idx[0]) for p in patches})
    y_starts = sorted({int(p.y_idx[0]) for p in patches})
    print(f"process {band.name}: patches={len(patches)} z_starts={z_starts}", flush=True)
    print(f"process {band.name}: x_starts={x_starts} y_starts={y_starts}", flush=True)

    band_coverage = np.zeros(FULL_SHAPE, dtype=np.uint8)
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

        pred_np = (pred.float() * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        label_np = (y_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        baseline_np = (b_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        data_slices, out_slices = crop_slices_for_patch(patch)
        z_sl, x_sl, y_sl = out_slices
        dz, dx, dy = data_slices
        acc["pred"][(slice(None), z_sl, x_sl, y_sl)] += pred_np[:, dz, dx, dy]
        acc["label"][(slice(None), z_sl, x_sl, y_sl)] += label_np[:, dz, dx, dy]
        acc["baseline"][(slice(None), z_sl, x_sl, y_sl)] += baseline_np[:, dz, dx, dy]
        weight[z_sl, x_sl, y_sl] += 1.0
        band_coverage[z_sl, x_sl, y_sl] += 1

        del x, y, b, x_norm, y_norm, b_norm, pred, pred_np, label_np, baseline_np, scale
        if device.type == "cuda" and i % 16 == 0:
            torch.cuda.empty_cache()
        if i == 1 or i % 32 == 0 or i == len(patches):
            print(f"process {band.name}: inferred {i:04d}/{len(patches)}", flush=True)

    nonzero = band_coverage > 0
    z_indices = np.where(np.any(nonzero, axis=(1, 2)))[0]
    effective_z0 = int(z_indices[0])
    effective_z1 = int(z_indices[-1])
    band_region = band_coverage[effective_z0 : effective_z1 + 1]
    summary = coverage_summary_from_array(band_region)
    summary.update(
        {
            "band": band.name,
            "z_range_generated_1based": f"{band.z_min}-{band.z_max}",
            "effective_z_range_0based": f"{effective_z0}-{effective_z1}",
            "effective_z_range_1based": f"{effective_z0 + 1}-{effective_z1 + 1}",
            "patch_count": len(patches),
        }
    )
    print(
        f"band_coverage {band.name}: effective_z0={effective_z0} effective_z1={effective_z1} "
        f"min/max/missing={summary['coverage_min']}/{summary['coverage_max']}/{summary['missing_voxels']}",
        flush=True,
    )
    assert summary["coverage_min"] >= 1 and summary["missing_voxels"] == 0
    return summary


def finalize_volumes(acc: dict[str, np.ndarray], weight: np.ndarray) -> dict[str, np.ndarray]:
    zero = int(np.count_nonzero(weight <= 0))
    final_cov = coverage_summary_from_array(weight.astype(np.uint16))
    print(
        f"final_coverage min/max/missing/overlap="
        f"{final_cov['coverage_min']}/{final_cov['coverage_max']}/{final_cov['missing_voxels']}/{final_cov['overlapped_voxels']}",
        flush=True,
    )
    print(f"final_coverage histogram={_format_histogram(final_cov['coverage_histogram'])}", flush=True)
    assert zero == 0, f"Final coverage has {zero} empty voxels"
    return {field: arr / weight[None, :, :, :] for field, arr in acc.items()}


def save_volumes(vols: dict[str, np.ndarray], final_cov: dict[str, Any], band_rows: list[dict[str, Any]]) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    for field, vol in vols.items():
        np.save(VIS_DIR / f"{field}.npy", vol.astype(np.float32, copy=False))
    meta = [
        f"task: {TASK_NAME}",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"case_id: {CASE_ID}",
        f"checkpoint: {WIDE_CKPT}",
        f"patch_size: {PATCH_SIZE}",
        f"stride: {STRIDE}",
        f"crop: {CROP}",
        f"full_shape: {FULL_SHAPE}",
        f"final_coverage: {final_cov}",
        f"bands: {len(band_rows)}",
        "",
    ]
    (VIS_DIR / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    print(f"saved_volume_dir={VIS_DIR}", flush=True)


def load_bn_volumes() -> dict[str, np.ndarray]:
    vols = {}
    for field in ("baseline", "pred", "label"):
        path = BN_DIR / f"{field}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        vols[field] = np.load(path).astype(np.float32, copy=False)
        print(f"loaded_bn_{field}={path} shape={vols[field].shape}", flush=True)
    return vols


def dynamic_metrics(vols: dict[str, np.ndarray], z_slice: slice | None = None, prefix: str = "") -> dict[str, float]:
    pred = vols["pred"] if z_slice is None else vols["pred"][:, z_slice, :, :]
    label = vols["label"] if z_slice is None else vols["label"][:, z_slice, :, :]
    pred_abs = complex_abs_numpy(pred)
    label_abs = complex_abs_numpy(label)
    eps = 1e-12
    return {
        f"{prefix}pred_p99": float(np.percentile(pred_abs, 99.0)),
        f"{prefix}label_p99": float(np.percentile(label_abs, 99.0)),
        f"{prefix}pred_p99_over_label_p99": float(
            np.percentile(pred_abs, 99.0) / (np.percentile(label_abs, 99.0) + eps)
        ),
        f"{prefix}abs_std_ratio": float(np.std(pred_abs) / (np.std(label_abs) + eps)),
    }


def metrics_for(model_name: str, vols: dict[str, np.ndarray]) -> dict[str, Any]:
    row = metric_row(
        CASE_ID,
        model_name,
        vols,
        tuple(DEFAULT_X_BOUNDARIES),
        Y_INDEX,
        PATCH_SIZE,
        BOUNDARY_MARGIN,
    )
    row.update(dynamic_metrics(vols))
    row.update(dynamic_metrics(vols, z_slice=slice(800, None), prefix="deep_z800_"))
    return row


def x_jump_profile(volume: np.ndarray) -> np.ndarray:
    mag = complex_abs_numpy(volume)
    return np.array(
        [float(np.mean(np.abs(mag[:, x, :] - mag[:, x - 1, :]))) for x in range(1, mag.shape[1])],
        dtype=np.float64,
    )


def save_x_jump_outputs(bn_vols: dict[str, np.ndarray], wide_vols: dict[str, np.ndarray]) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_profile = x_jump_profile(wide_vols["label"])
    rows = []
    profiles = {
        "label": label_profile,
        "BN-L1 pred": x_jump_profile(bn_vols["pred"]),
        "wide+SSIM crop8 pred": x_jump_profile(wide_vols["pred"]),
    }
    for name, profile in profiles.items():
        for x in range(1, FULL_SHAPE[1]):
            rows.append(
                {
                    "signal": name,
                    "x": x,
                    "jump": profile[x - 1],
                    "over_label": profile[x - 1] / (label_profile[x - 1] + 1e-12),
                    "is_old_dense64_boundary": x in DEFAULT_X_BOUNDARIES,
                    "is_crop8_stride16_boundary": x % STRIDE[1] == 0,
                }
            )
    csv_path = OUT_DIR / "fullvol_x_jump_profile.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
    for name, profile in profiles.items():
        y = profile / (label_profile + 1e-12) if name != "label" else np.ones_like(profile)
        ax.plot(np.arange(1, FULL_SHAPE[1]), y, label=name, linewidth=1.5)
    for x in DEFAULT_X_BOUNDARIES:
        ax.axvline(x, color="tab:red", linewidth=0.8, alpha=0.65)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{CASE_ID} | full-volume x-jump ratio profile")
    ax.set_xlabel("x boundary index")
    ax.set_ylabel("jump / label jump")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig_path = OUT_DIR / "figs" / f"{CASE_ID}_fullvol_x_jump_ratio_profile.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return csv_path, fig_path


def save_visuals(bn_vols: dict[str, np.ndarray], wide_vols: dict[str, np.ndarray]) -> list[Path]:
    out_dir = OUT_DIR / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    panels = [
        ("label", wide_vols["label"]),
        ("BN-L1 pred", bn_vols["pred"]),
        ("wide+SSIM crop8 pred", wide_vols["pred"]),
    ]
    ref = max(float(np.max(complex_abs_numpy(vol))) for _, vol in panels)
    specs = [
        ("xz", 32, f"{CASE_ID}_fullvol_compare_xz_y32.png"),
        ("xz", 64, f"{CASE_ID}_fullvol_compare_xz_y64.png"),
        ("xy", 512, f"{CASE_ID}_fullvol_compare_xy_z512.png"),
        ("xy", 850, f"{CASE_ID}_fullvol_compare_xy_z850.png"),
    ]
    for view, index, filename in specs:
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
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
            if view == "xy":
                for x in DEFAULT_X_BOUNDARIES:
                    ax.axhline(x - 0.5, color="tab:red", linewidth=0.7, alpha=0.65)
            if view == "xz":
                for x in DEFAULT_X_BOUNDARIES:
                    ax.axvline(x - 0.5, color="tab:red", linewidth=0.7, alpha=0.65)
            ax.set_title(f"{title} | {view}={used}", fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="dB")
        fig.suptitle(f"{CASE_ID} | shared dB ref | {physical_spacing_label(view)}", fontsize=10)
        path = out_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)
        print(f"saved_visual={path}", flush=True)
    return saved


def run(dry_run: bool = False) -> None:
    print_header()
    seed_everything(20260522)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bands = make_bands()
    print("stream_bands:", flush=True)
    for band in bands:
        print(
            f"  {band.name}: z_starts={list(band.z_starts)} "
            f"z_range={band.z_min}-{band.z_max} expected_patches={band.expected_patches}",
            flush=True,
        )
    if dry_run:
        return

    device = get_device()
    print(f"device={device}", flush=True)
    model = load_wide_model(device)
    acc = {
        "pred": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "label": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "baseline": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
    }
    weight = np.zeros(FULL_SHAPE, dtype=np.float32)

    band_rows: list[dict[str, Any]] = []
    try:
        for band in bands:
            run_matlab_generation(band)
            band_rows.append(process_band(band, model, device, acc, weight))
            safe_rmtree(band.output_root_wsl)
    finally:
        for band in bands:
            if band.output_root_wsl.exists():
                safe_rmtree(band.output_root_wsl)

    final_cov = coverage_summary_from_array(weight.astype(np.uint16))
    wide_vols = finalize_volumes(acc, weight)
    save_volumes(wide_vols, final_cov, band_rows)
    bn_vols = load_bn_volumes()

    metrics_rows = [metrics_for("BN-L1 no-overlap", bn_vols), metrics_for("wide+SSIM crop8 streamed", wide_vols)]
    for row in metrics_rows:
        row["case"] = CASE_ID
    metrics_path = OUT_DIR / "wide_ssim_fullvol_crop8_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"saved_metrics={metrics_path}", flush=True)

    band_path = OUT_DIR / "wide_ssim_fullvol_crop8_band_coverage.csv"
    pd.DataFrame(band_rows).to_csv(band_path, index=False)
    print(f"saved_band_coverage={band_path}", flush=True)

    profile_path, profile_fig = save_x_jump_outputs(bn_vols, wide_vols)
    visuals = save_visuals(bn_vols, wide_vols)

    summary_cols = [
        "model",
        "x_boundary_pred_over_label",
        "xz_y_boundary_pred_over_label",
        "pred_p99_over_label_p99",
        "abs_std_ratio",
        "deep_z800_pred_p99_over_label_p99",
        "deep_z800_abs_std_ratio",
        "nonboundary_complex_improvement",
        "nonboundary_abs_improvement",
    ]
    print("\nsummary_metrics", flush=True)
    print(pd.DataFrame(metrics_rows).reindex(columns=summary_cols).to_string(index=False), flush=True)
    print("\nsaved_outputs", flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"band_coverage={band_path}", flush=True)
    print(f"x_jump_profile={profile_path}", flush=True)
    print(f"x_jump_fig={profile_fig}", flush=True)
    for path in visuals:
        print(f"visual={path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
