from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rf_bn_multivolume_diagnostics import DEFAULT_X_BOUNDARIES, metric_row
from rf_stitch import DEFAULT_FULL_SHAPE, RFPatch, _format_histogram, check_coverage, load_dense_patches
from rf_visualization import complex_abs_numpy, extract_slice, physical_aspect_for_view, physical_spacing_label, volume_to_db


TASK_NAME = "wide+SSIM seam diagnosis + restitch feasibility check"
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "test_metrics" / "wide_ssim_seamfix"
PATCH_DIR = Path("/mnt/h/DAS/dense64/carotid/Carotid_008/test/carotid")
CASE_ID = "RF000489_Carotid_008"
PATCH_SIZE = (64, 32, 32)
FULL_SHAPE = DEFAULT_FULL_SHAPE
BOUNDARY_MARGIN = 3
Y_INDEX = 64
DB_MIN = -60.0
CROP = (8, 8, 8)

BN_DIR = (
    ROOT
    / "vis_best_model"
    / "tiny_bn_random64_full1500"
    / "full_volume_stitch_validation"
    / "RF000489_Carotid_008_frame1_dense64_hann"
)
WIDE_DIR = (
    ROOT
    / "vis_best_model"
    / "wide_ssim_random64_full1500"
    / "full_volume_stitch_validation"
    / "RF000489_Carotid_008_frame1_dense64_hann"
)


@dataclass(frozen=True)
class GridSpec:
    name: str
    patch_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    crop: tuple[int, int, int] | None = None


def print_header() -> None:
    print("=" * 104)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"task: {TASK_NAME}")
    print("=" * 104)


def load_volumes(volume_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for field in ("baseline", "pred", "label"):
        path = volume_dir / f"{field}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        out[field] = np.load(path)
    return out


def volume_dynamic_metrics(vols: dict[str, np.ndarray]) -> dict[str, float]:
    pred_abs = complex_abs_numpy(vols["pred"])
    label_abs = complex_abs_numpy(vols["label"])
    eps = 1e-12
    pred_p99 = float(np.percentile(pred_abs, 99.0))
    label_p99 = float(np.percentile(label_abs, 99.0))
    return {
        "pred_p99": pred_p99,
        "label_p99": label_p99,
        "pred_p99_over_label_p99": pred_p99 / (label_p99 + eps),
        "abs_std_ratio": float(np.std(pred_abs) / (np.std(label_abs) + eps)),
    }


def metric_for(model: str, stitch_config: str, vols: dict[str, np.ndarray], status: str = "ok") -> dict[str, Any]:
    row = metric_row(CASE_ID, model, vols, tuple(DEFAULT_X_BOUNDARIES), Y_INDEX, PATCH_SIZE, BOUNDARY_MARGIN)
    row.update(volume_dynamic_metrics(vols))
    row.update(
        {
            "stitch_config": stitch_config,
            "status": status,
            "patch_dir": str(PATCH_DIR),
        }
    )
    return row


def starts_for_dim(dim: int, patch: int, stride: int) -> list[int]:
    starts = list(range(1, dim - patch + 2, stride))
    last = dim - patch + 1
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def expected_grid_count(spec: GridSpec) -> dict[str, Any]:
    starts = [
        starts_for_dim(dim, patch, stride)
        for dim, patch, stride in zip(FULL_SHAPE, spec.patch_size, spec.stride)
    ]
    return {
        "config": spec.name,
        "patch_size": spec.patch_size,
        "stride": spec.stride,
        "crop": spec.crop,
        "z_windows": len(starts[0]),
        "x_windows": len(starts[1]),
        "y_windows": len(starts[2]),
        "expected_patches": len(starts[0]) * len(starts[1]) * len(starts[2]),
        "z_starts": starts[0],
        "x_starts": starts[1],
        "y_starts": starts[2],
    }


def observed_grid_summary(patches: list[RFPatch]) -> dict[str, Any]:
    z_starts = sorted({int(p.z_idx[0]) for p in patches})
    x_starts = sorted({int(p.x_idx[0]) for p in patches})
    y_starts = sorted({int(p.y_idx[0]) for p in patches})
    return {
        "observed_patches": len(patches),
        "z_windows": len(z_starts),
        "x_windows": len(x_starts),
        "y_windows": len(y_starts),
        "z_starts": z_starts,
        "x_starts": x_starts,
        "y_starts": y_starts,
    }


def coverage_for_cropped_current_grid(patches: list[RFPatch], crop: tuple[int, int, int]) -> dict[str, Any]:
    coverage = np.zeros(FULL_SHAPE, dtype=np.uint16)
    for patch in patches:
        starts = [int(patch.z_idx[0]) - 1, int(patch.x_idx[0]) - 1, int(patch.y_idx[0]) - 1]
        stops = [int(patch.z_idx[-1]), int(patch.x_idx[-1]), int(patch.y_idx[-1])]
        slices = []
        for axis, (start, stop, c, dim) in enumerate(zip(starts, stops, crop, FULL_SHAPE)):
            lo = start + (0 if start == 0 else c)
            hi = stop - (0 if stop == dim else c)
            if hi <= lo:
                raise ValueError(f"Invalid cropped slice axis={axis}: {lo}:{hi}")
            slices.append(slice(lo, hi))
        coverage[tuple(slices)] += 1

    unique, counts = np.unique(coverage, return_counts=True)
    hist = {int(k): int(v) for k, v in zip(unique, counts)}
    return {
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "coverage_histogram": hist,
        "missing_voxels": int(np.count_nonzero(coverage == 0)),
        "overlapped_voxels": int(np.count_nonzero(coverage > 1)),
    }


def add_unavailable_rows(rows: list[dict[str, Any]], config: str, reason: str, observed: dict[str, Any], expected: dict[str, Any]) -> None:
    for model in ("BN-L1", "wide+SSIM"):
        rows.append(
            {
                "case": CASE_ID,
                "model": model,
                "stitch_config": config,
                "status": "not_run",
                "reason": reason,
                "observed_patches": observed["observed_patches"],
                "expected_patches": expected["expected_patches"],
                "x_boundary_pred_over_label": np.nan,
                "xz_y_boundary_pred_over_label": np.nan,
                "pred_p99": np.nan,
                "label_p99": np.nan,
                "pred_p99_over_label_p99": np.nan,
                "abs_std_ratio": np.nan,
            }
        )


def x_jump_profile(volume: np.ndarray) -> np.ndarray:
    mag = complex_abs_numpy(volume)
    return np.array(
        [float(np.mean(np.abs(mag[:, x, :] - mag[:, x - 1, :]))) for x in range(1, mag.shape[1])],
        dtype=np.float64,
    )


def fft_period_stats(profile: np.ndarray, target_period: float = 32.0) -> dict[str, float]:
    centered = profile - float(np.mean(profile))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size <= 1 or float(np.sum(power[1:])) <= 0:
        return {
            "target_period": target_period,
            "target_bin": float("nan"),
            "target_period_est": float("nan"),
            "target_power_fraction": float("nan"),
            "target_power_rank": float("nan"),
        }
    freqs = np.fft.rfftfreq(profile.size, d=1.0)
    target_freq = 1.0 / target_period
    target_bin = int(np.argmin(np.abs(freqs - target_freq)))
    nonzero = power[1:]
    rank = 1 + int(np.sum(nonzero > power[target_bin]))
    period_est = float("inf") if freqs[target_bin] <= 0 else float(1.0 / freqs[target_bin])
    return {
        "target_period": target_period,
        "target_bin": float(target_bin),
        "target_period_est": period_est,
        "target_power_fraction": float(power[target_bin] / np.sum(nonzero)),
        "target_power_rank": float(rank),
    }


def save_boundary_profile_and_fft(volumes: dict[str, dict[str, np.ndarray]]) -> tuple[Path, Path, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = {
        "label": volumes["BN-L1"]["label"],
        "BN-L1 pred": volumes["BN-L1"]["pred"],
        "wide+SSIM pred": volumes["wide+SSIM"]["pred"],
    }
    profiles = {name: x_jump_profile(vol) for name, vol in labels.items()}
    xs = np.arange(1, 128)

    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    for name, profile in profiles.items():
        ax.plot(xs, profile, label=name, linewidth=1.5)
    for boundary in DEFAULT_X_BOUNDARIES:
        ax.axvline(boundary, color="tab:red", linewidth=0.8, alpha=0.65)
    ax.set_title(f"{CASE_ID} | x-adjacent jump profile | red=patch grid")
    ax.set_xlabel("x boundary index")
    ax.set_ylabel("mean |mag[x]-mag[x-1]|")
    ax.grid(True, alpha=0.25)
    ax.legend()
    profile_path = OUT_DIR / f"{CASE_ID}_x_jump_profile.png"
    fig.savefig(profile_path, dpi=200)
    plt.close(fig)

    rows = []
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    for name, profile in profiles.items():
        centered = profile - float(np.mean(profile))
        power = np.abs(np.fft.rfft(centered)) ** 2
        freqs = np.fft.rfftfreq(profile.size, d=1.0)
        periods = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
        ax.plot(periods[1:], power[1:] / max(float(np.sum(power[1:])), 1e-12), label=name, linewidth=1.5)
        row = {"signal": name}
        row.update(fft_period_stats(profile, target_period=32.0))
        rows.append(row)
    ax.axvline(32, color="tab:red", linewidth=0.8, alpha=0.65, label="period 32")
    ax.set_xlim(4, 80)
    ax.set_title(f"{CASE_ID} | FFT power of x-jump profile")
    ax.set_xlabel("period in x samples")
    ax.set_ylabel("normalized power")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fft_path = OUT_DIR / f"{CASE_ID}_x_jump_fft.png"
    fig.savefig(fft_path, dpi=200)
    plt.close(fig)

    return profile_path, fft_path, pd.DataFrame(rows)


def save_config1_vs_config3_feasibility(wide_vols: dict[str, np.ndarray], crop_coverage: dict[str, Any]) -> Path:
    out_dir = OUT_DIR / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_db = volume_to_db(complex_abs_numpy(wide_vols["pred"]), ref=None, db_min=DB_MIN)
    img, used, xlabel, ylabel = extract_slice(wide_db, view="xy", slice_index=512)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    im = axes[0].imshow(img, cmap="gray", vmin=DB_MIN, vmax=0.0, origin="upper", aspect=physical_aspect_for_view("xy"))
    for boundary in DEFAULT_X_BOUNDARIES:
        axes[0].axhline(boundary - 0.5, color="tab:red", linewidth=0.7, alpha=0.65)
    axes[0].set_title(f"config1 current | wide+SSIM pred | xy z={used}")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    axes[1].axis("off")
    text = (
        "config3 edge-crop restitch: NOT RUN\n\n"
        "Reason: current RF000489 grid is no-overlap only.\n"
        "Applying crop8 to existing 256 patches leaves holes.\n\n"
        f"coverage min/max: {crop_coverage['coverage_min']} / {crop_coverage['coverage_max']}\n"
        f"missing voxels: {crop_coverage['missing_voxels']:,}\n"
        f"overlapped voxels: {crop_coverage['overlapped_voxels']:,}\n\n"
        "Need denser overlap RF patches before valid config3 stitch."
    )
    axes[1].text(0.02, 0.98, text, ha="left", va="top", fontsize=10, family="monospace")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="dB")
    fig.suptitle(f"{CASE_ID} | config1 vs config3 feasibility | {physical_spacing_label('xy')}", fontsize=10)
    path = out_dir / f"{CASE_ID}_config1_vs_config3_feasibility_xy_z512.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print_header()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    patches = load_dense_patches(PATCH_DIR, fields=())
    observed = observed_grid_summary(patches)
    print("observed_grid:")
    print(observed)
    coverage = check_coverage(patches, FULL_SHAPE)
    print("config1 coverage: PASS")
    print(f"coverage min/max: {coverage['coverage_min']} / {coverage['coverage_max']}")
    print(f"coverage histogram: {_format_histogram(coverage['coverage_histogram'])}")
    print(f"missing voxels: {coverage['missing_voxels']}")
    print(f"overlapped voxels: {coverage['overlapped_voxels']}")

    config2 = GridSpec("config2_half_stride_overlap_requested", PATCH_SIZE, (32, 16, 16), None)
    config3 = GridSpec("config3_crop8_center_requested", PATCH_SIZE, (48, 16, 16), CROP)
    expected2 = expected_grid_count(config2)
    expected3 = expected_grid_count(config3)
    crop_coverage = coverage_for_cropped_current_grid(patches, CROP)

    feasibility_rows = [
        {"config": "observed_current_grid", **observed},
        expected2,
        expected3,
        {
            "config": "current_grid_with_crop8_attempt",
            "patch_size": PATCH_SIZE,
            "stride": (64, 32, 32),
            "crop": CROP,
            **crop_coverage,
        },
    ]
    feasibility_path = OUT_DIR / "restitch_feasibility.csv"
    pd.DataFrame(feasibility_rows).to_csv(feasibility_path, index=False)
    print(f"saved_feasibility={feasibility_path}")
    print("config2 required patches:", expected2["expected_patches"])
    print("config3 required patches:", expected3["expected_patches"])
    print("current crop8 coverage:", crop_coverage)

    volumes = {
        "BN-L1": load_volumes(BN_DIR),
        "wide+SSIM": load_volumes(WIDE_DIR),
    }

    rows: list[dict[str, Any]] = [
        metric_for("BN-L1", "config1_current_dense64_hann", volumes["BN-L1"]),
        metric_for("wide+SSIM", "config1_current_dense64_hann", volumes["wide+SSIM"]),
    ]
    add_unavailable_rows(
        rows,
        "config2_half_stride_overlap_requested",
        "missing_overlap_rf_patches_observed_256_expected_1519",
        observed,
        expected2,
    )
    add_unavailable_rows(
        rows,
        "config3_crop8_center_requested",
        f"crop8_current_grid_missing_voxels_{crop_coverage['missing_voxels']}",
        observed,
        expected3,
    )

    metrics_path = OUT_DIR / "wide_ssim_seamfix_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"saved_metrics={metrics_path}")

    profile_path, fft_path, fft_df = save_boundary_profile_and_fft(volumes)
    fft_path_csv = OUT_DIR / "x_jump_fft_period32_stats.csv"
    fft_df.to_csv(fft_path_csv, index=False)
    feasibility_fig = save_config1_vs_config3_feasibility(volumes["wide+SSIM"], crop_coverage)

    print("\nsummary")
    cols = [
        "model",
        "stitch_config",
        "status",
        "x_boundary_pred_over_label",
        "xz_y_boundary_pred_over_label",
        "pred_p99_over_label_p99",
        "abs_std_ratio",
        "reason",
    ]
    print(pd.DataFrame(rows).reindex(columns=cols).to_string(index=False))
    print("\nperiod32_fft_stats")
    print(fft_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"saved_profile={profile_path}")
    print(f"saved_fft={fft_path}")
    print(f"saved_fft_stats={fft_path_csv}")
    print(f"saved_feasibility_fig={feasibility_fig}")


if __name__ == "__main__":
    main()
