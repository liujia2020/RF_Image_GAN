from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rf_visualization import complex_abs_numpy, extract_slice, physical_aspect_for_view, physical_spacing_label, volume_to_db


DEFAULT_VOLUME_DIR = Path("vis_best_model/tiny_nonpoint_full_32/full_volume")
DEFAULT_OVERLAP_DIR = DEFAULT_VOLUME_DIR / "Carotid_012_frame1_overlap25_zband150-600"
DEFAULT_DB_MIN = -60.0
DEFAULT_Z_START0 = 149
DEFAULT_Z_STOP0 = 600
VOLUME_NAMES = ("baseline", "pred", "label")


def _load_full_volumes(volume_dir: Path) -> dict[str, np.ndarray]:
    volumes = {
        "baseline": np.load(volume_dir / "baseline.npy", mmap_mode="r"),
        "pred": np.load(volume_dir / "pred.npy", mmap_mode="r"),
        "label": np.load(volume_dir / "label.npy", mmap_mode="r"),
    }
    for name, volume in volumes.items():
        if volume.shape != (2, 1024, 128, 128):
            raise ValueError(f"{name} expected shape [2,1024,128,128], got {volume.shape}")
    return volumes


def _load_named_volumes(volume_dir: Path) -> dict[str, np.ndarray]:
    return {name: np.load(volume_dir / f"{name}.npy", mmap_mode="r") for name in VOLUME_NAMES}


def _crop_z(volumes: dict[str, np.ndarray], z_start0: int, z_stop0: int) -> dict[str, np.ndarray]:
    cropped = {}
    for name, volume in volumes.items():
        if volume.ndim != 4 or volume.shape[0] != 2:
            raise ValueError(f"{name} expected [2,Z,X,Y], got {volume.shape}")
        if z_start0 < 0 or z_stop0 > volume.shape[1] or z_stop0 <= z_start0:
            raise ValueError(f"Invalid z crop {z_start0}:{z_stop0} for {name} shape {volume.shape}")
        cropped[name] = volume[:, z_start0:z_stop0, :, :]
    return cropped


def _to_db_pair(
    noverlap: dict[str, np.ndarray],
    overlap: dict[str, np.ndarray],
    db_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    noverlap_mag = {name: complex_abs_numpy(volume) for name, volume in noverlap.items()}
    overlap_mag = {name: complex_abs_numpy(volume) for name, volume in overlap.items()}
    ref = max(
        max(float(np.max(mag)) for mag in noverlap_mag.values()),
        max(float(np.max(mag)) for mag in overlap_mag.values()),
    )
    print(f"shared comparison dB ref: {ref:.6e}")
    noverlap_db = {name: volume_to_db(mag, ref=ref, db_min=db_min) for name, mag in noverlap_mag.items()}
    overlap_db = {name: volume_to_db(mag, ref=ref, db_min=db_min) for name, mag in overlap_mag.items()}
    return noverlap_db, overlap_db, ref


def _to_db_volumes(volumes: dict[str, np.ndarray], db_min: float) -> dict[str, np.ndarray]:
    magnitudes = {name: complex_abs_numpy(volume) for name, volume in volumes.items()}
    ref = max(float(np.max(mag)) for mag in magnitudes.values())
    print(f"shared dB ref: {ref:.6e}")
    return {name: volume_to_db(mag, ref=ref, db_min=db_min) for name, mag in magnitudes.items()}


def _save_comparison(
    db_volumes: dict[str, np.ndarray],
    view: str,
    slice_index: int | None,
    save_path: Path,
    db_min: float,
) -> Path:
    names = ("baseline", "pred", "label")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)

    used_index = None
    xlabel = ""
    ylabel = ""
    images = []
    aspect = physical_aspect_for_view(view)
    for ax, name in zip(axes, names):
        img, used_index, xlabel, ylabel = extract_slice(db_volumes[name], view=view, slice_index=slice_index)
        images.append(img)
        im = ax.imshow(img, cmap="gray", vmin=db_min, vmax=0.0, origin="upper", aspect=aspect)
        ax.set_title(f"{view} index={used_index} | {name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle(physical_spacing_label(view), fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.84, label="dB")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    shapes = {img.shape for img in images}
    print(f"saved {save_path} view={view} index={used_index} slice_shapes={sorted(shapes)}")
    return save_path


def _save_noverlap_vs_overlap25_comparison(
    noverlap_db: dict[str, np.ndarray],
    overlap_db: dict[str, np.ndarray],
    view: str,
    slice_index: int,
    save_path: Path,
    db_min: float,
    z_start0: int,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
    row_specs = (("No overlap", noverlap_db), ("Overlap25 Hann", overlap_db))

    used_index = None
    xlabel = ""
    ylabel = ""
    images = []
    im = None
    aspect = physical_aspect_for_view(view)
    for row, (row_name, db_volumes) in enumerate(row_specs):
        for col, name in enumerate(VOLUME_NAMES):
            ax = axes[row, col]
            img, used_index, xlabel, ylabel = extract_slice(db_volumes[name], view=view, slice_index=slice_index)
            images.append(img)
            im = ax.imshow(img, cmap="gray", vmin=db_min, vmax=0.0, origin="upper", aspect=aspect)

            if view.lower() == "xy":
                pos_text = f"z={z_start0 + used_index} (local {used_index})"
            else:
                pos_text = f"{view.lower()} index={used_index}"
            ax.set_title(f"{row_name} | {name} | {pos_text}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    fig.suptitle(physical_spacing_label(view), fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="dB")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    shapes = {img.shape for img in images}
    print(f"saved {save_path} view={view} index={used_index} slice_shapes={sorted(shapes)}")
    return save_path


def generate_noverlap_vs_overlap25_figs(
    full_volume_dir: str | Path = DEFAULT_VOLUME_DIR,
    overlap_dir: str | Path = DEFAULT_OVERLAP_DIR,
    z_start0: int = DEFAULT_Z_START0,
    z_stop0: int = DEFAULT_Z_STOP0,
    db_min: float = DEFAULT_DB_MIN,
    source_stem: str = "Carotid_012",
) -> list[Path]:
    full_volume_dir = Path(full_volume_dir)
    overlap_dir = Path(overlap_dir)
    figs_dir = overlap_dir / "figs"

    noverlap_full = _load_full_volumes(full_volume_dir)
    noverlap = _crop_z(noverlap_full, z_start0=z_start0, z_stop0=z_stop0)
    overlap = _load_named_volumes(overlap_dir)

    expected_shape = next(iter(noverlap.values())).shape
    for name in VOLUME_NAMES:
        if noverlap[name].shape != expected_shape:
            raise ValueError(f"No-overlap {name} shape mismatch: {noverlap[name].shape} vs {expected_shape}")
        if overlap[name].shape != expected_shape:
            raise ValueError(f"Overlap {name} shape mismatch: {overlap[name].shape} vs {expected_shape}")

    print(f"no-overlap crop z0: {z_start0}-{z_stop0 - 1}, shape={expected_shape}")
    print(f"overlap25 z-band shape: {expected_shape}")

    noverlap_db, overlap_db, _ = _to_db_pair(noverlap, overlap, db_min=db_min)

    z_mid_local = expected_shape[1] // 2
    jobs = [
        ("xz", 64, f"{source_stem}_compare_noverlap_vs_overlap25_xz_y64.png"),
        ("xz", 32, f"{source_stem}_compare_noverlap_vs_overlap25_xz_y32.png"),
        ("xz", 96, f"{source_stem}_compare_noverlap_vs_overlap25_xz_y96.png"),
        ("xy", z_mid_local, f"{source_stem}_compare_noverlap_vs_overlap25_xy_z{z_start0 + z_mid_local}.png"),
    ]

    saved = []
    for view, slice_index, filename in jobs:
        saved.append(
            _save_noverlap_vs_overlap25_comparison(
                noverlap_db=noverlap_db,
                overlap_db=overlap_db,
                view=view,
                slice_index=slice_index,
                save_path=figs_dir / filename,
                db_min=db_min,
                z_start0=z_start0,
            )
        )

    return saved


def generate_full_volume_bmode_figs(
    volume_dir: str | Path = DEFAULT_VOLUME_DIR,
    db_min: float = DEFAULT_DB_MIN,
) -> list[Path]:
    volume_dir = Path(volume_dir)
    figs_dir = volume_dir / "figs"

    volumes = _load_full_volumes(volume_dir)
    db_volumes = _to_db_volumes(volumes, db_min=db_min)

    _, z_size, x_size, y_size = volumes["pred"].shape
    jobs = [
        ("xz", None, "bmode_xz_center_y.png"),
        ("xy", None, "bmode_xy_center_z.png"),
        ("zy", None, "bmode_zy_center_x.png"),
    ]

    for frac in (0.25, 0.50, 0.75):
        y_idx = min(max(int(round((y_size - 1) * frac)), 0), y_size - 1)
        jobs.append(("xz", y_idx, f"bmode_xz_y{y_idx:03d}_{int(frac * 100):02d}pct.png"))

    saved = []
    for view, slice_index, filename in jobs:
        saved.append(
            _save_comparison(
                db_volumes=db_volumes,
                view=view,
                slice_index=slice_index,
                save_path=figs_dir / filename,
                db_min=db_min,
            )
        )

    # Depth-specific views: in rf_visualization.extract_slice, z-indexed cuts are
    # xy views. These complement the requested xz planes for carotid depth checks.
    for frac in (0.25, 0.50, 0.75):
        z_idx = min(max(int(round((z_size - 1) * frac)), 0), z_size - 1)
        saved.append(
            _save_comparison(
                db_volumes=db_volumes,
                view="xy",
                slice_index=z_idx,
                save_path=figs_dir / f"bmode_xy_z{z_idx:04d}_{int(frac * 100):02d}pct_depth.png",
                db_min=db_min,
            )
        )

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate B-mode comparison figures for stitched full volumes.")
    parser.add_argument("--volume-dir", default=str(DEFAULT_VOLUME_DIR))
    parser.add_argument("--overlap-dir", default=str(DEFAULT_OVERLAP_DIR))
    parser.add_argument("--z-start0", type=int, default=DEFAULT_Z_START0)
    parser.add_argument("--z-stop0", type=int, default=DEFAULT_Z_STOP0)
    parser.add_argument("--db-min", type=float, default=DEFAULT_DB_MIN)
    parser.add_argument("--single", action="store_true", help="Generate the original single-run full-volume figures.")
    args = parser.parse_args()

    if args.single:
        saved = generate_full_volume_bmode_figs(args.volume_dir, db_min=args.db_min)
    else:
        saved = generate_noverlap_vs_overlap25_figs(
            full_volume_dir=args.volume_dir,
            overlap_dir=args.overlap_dir,
            z_start0=args.z_start0,
            z_stop0=args.z_stop0,
            db_min=args.db_min,
        )
    print("\nSaved figures:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
