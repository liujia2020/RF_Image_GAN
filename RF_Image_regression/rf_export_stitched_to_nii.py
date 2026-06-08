from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


TASK_NAME = "export stitched volumes to uncompressed NIfTI for 3D Slicer"
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "nii_for_slicer"

# Complex arrays are stored as [2, z, x, y].
COMPLEX_AXIS_ORDER = "[complex_channel, z, x, y]"
ENVELOPE_AXIS_ORDER = "[z, x, y]"
SPACING_ZXY_MM = (0.0362, 0.2, 0.2)
AFFINE_ZXY = np.diag([SPACING_ZXY_MM[0], SPACING_ZXY_MM[1], SPACING_ZXY_MM[2], 1.0]).astype(
    np.float32
)


@dataclass(frozen=True)
class ExportCase:
    case_name: str
    label_dir: Path
    bnl1_dir: Path | None
    widessim_dir: Path
    bnl1_note: str


CASES = [
    ExportCase(
        case_name="Carotid_008",
        label_dir=ROOT
        / "vis_best_model/wide_ssim_random64_full1500/full_volume_stream_crop8/RF000489_Carotid_008_frame1_dense64_crop8_streamed",
        bnl1_dir=ROOT
        / "vis_best_model/tiny_bn_random64_full1500/full_volume_stitch_validation/RF000489_Carotid_008_frame1_dense64_hann",
        widessim_dir=ROOT
        / "vis_best_model/wide_ssim_random64_full1500/full_volume_stream_crop8/RF000489_Carotid_008_frame1_dense64_crop8_streamed",
        bnl1_note="persisted BN-L1 validation stitch",
    ),
    ExportCase(
        case_name="Carotid_073",
        label_dir=ROOT
        / "vis_best_model/wide_ssim_random64_full1500/full_volume_stream_crop8_generalize/RF000487_Carotid_073_frame1_dense64_crop8_streamed",
        bnl1_dir=ROOT / "vis_best_model/tiny_bn_random64_full1500/seam_multi/RF000487",
        widessim_dir=ROOT
        / "vis_best_model/wide_ssim_random64_full1500/full_volume_stream_crop8_generalize/RF000487_Carotid_073_frame1_dense64_crop8_streamed",
        bnl1_note="persisted BN-L1 seam_multi stitch; crop8 BN-L1 volume was not persisted",
    ),
    ExportCase(
        case_name="Muscle_056",
        label_dir=ROOT
        / "vis_best_model/wide_ssim_random64_full1500/full_volume_stream_crop8_generalize/RF000386_Muscle_056_frame1_dense64_crop8_streamed",
        bnl1_dir=ROOT / "vis_best_model/tiny_bn_random64_full1500/seam_multi/RF000386",
        widessim_dir=ROOT
        / "vis_best_model/wide_ssim_random64_full1500/full_volume_stream_crop8_generalize/RF000386_Muscle_056_frame1_dense64_crop8_streamed",
        bnl1_note="persisted BN-L1 seam_multi stitch; crop8 BN-L1 volume was not persisted",
    ),
    ExportCase(
        case_name="Phantom_036",
        label_dir=ROOT
        / "vis_best_model/wide_ssim_random64_full1500/full_volume_stream_crop8_generalize/RF000587_Phantom_036_frame1_dense64_crop8_streamed",
        bnl1_dir=ROOT / "vis_best_model/tiny_bn_random64_full1500/seam_multi/RF000587",
        widessim_dir=ROOT
        / "vis_best_model/wide_ssim_random64_full1500/full_volume_stream_crop8_generalize/RF000587_Phantom_036_frame1_dense64_crop8_streamed",
        bnl1_note="persisted BN-L1 seam_multi stitch; crop8 BN-L1 volume was not persisted",
    ),
]


def load_complex(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 4 or arr.shape[0] != 2:
        raise ValueError(f"Expected [2,z,x,y] complex array, got {arr.shape}: {path}")
    return arr


def env_from_complex(arr: np.ndarray) -> np.ndarray:
    env = np.sqrt(arr[0].astype(np.float32) ** 2 + arr[1].astype(np.float32) ** 2)
    return env.astype(np.float32, copy=False)


def db_from_env(env: np.ndarray, ref: float) -> np.ndarray:
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError(f"Invalid dB reference: {ref}")
    ratio = np.maximum(env.astype(np.float32) / np.float32(ref), np.float32(1e-12))
    db = 20.0 * np.log10(ratio)
    return np.clip(db, -60.0, 0.0).astype(np.float32, copy=False)


def save_nii(data: np.ndarray, out_path: Path) -> dict[str, object]:
    if out_path.suffix != ".nii":
        raise ValueError(f"Output must be uncompressed .nii: {out_path}")
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), AFFINE_ZXY)
    image.header.set_zooms(SPACING_ZXY_MM)
    image.header.set_data_dtype(np.float32)
    nib.save(image, str(out_path))
    return {
        "file": str(out_path.relative_to(ROOT)),
        "shape": tuple(int(v) for v in data.shape),
        "spacing": SPACING_ZXY_MM,
        "dtype": "float32",
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
    }


def existing_volume_sets(case: ExportCase) -> Iterable[tuple[str, Path, str]]:
    yield "label", case.label_dir / "label.npy", "label"
    if case.bnl1_dir is not None:
        yield "bnl1", case.bnl1_dir / "pred.npy", case.bnl1_note
    yield "widessim", case.widessim_dir / "pred.npy", "wide+SSIM pred"


def print_inventory() -> list[ExportCase]:
    print("\nPersisted full-volume stitch outputs considered:")
    available: list[ExportCase] = []
    for case in CASES:
        print(f"\nCASE {case.case_name}")
        ok = True
        for role, path, note in existing_volume_sets(case):
            if not path.exists():
                ok = False
                print(f"  MISSING {role}: {path.relative_to(ROOT)} ({note})")
                continue
            arr = load_complex(path)
            print(
                f"  {role}: {path.relative_to(ROOT)} | shape={arr.shape} | "
                f"axis_order={COMPLEX_AXIS_ORDER} | dtype={arr.dtype} | note={note}"
            )
        if ok:
            available.append(case)
    return available


def export_case(case: ExportCase) -> list[dict[str, object]]:
    print(f"\nExporting {case.case_name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label_arr = load_complex(case.label_dir / "label.npy")
    label_env = env_from_complex(label_arr)
    ref = float(np.nanmax(label_env))
    print(
        f"  envelope_axis_order={ENVELOPE_AXIS_ORDER} | nifti_spacing_zxy_mm={SPACING_ZXY_MM} | "
        f"db_ref=label_global_max={ref:.9g}"
    )

    outputs: list[dict[str, object]] = []

    volumes = [
        ("label", label_env),
        ("bnl1", env_from_complex(load_complex(case.bnl1_dir / "pred.npy")) if case.bnl1_dir else None),
        ("widessim", env_from_complex(load_complex(case.widessim_dir / "pred.npy"))),
    ]

    for role, env in volumes:
        if env is None:
            continue
        env_path = OUT_DIR / f"{case.case_name}_{role}_env.nii"
        db_path = OUT_DIR / f"{case.case_name}_{role}_db.nii"
        outputs.append(save_nii(env, env_path))
        outputs.append(save_nii(db_from_env(env, ref), db_path))

    for row in outputs:
        print(
            f"  wrote {row['file']} | shape={row['shape']} | spacing={row['spacing']} | "
            f"dtype={row['dtype']} | min={row['min']:.9g} | max={row['max']:.9g}"
        )
    return outputs


def main() -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} | TASK: {TASK_NAME}")
    print(f"Output directory: {OUT_DIR.relative_to(ROOT)}")
    available = print_inventory()
    all_outputs: list[dict[str, object]] = []
    for case in available:
        all_outputs.extend(export_case(case))
    print(f"\nExported {len(all_outputs)} NIfTI files.")
    print("All exported files are uncompressed .nii.")


if __name__ == "__main__":
    main()
