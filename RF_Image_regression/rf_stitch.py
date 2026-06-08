from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch

from rf_eval_utils import _scale_to_broadcast
from rf_models import build_model


DEFAULT_SAMPLE_GROUP = "/sample_000001"
DEFAULT_FULL_SHAPE = (1024, 128, 128)
DEFAULT_DENSE_PATCH_DIR = r"F:\DAS\RF_LearningSamples_dense_carotid_test_32x16x16\test\carotid"
DEFAULT_CKPT_PATH = "checkpoint/tiny_nonpoint_full_32/best_model.pth"
DEFAULT_OUTPUT_DIR = "vis_best_model/tiny_nonpoint_full_32/full_volume"
DEFAULT_DENSE32_VOLUME_DIR = Path("vis_best_model/tiny_nonpoint_full_32/full_volume")
DEFAULT_DENSE64_VOLUME_DIR = Path(
    "vis_best_model/tiny_random64_pilot80/full_volume/Carotid_012_frame1_dense64_nooverlap"
)
PATCH_ID_RE = re.compile(r"_patch(\d+)\.h5$", re.IGNORECASE)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class RFPatch:
    path: Path
    z_idx: np.ndarray
    x_idx: np.ndarray
    y_idx: np.ndarray
    source_file: str = ""
    frame_id: int = 1
    data: dict[str, np.ndarray] = dataclass_field(default_factory=dict)

    def get(self, field: str) -> np.ndarray:
        if field not in self.data:
            raise KeyError(
                f"Patch field {field!r} was not loaded for {self.path}. "
                "Reload with load_dense_patches(..., fields=[...])."
            )
        return self.data[field]


def resolve_patch_dir(patch_dir: str | Path) -> Path:
    """Resolve Windows/WSL spelling for the dense patch directory."""
    p = Path(patch_dir)
    if p.exists():
        return p

    text = str(patch_dir)
    if text.startswith("/mnt/") and len(text) >= 7 and text[5].isalpha() and text[6] == "/":
        win = Path(f"{text[5].upper()}:\\" + text[7:].replace("/", "\\"))
        if win.exists():
            return win

    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        wsl = Path("/mnt") / drive / text[3:].replace("\\", "/")
        if wsl.exists():
            return wsl

    return p


def _patch_sort_key(path: Path) -> tuple[int, str]:
    match = PATCH_ID_RE.search(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def _read_float32(h5: h5py.File, path: str) -> np.ndarray:
    return np.asarray(h5[path], dtype=np.float32)


def _read_rf_tensor(h5: h5py.File, path: str) -> np.ndarray:
    arr = _read_float32(h5, path)
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D RF tensor at {path}, got shape {arr.shape}")
    return np.transpose(arr, (4, 3, 2, 1, 0))


def _read_volume(h5: h5py.File, path: str) -> np.ndarray:
    arr = _read_float32(h5, path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got shape {arr.shape}")
    return np.transpose(arr, (2, 1, 0))


def _read_input(h5: h5py.File, group: str) -> np.ndarray:
    f_rc_real = _read_rf_tensor(h5, f"{group}/input/F_RC_real")
    f_rc_imag = _read_rf_tensor(h5, f"{group}/input/F_RC_imag")
    f_cr_real = _read_rf_tensor(h5, f"{group}/input/F_CR_real")
    f_cr_imag = _read_rf_tensor(h5, f"{group}/input/F_CR_imag")

    # [Nz, Nx, Ny, Nch, Nangle, 4]
    x = np.stack([f_rc_real, f_rc_imag, f_cr_real, f_cr_imag], axis=-1)

    # [Nch, Nangle, 4, Nz, Nx, Ny] -> [1536, Nz, Nx, Ny]
    x = np.transpose(x, (3, 4, 5, 0, 1, 2))
    n_ch, n_angle, n_comp, nz, nx, ny = x.shape
    return np.ascontiguousarray(x.reshape(n_ch * n_angle * n_comp, nz, nx, ny))


def _read_label(h5: h5py.File, group: str) -> np.ndarray:
    real = _read_volume(h5, f"{group}/label/DAS_target_avg_real")
    imag = _read_volume(h5, f"{group}/label/DAS_target_avg_imag")
    return np.ascontiguousarray(np.stack([real, imag], axis=0))


def _read_baseline(h5: h5py.File, group: str) -> np.ndarray:
    real = _read_volume(h5, f"{group}/baseline/DAS_input_avg_real")
    imag = _read_volume(h5, f"{group}/baseline/DAS_input_avg_imag")
    return np.ascontiguousarray(np.stack([real, imag], axis=0))


def _read_meta(h5: h5py.File, group: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    z_idx = np.asarray(h5[f"{group}/meta/z_idx"], dtype=np.int32).reshape(-1)
    x_idx = np.asarray(h5[f"{group}/meta/x_idx"], dtype=np.int32).reshape(-1)
    y_idx = np.asarray(h5[f"{group}/meta/y_idx"], dtype=np.int32).reshape(-1)

    frame_id = int(np.asarray(h5[f"{group}/meta/frame_id"]).reshape(-1)[0])
    try:
        source_file = h5[f"{group}/meta"].attrs.get("source_file", "")
        if isinstance(source_file, bytes):
            source_file = source_file.decode("utf-8", errors="replace")
        else:
            source_file = str(source_file)
    except Exception:
        source_file = ""

    return z_idx, x_idx, y_idx, source_file, frame_id


def load_dense_patches(
    patch_dir: str | Path,
    fields: Iterable[str] = ("label",),
    sample_group: str = DEFAULT_SAMPLE_GROUP,
) -> list[RFPatch]:
    """
    Load dense RF patches and metadata.

    Supported fields are "input", "label", and "baseline". For coverage-only
    checks pass fields=() to load metadata without patch tensors. The input RF
    tensor is large, so callers that need inference should usually stream it
    patch-by-patch rather than load all inputs at once.
    """
    patch_dir = resolve_patch_dir(patch_dir)
    fields = tuple(fields)
    valid_fields = {"input", "label", "baseline"}
    unknown = set(fields) - valid_fields
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")

    files = sorted(patch_dir.glob("*.h5"), key=_patch_sort_key)
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {patch_dir}")

    group = sample_group.rstrip("/")
    patches: list[RFPatch] = []

    for path in files:
        with h5py.File(path, "r") as h5:
            z_idx, x_idx, y_idx, source_file, frame_id = _read_meta(h5, group)
            data: dict[str, np.ndarray] = {}
            if "input" in fields:
                data["input"] = _read_input(h5, group)
            if "label" in fields:
                data["label"] = _read_label(h5, group)
            if "baseline" in fields:
                data["baseline"] = _read_baseline(h5, group)

        patches.append(
            RFPatch(
                path=path,
                z_idx=z_idx,
                x_idx=x_idx,
                y_idx=y_idx,
                source_file=source_file,
                frame_id=frame_id,
                data=data,
            )
        )

    return patches


def _idx_to_slice(idx_1based: np.ndarray, dim_len: int, axis_name: str, patch_path: Path) -> slice:
    idx = np.asarray(idx_1based, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError(f"Empty {axis_name}_idx in {patch_path}")
    if idx.min() < 1 or idx.max() > dim_len:
        raise ValueError(
            f"{axis_name}_idx out of 1-based bounds for {patch_path}: "
            f"min={idx.min()}, max={idx.max()}, dim={dim_len}"
        )
    if not np.all(np.diff(idx) == 1):
        raise ValueError(f"{axis_name}_idx is not contiguous in {patch_path}")

    start0 = int(idx[0]) - 1
    stop0 = int(idx[-1])
    return slice(start0, stop0)


def _patch_slices(patch: RFPatch, full_shape: Sequence[int]) -> tuple[slice, slice, slice]:
    if len(full_shape) != 3:
        raise ValueError(f"full_shape must be [Nz, Nx, Ny], got {full_shape}")
    nz, nx, ny = (int(v) for v in full_shape)
    return (
        _idx_to_slice(patch.z_idx, nz, "z", patch.path),
        _idx_to_slice(patch.x_idx, nx, "x", patch.path),
        _idx_to_slice(patch.y_idx, ny, "y", patch.path),
    )


def check_coverage(patches: Sequence[RFPatch], full_shape: Sequence[int]) -> dict[str, object]:
    """Assert every voxel is covered exactly once by the dense patches."""
    full_shape = tuple(int(v) for v in full_shape)
    coverage = np.zeros(full_shape, dtype=np.uint16)

    for patch in patches:
        z_slice, x_slice, y_slice = _patch_slices(patch, full_shape)
        coverage[z_slice, x_slice, y_slice] += 1

    unique, counts = np.unique(coverage, return_counts=True)
    hist = {int(k): int(v) for k, v in zip(unique, counts)}
    missing = int(np.count_nonzero(coverage == 0))
    overlapped = int(np.count_nonzero(coverage > 1))

    summary = {
        "full_shape": full_shape,
        "num_patches": len(patches),
        "total_voxels": int(np.prod(full_shape)),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "coverage_histogram": hist,
        "missing_voxels": missing,
        "overlapped_voxels": overlapped,
    }

    if missing != 0 or overlapped != 0 or summary["coverage_min"] != 1 or summary["coverage_max"] != 1:
        raise AssertionError(f"Coverage is not exactly 1 everywhere: {summary}")

    return summary


def make_hann3d_weight(patch_shape: Sequence[int], floor: float = 0.01) -> np.ndarray:
    """Create a 3D Hann fusion weight for one spatial patch."""
    if len(patch_shape) != 3:
        raise ValueError(f"patch_shape must be [Pz, Px, Py], got {patch_shape}")

    pz, px, py = (int(v) for v in patch_shape)
    if min(pz, px, py) <= 0:
        raise ValueError(f"patch_shape entries must be positive, got {patch_shape}")

    wz = np.hanning(pz).astype(np.float32)
    wx = np.hanning(px).astype(np.float32)
    wy = np.hanning(py).astype(np.float32)
    hann3d = wz[:, None, None] * wx[None, :, None] * wy[None, None, :]
    return np.maximum(hann3d, float(floor)).astype(np.float32, copy=False)


def _shift_patch_to_zband(patch: RFPatch, z_min_1based: int) -> RFPatch:
    """Return a patch view whose z indices are local to a z-band sub-volume."""
    return RFPatch(
        path=patch.path,
        z_idx=(np.asarray(patch.z_idx, dtype=np.int32) - int(z_min_1based) + 1).astype(np.int32),
        x_idx=patch.x_idx,
        y_idx=patch.y_idx,
        source_file=patch.source_file,
        frame_id=patch.frame_id,
        data=patch.data,
    )


def _source_stem(source_file: str) -> str:
    name = Path(str(source_file).replace("\\", "/")).stem or "unknown_source"
    return SAFE_NAME_RE.sub("_", name).strip("_") or "unknown_source"


def _validate_patch_bounds(patches: Sequence[RFPatch], full_shape: Sequence[int]) -> None:
    for patch in patches:
        _patch_slices(patch, full_shape)


def stitch_volume(
    patches: Sequence[RFPatch],
    full_shape: Sequence[int],
    field: str,
    weight: np.ndarray | None = None,
    return_weight: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Stitch a loaded patch field using average or spatially weighted fusion."""
    if len(patches) == 0:
        raise ValueError("No patches to stitch.")

    full_shape = tuple(int(v) for v in full_shape)
    first = patches[0].get(field)
    if first.ndim == 3:
        out_shape = full_shape
        channel_first = False
    elif first.ndim == 4:
        out_shape = (first.shape[0],) + full_shape
        channel_first = True
    else:
        raise ValueError(f"Expected field {field!r} to be 3D or 4D, got {first.shape}")

    acc = np.zeros(out_shape, dtype=np.float32)
    weight_buffer = np.zeros(full_shape, dtype=np.float32)
    base_weight = None if weight is None else np.asarray(weight, dtype=np.float32)

    for patch in patches:
        arr = patch.get(field).astype(np.float32, copy=False)
        z_slice, x_slice, y_slice = _patch_slices(patch, full_shape)
        spatial_shape = (
            z_slice.stop - z_slice.start,
            x_slice.stop - x_slice.start,
            y_slice.stop - y_slice.start,
        )

        if base_weight is None:
            patch_weight = np.ones(spatial_shape, dtype=np.float32)
        else:
            if base_weight.shape != spatial_shape:
                raise ValueError(
                    f"Weight shape mismatch for {patch.path}: got {base_weight.shape}, expected {spatial_shape}"
                )
            patch_weight = base_weight

        if channel_first:
            expected = (out_shape[0],) + spatial_shape
            if arr.shape != expected:
                raise ValueError(f"{field!r} shape mismatch in {patch.path}: got {arr.shape}, expected {expected}")
            acc[(slice(None), z_slice, x_slice, y_slice)] += arr * patch_weight[None, :, :, :]
        else:
            if arr.shape != spatial_shape:
                raise ValueError(
                    f"{field!r} shape mismatch in {patch.path}: got {arr.shape}, expected {spatial_shape}"
                )
            acc[z_slice, x_slice, y_slice] += arr * patch_weight

        weight_buffer[z_slice, x_slice, y_slice] += patch_weight

    if np.any(weight_buffer <= 0):
        raise AssertionError("Cannot stitch: at least one voxel has zero accumulated weight.")

    if channel_first:
        out = acc / weight_buffer[None, :, :, :]
    else:
        out = acc / weight_buffer

    if return_weight:
        return out, weight_buffer
    return out


def _format_histogram(hist: dict[int, int]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in sorted(hist.items()))


def _load_model_for_inference(ckpt_path: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["model"]
    # Detect BN vs IN: BN state_dict has running_mean/running_var keys.
    use_bn = any("running_mean" in k for k in state_dict)

    model = build_model(
        "tiny",
        in_channels=1536,
        hidden=64,
        out_channels=2,
        use_batch_norm=use_bn,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _read_patch_tensors(
    patch_path: Path,
    sample_group: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = sample_group.rstrip("/")
    with h5py.File(patch_path, "r") as h5:
        x_np = _read_input(h5, group)
        y_np = _read_label(h5, group)
        b_np = _read_baseline(h5, group)

    x = torch.from_numpy(x_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(y_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(b_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    return x, y, b


def _volume_stats(name: str, volume: np.ndarray) -> dict[str, object]:
    finite = bool(np.isfinite(volume).all())
    return {
        "name": name,
        "shape": tuple(int(v) for v in volume.shape),
        "dtype": str(volume.dtype),
        "min": float(np.min(volume)),
        "max": float(np.max(volume)),
        "finite": finite,
    }


def _print_volume_stats(name: str, volume: np.ndarray) -> None:
    stats = _volume_stats(name, volume)
    print(
        f"{name} volume: "
        f"shape={stats['shape']}, dtype={stats['dtype']}, "
        f"min={stats['min']:.6e}, max={stats['max']:.6e}, "
        f"finite={stats['finite']}"
    )


def _complex_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False))))


def seam_metric(
    volume: np.ndarray,
    boundaries_x: Sequence[int],
    half_width: int = 1,
) -> dict[int, float]:
    """
    Compute x-boundary seam jumps for a complex two-channel volume.

    Parameters
    ----------
    volume:
        Complex RF volume with shape [2, Z, X, Y], where channel 0 is real and
        channel 1 is imaginary.
    boundaries_x:
        Zero-based x boundary indices. For a 64x32x32 dense grid over X=128,
        use (32, 64, 96). Boundary b compares columns immediately to the left
        of b with columns starting at b.
    half_width:
        Number of columns to average on each side of the boundary.

    Returns
    -------
    dict[int, float]
        Mapping from boundary x index to seam jump value.

    Definition
    ----------
    Selected as canonical seam metric; reproduces historical 1.46x
    (Tiny pred/label mean = 1.4618). It averages the left and right slabs
    separately, takes abs(left_avg - right_avg) on real and imaginary channels,
    then averages over [2, Z, Y]. This is the mean absolute real/imag component
    jump and matches the component-wise convention used by _complex_l1.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 4 or volume.shape[0] != 2:
        raise ValueError(f"Expected volume shape [2,Z,X,Y], got {volume.shape}")
    if half_width < 1:
        raise ValueError(f"half_width must be >= 1, got {half_width}")

    _, _, nx, _ = volume.shape
    jumps: dict[int, float] = {}

    for boundary in boundaries_x:
        b = int(boundary)
        left_start = b - int(half_width)
        left_stop = b
        right_start = b
        right_stop = b + int(half_width)

        if left_start < 0 or right_stop > nx:
            raise ValueError(
                f"Boundary x={b} with half_width={half_width} is out of bounds for X={nx}."
            )

        left_avg = np.mean(volume[:, :, left_start:left_stop, :], axis=2)
        right_avg = np.mean(volume[:, :, right_start:right_stop, :], axis=2)
        jumps[b] = float(np.mean(np.abs(left_avg - right_avg)))

    return jumps


def seam_metric_complex_magnitude(
    volume: np.ndarray,
    boundaries_x: Sequence[int],
    half_width: int = 1,
) -> dict[int, float]:
    """
    Compute x-boundary seam jumps using complex magnitude of the jump vector.

    For each boundary x=b, this averages left/right slabs separately, forms the
    complex difference between the two averaged slabs, computes
    sqrt(real_diff**2 + imag_diff**2), then averages over [Z, Y].

    Retained only for cross-validation; it also reproduces the historical
    1.461x Tiny pred/label ratio under the complex-magnitude convention.
    seam_metric is the canonical reported metric.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 4 or volume.shape[0] != 2:
        raise ValueError(f"Expected volume shape [2,Z,X,Y], got {volume.shape}")
    if half_width < 1:
        raise ValueError(f"half_width must be >= 1, got {half_width}")

    _, _, nx, _ = volume.shape
    jumps: dict[int, float] = {}

    for boundary in boundaries_x:
        b = int(boundary)
        left_start = b - int(half_width)
        left_stop = b
        right_start = b
        right_stop = b + int(half_width)

        if left_start < 0 or right_stop > nx:
            raise ValueError(
                f"Boundary x={b} with half_width={half_width} is out of bounds for X={nx}."
            )

        left_avg = np.mean(volume[:, :, left_start:left_stop, :], axis=2)
        right_avg = np.mean(volume[:, :, right_start:right_stop, :], axis=2)
        diff = left_avg - right_avg
        jumps[b] = float(np.mean(np.sqrt(diff[0] ** 2 + diff[1] ** 2)))

    return jumps


@torch.no_grad()
def infer_full_volume(
    patch_dir: str | Path,
    ckpt_path: str | Path,
    device: str | torch.device,
    full_shape: Sequence[int] = DEFAULT_FULL_SHAPE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_group: str = DEFAULT_SAMPLE_GROUP,
    patch_size: Sequence[int] | None = None,
    stride: Sequence[int] | None = None,
    z_range: Sequence[int] | None = None,
    weighted: bool = False,
    weight_floor: float = 0.01,
    output_tag: str | None = None,
) -> dict[str, object]:
    """
    Stream dense patches through the trained Tiny model and stitch full volumes.

    Inputs are read one patch at a time to avoid loading the full dense RF tensor
    set into memory. Normalization and denormalization mirror
    rf_eval_utils.evaluate_full_test_set:
        pred_raw = pred * scale
        y_raw = y * scale
        b_raw = b * scale
    where scale is computed like RFLearningDataset(normalize=True).
    """
    patch_dir = resolve_patch_dir(patch_dir)
    ckpt_path = Path(ckpt_path)
    base_output_dir = Path(output_dir)
    full_shape = tuple(int(v) for v in full_shape)
    device = torch.device(device)

    model = _load_model_for_inference(ckpt_path, device=device)

    patches = load_dense_patches(patch_dir, fields=(), sample_group=sample_group)
    print(f"patch_dir: {patch_dir}")
    print(f"patches loaded: {len(patches)}")
    print(f"ckpt_path: {ckpt_path}")
    print(f"device: {device}")

    first = patches[0]
    actual_patch_size = (len(first.z_idx), len(first.x_idx), len(first.y_idx))
    if patch_size is None:
        patch_size = actual_patch_size
    else:
        patch_size = tuple(int(v) for v in patch_size)
        if patch_size != actual_patch_size:
            raise ValueError(f"patch_size={patch_size} does not match patch metadata {actual_patch_size}")
    stride_tuple = None if stride is None else tuple(int(v) for v in stride)
    print(f"patch_size: {tuple(int(v) for v in patch_size)}")
    print(f"stride: {stride_tuple}")

    stitch_patches: Sequence[RFPatch] = patches
    stitch_shape = full_shape
    summary: dict[str, object] | None = None
    z_coverage_1based: tuple[int, int] | None = None
    z_coverage_0based: tuple[int, int] | None = None

    if z_range is None:
        if not weighted:
            summary = check_coverage(patches, full_shape)
            print("coverage check: PASS")
            print(f"coverage min/max: {summary['coverage_min']} / {summary['coverage_max']}")
            print(f"coverage histogram: {_format_histogram(summary['coverage_histogram'])}")
            print(f"missing voxels: {summary['missing_voxels']}")
            print(f"overlapped voxels: {summary['overlapped_voxels']}")
        else:
            _validate_patch_bounds(patches, full_shape)
    else:
        z_min, z_max = (int(v) for v in z_range)
        if z_min < 1 or z_max < z_min or z_max > full_shape[0]:
            raise ValueError(f"Invalid 1-based z_range={z_range} for full_shape={full_shape}")
        raw_z_min = min(int(p.z_idx[0]) for p in patches)
        raw_z_max = max(int(p.z_idx[-1]) for p in patches)
        if raw_z_min < z_min or raw_z_max > z_max:
            raise ValueError(
                f"Patch z indices {raw_z_min}-{raw_z_max} fall outside requested z_range {z_min}-{z_max}"
            )

        stitch_shape = (z_max - z_min + 1, full_shape[1], full_shape[2])
        stitch_patches = [_shift_patch_to_zband(p, z_min_1based=z_min) for p in patches]
        _validate_patch_bounds(stitch_patches, stitch_shape)
        z_coverage_1based = (z_min, z_max)
        z_coverage_0based = (z_min - 1, z_max - 1)

        print("z-band stitch mode: ON")
        print(f"z_range_1based: {z_coverage_1based[0]}-{z_coverage_1based[1]}")
        print(f"z_range_0based: {z_coverage_0based[0]}-{z_coverage_0based[1]}")
        print(f"stitch_shape: {stitch_shape}")

    hann_weight = make_hann3d_weight(patch_size, floor=weight_floor) if weighted else None
    if hann_weight is not None:
        print("weighted fusion: Hann 3D")
        print(
            f"hann weight: shape={hann_weight.shape}, "
            f"min={float(hann_weight.min()):.6e}, max={float(hann_weight.max()):.6e}"
        )

    if output_tag is None and z_range is not None:
        z_min, z_max = (int(v) for v in z_range)
        if stride_tuple is not None and patch_size is not None:
            overlaps = [round((1.0 - (s / p)) * 100) for p, s in zip(patch_size, stride_tuple)]
            overlap_tag = f"overlap{overlaps[0]}" if len(set(overlaps)) == 1 else "overlap"
        else:
            overlap_tag = "weighted" if weighted else "zband"
        output_tag = f"{_source_stem(first.source_file)}_frame{first.frame_id}_{overlap_tag}_zband{z_min}-{z_max}"
    elif output_tag is None and stride_tuple is not None and patch_size is not None:
        if tuple(stride_tuple) == tuple(int(v) for v in patch_size):
            size_tag = f"dense{int(patch_size[0])}_nooverlap"
        else:
            size_tag = "dense_custom"
        output_tag = f"{_source_stem(first.source_file)}_frame{first.frame_id}_{size_tag}"

    output_dir = base_output_dir / output_tag if output_tag else base_output_dir

    for i, patch in enumerate(patches, start=1):
        x, y, b = _read_patch_tensors(patch.path, sample_group=sample_group, device=device)

        scale = torch.amax(torch.abs(y)) + 1e-8
        scale = _scale_to_broadcast(scale, device=device)

        x_norm = x / scale
        y_norm = y / scale
        b_norm = b / scale

        pred = model(x_norm, b_norm)

        patch.data["pred"] = (pred * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        patch.data["label"] = (y_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        patch.data["baseline"] = (b_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        del x, y, b, x_norm, y_norm, b_norm, pred, scale

        if i == 1 or i % 100 == 0 or i == len(patches):
            print(f"inferred {i:04d}/{len(patches)} patches")

    pred_volume, weight_buffer = stitch_volume(
        stitch_patches,
        stitch_shape,
        "pred",
        weight=hann_weight,
        return_weight=True,
    )
    label_volume = stitch_volume(stitch_patches, stitch_shape, "label", weight=hann_weight)
    baseline_volume = stitch_volume(stitch_patches, stitch_shape, "baseline", weight=hann_weight)

    weight_zero_count = int(np.count_nonzero(weight_buffer <= 0))
    if weight_zero_count != 0:
        raise AssertionError(f"Weight-buffer self-check failed: {weight_zero_count} voxels have zero weight.")
    print("weight-buffer self-check: PASS")
    print(
        f"weight buffer: shape={weight_buffer.shape}, "
        f"min={float(weight_buffer.min()):.6e}, max={float(weight_buffer.max()):.6e}, "
        f"zero_count={weight_zero_count}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pred.npy", pred_volume)
    np.save(output_dir / "label.npy", label_volume)
    np.save(output_dir / "baseline.npy", baseline_volume)

    pred_l1 = _complex_l1(pred_volume, label_volume)
    baseline_l1 = _complex_l1(baseline_volume, label_volume)

    meta_text = "\n".join(
        [
            f"source_file: {first.source_file}",
            f"frame_id: {first.frame_id}",
            f"patch_dir: {patch_dir}",
            f"ckpt_path: {ckpt_path}",
            f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
            f"num_patches: {len(patches)}",
            f"full_shape: {full_shape}",
            f"stitch_shape: {stitch_shape}",
            f"patch_size: {tuple(int(v) for v in patch_size)}",
            f"stride: {stride_tuple}",
            f"z_range_1based: {z_coverage_1based}",
            f"z_range_0based: {z_coverage_0based}",
            f"weighted_fusion: {weighted}",
            f"weight_type: {'hann3d' if weighted else 'uniform'}",
            f"weight_floor: {weight_floor}",
            f"weight_buffer_min: {float(weight_buffer.min()):.9e}",
            f"weight_buffer_max: {float(weight_buffer.max()):.9e}",
            f"weight_buffer_zero_count: {weight_zero_count}",
            f"pred_shape: {pred_volume.shape}",
            f"label_shape: {label_volume.shape}",
            f"baseline_shape: {baseline_volume.shape}",
            f"pred_complex_l1: {pred_l1:.9e}",
            f"baseline_complex_l1: {baseline_l1:.9e}",
            f"pred_better_than_baseline: {pred_l1 < baseline_l1}",
            "",
        ]
    )
    (output_dir / "meta.txt").write_text(meta_text, encoding="utf-8")

    print(f"saved pred: {output_dir / 'pred.npy'}")
    print(f"saved label: {output_dir / 'label.npy'}")
    print(f"saved baseline: {output_dir / 'baseline.npy'}")
    print(f"saved meta: {output_dir / 'meta.txt'}")
    print(f"source_file: {first.source_file}")
    print(f"frame_id: {first.frame_id}")

    _print_volume_stats("pred", pred_volume)
    _print_volume_stats("label", label_volume)
    _print_volume_stats("baseline", baseline_volume)
    print(f"full-volume complex L1(pred, label): {pred_l1:.9e}")
    print(f"full-volume complex L1(baseline, label): {baseline_l1:.9e}")
    print(f"pred lower than baseline: {pred_l1 < baseline_l1}")

    return {
        "pred": pred_volume,
        "label": label_volume,
        "baseline": baseline_volume,
        "pred_l1": pred_l1,
        "baseline_l1": baseline_l1,
        "output_dir": output_dir,
        "coverage": summary,
        "weight_buffer": weight_buffer,
    }


def _load_stitched_volumes(volume_dir: str | Path) -> dict[str, np.ndarray]:
    volume_dir = Path(volume_dir)
    volumes = {
        "baseline": np.load(volume_dir / "baseline.npy", mmap_mode="r"),
        "pred": np.load(volume_dir / "pred.npy", mmap_mode="r"),
        "label": np.load(volume_dir / "label.npy", mmap_mode="r"),
    }

    for name, volume in volumes.items():
        if volume.ndim != 4 or volume.shape[0] != 2:
            raise ValueError(f"{name} in {volume_dir} must be [2,Z,X,Y], got {volume.shape}")

    return volumes


def _crop_volume_z(volumes: dict[str, np.ndarray], z_start0: int, z_stop0: int) -> dict[str, np.ndarray]:
    cropped = {}
    for name, volume in volumes.items():
        if z_start0 < 0 or z_stop0 > volume.shape[1] or z_stop0 <= z_start0:
            raise ValueError(f"Invalid z crop {z_start0}:{z_stop0} for {name} shape {volume.shape}")
        cropped[name] = volume[:, z_start0:z_stop0, :, :]
    return cropped


def generate_dense64_vs_dense32_bmode(
    dense32_dir: str | Path = DEFAULT_DENSE32_VOLUME_DIR,
    dense64_dir: str | Path = DEFAULT_DENSE64_VOLUME_DIR,
    view: str = "xz",
    slice_index: int = 64,
    z_start0: int = 0,
    z_stop0: int = DEFAULT_FULL_SHAPE[0],
    db_min: float = -60.0,
    source_stem: str = "Carotid_012",
) -> Path:
    """Generate a 2x3 B-mode comparison for dense32 vs dense64 no-overlap outputs."""
    import matplotlib.pyplot as plt
    from rf_visualization import (
        complex_abs_numpy,
        extract_slice,
        physical_aspect_for_view,
        physical_spacing_label,
        volume_to_db,
    )

    dense32_dir = Path(dense32_dir)
    dense64_dir = Path(dense64_dir)
    names = ("baseline", "pred", "label")

    dense32 = _crop_volume_z(_load_stitched_volumes(dense32_dir), z_start0=z_start0, z_stop0=z_stop0)
    dense64 = _crop_volume_z(_load_stitched_volumes(dense64_dir), z_start0=z_start0, z_stop0=z_stop0)

    expected_shape = dense64["pred"].shape
    for name in names:
        if dense32[name].shape != expected_shape:
            raise ValueError(f"dense32 {name} shape {dense32[name].shape} != dense64 shape {expected_shape}")
        if dense64[name].shape != expected_shape:
            raise ValueError(f"dense64 {name} shape {dense64[name].shape} != dense64 pred shape {expected_shape}")

    dense32_mag = {name: complex_abs_numpy(volume) for name, volume in dense32.items()}
    dense64_mag = {name: complex_abs_numpy(volume) for name, volume in dense64.items()}
    ref = max(
        max(float(np.max(mag)) for mag in dense32_mag.values()),
        max(float(np.max(mag)) for mag in dense64_mag.values()),
    )
    dense32_db = {name: volume_to_db(mag, ref=ref, db_min=db_min) for name, mag in dense32_mag.items()}
    dense64_db = {name: volume_to_db(mag, ref=ref, db_min=db_min) for name, mag in dense64_mag.items()}

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
    aspect = physical_aspect_for_view(view)
    row_specs = (("32x16x16 no-overlap", dense32_db), ("64x32x32 no-overlap", dense64_db))

    im = None
    used_index = slice_index
    for row, (row_name, db_volumes) in enumerate(row_specs):
        for col, name in enumerate(names):
            ax = axes[row, col]
            img, used_index, xlabel, ylabel = extract_slice(db_volumes[name], view=view, slice_index=slice_index)
            im = ax.imshow(img, cmap="gray", vmin=db_min, vmax=0.0, origin="upper", aspect=aspect)
            ax.set_title(f"{row_name} | {name} | {view} index={used_index}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="dB")

    fig.suptitle(
        f"{source_stem} dense32 vs dense64 | z0={z_start0}-{z_stop0 - 1} | "
        f"shared ref={ref:.6e}\n{physical_spacing_label(view)}",
        fontsize=10,
    )

    figs_dir = dense64_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    save_path = figs_dir / f"{source_stem}_compare_dense32_vs_dense64_nooverlap_{view}_y{used_index}.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"dense32 dir: {dense32_dir}")
    print(f"dense64 dir: {dense64_dir}")
    print(f"z crop 0-based: {z_start0}-{z_stop0 - 1}")
    print(f"comparison shape: {expected_shape}")
    print(f"shared dB ref: {ref:.6e}")
    print(f"imshow aspect for {view}: {aspect:.6f}")
    print(f"saved comparison figure: {save_path}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dense RF patch coverage and stitch a field.")
    parser.add_argument(
        "--patch-dir",
        default=DEFAULT_DENSE_PATCH_DIR,
    )
    parser.add_argument("--full-shape", nargs=3, type=int, default=DEFAULT_FULL_SHAPE)
    parser.add_argument("--field", default="label", choices=("label", "baseline", "input"))
    parser.add_argument("--sample-group", default=DEFAULT_SAMPLE_GROUP)
    parser.add_argument("--infer", action="store_true", help="Run streaming model inference and stitch full volumes.")
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch-size", nargs=3, type=int, default=None)
    parser.add_argument("--stride", nargs=3, type=int, default=None)
    parser.add_argument("--z-range", nargs=2, type=int, default=None)
    parser.add_argument("--weighted", action="store_true", help="Use 3D Hann weighted patch fusion.")
    parser.add_argument("--weight-floor", type=float, default=0.01)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument(
        "--plot-dense64-comparison",
        action="store_true",
        help="Generate dense32-vs-dense64 no-overlap B-mode comparison figure.",
    )
    parser.add_argument("--dense32-dir", default=str(DEFAULT_DENSE32_VOLUME_DIR))
    parser.add_argument("--dense64-dir", default=str(DEFAULT_DENSE64_VOLUME_DIR))
    parser.add_argument("--compare-view", default="xz", choices=("xz", "xy", "zy"))
    parser.add_argument("--compare-slice-index", type=int, default=64)
    parser.add_argument("--compare-z-start0", type=int, default=0)
    parser.add_argument("--compare-z-stop0", type=int, default=DEFAULT_FULL_SHAPE[0])
    parser.add_argument("--db-min", type=float, default=-60.0)
    args = parser.parse_args()

    if args.plot_dense64_comparison:
        generate_dense64_vs_dense32_bmode(
            dense32_dir=args.dense32_dir,
            dense64_dir=args.dense64_dir,
            view=args.compare_view,
            slice_index=args.compare_slice_index,
            z_start0=args.compare_z_start0,
            z_stop0=args.compare_z_stop0,
            db_min=args.db_min,
        )
        return

    if args.infer:
        infer_full_volume(
            patch_dir=args.patch_dir,
            ckpt_path=args.ckpt_path,
            device=args.device,
            full_shape=tuple(args.full_shape),
            output_dir=args.output_dir,
            sample_group=args.sample_group,
            patch_size=args.patch_size,
            stride=args.stride,
            z_range=args.z_range,
            weighted=args.weighted,
            weight_floor=args.weight_floor,
            output_tag=args.output_tag,
        )
        return

    patches = load_dense_patches(args.patch_dir, fields=(args.field,), sample_group=args.sample_group)
    print(f"patch_dir: {resolve_patch_dir(args.patch_dir)}")
    print(f"patches loaded: {len(patches)}")
    print(f"field loaded: {args.field}")

    summary = check_coverage(patches, tuple(args.full_shape))
    print("coverage check: PASS")
    print(f"coverage min/max: {summary['coverage_min']} / {summary['coverage_max']}")
    print(f"coverage histogram: {_format_histogram(summary['coverage_histogram'])}")
    print(f"missing voxels: {summary['missing_voxels']}")
    print(f"overlapped voxels: {summary['overlapped_voxels']}")

    stitched = stitch_volume(patches, tuple(args.full_shape), args.field)
    print(f"stitched {args.field} shape: {stitched.shape}")
    print(f"stitched {args.field} dtype: {stitched.dtype}")
    print(f"stitched {args.field} finite: {bool(np.isfinite(stitched).all())}")
    print(f"stitched {args.field} min/max: {float(stitched.min()):.6e} / {float(stitched.max()):.6e}")

    first = patches[0]
    last = patches[-1]
    print(
        "first patch idx: "
        f"z={int(first.z_idx[0])}-{int(first.z_idx[-1])}, "
        f"x={int(first.x_idx[0])}-{int(first.x_idx[-1])}, "
        f"y={int(first.y_idx[0])}-{int(first.y_idx[-1])}"
    )
    print(
        "last patch idx: "
        f"z={int(last.z_idx[0])}-{int(last.z_idx[-1])}, "
        f"x={int(last.x_idx[0])}-{int(last.x_idx[-1])}, "
        f"y={int(last.y_idx[0])}-{int(last.y_idx[-1])}"
    )
    print(f"source_file: {first.source_file}")
    print(f"frame_id: {first.frame_id}")


if __name__ == "__main__":
    main()
