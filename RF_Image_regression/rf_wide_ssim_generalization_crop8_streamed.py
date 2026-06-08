from __future__ import annotations

import argparse
from collections import OrderedDict
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rf_bn_multivolume_diagnostics import DEFAULT_X_BOUNDARIES, metric_row
from rf_models import build_model
from rf_stitch import (
    DEFAULT_SAMPLE_GROUP,
    _format_histogram,
    _read_baseline,
    _read_input,
    _read_label,
    _read_patch_tensors,
    load_dense_patches,
)
from rf_train_utils import count_trainable_parameters, get_device, seed_everything
from rf_visualization import complex_abs_numpy, extract_slice, physical_aspect_for_view, physical_spacing_label, volume_to_db


TASK_NAME = "wide+SSIM crop8 generalization across tissues/volumes"
ROOT = Path(__file__).resolve().parent
MATLAB_TEMPLATE = ROOT / "MATLAB_dense_generation" / "build_RF_dense_zband_RF000489_crop8.m"
MATLAB_TMP_DIR = ROOT / "MATLAB_dense_generation" / "_generalize_stream_tmp"
MATLAB_LOG_DIR = ROOT / "MATLAB_dense_generation" / "_generalize_stream_logs"
OUT_DIR = ROOT / "test_metrics" / "wide_ssim_generalize"
WIDE_VIS_ROOT = (
    ROOT
    / "vis_best_model"
    / "wide_ssim_random64_full1500"
    / "full_volume_stream_crop8_generalize"
)
WIDE_CKPT = ROOT / "checkpoint" / "wide_ssim_random64_full1500" / "best_by_val_ssim.pth"
BN_CKPT = ROOT / "checkpoint" / "tiny_bn_random64_full1500" / "best_model.pth"

FULL_SHAPE = (1024, 128, 128)
PATCH_SIZE = (64, 32, 32)
STRIDE = (48, 16, 16)
CROP = (8, 8, 8)
Y_INDEX = 64
BOUNDARY_MARGIN = 3
DB_MIN = -60.0
TEMP_ROOT_PREFIX_WIN = r"F:\DAS\RF_LearningSamples_dense_generalize_crop8_"
TEMP_ROOT_PREFIX_WSL = "/mnt/f/DAS/RF_LearningSamples_dense_generalize_crop8_"
MEMMAP_TEMP_PREFIX_WSL = "/mnt/f/DAS/RF_LearningSamples_dense_generalize_memmap_"
MANIFEST_DIR_WIN = r"F:\Data_0110_RFdata\_manifest"


DEFAULT_CASE_FILE_IDS = (
    "RF000487",  # Carotid_073, held-out carotid distinct from RF000489/Carotid_008.
    "RF000386",  # Muscle_056.
    "RF000587",  # Phantom_036.
)


@dataclass(frozen=True)
class CaseSpec:
    file_id: str
    category: str
    source_file: str
    nooverlap_patch_dir: Path

    @property
    def source_stem(self) -> str:
        return Path(self.source_file).stem

    @property
    def case_id(self) -> str:
        return f"{self.file_id}_{self.source_stem}"

    @property
    def manifest_name(self) -> str:
        return f"RF_manifest_dense_{self.category}_{self.file_id}_64x32x32.csv"

    @property
    def volume_dir(self) -> Path:
        return WIDE_VIS_ROOT / f"{self.case_id}_frame1_dense64_crop8_streamed"


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


def print_header() -> None:
    print("=" * 112, flush=True)
    print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"task: {TASK_NAME}", flush=True)
    print("=" * 112, flush=True)


def z_start_grid() -> list[int]:
    starts = list(range(1, FULL_SHAPE[0] - PATCH_SIZE[0] + 2, STRIDE[0]))
    assert starts[-1] == FULL_SHAPE[0] - PATCH_SIZE[0] + 1
    return starts


def make_bands(starts_per_band: int = 3) -> list[Band]:
    starts = z_start_grid()
    return [
        Band(index, tuple(starts[i : i + starts_per_band]))
        for index, i in enumerate(range(0, len(starts), starts_per_band), start=1)
    ]


def first_norm_name(model: torch.nn.Module) -> str:
    norm = next((m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d))), None)
    return type(norm).__name__ if norm is not None else "MISSING"


def load_wide_model(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(WIDE_CKPT, map_location=device)
    config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    model_name = str(config.get("model_name", "wide_deep"))
    hidden = int(config.get("hidden", 128))
    model = build_model(
        model_name,
        in_channels=1536,
        hidden=hidden,
        head_channels=32,
        out_channels=2,
        num_blocks=4,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    model_class = type(model).__name__
    norm_class = first_norm_name(model)
    print(f"wide_startup ckpt={WIDE_CKPT}", flush=True)
    print(f"wide_startup model_class={model_class}", flush=True)
    print(f"wide_startup first_norm_class={norm_class}", flush=True)
    print(f"wide_startup trainable_parameters={count_trainable_parameters(model):,}", flush=True)
    print(f"wide_startup checkpoint_epoch={ckpt.get('epoch', 'NA')}", flush=True)
    assert model_class == "WideDeepResidualRFNet", model_class
    assert isinstance(next(m for m in model.modules() if isinstance(m, nn.BatchNorm3d)), nn.BatchNorm3d)
    return model


def load_bn_model(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(BN_CKPT, map_location=device)
    state_dict = ckpt["model"]
    use_bn = any("running_mean" in key for key in state_dict)
    model = build_model(
        "tiny",
        in_channels=1536,
        hidden=64,
        out_channels=2,
        use_batch_norm=use_bn,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    model_class = type(model).__name__
    norm_class = first_norm_name(model)
    print(f"bn_startup ckpt={BN_CKPT}", flush=True)
    print(f"bn_startup model_class={model_class}", flush=True)
    print(f"bn_startup first_norm_class={norm_class}", flush=True)
    print(f"bn_startup trainable_parameters={count_trainable_parameters(model):,}", flush=True)
    print(f"bn_startup checkpoint_epoch={ckpt.get('epoch', 'NA')}", flush=True)
    assert model_class == "TinyResidualRFNet", model_class
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
        raise ValueError(f"Refusing to delete outside temp root: {path}")
    valid_prefixes = (Path(TEMP_ROOT_PREFIX_WSL).name, Path(MEMMAP_TEMP_PREFIX_WSL).name)
    if not path.name.startswith(valid_prefixes):
        raise ValueError(f"Refusing to delete unexpected temp directory: {path}")
    if path.exists():
        print(f"delete_temp_dir={path}", flush=True)
        shutil.rmtree(path)


def temp_output_root_win(case: CaseSpec, band: Band) -> str:
    return TEMP_ROOT_PREFIX_WIN + f"{case.case_id}_{band.name}"


def temp_output_root_wsl(case: CaseSpec, band: Band) -> Path:
    return Path(TEMP_ROOT_PREFIX_WSL + f"{case.case_id}_{band.name}")


def temp_patch_dir_wsl(case: CaseSpec, band: Band) -> Path:
    return temp_output_root_wsl(case, band) / "test" / case.category


def render_matlab_script(case: CaseSpec, band: Band) -> Path:
    MATLAB_TMP_DIR.mkdir(parents=True, exist_ok=True)
    text = MATLAB_TEMPLATE.read_text(encoding="utf-8-sig")
    dense_manifest_name = f"RF_manifest_dense_{case.case_id}_generalize_crop8_{band.name}.csv"
    text = text.replace(
        "'RF_manifest_dense_carotid_RF000489_64x32x32.csv'",
        f"'{case.manifest_name}'",
    )
    text = text.replace(
        "OutputRoot = 'F:\\DAS\\RF_LearningSamples_dense_RF000489_zband448-576_crop8';",
        f"OutputRoot = '{temp_output_root_win(case, band)}';",
    )
    text = text.replace(
        "'RF_manifest_dense_RF000489_zband448-576_crop8.csv'",
        f"'{dense_manifest_name}'",
    )
    text = text.replace("z_range = [448, 576];", f"z_range = [{band.z_min}, {band.z_max}];")
    text = text.replace("dense_test_file_id = 'RF000489';", f"dense_test_file_id = '{case.file_id}';")
    text = text.replace("strcmp(string(manifest.category), 'carotid')", f"strcmp(string(manifest.category), '{case.category}')")
    text = text.replace(
        "error('No carotid test RF file found for dense tiled generation.');",
        "error('No selected test RF file found for dense tiled generation.');",
    )
    # MATLAB run() evaluates the script stem and silently truncates names
    # longer than namelengthmax (63 chars), so keep generated script names short.
    path = MATLAB_TMP_DIR / f"g_{case.file_id}_b{band.index:02d}.m"
    path.write_text(text, encoding="utf-8")
    return path


def run_matlab_generation(case: CaseSpec, band: Band) -> None:
    out_root = temp_output_root_wsl(case, band)
    safe_rmtree(out_root)
    tmp_script = render_matlab_script(case, band)
    MATLAB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout = MATLAB_LOG_DIR / f"{case.case_id}_{band.name}_stdout.txt"
    stderr = MATLAB_LOG_DIR / f"{case.case_id}_{band.name}_stderr.txt"
    repo_win = r"\\wsl.localhost\Ubuntu\home\liujia\RF_Image"
    script_win = rf"\\wsl.localhost\Ubuntu\home\liujia\RF_Image\MATLAB_dense_generation\_generalize_stream_tmp\{tmp_script.name}"
    ps = (
        "$matlab = 'D:\\Program Files\\MATLAB\\R2024b\\bin\\matlab.exe'; "
        f"$repo = '{repo_win}'; "
        f"$script = '{script_win}'; "
        f"$stdout = '{repo_win}\\MATLAB_dense_generation\\_generalize_stream_logs\\{stdout.name}'; "
        f"$stderr = '{repo_win}\\MATLAB_dense_generation\\_generalize_stream_logs\\{stderr.name}'; "
        "$cmd = \"cd('$repo'); run('$script');\"; "
        "& $matlab -batch $cmd 1> $stdout 2> $stderr; "
        "exit $LASTEXITCODE"
    )
    print(
        f"\nmatlab_generate {case.case_id} {band.name}: "
        f"category={case.category} z_range=[{band.z_min},{band.z_max}] "
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
        raise RuntimeError(f"MATLAB generation failed for {case.case_id} {band.name}: code={result.returncode}{tail}")

    patch_dir = temp_patch_dir_wsl(case, band)
    count = len(list(patch_dir.glob("*.h5")))
    print(f"matlab_done {case.case_id} {band.name}: h5_count={count}", flush=True)
    if count != band.expected_patches:
        raise AssertionError(f"{case.case_id} {band.name}: expected {band.expected_patches}, got {count}")


def crop_slices_for_patch(patch: Any) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    z_idx = np.asarray(patch.z_idx, dtype=np.int32)
    x_idx = np.asarray(patch.x_idx, dtype=np.int32)
    y_idx = np.asarray(patch.y_idx, dtype=np.int32)
    data_slices = []
    out_slices = []
    for idx, (lo, hi), crop in zip(
        (z_idx, x_idx, y_idx),
        ((1, FULL_SHAPE[0]), (1, FULL_SHAPE[1]), (1, FULL_SHAPE[2])),
        CROP,
    ):
        before = 0 if int(idx[0]) == lo else int(crop)
        after = 0 if int(idx[-1]) == hi else int(crop)
        stop = len(idx) - after
        if before >= stop:
            raise ValueError(f"Invalid crop idx={idx[0]}..{idx[-1]} crop={crop}")
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


def normalized_infer(
    model: torch.nn.Module,
    patch_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, b = _read_patch_tensors(patch_path, DEFAULT_SAMPLE_GROUP, device)
    scale = torch.amax(torch.abs(y)) + 1e-8
    scale = scale_to_broadcast(scale, y)
    x_norm = x / scale
    y_norm = y / scale
    b_norm = b / scale
    use_amp = device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        pred = model(x_norm, b_norm)
    pred_np = (pred.float() * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    label_np = (y_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_np = (b_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    del x, y, b, x_norm, y_norm, b_norm, pred, scale
    return pred_np, label_np, baseline_np


def normalized_infer_arrays(
    wide_model: torch.nn.Module,
    bn_model: torch.nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    b_np: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(np.ascontiguousarray(y_np)).unsqueeze(0).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(np.ascontiguousarray(b_np)).unsqueeze(0).to(device=device, dtype=torch.float32)
    scale = torch.amax(torch.abs(y)) + 1e-8
    scale = scale_to_broadcast(scale, y)
    x_norm = x / scale
    y_norm = y / scale
    b_norm = b / scale
    use_amp = device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        wide_pred = wide_model(x_norm, b_norm)
        bn_pred = bn_model(x_norm, b_norm)
    wide_np = (wide_pred.float() * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    bn_np = (bn_pred.float() * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    label_np = (y_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_np = (b_norm * scale).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    del x, y, b, x_norm, y_norm, b_norm, wide_pred, bn_pred, scale
    return wide_np, bn_np, label_np, baseline_np


def overlap_starts(dim: int, patch: int, stride: int) -> list[int]:
    starts = list(range(1, dim - patch + 2, stride))
    if starts[-1] != dim - patch + 1:
        starts.append(dim - patch + 1)
    return sorted(set(starts))


def memmap_temp_dir(case: CaseSpec) -> Path:
    return Path(MEMMAP_TEMP_PREFIX_WSL + case.case_id)


def build_full_volume_memmap(case: CaseSpec) -> tuple[np.memmap, dict[str, np.ndarray], dict[str, Any], Path]:
    temp_dir = memmap_temp_dir(case)
    safe_rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    input_path = temp_dir / "input_float32.dat"
    input_map = np.memmap(input_path, dtype=np.float32, mode="w+", shape=(1536, *FULL_SHAPE))
    vols = {
        "label": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "baseline": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
    }
    coverage = np.zeros(FULL_SHAPE, dtype=np.uint8)
    patches = load_dense_patches(case.nooverlap_patch_dir, fields=(), sample_group=DEFAULT_SAMPLE_GROUP)
    print(f"build_memmap {case.case_id}: source_patches={len(patches)} temp_dir={temp_dir}", flush=True)
    if len(patches) != 256:
        raise AssertionError(f"{case.case_id}: expected 256 no-overlap patches, got {len(patches)}")
    for i, patch in enumerate(patches, start=1):
        with h5py.File(patch.path, "r") as h5:
            x_np = _read_input(h5, DEFAULT_SAMPLE_GROUP)
            y_np = _read_label(h5, DEFAULT_SAMPLE_GROUP)
            b_np = _read_baseline(h5, DEFAULT_SAMPLE_GROUP)
        z0, z1 = int(patch.z_idx[0]) - 1, int(patch.z_idx[-1])
        x0, x1 = int(patch.x_idx[0]) - 1, int(patch.x_idx[-1])
        y0, y1 = int(patch.y_idx[0]) - 1, int(patch.y_idx[-1])
        input_map[:, z0:z1, x0:x1, y0:y1] = x_np
        vols["label"][:, z0:z1, x0:x1, y0:y1] = y_np
        vols["baseline"][:, z0:z1, x0:x1, y0:y1] = b_np
        coverage[z0:z1, x0:x1, y0:y1] += 1
        if i == 1 or i % 32 == 0 or i == len(patches):
            print(f"build_memmap {case.case_id}: loaded {i:04d}/{len(patches)}", flush=True)
    input_map.flush()
    summary = coverage_summary_from_array(coverage)
    print(
        f"memmap_source_coverage {case.case_id}: "
        f"min/max/missing/overlap={summary['coverage_min']}/{summary['coverage_max']}/"
        f"{summary['missing_voxels']}/{summary['overlapped_voxels']}",
        flush=True,
    )
    print(f"memmap_source_coverage_hist {case.case_id}: {_format_histogram(summary['coverage_histogram'])}", flush=True)
    assert summary["coverage_min"] == 1 and summary["coverage_max"] == 1 and summary["missing_voxels"] == 0
    return input_map, vols, summary, temp_dir


def crop_slices_for_starts(z_start: int, x_start: int, y_start: int) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    class PatchLike:
        pass

    p = PatchLike()
    p.z_idx = np.arange(z_start, z_start + PATCH_SIZE[0], dtype=np.int32)
    p.x_idx = np.arange(x_start, x_start + PATCH_SIZE[1], dtype=np.int32)
    p.y_idx = np.arange(y_start, y_start + PATCH_SIZE[2], dtype=np.int32)
    return crop_slices_for_patch(p)


class DenseTileCache:
    def __init__(self, tile_map: dict[tuple[int, int, int], Path], max_tiles: int = 6):
        self.tile_map = tile_map
        self.max_tiles = int(max_tiles)
        self.cache: OrderedDict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, key: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.tile_map[key]
        with h5py.File(path, "r") as h5:
            tile = (
                _read_input(h5, DEFAULT_SAMPLE_GROUP),
                _read_label(h5, DEFAULT_SAMPLE_GROUP),
                _read_baseline(h5, DEFAULT_SAMPLE_GROUP),
            )
        self.cache[key] = tile
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_tiles:
            self.cache.popitem(last=False)
        return tile

    def clear(self) -> None:
        self.cache.clear()


def build_tile_index(case: CaseSpec) -> tuple[dict[tuple[int, int, int], Path], dict[str, Any]]:
    patches = load_dense_patches(case.nooverlap_patch_dir, fields=(), sample_group=DEFAULT_SAMPLE_GROUP)
    print(f"build_tile_index {case.case_id}: source_patches={len(patches)}", flush=True)
    if len(patches) != 256:
        raise AssertionError(f"{case.case_id}: expected 256 no-overlap patches, got {len(patches)}")
    tile_map: dict[tuple[int, int, int], Path] = {}
    coverage = np.zeros(FULL_SHAPE, dtype=np.uint8)
    for patch in patches:
        key = (int(patch.z_idx[0]), int(patch.x_idx[0]), int(patch.y_idx[0]))
        if key in tile_map:
            raise AssertionError(f"Duplicate tile key for {case.case_id}: {key}")
        tile_map[key] = patch.path
        z0, z1 = int(patch.z_idx[0]) - 1, int(patch.z_idx[-1])
        x0, x1 = int(patch.x_idx[0]) - 1, int(patch.x_idx[-1])
        y0, y1 = int(patch.y_idx[0]) - 1, int(patch.y_idx[-1])
        coverage[z0:z1, x0:x1, y0:y1] += 1
    summary = coverage_summary_from_array(coverage)
    print(
        f"tile_source_coverage {case.case_id}: "
        f"min/max/missing/overlap={summary['coverage_min']}/{summary['coverage_max']}/"
        f"{summary['missing_voxels']}/{summary['overlapped_voxels']}",
        flush=True,
    )
    print(f"tile_source_coverage_hist {case.case_id}: {_format_histogram(summary['coverage_histogram'])}", flush=True)
    assert summary["coverage_min"] == 1 and summary["coverage_max"] == 1 and summary["missing_voxels"] == 0
    return tile_map, summary


def covering_tile_starts(start: int, patch_len: int, tile_len: int, full_len: int) -> list[int]:
    first = ((start - 1) // tile_len) * tile_len + 1
    last_voxel = start + patch_len - 1
    starts = []
    current = first
    while current <= last_voxel:
        if current < 1 or current > full_len - tile_len + 1:
            raise ValueError(f"Invalid tile start {current} for full_len={full_len}")
        starts.append(current)
        current += tile_len
    return starts


def assemble_patch_from_tiles(
    cache: DenseTileCache,
    z_start: int,
    x_start: int,
    y_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_patch = np.empty((1536, *PATCH_SIZE), dtype=np.float32)
    y_patch = np.empty((2, *PATCH_SIZE), dtype=np.float32)
    b_patch = np.empty((2, *PATCH_SIZE), dtype=np.float32)

    z0, z1 = z_start - 1, z_start - 1 + PATCH_SIZE[0]
    x0, x1 = x_start - 1, x_start - 1 + PATCH_SIZE[1]
    y0, y1 = y_start - 1, y_start - 1 + PATCH_SIZE[2]

    z_tiles = covering_tile_starts(z_start, PATCH_SIZE[0], PATCH_SIZE[0], FULL_SHAPE[0])
    x_tiles = covering_tile_starts(x_start, PATCH_SIZE[1], PATCH_SIZE[1], FULL_SHAPE[1])
    y_tiles = covering_tile_starts(y_start, PATCH_SIZE[2], PATCH_SIZE[2], FULL_SHAPE[2])

    for tz in z_tiles:
        tz0, tz1 = tz - 1, tz - 1 + PATCH_SIZE[0]
        gz0, gz1 = max(z0, tz0), min(z1, tz1)
        for tx in x_tiles:
            tx0, tx1 = tx - 1, tx - 1 + PATCH_SIZE[1]
            gx0, gx1 = max(x0, tx0), min(x1, tx1)
            for ty in y_tiles:
                ty0, ty1 = ty - 1, ty - 1 + PATCH_SIZE[2]
                gy0, gy1 = max(y0, ty0), min(y1, ty1)
                tile_x, tile_y, tile_b = cache.get((tz, tx, ty))
                dst = (
                    slice(gz0 - z0, gz1 - z0),
                    slice(gx0 - x0, gx1 - x0),
                    slice(gy0 - y0, gy1 - y0),
                )
                src = (
                    slice(gz0 - tz0, gz1 - tz0),
                    slice(gx0 - tx0, gx1 - tx0),
                    slice(gy0 - ty0, gy1 - ty0),
                )
                x_patch[(slice(None), *dst)] = tile_x[(slice(None), *src)]
                y_patch[(slice(None), *dst)] = tile_y[(slice(None), *src)]
                b_patch[(slice(None), *dst)] = tile_b[(slice(None), *src)]

    return x_patch, y_patch, b_patch


@torch.no_grad()
def process_dual_band(
    case: CaseSpec,
    band: Band,
    wide_model: torch.nn.Module,
    bn_model: torch.nn.Module,
    device: torch.device,
    wide_acc: dict[str, np.ndarray],
    bn_acc: dict[str, np.ndarray],
    weight: np.ndarray,
) -> dict[str, Any]:
    patch_dir = temp_patch_dir_wsl(case, band)
    patches = load_dense_patches(patch_dir, fields=(), sample_group=DEFAULT_SAMPLE_GROUP)
    z_starts = sorted({int(p.z_idx[0]) for p in patches})
    x_starts = sorted({int(p.x_idx[0]) for p in patches})
    y_starts = sorted({int(p.y_idx[0]) for p in patches})
    print(f"process_dual {case.case_id} {band.name}: patches={len(patches)} z_starts={z_starts}", flush=True)
    print(f"process_dual {case.case_id} {band.name}: x_starts={x_starts} y_starts={y_starts}", flush=True)

    band_coverage = np.zeros(FULL_SHAPE, dtype=np.uint8)
    for i, patch in enumerate(patches, start=1):
        wide_pred_np, label_np, baseline_np = normalized_infer(wide_model, patch.path, device)
        bn_pred_np, _, _ = normalized_infer(bn_model, patch.path, device)
        data_slices, out_slices = crop_slices_for_patch(patch)
        z_sl, x_sl, y_sl = out_slices
        dz, dx, dy = data_slices
        wide_acc["pred"][(slice(None), z_sl, x_sl, y_sl)] += wide_pred_np[:, dz, dx, dy]
        wide_acc["label"][(slice(None), z_sl, x_sl, y_sl)] += label_np[:, dz, dx, dy]
        wide_acc["baseline"][(slice(None), z_sl, x_sl, y_sl)] += baseline_np[:, dz, dx, dy]
        bn_acc["pred"][(slice(None), z_sl, x_sl, y_sl)] += bn_pred_np[:, dz, dx, dy]
        bn_acc["label"][(slice(None), z_sl, x_sl, y_sl)] += label_np[:, dz, dx, dy]
        bn_acc["baseline"][(slice(None), z_sl, x_sl, y_sl)] += baseline_np[:, dz, dx, dy]
        weight[z_sl, x_sl, y_sl] += 1.0
        band_coverage[z_sl, x_sl, y_sl] += 1
        if device.type == "cuda" and i % 16 == 0:
            torch.cuda.empty_cache()
        if i == 1 or i % 32 == 0 or i == len(patches):
            print(f"process_dual {case.case_id} {band.name}: inferred {i:04d}/{len(patches)}", flush=True)

    nonzero = band_coverage > 0
    z_indices = np.where(np.any(nonzero, axis=(1, 2)))[0]
    effective_z0 = int(z_indices[0])
    effective_z1 = int(z_indices[-1])
    summary = coverage_summary_from_array(band_coverage[effective_z0 : effective_z1 + 1])
    summary.update(
        {
            "case": case.case_id,
            "category": case.category,
            "source_file": case.source_file,
            "band": band.name,
            "z_range_generated_1based": f"{band.z_min}-{band.z_max}",
            "effective_z_range_0based": f"{effective_z0}-{effective_z1}",
            "effective_z_range_1based": f"{effective_z0 + 1}-{effective_z1 + 1}",
            "patch_count": len(patches),
        }
    )
    print(
        f"band_coverage {case.case_id} {band.name}: effective_z0={effective_z0} effective_z1={effective_z1} "
        f"min/max/missing={summary['coverage_min']}/{summary['coverage_max']}/{summary['missing_voxels']}",
        flush=True,
    )
    assert summary["coverage_min"] >= 1 and summary["missing_voxels"] == 0
    return summary


@torch.no_grad()
def process_dual_band_from_memmap(
    case: CaseSpec,
    band: Band,
    input_map: np.memmap,
    source_vols: dict[str, np.ndarray],
    wide_model: torch.nn.Module,
    bn_model: torch.nn.Module,
    device: torch.device,
    wide_acc: dict[str, np.ndarray],
    bn_acc: dict[str, np.ndarray],
    weight: np.ndarray,
) -> dict[str, Any]:
    x_starts = overlap_starts(FULL_SHAPE[1], PATCH_SIZE[1], STRIDE[1])
    y_starts = overlap_starts(FULL_SHAPE[2], PATCH_SIZE[2], STRIDE[2])
    total = len(band.z_starts) * len(x_starts) * len(y_starts)
    print(
        f"process_memmap {case.case_id} {band.name}: "
        f"patches={total} z_starts={list(band.z_starts)} x_starts={x_starts} y_starts={y_starts}",
        flush=True,
    )
    band_coverage = np.zeros(FULL_SHAPE, dtype=np.uint8)
    i = 0
    for z_start in band.z_starts:
        z0 = z_start - 1
        z1 = z0 + PATCH_SIZE[0]
        for x_start in x_starts:
            x0 = x_start - 1
            x1 = x0 + PATCH_SIZE[1]
            for y_start in y_starts:
                y0 = y_start - 1
                y1 = y0 + PATCH_SIZE[2]
                i += 1
                x_np = np.asarray(input_map[:, z0:z1, x0:x1, y0:y1], dtype=np.float32).copy()
                y_np = source_vols["label"][:, z0:z1, x0:x1, y0:y1]
                b_np = source_vols["baseline"][:, z0:z1, x0:x1, y0:y1]
                wide_pred_np, bn_pred_np, label_np, baseline_np = normalized_infer_arrays(
                    wide_model, bn_model, x_np, y_np, b_np, device
                )
                data_slices, out_slices = crop_slices_for_starts(z_start, x_start, y_start)
                z_sl, x_sl, y_sl = out_slices
                dz, dx, dy = data_slices
                wide_acc["pred"][(slice(None), z_sl, x_sl, y_sl)] += wide_pred_np[:, dz, dx, dy]
                wide_acc["label"][(slice(None), z_sl, x_sl, y_sl)] += label_np[:, dz, dx, dy]
                wide_acc["baseline"][(slice(None), z_sl, x_sl, y_sl)] += baseline_np[:, dz, dx, dy]
                bn_acc["pred"][(slice(None), z_sl, x_sl, y_sl)] += bn_pred_np[:, dz, dx, dy]
                bn_acc["label"][(slice(None), z_sl, x_sl, y_sl)] += label_np[:, dz, dx, dy]
                bn_acc["baseline"][(slice(None), z_sl, x_sl, y_sl)] += baseline_np[:, dz, dx, dy]
                weight[z_sl, x_sl, y_sl] += 1.0
                band_coverage[z_sl, x_sl, y_sl] += 1
                if device.type == "cuda" and i % 16 == 0:
                    torch.cuda.empty_cache()
                if i == 1 or i % 32 == 0 or i == total:
                    print(f"process_memmap {case.case_id} {band.name}: inferred {i:04d}/{total}", flush=True)

    nonzero = band_coverage > 0
    z_indices = np.where(np.any(nonzero, axis=(1, 2)))[0]
    effective_z0 = int(z_indices[0])
    effective_z1 = int(z_indices[-1])
    summary = coverage_summary_from_array(band_coverage[effective_z0 : effective_z1 + 1])
    summary.update(
        {
            "case": case.case_id,
            "category": case.category,
            "source_file": case.source_file,
            "band": band.name,
            "z_range_generated_1based": f"{band.z_min}-{band.z_max}",
            "effective_z_range_0based": f"{effective_z0}-{effective_z1}",
            "effective_z_range_1based": f"{effective_z0 + 1}-{effective_z1 + 1}",
            "patch_count": total,
        }
    )
    print(
        f"band_coverage {case.case_id} {band.name}: effective_z0={effective_z0} effective_z1={effective_z1} "
        f"min/max/missing={summary['coverage_min']}/{summary['coverage_max']}/{summary['missing_voxels']}",
        flush=True,
    )
    assert summary["coverage_min"] >= 1 and summary["missing_voxels"] == 0
    return summary


@torch.no_grad()
def process_dual_band_from_tiles(
    case: CaseSpec,
    band: Band,
    cache: DenseTileCache,
    wide_model: torch.nn.Module,
    bn_model: torch.nn.Module,
    device: torch.device,
    wide_acc: dict[str, np.ndarray],
    bn_acc: dict[str, np.ndarray],
    weight: np.ndarray,
) -> dict[str, Any]:
    x_starts = overlap_starts(FULL_SHAPE[1], PATCH_SIZE[1], STRIDE[1])
    y_starts = overlap_starts(FULL_SHAPE[2], PATCH_SIZE[2], STRIDE[2])
    total = len(band.z_starts) * len(x_starts) * len(y_starts)
    print(
        f"process_tiles {case.case_id} {band.name}: "
        f"patches={total} z_starts={list(band.z_starts)} x_starts={x_starts} y_starts={y_starts}",
        flush=True,
    )
    band_coverage = np.zeros(FULL_SHAPE, dtype=np.uint8)
    i = 0
    for z_start in band.z_starts:
        for x_start in x_starts:
            for y_start in y_starts:
                i += 1
                x_np, y_np, b_np = assemble_patch_from_tiles(cache, z_start, x_start, y_start)
                wide_pred_np, bn_pred_np, label_np, baseline_np = normalized_infer_arrays(
                    wide_model, bn_model, x_np, y_np, b_np, device
                )
                data_slices, out_slices = crop_slices_for_starts(z_start, x_start, y_start)
                z_sl, x_sl, y_sl = out_slices
                dz, dx, dy = data_slices
                wide_acc["pred"][(slice(None), z_sl, x_sl, y_sl)] += wide_pred_np[:, dz, dx, dy]
                wide_acc["label"][(slice(None), z_sl, x_sl, y_sl)] += label_np[:, dz, dx, dy]
                wide_acc["baseline"][(slice(None), z_sl, x_sl, y_sl)] += baseline_np[:, dz, dx, dy]
                bn_acc["pred"][(slice(None), z_sl, x_sl, y_sl)] += bn_pred_np[:, dz, dx, dy]
                bn_acc["label"][(slice(None), z_sl, x_sl, y_sl)] += label_np[:, dz, dx, dy]
                bn_acc["baseline"][(slice(None), z_sl, x_sl, y_sl)] += baseline_np[:, dz, dx, dy]
                weight[z_sl, x_sl, y_sl] += 1.0
                band_coverage[z_sl, x_sl, y_sl] += 1
                del x_np, y_np, b_np, wide_pred_np, bn_pred_np, label_np, baseline_np
                if device.type == "cuda" and i % 16 == 0:
                    torch.cuda.empty_cache()
                if i == 1 or i % 32 == 0 or i == total:
                    print(f"process_tiles {case.case_id} {band.name}: inferred {i:04d}/{total}", flush=True)

    nonzero = band_coverage > 0
    z_indices = np.where(np.any(nonzero, axis=(1, 2)))[0]
    effective_z0 = int(z_indices[0])
    effective_z1 = int(z_indices[-1])
    summary = coverage_summary_from_array(band_coverage[effective_z0 : effective_z1 + 1])
    summary.update(
        {
            "case": case.case_id,
            "category": case.category,
            "source_file": case.source_file,
            "band": band.name,
            "z_range_generated_1based": f"{band.z_min}-{band.z_max}",
            "effective_z_range_0based": f"{effective_z0}-{effective_z1}",
            "effective_z_range_1based": f"{effective_z0 + 1}-{effective_z1 + 1}",
            "patch_count": total,
        }
    )
    print(
        f"band_coverage {case.case_id} {band.name}: effective_z0={effective_z0} effective_z1={effective_z1} "
        f"min/max/missing={summary['coverage_min']}/{summary['coverage_max']}/{summary['missing_voxels']}",
        flush=True,
    )
    assert summary["coverage_min"] >= 1 and summary["missing_voxels"] == 0
    return summary


@torch.no_grad()
def infer_bn_nooverlap(case: CaseSpec, model: torch.nn.Module, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    patches = load_dense_patches(case.nooverlap_patch_dir, fields=(), sample_group=DEFAULT_SAMPLE_GROUP)
    print(f"\ninfer_bn_nooverlap {case.case_id}: patch_dir={case.nooverlap_patch_dir} patches={len(patches)}", flush=True)
    vols = {
        "pred": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "label": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "baseline": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
    }
    coverage = np.zeros(FULL_SHAPE, dtype=np.uint8)
    for i, patch in enumerate(patches, start=1):
        pred_np, label_np, baseline_np = normalized_infer(model, patch.path, device)
        z0, z1 = int(patch.z_idx[0]) - 1, int(patch.z_idx[-1])
        x0, x1 = int(patch.x_idx[0]) - 1, int(patch.x_idx[-1])
        y0, y1 = int(patch.y_idx[0]) - 1, int(patch.y_idx[-1])
        vols["pred"][:, z0:z1, x0:x1, y0:y1] = pred_np
        vols["label"][:, z0:z1, x0:x1, y0:y1] = label_np
        vols["baseline"][:, z0:z1, x0:x1, y0:y1] = baseline_np
        coverage[z0:z1, x0:x1, y0:y1] += 1
        if device.type == "cuda" and i % 16 == 0:
            torch.cuda.empty_cache()
        if i == 1 or i % 64 == 0 or i == len(patches):
            print(f"infer_bn_nooverlap {case.case_id}: inferred {i:04d}/{len(patches)}", flush=True)
    summary = coverage_summary_from_array(coverage)
    print(
        f"bn_nooverlap_coverage {case.case_id}: "
        f"min/max/missing/overlap={summary['coverage_min']}/{summary['coverage_max']}/"
        f"{summary['missing_voxels']}/{summary['overlapped_voxels']}",
        flush=True,
    )
    print(f"bn_nooverlap_coverage_hist {case.case_id}: {_format_histogram(summary['coverage_histogram'])}", flush=True)
    assert summary["coverage_min"] == 1 and summary["coverage_max"] == 1 and summary["missing_voxels"] == 0
    summary.update({"case": case.case_id, "category": case.category, "source_file": case.source_file})
    return vols, summary


def finalize_wide_volumes(acc: dict[str, np.ndarray], weight: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    final_cov = coverage_summary_from_array(weight.astype(np.uint16))
    print(
        "wide_final_coverage min/max/missing/overlap="
        f"{final_cov['coverage_min']}/{final_cov['coverage_max']}/"
        f"{final_cov['missing_voxels']}/{final_cov['overlapped_voxels']}",
        flush=True,
    )
    print(f"wide_final_coverage histogram={_format_histogram(final_cov['coverage_histogram'])}", flush=True)
    assert final_cov["coverage_min"] == 1 and final_cov["coverage_max"] == 1 and final_cov["missing_voxels"] == 0
    vols = {field: arr / weight[None, :, :, :] for field, arr in acc.items()}
    return vols, final_cov


def save_wide_volumes(case: CaseSpec, vols: dict[str, np.ndarray], final_cov: dict[str, Any], band_rows: list[dict[str, Any]]) -> None:
    case.volume_dir.mkdir(parents=True, exist_ok=True)
    for field, volume in vols.items():
        np.save(case.volume_dir / f"{field}.npy", volume.astype(np.float32, copy=False))
    meta = [
        f"task: {TASK_NAME}",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"case_id: {case.case_id}",
        f"category: {case.category}",
        f"source_file: {case.source_file}",
        f"checkpoint: {WIDE_CKPT}",
        f"patch_size: {PATCH_SIZE}",
        f"stride: {STRIDE}",
        f"crop: {CROP}",
        f"full_shape: {FULL_SHAPE}",
        f"final_coverage: {final_cov}",
        f"bands: {len(band_rows)}",
        "",
    ]
    (case.volume_dir / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    print(f"saved_wide_volume_dir={case.volume_dir}", flush=True)


def dynamic_metrics(vols: dict[str, np.ndarray], z_slice: slice | None = None, prefix: str = "") -> dict[str, float]:
    pred = vols["pred"] if z_slice is None else vols["pred"][:, z_slice, :, :]
    label = vols["label"] if z_slice is None else vols["label"][:, z_slice, :, :]
    pred_abs = complex_abs_numpy(pred)
    label_abs = complex_abs_numpy(label)
    pred_p99 = float(np.percentile(pred_abs, 99.0))
    label_p99 = float(np.percentile(label_abs, 99.0))
    eps = 1e-12
    return {
        f"{prefix}pred_p99": pred_p99,
        f"{prefix}label_p99": label_p99,
        f"{prefix}pred_p99_over_label_p99": pred_p99 / (label_p99 + eps),
        f"{prefix}abs_std_ratio": float(np.std(pred_abs) / (np.std(label_abs) + eps)),
    }


def metrics_for(case: CaseSpec, model_name: str, vols: dict[str, np.ndarray]) -> dict[str, Any]:
    row = metric_row(
        case.case_id,
        model_name,
        vols,
        tuple(DEFAULT_X_BOUNDARIES),
        Y_INDEX,
        PATCH_SIZE,
        BOUNDARY_MARGIN,
    )
    row["file_id"] = case.file_id
    row["category"] = case.category
    row["source_file"] = case.source_file
    row.update(dynamic_metrics(vols))
    row.update(dynamic_metrics(vols, z_slice=slice(800, None), prefix="deep_z800_"))
    return row


def x_jump_profile(volume: np.ndarray) -> np.ndarray:
    mag = complex_abs_numpy(volume)
    return np.array(
        [float(np.mean(np.abs(mag[:, x, :] - mag[:, x - 1, :]))) for x in range(1, mag.shape[1])],
        dtype=np.float64,
    )


def save_x_jump_outputs(case: CaseSpec, bn_vols: dict[str, np.ndarray], wide_vols: dict[str, np.ndarray]) -> tuple[Path, Path]:
    out_dir = OUT_DIR / "x_jump"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_profile = x_jump_profile(wide_vols["label"])
    profiles = {
        "label": label_profile,
        "BN-L1 pred": x_jump_profile(bn_vols["pred"]),
        "wide+SSIM crop8 pred": x_jump_profile(wide_vols["pred"]),
    }
    rows = []
    for name, profile in profiles.items():
        for x in range(1, FULL_SHAPE[1]):
            rows.append(
                {
                    "case": case.case_id,
                    "category": case.category,
                    "source_file": case.source_file,
                    "signal": name,
                    "x": x,
                    "jump": profile[x - 1],
                    "over_label": profile[x - 1] / (label_profile[x - 1] + 1e-12),
                    "is_old_dense64_boundary": x in DEFAULT_X_BOUNDARIES,
                    "is_crop8_stride16_boundary": x % STRIDE[1] == 0,
                }
            )
    csv_path = out_dir / f"{case.case_id}_x_jump_profile.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
    for name, profile in profiles.items():
        y = profile / (label_profile + 1e-12) if name != "label" else np.ones_like(profile)
        ax.plot(np.arange(1, FULL_SHAPE[1]), y, label=name, linewidth=1.5)
    for x in DEFAULT_X_BOUNDARIES:
        ax.axvline(x, color="tab:red", linewidth=0.8, alpha=0.65)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{case.case_id} | x-jump ratio profile")
    ax.set_xlabel("x boundary index")
    ax.set_ylabel("jump / label jump")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig_path = OUT_DIR / "figs" / f"{case.case_id}_x_jump_ratio_profile.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return csv_path, fig_path


def save_visuals(case: CaseSpec, bn_vols: dict[str, np.ndarray], wide_vols: dict[str, np.ndarray]) -> list[Path]:
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
        ("xz", 64, f"{case.case_id}_compare_xz_y64.png"),
        ("xy", 850, f"{case.case_id}_compare_xy_z850.png"),
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
        fig.suptitle(f"{case.case_id} | shared dB ref | {physical_spacing_label(view)}", fontsize=10)
        path = out_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)
        print(f"saved_visual={path}", flush=True)
    return saved


def load_cases(case_ids: tuple[str, ...]) -> list[CaseSpec]:
    index = pd.read_csv(ROOT / "dense64_index.csv")
    rows = []
    for file_id in case_ids:
        match = index[index["file_id"] == file_id]
        if match.empty:
            raise ValueError(f"file_id {file_id} not found in dense64_index.csv")
        row = match.iloc[0]
        patch_dir = Path(str(row["patch_dir"]))
        rows.append(
            CaseSpec(
                file_id=str(row["file_id"]),
                category=str(row["category"]),
                source_file=str(row["source_file"]),
                nooverlap_patch_dir=patch_dir,
            )
        )
    return rows


def run_case(
    case: CaseSpec,
    bands: list[Band],
    wide_model: torch.nn.Module,
    bn_model: torch.nn.Module,
    device: torch.device,
    skip_matlab: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    print("\n" + "=" * 112, flush=True)
    print(f"case_start {case.case_id} category={case.category} source_file={case.source_file}", flush=True)
    print(f"dense64_index_patch_dir={case.nooverlap_patch_dir}", flush=True)
    print(
        "comparison_protocol=assemble virtual crop8 patches from existing no-overlap dense64 tiles; "
        "both models use the same streamed crop8 patches",
        flush=True,
    )

    wide_acc = {
        "pred": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "label": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "baseline": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
    }
    bn_acc = {
        "pred": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "label": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
        "baseline": np.zeros((2, *FULL_SHAPE), dtype=np.float32),
    }
    weight = np.zeros(FULL_SHAPE, dtype=np.float32)
    band_rows: list[dict[str, Any]] = []
    cache: DenseTileCache | None = None
    try:
        tile_map, source_cov = build_tile_index(case)
        cache = DenseTileCache(tile_map, max_tiles=16)
        for band in bands:
            if skip_matlab:
                band_rows.append(
                    process_dual_band_from_tiles(
                        case,
                        band,
                        cache,
                        wide_model,
                        bn_model,
                        device,
                        wide_acc,
                        bn_acc,
                        weight,
                    )
                )
            else:
                run_matlab_generation(case, band)
                band_rows.append(process_dual_band(case, band, wide_model, bn_model, device, wide_acc, bn_acc, weight))
                safe_rmtree(temp_output_root_wsl(case, band))
    finally:
        for band in bands:
            path = temp_output_root_wsl(case, band)
            if path.exists():
                safe_rmtree(path)
        if cache is not None:
            cache.clear()

    wide_vols, wide_cov = finalize_wide_volumes(wide_acc, weight)
    bn_vols = {field: arr / weight[None, :, :, :] for field, arr in bn_acc.items()}
    save_wide_volumes(case, wide_vols, wide_cov, band_rows)
    metrics_rows = [
        metrics_for(case, "BN-L1 crop8 streamed", bn_vols),
        metrics_for(case, "wide+SSIM crop8 streamed", wide_vols),
    ]
    metrics_rows[0].update({"coverage_min": wide_cov["coverage_min"], "coverage_max": wide_cov["coverage_max"], "missing_voxels": wide_cov["missing_voxels"]})
    metrics_rows[1].update({"coverage_min": wide_cov["coverage_min"], "coverage_max": wide_cov["coverage_max"], "missing_voxels": wide_cov["missing_voxels"]})

    x_csv, x_fig = save_x_jump_outputs(case, bn_vols, wide_vols)
    visuals = save_visuals(case, bn_vols, wide_vols)
    print(f"x_jump_csv={x_csv}", flush=True)
    print(f"x_jump_fig={x_fig}", flush=True)
    print(f"case_done {case.case_id}", flush=True)
    return metrics_rows, band_rows, [x_csv, x_fig, *visuals]


def run(case_ids: tuple[str, ...], dry_run: bool = False, skip_matlab: bool = False) -> None:
    print_header()
    seed_everything(20260522)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bands = make_bands()
    cases = load_cases(case_ids)
    print("selected_cases:", flush=True)
    for case in cases:
        print(f"  {case.case_id}: category={case.category} source_file={case.source_file}", flush=True)
    print("stream_bands:", flush=True)
    for band in bands:
        print(f"  {band.name}: z_starts={list(band.z_starts)} z_range={band.z_min}-{band.z_max}", flush=True)
    if dry_run:
        return

    device = get_device()
    print(f"device={device}", flush=True)
    wide_model = load_wide_model(device)
    bn_model = load_bn_model(device)

    all_metric_rows: list[dict[str, Any]] = []
    all_band_rows: list[dict[str, Any]] = []
    all_outputs: list[Path] = []
    for case in cases:
        metric_rows, band_rows, outputs = run_case(case, bands, wide_model, bn_model, device, skip_matlab=skip_matlab)
        all_metric_rows.extend(metric_rows)
        all_band_rows.extend(band_rows)
        all_outputs.extend(outputs)
        del metric_rows, band_rows, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_path = OUT_DIR / "wide_ssim_generalize_metrics.csv"
    band_path = OUT_DIR / "wide_ssim_generalize_band_coverage.csv"
    pd.DataFrame(all_metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(all_band_rows).to_csv(band_path, index=False)

    summary_cols = [
        "case",
        "category",
        "model",
        "x_boundary_pred_over_label",
        "xz_y_boundary_pred_over_label",
        "pred_p99_over_label_p99",
        "abs_std_ratio",
        "deep_z800_pred_p99_over_label_p99",
        "deep_z800_abs_std_ratio",
        "nonboundary_complex_improvement",
        "nonboundary_abs_improvement",
        "coverage_min",
        "coverage_max",
        "missing_voxels",
    ]
    print("\nsummary_metrics", flush=True)
    print(pd.DataFrame(all_metric_rows).reindex(columns=summary_cols).to_string(index=False), flush=True)
    print("\nsaved_outputs", flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"band_coverage={band_path}", flush=True)
    for path in all_outputs:
        print(f"output={path}", flush=True)


def parse_case_ids(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_CASE_FILE_IDS
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="", help="Comma-separated file_id list. Default: curated generalization set.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--use-matlab-bands",
        dest="skip_matlab",
        action="store_false",
        help="Use raw MATLAB generation for overlap bands instead of virtual crop8 patches from dense64 memmap.",
    )
    parser.set_defaults(skip_matlab=True)
    args = parser.parse_args()
    run(parse_case_ids(args.cases), dry_run=args.dry_run, skip_matlab=args.skip_matlab)


if __name__ == "__main__":
    main()
