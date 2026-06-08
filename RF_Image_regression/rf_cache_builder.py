from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from rf_cached_dataset import RFCachedDataset
from rf_learning_dataset import RFLearningDataset


CACHE_FILES = ("input.dat", "label.dat", "baseline.dat", "scale.dat", "meta.npz")


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _ensure_output_dir(cache_dir: Path, overwrite: bool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = [cache_dir / name for name in CACHE_FILES if (cache_dir / name).exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Cache files already exist under {cache_dir}. "
            f"Pass --overwrite to replace: {paths}"
        )
    for path in existing:
        path.unlink()


def _cache_size_bytes(cache_dir: Path) -> int:
    return sum(path.stat().st_size for path in cache_dir.iterdir() if path.is_file())


def _format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def build_split_cache(
    root_dir: str | Path,
    split: str,
    out_root: str | Path = "Data_cache",
    sample_group: str = "/sample_000001",
    normalize: bool = False,
    include_categories: Optional[Iterable[str]] = None,
    exclude_categories: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> Path:
    root_dir = Path(root_dir)
    split_dir = root_dir / split
    cache_dir = Path(out_root) / split

    dataset = RFLearningDataset(
        split_dir,
        sample_group=sample_group,
        normalize=normalize,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        verbose=True,
    )

    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError(f"No samples found for split: {split_dir}")

    _ensure_output_dir(cache_dir, overwrite=overwrite)

    first = dataset[0]
    input_shape = tuple(first["input"].shape)
    label_shape = tuple(first["label"].shape)
    baseline_shape = tuple(first["baseline"].shape)

    input_full_shape = (n_samples,) + input_shape
    label_full_shape = (n_samples,) + label_shape
    baseline_full_shape = (n_samples,) + baseline_shape

    input_mm = np.memmap(
        cache_dir / "input.dat",
        dtype=np.float16,
        mode="w+",
        shape=input_full_shape,
    )
    label_mm = np.memmap(
        cache_dir / "label.dat",
        dtype=np.float16,
        mode="w+",
        shape=label_full_shape,
    )
    baseline_mm = np.memmap(
        cache_dir / "baseline.dat",
        dtype=np.float16,
        mode="w+",
        shape=baseline_full_shape,
    )
    scale_mm = np.memmap(
        cache_dir / "scale.dat",
        dtype=np.float32,
        mode="w+",
        shape=(n_samples,),
    )

    paths = []
    categories = []
    z_indices = []
    x_indices = []
    y_indices = []

    start_time = time.time()
    for idx in range(n_samples):
        item = first if idx == 0 else dataset[idx]

        input_mm[idx] = _as_numpy(item["input"]).astype(np.float16)
        label_mm[idx] = _as_numpy(item["label"]).astype(np.float16)
        baseline_mm[idx] = _as_numpy(item["baseline"]).astype(np.float16)
        scale_mm[idx] = np.float32(item["scale"].item())

        paths.append(item["path"])
        categories.append(item["category"])
        z_indices.append(_as_numpy(item["z_idx"]).astype(np.int32))
        x_indices.append(_as_numpy(item["x_idx"]).astype(np.int32))
        y_indices.append(_as_numpy(item["y_idx"]).astype(np.int32))

        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == n_samples:
            elapsed = time.time() - start_time
            print(f"cached {idx + 1:04d}/{n_samples} samples ({elapsed:.1f}s)")

    input_mm.flush()
    label_mm.flush()
    baseline_mm.flush()
    scale_mm.flush()

    np.savez_compressed(
        cache_dir / "meta.npz",
        input_shape=np.array(input_full_shape, dtype=np.int64),
        label_shape=np.array(label_full_shape, dtype=np.int64),
        baseline_shape=np.array(baseline_full_shape, dtype=np.int64),
        input_dtype=np.array("float16"),
        label_dtype=np.array("float16"),
        baseline_dtype=np.array("float16"),
        scale_dtype=np.array("float32"),
        path=np.array(paths, dtype=str),
        category=np.array(categories, dtype=str),
        z_idx=np.stack(z_indices).astype(np.int32),
        x_idx=np.stack(x_indices).astype(np.int32),
        y_idx=np.stack(y_indices).astype(np.int32),
        root_dir=np.array(str(root_dir)),
        split=np.array(split),
        sample_group=np.array(sample_group),
        normalize=np.array(bool(normalize)),
    )

    print(f"Cache written: {cache_dir}")
    print(f"Cache size   : {_format_bytes(_cache_size_bytes(cache_dir))}")
    return cache_dir


def _assert_tensor_close(
    field: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    atol: float,
    rtol: float,
) -> float:
    if expected.shape != actual.shape:
        raise AssertionError(
            f"{field} shape mismatch: expected {tuple(expected.shape)}, "
            f"got {tuple(actual.shape)}"
        )
    if expected.dtype != actual.dtype:
        raise AssertionError(
            f"{field} dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
        )
    if not torch.allclose(expected, actual, atol=atol, rtol=rtol):
        max_diff = torch.max(torch.abs(expected - actual)).item()
        raise AssertionError(
            f"{field} mismatch: max_abs_diff={max_diff:.6e}, "
            f"atol={atol}, rtol={rtol}"
        )
    return torch.max(torch.abs(expected - actual)).item()


def verify_split_cache(
    root_dir: str | Path,
    split: str,
    cache_dir: str | Path,
    sample_group: str = "/sample_000001",
    normalize: bool = False,
    include_categories: Optional[Iterable[str]] = None,
    exclude_categories: Optional[Iterable[str]] = None,
    num_checks: int = 20,
    seed: int = 20260525,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    root_dir = Path(root_dir)
    original = RFLearningDataset(
        root_dir / split,
        sample_group=sample_group,
        normalize=normalize,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        verbose=False,
    )
    cached = RFCachedDataset(cache_dir)

    if len(original) != len(cached):
        raise AssertionError(f"length mismatch: original={len(original)}, cached={len(cached)}")

    rng = np.random.default_rng(seed)
    n_checks = min(num_checks, len(original))
    indices = rng.choice(len(original), size=n_checks, replace=False)

    max_diffs = {"input": 0.0, "label": 0.0, "baseline": 0.0, "scale": 0.0}
    for idx in indices:
        expected = original[int(idx)]
        actual = cached[int(idx)]

        for field in ("input", "label", "baseline"):
            diff = _assert_tensor_close(field, expected[field], actual[field], atol=atol, rtol=rtol)
            max_diffs[field] = max(max_diffs[field], diff)

        scale_diff = _assert_tensor_close(
            "scale",
            expected["scale"],
            actual["scale"],
            atol=atol,
            rtol=rtol,
        )
        max_diffs["scale"] = max(max_diffs["scale"], scale_diff)

        for field in ("path", "category"):
            if expected[field] != actual[field]:
                raise AssertionError(
                    f"{field} mismatch at index {idx}: "
                    f"expected={expected[field]!r}, actual={actual[field]!r}"
                )

        for field in ("z_idx", "x_idx", "y_idx"):
            if not torch.equal(expected[field], actual[field]):
                raise AssertionError(f"{field} mismatch at index {idx}")

    print("Cache self-test passed.")
    print(f"  checked indices: {sorted(int(i) for i in indices)}")
    print(
        "  max_abs_diff  : "
        f"input={max_diffs['input']:.6e}, "
        f"label={max_diffs['label']:.6e}, "
        f"baseline={max_diffs['baseline']:.6e}, "
        f"scale={max_diffs['scale']:.6e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RF memmap cache for one split.")
    parser.add_argument("--root-dir", default="Data", help="Dataset root containing split folders.")
    parser.add_argument("--split", required=True, help="Split name, e.g. train, val, or test.")
    parser.add_argument("--out-root", default="Data_cache", help="Cache root directory.")
    parser.add_argument("--sample-group", default="/sample_000001")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--include-categories", nargs="*", default=None)
    parser.add_argument("--exclude-categories", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--num-checks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = build_split_cache(
        root_dir=args.root_dir,
        split=args.split,
        out_root=args.out_root,
        sample_group=args.sample_group,
        normalize=args.normalize,
        include_categories=args.include_categories,
        exclude_categories=args.exclude_categories,
        overwrite=args.overwrite,
    )

    if not args.skip_verify:
        verify_split_cache(
            root_dir=args.root_dir,
            split=args.split,
            cache_dir=cache_dir,
            sample_group=args.sample_group,
            normalize=args.normalize,
            include_categories=args.include_categories,
            exclude_categories=args.exclude_categories,
            num_checks=args.num_checks,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
