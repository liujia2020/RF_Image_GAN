from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path("/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607")
TARGET_ROOT = Path("/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_bmode_gt_ref64407p58_20260610")
REF = np.float32(64407.58)
EPS = np.float32(1e-12)
SPLITS = ("train", "val", "test")
CATEGORIES = ("carotid", "muscle", "phantom")


def load_meta(split: str) -> dict[str, np.ndarray]:
    with np.load(SOURCE_ROOT / split / "meta.npz", allow_pickle=False) as meta:
        return {key: meta[key] for key in meta.files}


def open_label_and_scale(split: str) -> tuple[np.memmap, np.memmap, dict[str, np.ndarray]]:
    meta = load_meta(split)
    shape = tuple(int(x) for x in meta["label_shape"])
    label = np.memmap(SOURCE_ROOT / split / "label.dat", dtype=np.float16, mode="r", shape=shape)
    scale = np.memmap(SOURCE_ROOT / split / "scale.dat", dtype=np.float32, mode="r", shape=(shape[0],))
    return label, scale, meta


def restored_envelope(label: np.memmap, scale: np.memmap, idx: int) -> np.ndarray:
    s = np.float32(scale[idx])
    real = np.asarray(label[idx, 0], dtype=np.float32) * s
    imag = np.asarray(label[idx, 1], dtype=np.float32) * s
    return np.sqrt(real * real + imag * imag, dtype=np.float32)


def env_to_bmode(env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    db = 20.0 * np.log10(env / REF + EPS)
    db = np.clip(db, -60.0, 0.0).astype(np.float32, copy=False)
    bmode = ((db + 60.0) / 60.0).astype(np.float32, copy=False)
    return db, bmode


def select_self_check_samples() -> list[dict[str, object]]:
    selected = []
    phantom_candidates = []
    for split in SPLITS:
        label, scale, meta = open_label_and_scale(split)
        categories = meta["category"].astype(str)
        paths = meta["path"].astype(str)
        for idx, category in enumerate(categories):
            if category not in CATEGORIES:
                continue
            env = restored_envelope(label, scale, idx)
            item = {
                "split": split,
                "idx": int(idx),
                "category": category,
                "path": str(paths[idx]),
                "env_max": float(env.max()),
                "env_mean": float(env.mean()),
                "env_median": float(np.percentile(env, 50)),
            }
            if category == "phantom":
                phantom_candidates.append(item)
            selected.append(item)

    by_cat = defaultdict(list)
    for item in selected:
        by_cat[item["category"]].append(item)
    return [
        max(by_cat["carotid"], key=lambda x: float(x["env_max"])),
        max(by_cat["muscle"], key=lambda x: float(x["env_max"])),
        min(phantom_candidates, key=lambda x: float(x["env_mean"])),
    ]


def manual_pixel_check(env: np.ndarray, bmode: np.ndarray) -> dict[str, float | list[int]]:
    flat_idx = int(np.argmax(env))
    zxy = [int(x) for x in np.unravel_index(flat_idx, env.shape)]
    value = float(env[tuple(zxy)])
    env_norm = value / float(REF)
    db_raw = 20.0 * np.log10(env_norm + float(EPS))
    db_clipped = float(np.clip(db_raw, -60.0, 0.0))
    mapped = (db_clipped + 60.0) / 60.0
    return {
        "zxy": zxy,
        "env": value,
        "env_norm": float(env_norm),
        "db_raw": float(db_raw),
        "db_clipped": db_clipped,
        "bmode_manual": float(mapped),
        "bmode_array": float(bmode[tuple(zxy)]),
        "abs_error": float(abs(mapped - float(bmode[tuple(zxy)]))),
    }


def run_self_check(out_dir: Path) -> list[dict[str, object]]:
    rows = []
    for sample in select_self_check_samples():
        label, scale, _ = open_label_and_scale(str(sample["split"]))
        env = restored_envelope(label, scale, int(sample["idx"]))
        db, bmode = env_to_bmode(env)
        above_ref = env > REF
        row = {
            **sample,
            "ref": float(REF),
            "bmode_min": float(bmode.min()),
            "bmode_max": float(bmode.max()),
            "bmode_mean": float(bmode.mean()),
            "bmode_zero_fraction": float(np.mean(bmode == 0.0)),
            "bmode_one_fraction": float(np.mean(bmode == 1.0)),
            "nan_count": int(np.isnan(bmode).sum()),
            "inf_count": int(np.isinf(bmode).sum()),
            "pixels_above_ref": int(np.count_nonzero(above_ref)),
            "pixels_above_ref_fraction": float(np.mean(above_ref)),
            "above_ref_all_clipped_to_one": bool(np.all(bmode[above_ref] == 1.0)) if np.any(above_ref) else True,
            "manual_pixel_check": manual_pixel_check(env, bmode),
            "db_min": float(db.min()),
            "db_max": float(db.max()),
        }
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "small_sample_self_check.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# B-mode GT cache",
        "",
        "生成日期：2026-06-10",
        "",
        "## 来源",
        "",
        f"- source cache: `{SOURCE_ROOT}`",
        "- source field: complex `label` after restoring scale",
        "- original complex label is not overwritten",
        "",
        "## 固定转换",
        "",
        f"- REF: `{float(REF)}`",
        "- env = sqrt(real^2 + imag^2) after fp32 scale restore",
        "- env_norm = env / REF",
        "- db = 20*log10(env_norm + 1e-12)",
        "- db = clip(db, -60, 0)",
        "- bmode = (db + 60) / 60",
        "",
        "## 输出",
        "",
        "- per split: `label_bmode.dat`, dtype fp16",
        "- shape: `[N, 1, 64, 32, 32]`",
        "- expected range: `[0, 1]`",
        "",
        "## 生成摘要",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
    ]
    (TARGET_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def convert_all(overwrite: bool) -> dict[str, object]:
    if TARGET_ROOT.exists():
        if not overwrite:
            raise FileExistsError(f"Target exists: {TARGET_ROOT}")
        shutil.rmtree(TARGET_ROOT)
    TARGET_ROOT.mkdir(parents=True)

    start = time.time()
    all_values = []
    global_nan = 0
    global_inf = 0
    all_zero_samples = []
    all_one_samples = []
    by_category_sum = Counter()
    by_category_count = Counter()
    split_summaries = {}

    for split in SPLITS:
        src_label, src_scale, meta = open_label_and_scale(split)
        categories = meta["category"].astype(str)
        paths = meta["path"].astype(str)
        shape = tuple(int(x) for x in meta["label_shape"])
        out_shape = (shape[0], 1, shape[2], shape[3], shape[4])

        split_dir = TARGET_ROOT / split
        split_dir.mkdir(parents=True)
        out = np.memmap(split_dir / "label_bmode.dat", dtype=np.float16, mode="w+", shape=out_shape)

        split_values = []
        for idx in range(shape[0]):
            env = restored_envelope(src_label, src_scale, idx)
            _, bmode = env_to_bmode(env)
            out[idx, 0] = bmode.astype(np.float16)
            bmode_fp32 = np.asarray(out[idx, 0], dtype=np.float32)

            nan_count = int(np.isnan(bmode_fp32).sum())
            inf_count = int(np.isinf(bmode_fp32).sum())
            global_nan += nan_count
            global_inf += inf_count
            if bool(np.all(bmode_fp32 == 0.0)):
                all_zero_samples.append({"split": split, "idx": int(idx), "category": str(categories[idx]), "path": str(paths[idx])})
            if bool(np.all(bmode_fp32 == 1.0)):
                all_one_samples.append({"split": split, "idx": int(idx), "category": str(categories[idx]), "path": str(paths[idx])})

            cat = str(categories[idx])
            by_category_sum[cat] += float(bmode_fp32.sum())
            by_category_count[cat] += int(bmode_fp32.size)
            split_values.append(bmode_fp32.ravel().copy())

        out.flush()
        split_arr = np.concatenate(split_values)
        all_values.append(split_arr)

        split_meta_path = split_dir / "meta.npz"
        np.savez(
            split_meta_path,
            label_bmode_shape=np.asarray(out_shape, dtype=np.int64),
            label_bmode_dtype=np.asarray("float16"),
            source_cache_dir=np.asarray(str(SOURCE_ROOT)),
            source_split=np.asarray(split),
            source_label_shape=np.asarray(shape, dtype=np.int64),
            source_label_dtype=np.asarray("float16"),
            reference_value=np.asarray(float(REF), dtype=np.float32),
            eps=np.asarray(float(EPS), dtype=np.float32),
            db_clip_min=np.asarray(-60.0, dtype=np.float32),
            db_clip_max=np.asarray(0.0, dtype=np.float32),
            transform=np.asarray("restore_scale_fp32 -> abs_complex -> divide_global_REF -> 20log10(+1e-12) -> clip[-60,0] -> map[0,1]"),
            path=meta["path"],
            category=meta["category"],
            z_idx=meta["z_idx"],
            x_idx=meta["x_idx"],
            y_idx=meta["y_idx"],
        )

        split_summaries[split] = {
            "samples": int(out_shape[0]),
            "shape": list(out_shape),
            "dtype": "float16",
            "min": float(split_arr.min()),
            "max": float(split_arr.max()),
            "mean": float(split_arr.mean()),
        }

    all_arr = np.concatenate(all_values)
    summary = {
        "target_root": str(TARGET_ROOT),
        "source_root": str(SOURCE_ROOT),
        "reference_value": float(REF),
        "eps": float(EPS),
        "samples": int(sum(s["samples"] for s in split_summaries.values())),
        "shape_by_split": {split: split_summaries[split]["shape"] for split in SPLITS},
        "dtype": "float16",
        "global_min": float(all_arr.min()),
        "global_max": float(all_arr.max()),
        "global_mean": float(all_arr.mean()),
        "nan_count": int(global_nan),
        "inf_count": int(global_inf),
        "all_zero_samples": all_zero_samples,
        "all_one_samples": all_one_samples,
        "category_mean": {
            cat: float(by_category_sum[cat] / by_category_count[cat])
            for cat in sorted(by_category_count)
        },
        "split_summaries": split_summaries,
        "elapsed_sec": float(time.time() - start),
    }
    (TARGET_ROOT / "conversion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path("experiments/cgan_v1/bmode_gt_conversion_2026-06-10/results")
    if args.self_check:
        rows = run_self_check(out_dir)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    if args.convert:
        summary = convert_all(overwrite=args.overwrite)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.self_check and not args.convert:
        parser.error("Use --self-check and/or --convert")


if __name__ == "__main__":
    main()
