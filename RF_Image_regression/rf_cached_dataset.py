from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RFCachedDataset(Dataset):
    """
    Dataset backed by preprocessed RF memmap cache.

    The returned sample dictionary matches RFLearningDataset.__getitem__:
        input, label, baseline, scale, path, category, z_idx, x_idx, y_idx
    """

    def __init__(self, cache_dir, return_fp16: bool = False):
        self.cache_dir = Path(cache_dir)
        self.return_fp16 = return_fp16
        meta_path = self.cache_dir / "meta.npz"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing cache metadata: {meta_path}")

        with np.load(meta_path, allow_pickle=False) as meta:
            self.input_shape = tuple(int(x) for x in meta["input_shape"])
            self.label_shape = tuple(int(x) for x in meta["label_shape"])
            self.baseline_shape = tuple(int(x) for x in meta["baseline_shape"])
            self.paths = meta["path"].astype(str).tolist()
            self.categories = meta["category"].astype(str).tolist()
            self.z_idx = meta["z_idx"].astype(np.int32, copy=False)
            self.x_idx = meta["x_idx"].astype(np.int32, copy=False)
            self.y_idx = meta["y_idx"].astype(np.int32, copy=False)

        self.n_samples = self.input_shape[0]
        self.files = [Path(path) for path in self.paths]
        if len(self.paths) != self.n_samples:
            raise ValueError(
                f"Metadata length mismatch: {len(self.paths)} paths for "
                f"{self.n_samples} cached samples"
            )

        self.input = np.memmap(
            self.cache_dir / "input.dat",
            dtype=np.float16,
            mode="r",
            shape=self.input_shape,
        )
        self.label = np.memmap(
            self.cache_dir / "label.dat",
            dtype=np.float16,
            mode="r",
            shape=self.label_shape,
        )
        self.baseline = np.memmap(
            self.cache_dir / "baseline.dat",
            dtype=np.float16,
            mode="r",
            shape=self.baseline_shape,
        )
        self.scale = np.memmap(
            self.cache_dir / "scale.dat",
            dtype=np.float32,
            mode="r",
            shape=(self.n_samples,),
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        out_dtype = np.float16 if self.return_fp16 else np.float32
        torch_dtype = torch.float16 if self.return_fp16 else torch.float32
        return {
            "input": torch.from_numpy(np.array(self.input[idx], dtype=out_dtype, copy=True)),
            "label": torch.from_numpy(np.array(self.label[idx], dtype=out_dtype, copy=True)),
            "baseline": torch.from_numpy(np.array(self.baseline[idx], dtype=out_dtype, copy=True)),
            "scale": torch.tensor(float(self.scale[idx]), dtype=torch.float32),
            "path": self.paths[idx],
            "category": self.categories[idx],
            "z_idx": torch.from_numpy(np.array(self.z_idx[idx], dtype=np.int32, copy=True)),
            "x_idx": torch.from_numpy(np.array(self.x_idx[idx], dtype=np.int32, copy=True)),
            "y_idx": torch.from_numpy(np.array(self.y_idx[idx], dtype=np.int32, copy=True)),
        }
