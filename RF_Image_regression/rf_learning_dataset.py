from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


KNOWN_CATEGORIES = ("carotid", "muscle", "phantom", "simu_point")


def infer_category_from_path(path) -> str:
    """Infer sample category from file path."""
    p = Path(path)
    parts = [part.lower() for part in p.parts]
    stem = p.stem.lower()

    for cat in KNOWN_CATEGORIES:
        if cat in parts or cat in stem:
            return cat

    return "unknown"


class RFLearningDataset(Dataset):
    """
    Dataset for delay-aligned RF learning samples.

    Expected HDF5 structure:
        /sample_000001/input/F_RC_real
        /sample_000001/input/F_RC_imag
        /sample_000001/input/F_CR_real
        /sample_000001/input/F_CR_imag
        /sample_000001/label/DAS_target_avg_real
        /sample_000001/label/DAS_target_avg_imag
        /sample_000001/baseline/DAS_input_avg_real
        /sample_000001/baseline/DAS_input_avg_imag

    MATLAB logical array order:
        RF tensor: [Nz, Nx, Ny, Nch, Nangle]
        DAS volume: [Nz, Nx, Ny]

    h5py observed order:
        RF tensor: [Nangle, Nch, Ny, Nx, Nz]
        DAS volume: [Ny, Nx, Nz]

    Returned tensors:
        input:    [1536, Nz, Nx, Ny]
        label:    [2, Nz, Nx, Ny]
        baseline: [2, Nz, Nx, Ny]
    """

    def __init__(
        self,
        root_dir,
        sample_group: str = "/sample_000001",
        normalize: bool = False,
        include_categories: Optional[Iterable[str]] = None,
        exclude_categories: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.sample_group = sample_group.rstrip("/")
        self.normalize = normalize

        self.include_categories = None if include_categories is None else set(include_categories)
        self.exclude_categories = set() if exclude_categories is None else set(exclude_categories)

        files = sorted(self.root_dir.rglob("*.h5"))
        files = [f for f in files if self._keep_file(f)]

        if max_samples is not None:
            files = files[:max_samples]

        if len(files) == 0:
            raise FileNotFoundError(
                f"No .h5 files found under {self.root_dir} with "
                f"include_categories={include_categories}, exclude_categories={exclude_categories}"
            )

        self.files = files
        self.categories = [infer_category_from_path(f) for f in self.files]

        if verbose:
            self.print_summary()

    def _keep_file(self, file_path: Path) -> bool:
        cat = infer_category_from_path(file_path)

        if self.include_categories is not None and cat not in self.include_categories:
            return False

        if cat in self.exclude_categories:
            return False

        return True

    def print_summary(self):
        print(f"RFLearningDataset")
        print(f"  root_dir   : {self.root_dir}")
        print(f"  samples    : {len(self.files)}")
        print(f"  normalize  : {self.normalize}")
        print(f"  include    : {self.include_categories}")
        print(f"  exclude    : {self.exclude_categories}")
        for cat in KNOWN_CATEGORIES + ("unknown",):
            n = sum(c == cat for c in self.categories)
            if n > 0:
                print(f"  {cat:10s}: {n}")

    def __len__(self):
        return len(self.files)

    def _read(self, h5, path):
        return np.asarray(h5[path], dtype=np.float32)

    def _read_rf_tensor(self, h5, path):
        arr = self._read(h5, path)

        if arr.ndim != 5:
            raise ValueError(f"Expected 5D RF tensor at {path}, got shape {arr.shape}")

        # h5py: [Nangle, Nch, Ny, Nx, Nz] -> [Nz, Nx, Ny, Nch, Nangle]
        arr = np.transpose(arr, (4, 3, 2, 1, 0))
        return arr

    def _read_volume(self, h5, path):
        arr = self._read(h5, path)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume at {path}, got shape {arr.shape}")

        # h5py: [Ny, Nx, Nz] -> [Nz, Nx, Ny]
        arr = np.transpose(arr, (2, 1, 0))
        return arr

    def __getitem__(self, idx):
        file_path = self.files[idx]
        category = infer_category_from_path(file_path)

        with h5py.File(file_path, "r") as h5:
            g = self.sample_group

            F_RC_real = self._read_rf_tensor(h5, f"{g}/input/F_RC_real")
            F_RC_imag = self._read_rf_tensor(h5, f"{g}/input/F_RC_imag")
            F_CR_real = self._read_rf_tensor(h5, f"{g}/input/F_CR_real")
            F_CR_imag = self._read_rf_tensor(h5, f"{g}/input/F_CR_imag")

            # [Nz, Nx, Ny, Nch, Nangle, 4]
            X = np.stack([F_RC_real, F_RC_imag, F_CR_real, F_CR_imag], axis=-1)

            # [Nch, Nangle, 4, Nz, Nx, Ny]
            X = np.transpose(X, (3, 4, 5, 0, 1, 2))

            # [Nch * Nangle * 4, Nz, Nx, Ny]
            Nch, Nang, Ncomp, Nz, Nx, Ny = X.shape
            X = X.reshape(Nch * Nang * Ncomp, Nz, Nx, Ny)

            Y_real = self._read_volume(h5, f"{g}/label/DAS_target_avg_real")
            Y_imag = self._read_volume(h5, f"{g}/label/DAS_target_avg_imag")
            Y = np.stack([Y_real, Y_imag], axis=0)

            B_real = self._read_volume(h5, f"{g}/baseline/DAS_input_avg_real")
            B_imag = self._read_volume(h5, f"{g}/baseline/DAS_input_avg_imag")
            B = np.stack([B_real, B_imag], axis=0)

            z_idx = np.asarray(h5[f"{g}/meta/z_idx"], dtype=np.int32).reshape(-1)
            x_idx = np.asarray(h5[f"{g}/meta/x_idx"], dtype=np.int32).reshape(-1)
            y_idx = np.asarray(h5[f"{g}/meta/y_idx"], dtype=np.int32).reshape(-1)

        X = torch.from_numpy(np.ascontiguousarray(X)).float()
        Y = torch.from_numpy(np.ascontiguousarray(Y)).float()
        B = torch.from_numpy(np.ascontiguousarray(B)).float()

        if self.normalize:
            scale = torch.amax(torch.abs(Y)) + 1e-8
            X = X / scale
            Y = Y / scale
            B = B / scale
        else:
            scale = torch.tensor(1.0, dtype=torch.float32)

        return {
            "input": X,
            "label": Y,
            "baseline": B,
            "scale": scale,
            "path": str(file_path),
            "category": category,
            "z_idx": torch.from_numpy(z_idx),
            "x_idx": torch.from_numpy(x_idx),
            "y_idx": torch.from_numpy(y_idx),
        }
