import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random
from rf_learning_dataset import RFLearningDataset
import numpy as np
import pandas as pd
from pathlib import Path
class TinyResidualRFNet(nn.Module):

def to_numpy_volume(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def complex_abs_numpy(x_2ch):
    """
    x_2ch: [2,Z,X,Y]
    """
    return np.sqrt(x_2ch[0] ** 2 + x_2ch[1] ** 2 + 1e-12)


def volume_to_db(img, ref=None, db_min=-60):
    if ref is None:
        ref = np.max(img)
    if ref <= 0:
        ref = 1.0

    out = 20 * np.log10(img / ref + 1e-12)
    out[out < db_min] = db_min
    out[out > 0] = 0
    return out


@torch.no_grad()
def visualize_best_model_samples(
    model,
    dataset,
    device,
    save_dir,
    indices=None,
    n_per_category=3,
    y_slice=None,
    db_min=-60,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # If indices are not provided, choose n_per_category samples per category.
    if indices is None:
        cat_to_indices = {}

        for i, path in enumerate(dataset.files):
            cat = infer_category_from_path(str(path))
            cat_to_indices.setdefault(cat, []).append(i)

        indices = []
        for cat, idxs in cat_to_indices.items():
            indices.extend(idxs[:n_per_category])

    print("Selected indices:", indices)

    saved_paths = []

    for count, idx in enumerate(indices, start=1):
        sample = dataset[idx]

        x = sample["input"].unsqueeze(0).to(device)
        y = sample["label"].unsqueeze(0).to(device)
        b = sample["baseline"].unsqueeze(0).to(device)

        scale = sample["scale"]
        if torch.is_tensor(scale):
            scale = scale.to(device).view(1, 1, 1, 1, 1)
        else:
            scale = torch.tensor(scale, device=device).view(1, 1, 1, 1, 1)

        pred = model(x, b)

        pred = (pred * scale)[0].detach().cpu()
        label = (y * scale)[0].detach().cpu()
        base = (b * scale)[0].detach().cpu()

        pred_np = pred.numpy()
        label_np = label.numpy()
        base_np = base.numpy()

        pred_abs = complex_abs_numpy(pred_np)
        label_abs = complex_abs_numpy(label_np)
        base_abs = complex_abs_numpy(base_np)

        err_pred = np.abs(pred_abs - label_abs)
        err_base = np.abs(base_abs - label_abs)

        Z, X, Y = label_abs.shape

        if y_slice is None:
            iy = Y // 2
        else:
            iy = y_slice

        ref = np.max([base_abs.max(), pred_abs.max(), label_abs.max()])

        imgs = [
            volume_to_db(base_abs[:, :, iy], ref=ref, db_min=db_min),
            volume_to_db(pred_abs[:, :, iy], ref=ref, db_min=db_min),
            volume_to_db(label_abs[:, :, iy], ref=ref, db_min=db_min),
            volume_to_db(err_pred[:, :, iy], ref=ref, db_min=db_min),
            volume_to_db(err_base[:, :, iy], ref=ref, db_min=db_min),
        ]

        titles = [
            "3-angle baseline",
            "network prediction",
            "33-angle label",
            "|pred-label|",
            "|baseline-label|",
        ]

        path = Path(sample["path"])
        cat = infer_category_from_path(str(path))
        name = path.stem

        pred_l1 = np.mean(np.abs(pred_np - label_np))
        base_l1 = np.mean(np.abs(base_np - label_np))

        pred_abs_l1 = np.mean(np.abs(pred_abs - label_abs))
        base_abs_l1 = np.mean(np.abs(base_abs - label_abs))

        fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

        for ax, img, title in zip(axes, imgs, titles):
            im = ax.imshow(img, cmap="gray", vmin=db_min, vmax=0, aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("x index")
            ax.set_ylabel("z index")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"{cat} | {name} | y-slice={iy}\n"
            f"complex L1 pred/base: {pred_l1:.3e} / {base_l1:.3e} | "
            f"abs L1 pred/base: {pred_abs_l1:.3e} / {base_abs_l1:.3e}",
            fontsize=10,
        )

        fig.tight_layout()

        save_path = save_dir / f"{count:03d}_{cat}_{name}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        saved_paths.append(save_path)

        print(f"Saved: {save_path}")
        print(f"  complex L1 pred/base: {pred_l1:.4e} / {base_l1:.4e}")
        print(f"  abs     L1 pred/base: {pred_abs_l1:.4e} / {base_abs_l1:.4e}")

    return saved_paths