"""
Probe generation utilities for RF_Image_GAN training.
Drop this into the project and import from training notebook.

Usage in train.py / train.ipynb:

    from probe_utils import generate_epoch_probes

    # After each epoch's training + validation, call:
    generate_epoch_probes(
        epoch=epoch,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_indices=TRAIN_FIXED_INDICES,   # {cat: idx}
        val_indices=VAL_FIXED_INDICES,       # {cat: idx}
        categories=["carotid", "muscle", "phantom"],
        y_idx=16,
        aspect=0.181,  # z_spacing / x_spacing
        train_out_dir=Path("02_train/probes/train"),
        val_out_dir=Path("02_train/probes/val"),
    )
"""
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_epoch_probes(
    epoch: int,
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    train_indices: dict[str, int],
    val_indices: dict[str, int],
    categories: list[str],
    y_idx: int = 16,
    aspect: float = 0.181,
    train_out_dir: Path = Path("02_train/probes/train"),
    val_out_dir: Path = Path("02_train/probes/val"),
    dpi: int = 160,
) -> None:
    """
    Generate one combined probe image per split for a given epoch.

    Each output is a 3×3 grid: rows = categories, cols = (baseline | pred | gt).
    Saved as {train|val}_epoch{NNN}_y{YY}.png in separate folders.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    train_out_dir.mkdir(parents=True, exist_ok=True)
    val_out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for ds, indices, out_dir, tag in [
            (train_dataset, train_indices, train_out_dir, "train"),
            (val_dataset, val_indices, val_out_dir, "val"),
        ]:
            fig, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)

            for row, cat in enumerate(categories):
                sample = ds[indices[cat]]
                x = sample["input"].unsqueeze(0).to(device)
                pred = model(x)[0, 0].cpu().numpy()
                gt = sample["gt_bmode"][0].numpy()

                # baseline may be precomputed or computed on-the-fly
                if "baseline_bmode" in sample:
                    baseline = sample["baseline_bmode"].numpy()
                else:
                    baseline = np.zeros_like(gt)  # fallback, shouldn't happen

                for col, (title, img) in enumerate([
                    ("baseline", baseline[:, :, y_idx]),
                    ("pred", pred[:, :, y_idx]),
                    ("gt", gt[:, :, y_idx]),
                ]):
                    ax = axes[row, col]
                    im = ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0,
                                   aspect=aspect, origin="lower")
                    ax.set_title(f"{cat} {title}", fontsize=8)
                    ax.set_axis_off()

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.80)
            fig.suptitle(f"{tag}  epoch {epoch:03d}  y={y_idx}",
                         fontsize=10, fontweight="bold")
            fname = out_dir / f"{tag}_epoch{epoch:03d}_y{y_idx}.png"
            fig.savefig(fname, dpi=dpi)
            plt.close(fig)

    if was_training:
        model.train()


def pick_first_per_category(dataset, categories: list[str]) -> dict[str, int]:
    """Return {category: first_sample_index} for each category in the dataset."""
    grouped: dict[str, list[int]] = {}
    for idx, cat in enumerate(dataset.categories):
        grouped.setdefault(str(cat), []).append(idx)
    return {cat: grouped[cat][0] for cat in categories}
