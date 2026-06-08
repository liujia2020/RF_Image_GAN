from pathlib import Path
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from rf_learning_dataset import RFLearningDataset


class TinyResidualRFNet(nn.Module):
    """
    Same model as train_pilot_tiny_residual.py
    """

    def __init__(self, in_channels=1536, hidden=64, out_channels=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, padding=0),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(hidden, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(32, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x, baseline):
        residual = self.net(x)
        pred = baseline + residual
        return pred


def load_best_model(ckpt_path, device):
    model = TinyResidualRFNet(
        in_channels=1536,
        hidden=64,
        out_channels=2,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print(f"Loaded checkpoint dict. Epoch = {ckpt.get('epoch', 'unknown')}")
        print(f"Best val L1 = {ckpt.get('best_val_l1', 'unknown')}")
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    return model


def complex_abs_2ch(x):
    """
    x: [2, Z, X, Y] or [B, 2, Z, X, Y]
    """
    if x.ndim == 4:
        return torch.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-8)
    elif x.ndim == 5:
        return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-8)
    else:
        raise ValueError(f"Unsupported shape: {x.shape}")


def to_db(img, ref=None, dynamic_range=60):
    """
    img: numpy magnitude image
    """
    img = np.asarray(img)

    if ref is None:
        ref = np.max(img)

    if ref <= 0:
        ref = 1.0

    db = 20 * np.log10(img / ref + 1e-12)
    db = np.clip(db, -dynamic_range, 0)

    return db


def get_category_from_path(path):
    path = Path(path)
    parts = [p.lower() for p in path.parts]

    for name in ["carotid", "muscle", "phantom", "simu_point"]:
        if name in parts:
            return name

    return "unknown"


@torch.no_grad()
def predict_one_sample(model, sample, device):
    """
    Dataset uses normalize=True.
    So input, label, baseline are normalized by sample scale.
    We multiply them back before visualization.
    """

    x = sample["input"].unsqueeze(0).to(device)       # [1,1536,Z,X,Y]
    y = sample["label"].unsqueeze(0).to(device)       # [1,2,Z,X,Y]
    b = sample["baseline"].unsqueeze(0).to(device)    # [1,2,Z,X,Y]
    scale = sample["scale"].to(device)

    if scale.ndim == 0:
        scale = scale.view(1, 1, 1, 1, 1)
    else:
        scale = scale.view(-1, 1, 1, 1, 1)

    pred = model(x, b)

    # Restore original magnitude scale
    pred = pred * scale
    y = y * scale
    b = b * scale

    pred = pred.squeeze(0).cpu()
    y = y.squeeze(0).cpu()
    b = b.squeeze(0).cpu()

    return b, pred, y


def plot_sample_result(sample, baseline, pred, label, save_path):
    """
    baseline, pred, label: [2, Z, X, Y]
    """

    baseline_abs = complex_abs_2ch(baseline).numpy()
    pred_abs = complex_abs_2ch(pred).numpy()
    label_abs = complex_abs_2ch(label).numpy()

    err_pred = np.abs(pred_abs - label_abs)
    err_base = np.abs(baseline_abs - label_abs)

    Z, X, Y = label_abs.shape

    z_mid = Z // 2
    y_mid = Y // 2
    x_mid = X // 2

    # Main view: XZ slice at middle Y
    base_xz = baseline_abs[:, :, y_mid]
    pred_xz = pred_abs[:, :, y_mid]
    label_xz = label_abs[:, :, y_mid]
    err_pred_xz = err_pred[:, :, y_mid]
    err_base_xz = err_base[:, :, y_mid]

    ref = max(np.max(base_xz), np.max(pred_xz), np.max(label_xz), 1.0)

    base_db = to_db(base_xz, ref=ref)
    pred_db = to_db(pred_xz, ref=ref)
    label_db = to_db(label_xz, ref=ref)
    err_pred_db = to_db(err_pred_xz, ref=ref)
    err_base_db = to_db(err_base_xz, ref=ref)

    path = sample["path"]
    category = get_category_from_path(path)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    imgs = [
        (base_db, "3-angle baseline"),
        (pred_db, "network prediction"),
        (label_db, "33-angle label"),
        (err_pred_db, "|pred - label|"),
        (err_base_db, "|baseline - label|"),
    ]

    for ax, (img, title) in zip(axes, imgs):
        im = ax.imshow(img, aspect="auto", vmin=-60, vmax=0)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{category} | {Path(path).name}\n"
        f"z_idx={sample['z_idx'][0].item()}~{sample['z_idx'][-1].item()}, "
        f"x_idx={sample['x_idx'][0].item()}~{sample['x_idx'][-1].item()}, "
        f"y_idx={sample['y_idx'][0].item()}~{sample['y_idx'][-1].item()}",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    # Print simple metrics
    l1_pred = torch.mean(torch.abs(pred - label)).item()
    l1_base = torch.mean(torch.abs(baseline - label)).item()

    abs_l1_pred = np.mean(np.abs(pred_abs - label_abs))
    abs_l1_base = np.mean(np.abs(baseline_abs - label_abs))

    print(f"Saved: {save_path}")
    print(f"  complex L1 pred/base: {l1_pred:.4e} / {l1_base:.4e}")
    print(f"  abs     L1 pred/base: {abs_l1_pred:.4e} / {abs_l1_base:.4e}")


def select_samples_by_category(dataset, n_per_category=2):
    """
    Pick a few samples from each category for visualization.
    """

    category_to_indices = {}

    for i, path in enumerate(dataset.files):
        cat = get_category_from_path(path)
        category_to_indices.setdefault(cat, []).append(i)

    selected = []

    for cat, indices in category_to_indices.items():
        random.shuffle(indices)
        selected.extend(indices[:n_per_category])

    return selected


def main():
    root_dir = r"/home/liujia/3DSSIM_1/RF_Image/Data/val"
    ckpt_path = r"checkpoints_pilot_tiny/best_tiny_residual_rfnet.pth"
    save_dir = Path("vis_best_tiny_residual")
    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(20260522)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = RFLearningDataset(
        root_dir=root_dir,
        sample_group="/sample_000001",
        normalize=True,
    )

    print("Number of samples:", len(dataset))

    model = load_best_model(ckpt_path, device)

    selected_indices = select_samples_by_category(dataset, n_per_category=2)

    print("Selected samples:", selected_indices)

    for count, idx in enumerate(selected_indices, start=1):
        sample = dataset[idx]

        baseline, pred, label = predict_one_sample(model, sample, device)

        category = get_category_from_path(sample["path"])
        save_name = f"{count:03d}_{category}_{Path(sample['path']).stem}.png"
        save_path = save_dir / save_name

        plot_sample_result(sample, baseline, pred, label, save_path)


if __name__ == "__main__":
    main()