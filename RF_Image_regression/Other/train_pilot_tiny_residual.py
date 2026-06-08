import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from rf_learning_dataset import RFLearningDataset


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int = 20260522):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
# ============================================================

class TinyResidualRFNet(nn.Module):
    """
    Pilot network.

    Input:
        x        [B,1536,16,8,8]
        baseline [B,2,16,8,8]

    Output:
        pred     [B,2,16,8,8]

    The network predicts a residual correction:
        pred = baseline + residual
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

        # Start close to baseline.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, baseline):
        residual = self.net(x)
        pred = baseline + residual
        return pred


# ============================================================
# Loss helpers
# ============================================================

def complex_abs_2ch(x):
    """
    x: [B,2,Z,X,Y], real/imag
    return: [B,1,Z,X,Y]
    """
    return torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-8)


def compute_loss(pred, label):
    """
    Main loss:
        complex real/imag L1
        + small envelope L1
    """
    loss_l1 = F.l1_loss(pred, label)

    pred_abs = complex_abs_2ch(pred)
    label_abs = complex_abs_2ch(label)
    loss_abs = F.l1_loss(pred_abs, label_abs)

    loss = loss_l1 + 0.1 * loss_abs

    return loss, loss_l1, loss_abs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_abs = 0.0
    total_baseline_l1 = 0.0
    total_baseline_abs = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        b = batch["baseline"].to(device, non_blocking=True)

        pred = model(x, b)

        loss, loss_l1, loss_abs = compute_loss(pred, y)

        baseline_l1 = F.l1_loss(b, y)
        baseline_abs = F.l1_loss(complex_abs_2ch(b), complex_abs_2ch(y))

        total_loss += loss.item()
        total_l1 += loss_l1.item()
        total_abs += loss_abs.item()
        total_baseline_l1 += baseline_l1.item()
        total_baseline_abs += baseline_abs.item()
        n_batches += 1

    total_loss /= max(n_batches, 1)
    total_l1 /= max(n_batches, 1)
    total_abs /= max(n_batches, 1)
    total_baseline_l1 /= max(n_batches, 1)
    total_baseline_abs /= max(n_batches, 1)

    improvement = 1.0 - total_l1 / (total_baseline_l1 + 1e-12)

    return {
        "loss": total_loss,
        "l1": total_l1,
        "abs": total_abs,
        "baseline_l1": total_baseline_l1,
        "baseline_abs": total_baseline_abs,
        "improvement": improvement,
    }


# ============================================================
# Training
# ============================================================

def main():
    seed_everything(20260522)

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------
    root_dir = r"/home/liujia/3DSSIM_1/RF_Image/Data"

    save_dir = Path("./checkpoints_pilot_tiny")
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 4
    num_epochs = 100
    lr = 1e-3
    weight_decay = 1e-5
    val_ratio = 0.2

    normalize = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Root dir:", root_dir)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    dataset = RFLearningDataset(
        root_dir=root_dir,
        sample_group="/sample_000001",
        normalize=normalize,
    )

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(20260522),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Total samples: {n_total}")
    print(f"Train samples: {n_train}")
    print(f"Val samples  : {n_val}")

    # ------------------------------------------------------------
    # Shape check
    # ------------------------------------------------------------
    one_batch = next(iter(train_loader))
    print("\nOne batch shape:")
    print("input   :", one_batch["input"].shape)
    print("label   :", one_batch["label"].shape)
    print("baseline:", one_batch["baseline"].shape)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = TinyResidualRFNet(
        in_channels=1536,
        hidden=64,
        out_channels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5,
    )

    best_val_l1 = float("inf")
    best_epoch = -1

    # Initial validation
    val_metrics = evaluate(model, val_loader, device)
    print("\nInitial validation:")
    print(
        f"val_L1={val_metrics['l1']:.6e} | "
        f"baseline_L1={val_metrics['baseline_l1']:.6e} | "
        f"improvement={val_metrics['improvement']*100:.2f}%"
    )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()

        train_loss = 0.0
        train_l1 = 0.0
        train_abs = 0.0
        train_base = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            b = batch["baseline"].to(device, non_blocking=True)

            pred = model(x, b)

            loss, loss_l1, loss_abs = compute_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                baseline_l1 = F.l1_loss(b, y)

            train_loss += loss.item()
            train_l1 += loss_l1.item()
            train_abs += loss_abs.item()
            train_base += baseline_l1.item()
            n_batches += 1

        scheduler.step()

        train_loss /= max(n_batches, 1)
        train_l1 /= max(n_batches, 1)
        train_abs /= max(n_batches, 1)
        train_base /= max(n_batches, 1)

        train_improvement = 1.0 - train_l1 / (train_base + 1e-12)

        val_metrics = evaluate(model, val_loader, device)

        # Save best model by val L1
        if val_metrics["l1"] < best_val_l1:
            best_val_l1 = val_metrics["l1"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_l1": best_val_l1,
                    "config": {
                        "root_dir": root_dir,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "normalize": normalize,
                    },
                },
                save_dir / "best_tiny_residual_rfnet.pth",
            )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train_L1={train_l1:.6e} | "
                f"train_base={train_base:.6e} | "
                f"train_impr={train_improvement*100:6.2f}% | "
                f"val_L1={val_metrics['l1']:.6e} | "
                f"val_base={val_metrics['baseline_l1']:.6e} | "
                f"val_impr={val_metrics['improvement']*100:6.2f}% | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    # Save final
    torch.save(model.state_dict(), save_dir / "final_tiny_residual_rfnet.pth")

    print("\nTraining finished.")
    print(f"Best val L1: {best_val_l1:.6e} at epoch {best_epoch}")
    print(f"Best model saved to: {save_dir / 'best_tiny_residual_rfnet.pth'}")
    print(f"Final model saved to: {save_dir / 'final_tiny_residual_rfnet.pth'}")


if __name__ == "__main__":
    main()