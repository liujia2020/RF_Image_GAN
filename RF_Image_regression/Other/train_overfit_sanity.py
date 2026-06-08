import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from rf_learning_dataset import RFLearningDataset


class TinyResidualRFNet(nn.Module):
    """
    Minimal sanity-check network.

    Input:
        x        [B,1536,16,8,8]
        baseline [B,2,16,8,8]

    Output:
        pred     [B,2,16,8,8]

    The network predicts a residual correction added to baseline.
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

        # Start close to baseline: final residual initially near zero.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, baseline):
        residual = self.net(x)
        pred = baseline + residual
        return pred


def complex_abs_2ch(x):
    """
    x: [B,2,Z,X,Y], real/imag
    return: [B,1,Z,X,Y]
    """
    return torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-8)


def main():
    root_dir = r"/home/liujia/3DSSIM_1/RF_Image/Data"

    dataset = RFLearningDataset(
        root_dir=root_dir,
        sample_group="/sample_000001",
        normalize=True,
    )

    # Only overfit first 4 samples.
    subset = Subset(dataset, list(range(4)))

    loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Overfit samples:", len(subset))

    model = TinyResidualRFNet(
        in_channels=1536,
        hidden=64,
        out_channels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    num_epochs = 300

    for epoch in range(1, num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_abs = 0.0
        epoch_base = 0.0
        n_batches = 0

        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            b = batch["baseline"].to(device, non_blocking=True)

            pred = model(x, b)

            # Complex real/imag L1 loss
            loss_l1 = F.l1_loss(pred, y)

            # Envelope magnitude loss, small weight
            pred_abs = complex_abs_2ch(pred)
            y_abs = complex_abs_2ch(y)
            loss_abs = F.l1_loss(pred_abs, y_abs)

            loss = loss_l1 + 0.1 * loss_abs

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                baseline_l1 = F.l1_loss(b, y)

            epoch_loss += loss.item()
            epoch_l1 += loss_l1.item()
            epoch_abs += loss_abs.item()
            epoch_base += baseline_l1.item()
            n_batches += 1

        epoch_loss /= n_batches
        epoch_l1 /= n_batches
        epoch_abs /= n_batches
        epoch_base /= n_batches

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"loss={epoch_loss:.6e} | "
                f"L1={epoch_l1:.6e} | "
                f"abs={epoch_abs:.6e} | "
                f"baseline_L1={epoch_base:.6e}"
            )

    torch.save(model.state_dict(), "tiny_residual_rfnet_overfit.pth")
    print("Saved model: tiny_residual_rfnet_overfit.pth")


if __name__ == "__main__":
    main()