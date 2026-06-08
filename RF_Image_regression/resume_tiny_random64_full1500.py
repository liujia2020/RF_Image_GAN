from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rf_cached_dataset import RFCachedDataset
from rf_models import build_model
from rf_train_utils import (
    complex_abs_2ch,
    compute_loss,
    count_trainable_parameters,
    evaluate_normalized,
    get_device,
    seed_everything,
)


EXP = {
    "experiment_name": "tiny_random64_full1500",
    "project_root": "/home/liujia/RF_Image",
    "include_categories": ["carotid", "muscle", "phantom"],
    "batch_size": 4,
    "num_epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "normalize": True,
    "abs_weight": 0.1,
    "model_name": "tiny",
    "hidden": 64,
    "seed": 20260522,
    "grad_clip": 1.0,
    "eta_min": 1e-5,
    "patience": 15,
}


def cosine_lr(epoch: int, base_lr: float, eta_min: float, t_max: int) -> float:
    return eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * epoch / t_max)) / 2.0


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main() -> None:
    seed_everything(EXP["seed"])

    project_root = Path(EXP["project_root"])
    data_cache_root = project_root / "Data_cache_random64_full1500"
    ckpt_dir = project_root / "checkpoint" / EXP["experiment_name"]
    ckpt_path = ckpt_dir / "best_model.pth"
    history_path = ckpt_dir / "training_history.csv"

    device = get_device()
    train_set = RFCachedDataset(data_cache_root / "train")
    val_set = RFCachedDataset(data_cache_root / "val")

    loader_kwargs = {
        "batch_size": EXP["batch_size"],
        "num_workers": 0,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = build_model(
        EXP["model_name"],
        in_channels=1536,
        hidden=EXP["hidden"],
        out_channels=2,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXP["lr"],
        weight_decay=EXP["weight_decay"],
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_state_to_device(optimizer, device)

    start_epoch = int(checkpoint["epoch"]) + 1
    best_epoch = int(checkpoint["epoch"])
    best_val_l1 = float(checkpoint["best_val_l1"])
    history = list(checkpoint.get("history", []))
    no_improve_epochs = 0
    model_class = type(model).__name__

    if history_path.exists():
        current = pd.read_csv(history_path)
        if len(current) > len(history):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = ckpt_dir / f"training_history_interrupted_backup_{stamp}.csv"
            shutil.copy2(history_path, backup_path)
            print(f"Backed up interrupted history: {backup_path}", flush=True)

    run_config = dict(EXP)
    run_config.update({
        "model_class": model_class,
        "resume_from_checkpoint": str(ckpt_path),
        "resume_start_epoch": start_epoch,
        "resume_best_epoch": best_epoch,
        "resume_best_val_l1": best_val_l1,
    })
    with (ckpt_dir / "run_config_resume.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    separator = "=" * 72
    print(f"\n{separator}", flush=True)
    print("RF training resume check", flush=True)
    print(f"{separator}", flush=True)
    print(f"experiment_name      : {EXP['experiment_name']}", flush=True)
    print(f"ckpt_dir             : {ckpt_dir}", flush=True)
    print(f"resume_checkpoint    : {ckpt_path}", flush=True)
    print(f"model_class          : {model_class}", flush=True)
    print(f"trainable_parameters : {count_trainable_parameters(model):,}", flush=True)
    print(f"resume_from_epoch    : {best_epoch}", flush=True)
    print(f"resume_start_epoch   : {start_epoch}", flush=True)
    print(f"best_val_l1          : {best_val_l1:.12e}", flush=True)
    print(f"{separator}\n", flush=True)

    if EXP["experiment_name"] not in str(ckpt_dir):
        raise AssertionError(
            f"ckpt_dir does not include experiment_name: {ckpt_dir} vs {EXP['experiment_name']}"
        )

    expected_class = "TinyResidualRFNet"
    if model_class != expected_class:
        raise AssertionError(
            f"model_name={EXP['model_name']} expected {expected_class}, got {model_class}"
        )

    init_val = evaluate_normalized(model, val_loader, device, abs_weight=EXP["abs_weight"])
    print("Validation at resume checkpoint:", flush=True)
    print(
        f"val_L1={init_val['l1']:.6e} | "
        f"baseline_L1={init_val['baseline_l1']:.6e} | "
        f"improvement={init_val['improvement']*100:.2f}%",
        flush=True,
    )

    for epoch in range(start_epoch, EXP["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_abs = 0.0
        train_base_l1 = 0.0
        train_base_abs = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            baseline = batch["baseline"].to(device, non_blocking=True)

            pred = model(x, baseline)
            loss, loss_l1, loss_abs = compute_loss(pred, y, abs_weight=EXP["abs_weight"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if EXP["grad_clip"] is not None and EXP["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), EXP["grad_clip"])
            optimizer.step()

            with torch.no_grad():
                base_l1 = F.l1_loss(baseline, y)
                base_abs = F.l1_loss(complex_abs_2ch(baseline), complex_abs_2ch(y))

            train_loss += float(loss.item())
            train_l1 += float(loss_l1.item())
            train_abs += float(loss_abs.item())
            train_base_l1 += float(base_l1.item())
            train_base_abs += float(base_abs.item())
            n_batches += 1

        lr = cosine_lr(epoch, EXP["lr"], EXP["eta_min"], EXP["num_epochs"])
        set_optimizer_lr(optimizer, lr)

        n = max(n_batches, 1)
        train_loss /= n
        train_l1 /= n
        train_abs /= n
        train_base_l1 /= n
        train_base_abs /= n

        train_impr = 1.0 - train_l1 / (train_base_l1 + 1e-12)
        train_abs_impr = 1.0 - train_abs / (train_base_abs + 1e-12)
        val_metrics = evaluate_normalized(model, val_loader, device, abs_weight=EXP["abs_weight"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l1": train_l1,
            "train_abs": train_abs,
            "train_base_l1": train_base_l1,
            "train_base_abs": train_base_abs,
            "train_impr": train_impr,
            "train_abs_impr": train_abs_impr,
            "val_loss": val_metrics["loss"],
            "val_l1": val_metrics["l1"],
            "val_abs": val_metrics["abs"],
            "val_base_l1": val_metrics["baseline_l1"],
            "val_base_abs": val_metrics["baseline_abs"],
            "val_impr": val_metrics["improvement"],
            "val_abs_impr": val_metrics["abs_improvement"],
            "lr": lr,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_class": model_class,
            "optimizer": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_l1": best_val_l1,
            "no_improve_epochs": no_improve_epochs,
            "history": history,
            "config": run_config,
        }
        torch.save(last_ckpt, ckpt_dir / "last_model.pth")

        if val_metrics["l1"] < best_val_l1:
            best_val_l1 = val_metrics["l1"]
            best_epoch = epoch
            no_improve_epochs = 0
            best_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "model_class": model_class,
                "optimizer": optimizer.state_dict(),
                "best_val_l1": best_val_l1,
                "history": history,
                "config": run_config,
            }
            torch.save(best_ckpt, ckpt_dir / "best_model.pth")
            torch.save(model.state_dict(), ckpt_dir / "best_state_dict.pth")
        else:
            no_improve_epochs += 1

        print(
            f"Epoch {epoch:04d} | "
            f"train_L1={train_l1:.6e} | "
            f"train_base={train_base_l1:.6e} | "
            f"train_impr={train_impr*100:6.2f}% | "
            f"val_L1={val_metrics['l1']:.6e} | "
            f"val_base={val_metrics['baseline_l1']:.6e} | "
            f"val_impr={val_metrics['improvement']*100:6.2f}% | "
            f"best_epoch={best_epoch} | "
            f"no_improve={no_improve_epochs} | "
            f"lr={lr:.2e}",
            flush=True,
        )

        if no_improve_epochs >= EXP["patience"]:
            print(
                "Early stopping triggered: "
                f"val L1 did not improve for {EXP['patience']} consecutive epoch(s). "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break

    torch.save(model.state_dict(), ckpt_dir / "final_state_dict.pth")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print("Training resume finished.", flush=True)
    print(f"Best val L1 = {best_val_l1:.6e} at epoch {best_epoch}", flush=True)
    print(f"Best checkpoint: {ckpt_dir / 'best_model.pth'}", flush=True)


if __name__ == "__main__":
    main()
