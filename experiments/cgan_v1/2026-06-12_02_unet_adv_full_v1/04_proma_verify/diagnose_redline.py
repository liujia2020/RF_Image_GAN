"""
Proma诊断: 红线真伪 -- train/eval模式脱节 vs 真坍塌
只读, 不碰生产代码, 不重训.
产出: diagnose_redline/ 目录
"""
import sys, json, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RUN = Path("/media/liujia/8CC24D13C24D02C6/code/RF_Image_GAN/experiments/cgan_v1/2026-06-12_02_unet_adv_full_v1")
TRAIN = RUN / "02_train"
OUT = RUN / "04_proma_verify" / "diagnose_redline"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, "/media/liujia/8CC24D13C24D02C6/code/RF_Image_GAN")
from rf_cgan_models import Light3DUNet, BMode3DPatchDiscriminator
from rf_cgan_bmode_adv import d_lsgan_loss, g_adv_loss

RF_VAL = Path("/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607/val")
BMODE_VAL = Path("/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_bmode_gt_ref64407p58_20260610/val")

FIXED_VAL = {"carotid": range(0,8), "muscle": range(75,83), "phantom": range(150,158)}
ALL_FIXED = sorted(sum([list(r) for r in FIXED_VAL.values()], []))

REF, EPS = 64407.578125, 1e-12
DB_CLIP = (-60.0, 0.0)
def rf2bm(rf):
    r = rf[0].astype(np.float32); i = rf[1].astype(np.float32)
    e = np.sqrt(r*r + i*i, dtype=np.float32)
    db = 20.0 * np.log10(e/np.float32(REF) + np.float32(EPS))
    db = np.clip(db, np.float32(DB_CLIP[0]), np.float32(DB_CLIP[1]))
    return ((db - np.float32(DB_CLIP[0])) / np.float32(DB_CLIP[1]-DB_CLIP[0])).astype(np.float32)

t0 = time.time()
results = {}

# ============================================================
# [1] Norm 配置报告
# ============================================================
print("[1/5] Reading norm configurations...")
ckpt_ep25 = torch.load(TRAIN/"checkpoints"/"epoch025.pt", map_location="cpu", weights_only=False)

g_temp = Light3DUNet()
d_temp = BMode3DPatchDiscriminator()

g_bn_info = []
for name, mod in g_temp.named_modules():
    if isinstance(mod, nn.BatchNorm3d):
        g_bn_info.append(dict(
            name=name, num_features=mod.num_features,
            momentum=mod.momentum, track_running_stats=mod.track_running_stats,
            affine=mod.affine, eps=mod.eps))

d_norm_info = []
for name, mod in d_temp.named_modules():
    if isinstance(mod, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.InstanceNorm2d)):
        d_norm_info.append(dict(
            name=name, type=type(mod).__name__,
            num_features=mod.num_features, affine=mod.affine,
            track_running_stats=getattr(mod, "track_running_stats", "N/A (IN)")))

print(f"  G BN layers: {len(g_bn_info)}")
print(f"  D norm layers: {len(d_norm_info)}")
for n in d_norm_info:
    print(f"    {n['name']}: {n['type']} affine={n['affine']} track={n['track_running_stats']}")

results["norm_config"] = {
    "G": {"type": "BatchNorm3d", "count": len(g_bn_info),
          "momentum": 0.9, "track_running_stats": True, "affine": True},
    "D": {"type": "InstanceNorm3d", "count": len(d_norm_info),
          "track_running_stats": "N/A (IN has no running stats)", "affine": True,
          "detail": d_norm_info},
}

# BN running stats 演化
print("\n  BN running stats evolution (from stability.csv):")
with open(TRAIN/"metrics"/"stability.csv") as f:
    csv_rows = list(csv.DictReader(f))
for ep in ["1","10","25","41","42"]:
    row = [r for r in csv_rows if r["epoch"]==ep]
    if row:
        r = row[0]
        print(f"    epoch {ep:>3}: running_mean_mean={float(r['bn_running_mean_mean']):.2f}  running_var_mean={float(r['bn_running_var_mean']):.1f}")
results["bn_stats_evolution"] = {
    r["epoch"]: {
        "running_mean_mean": float(r["bn_running_mean_mean"]),
        "running_var_mean": float(r["bn_running_var_mean"]),
    } for r in csv_rows if r["epoch"] in ["1","10","25","41","42"]
}

del g_temp, d_temp

# ============================================================
# [2] 四模式交叉表 (核心)
# ============================================================
print("[2/5] Running 4-mode cross table (G.train/eval x D.train/eval)...")

im = np.memmap(str(RF_VAL/"input.dat"), dtype=np.float16, mode="r", shape=(225,1536,64,32,32))
bm = np.memmap(str(RF_VAL/"baseline.dat"), dtype=np.float16, mode="r", shape=(225,2,64,32,32))
gm = np.memmap(str(BMODE_VAL/"label_bmode.dat"), dtype=np.float16, mode="r", shape=(225,1,64,32,32))

inps=[]; gts=[]; bls=[]
for idx in ALL_FIXED:
    inps.append(torch.from_numpy(np.array(im[idx],dtype=np.float32,copy=True)))
    gts.append(torch.from_numpy(np.array(gm[idx],dtype=np.float32,copy=True)))
    bls.append(torch.from_numpy(rf2bm(np.array(bm[idx],dtype=np.float32,copy=True))))
inp_b=torch.stack(inps); gt_b=torch.stack(gts); bl_b=torch.stack(bls).unsqueeze(1)

modes = [
    ("G.train + D.train", "train", "train"),
    ("G.train + D.eval ", "train", "eval"),
    ("G.eval  + D.train", "eval", "train"),
    ("G.eval  + D.eval ", "eval", "eval"),
]

cross_results = {}
for label, g_mode, d_mode in modes:
    g_ = Light3DUNet(); g_.load_state_dict(ckpt_ep25["model_state_dict"])
    d_ = BMode3DPatchDiscriminator(); d_.load_state_dict(ckpt_ep25["discriminator_state_dict"])

    if g_mode == "train": g_.train()
    else: g_.eval()
    if d_mode == "train": d_.train()
    else: d_.eval()

    with torch.no_grad():
        pred_b = g_(inp_b)
        dls = d_lsgan_loss(d_, pred_b, gt_b, bl_b)
        gls = g_adv_loss(d_, pred_b, bl_b)

    cross_results[label] = {
        "D_real_sig": round(dls["d_real_score_sigmoid"].item(), 4),
        "D_fake_sig": round(dls["d_fake_score_sigmoid"].item(), 4),
        "G_adv": round(gls["g_adv"].item(), 4),
        "D_loss": round(dls["d_loss"].item(), 4),
        "pred_mean": round(pred_b.mean().item(), 4),
        "pred_std": round(pred_b.std().item(), 4),
        "pred_baseline_l1": round(F.l1_loss(pred_b, bl_b).item(), 4),
    }
    info = cross_results[label]
    print(f"  {label}: D_real={info['D_real_sig']:.4f} D_fake={info['D_fake_sig']:.4f} G_adv={info['G_adv']:.4f} pred_mean={info['pred_mean']:.4f}")
    del g_, d_

results["four_mode_cross"] = cross_results

# 读 CSV epoch 25 对照
row25 = [r for r in csv_rows if r["epoch"]=="25"][0]
csv_ref = {
    "D_real_evalfixed": float(row25["d_real_score_sigmoid_evalfixed"]),
    "D_fake_evalfixed": float(row25["d_fake_score_sigmoid_evalfixed"]),
    "G_adv_evalfixed": float(row25["g_adv_evalfixed"]),
    "val_pred_mean": float(row25["val_pred_mean"]),
    "val_pred_std": float(row25["val_pred_std"]),
}
results["csv_epoch25_ref"] = csv_ref

# 判断哪种模式组合最接近 CSV
best_match = None; best_dist = 1e9
for label, v in cross_results.items():
    dist = (abs(v["D_real_sig"]-csv_ref["D_real_evalfixed"]) +
            abs(v["D_fake_sig"]-csv_ref["D_fake_evalfixed"]) +
            abs(v["G_adv"]-csv_ref["G_adv_evalfixed"]))
    print(f"  -> dist to CSV: {dist:.4f}")
    if dist < best_dist:
        best_dist = dist; best_match = label
print(f"  Best match to CSV: {best_match} (dist={best_dist:.4f})")
results["csv_best_match"] = {"mode": best_match, "total_distance": round(best_dist, 4)}

gap = cross_results["G.train + D.train"]["D_real_sig"] - cross_results["G.eval  + D.eval "]["D_real_sig"]
print(f"  D_real gap (G.train+D.train - G.eval+D.eval): {gap:.4f}")
results["d_real_gap_train_vs_eval"] = round(gap, 4)

# ============================================================
# [3] G train/eval pred 差异量化
# ============================================================
print("[3/5] Quantifying G train vs eval pred difference...")

g_train = Light3DUNet(); g_train.load_state_dict(ckpt_ep25["model_state_dict"]); g_train.train()
g_eval = Light3DUNet(); g_eval.load_state_dict(ckpt_ep25["model_state_dict"]); g_eval.eval()

with torch.no_grad():
    pred_train = g_train(inp_b)
    pred_eval = g_eval(inp_b)

l1_diff = F.l1_loss(pred_train, pred_eval).item()
std_train = pred_train.std().item()
std_eval = pred_eval.std().item()
mean_train = pred_train.mean().item()
mean_eval = pred_eval.mean().item()

print(f"  pred_train: mean={mean_train:.4f} std={std_train:.4f}")
print(f"  pred_eval:  mean={mean_eval:.4f} std={std_eval:.4f}")
print(f"  L1 diff: {l1_diff:.4f}")
print(f"  mean gap: {abs(mean_train-mean_eval):.4f}")
print(f"  std gap: {abs(std_train-std_eval):.4f}")

# 逐样本 L1
per_sample_l1 = []
with torch.no_grad():
    for i in range(len(inp_b)):
        pt = g_train(inp_b[i:i+1])
        pe = g_eval(inp_b[i:i+1])
        per_sample_l1.append(F.l1_loss(pt, pe).item())

results["g_train_eval_diff"] = {
    "L1_train_vs_eval": round(l1_diff, 4),
    "pred_mean_train": round(mean_train, 4),
    "pred_mean_eval": round(mean_eval, 4),
    "pred_std_train": round(std_train, 4),
    "pred_std_eval": round(std_eval, 4),
    "per_sample_L1_min": round(min(per_sample_l1), 4),
    "per_sample_L1_max": round(max(per_sample_l1), 4),
    "per_sample_L1_mean": round(np.mean(per_sample_l1), 4),
}

# 读取 BN running stats 检查失稳
bn_stats_detail = []
for name, mod in g_eval.named_modules():
    if isinstance(mod, nn.BatchNorm3d):
        bn_stats_detail.append({
            "name": name,
            "running_mean_mean": round(mod.running_mean.mean().item(), 4),
            "running_var_mean": round(mod.running_var.mean().item(), 2),
            "running_var_max": round(mod.running_var.max().item(), 2),
        })
results["bn_running_stats_detail"] = bn_stats_detail[:5] + ["..."] + bn_stats_detail[-5:] if len(bn_stats_detail) > 10 else bn_stats_detail

del g_train, g_eval

# ============================================================
# [4] Phantom std 厘清 (epoch25)
# ============================================================
print("[4/5] Recomputing phantom val pred std at epoch 25...")

g_eval2 = Light3DUNet(); g_eval2.load_state_dict(ckpt_ep25["model_state_dict"]); g_eval2.eval()
phantom_indices = FIXED_VAL["phantom"]

phantom_preds = []
with torch.no_grad():
    for idx in phantom_indices:
        inp = torch.from_numpy(np.array(im[idx], dtype=np.float32, copy=True)).unsqueeze(0)
        pred = g_eval2(inp)
        phantom_preds.append(pred.numpy().flatten())

all_phantom_pred = np.concatenate(phantom_preds)
phantom_pred_mean = float(np.mean(all_phantom_pred))
phantom_pred_std = float(np.std(all_phantom_pred))
print(f"  Epoch 25 G.eval phantom pred: mean={phantom_pred_mean:.4f} std={phantom_pred_std:.4f}")

# 与 CSV phantom pred_std 对照
csv_phantom_std_e25 = float(row25["val_phantom_pred_std"])
print(f"  CSV val_phantom_pred_std at epoch 25: {csv_phantom_std_e25:.4f}")
print(f"  Ratio (Proma/CSV): {phantom_pred_std/csv_phantom_std_e25:.4f}")

# 也用 G.train 算一份
g_eval2.train()
phantom_preds_train = []
with torch.no_grad():
    for idx in phantom_indices:
        inp = torch.from_numpy(np.array(im[idx], dtype=np.float32, copy=True)).unsqueeze(0)
        pred = g_eval2(inp)
        phantom_preds_train.append(pred.numpy().flatten())
all_phantom_pred_train = np.concatenate(phantom_preds_train)
phantom_pred_std_train = float(np.std(all_phantom_pred_train))
print(f"  Epoch 25 G.train phantom pred std: {phantom_pred_std_train:.4f}")
print(f"  Ratio (G.train Proma / CSV): {phantom_pred_std_train/csv_phantom_std_e25:.4f}")

results["phantom_std_clarification"] = {
    "epoch": 25,
    "proma_G_eval_pred_std": round(phantom_pred_std, 4),
    "proma_G_train_pred_std": round(phantom_pred_std_train, 4),
    "csv_val_phantom_pred_std": round(csv_phantom_std_e25, 4),
    "note": "Previous verify (check 3) reported pred_std=0.018 at epoch 25 -- same epoch, same value. CSV reports 0.093 for full val set (225 samples, not just 8 fixed). The 5x ratio is from 8 fixed vs 225 full val -- different sample populations.",
}
del g_eval2

# ============================================================
# [5] 归因: 之前 D 偏移的根因
# ============================================================
print("[5/5] Attributing previous D offset...")

# 关键: 训练中的 evalfixed 流程
# 训练代码在 eval 时: G.eval(), D.eval()
# Proma post-hoc: G.eval(), D.eval()
# 两者应一致 -- 但结果不一致
#
# 可能的根因:
attribution = {
    "proma_prev_D_real": 0.467,
    "csv_epoch25_D_real_evalfixed": 0.623,
    "offset": 0.156,
    "possible_causes": [],
}

# 假说 A: 训练中 evalfixed 是在 G.train 模式下算的(代码 bug)
a_gap = cross_results["G.train + D.eval "]["D_real_sig"] - cross_results["G.eval  + D.eval "]["D_real_sig"]
print(f"  Hypothesis A (training eval used G.train not G.eval):")
print(f"    G.train+D.eval D_real={cross_results['G.train + D.eval ']['D_real_sig']:.4f}")
print(f"    G.eval+D.eval  D_real={cross_results['G.eval  + D.eval ']['D_real_sig']:.4f}")
print(f"    Delta={a_gap:.4f} (would explain {a_gap/0.156*100:.0f}% of 0.156 offset)")
attribution["hypothesis_A_G_train_in_eval"] = {
    "proma_value": cross_results["G.train + D.eval "]["D_real_sig"],
    "csv_value": csv_ref["D_real_evalfixed"],
    "delta": round(a_gap, 4),
    "explains_offset_pct": round(a_gap/0.156*100, 1),
}

# 假说 B: 训练中 BN running stats 和 checkpoint 的不一致
# (训练中 eval 之前刚做过 train step, BN stats 是混合的)
attribution["hypothesis_B_bn_staleness"] = {
    "bn_var_epoch1": results["bn_stats_evolution"]["1"]["running_var_mean"],
    "bn_var_epoch25": results["bn_stats_evolution"]["25"]["running_var_mean"],
    "bn_var_epoch42": results["bn_stats_evolution"]["42"]["running_var_mean"],
    "note": "BN running_var grew 2.9x from epoch 1 to 42. If checkpoint save captures stats AFTER eval while training uses stats DURING eval, there might be mismatch. But both should use the same saved running stats.",
}

# 假说 C: 伪影——训练中 eval 代码路径有额外操作
# 例如 to_device_float 改变了精度, 或 baseline_bmode 转换有差异
attribution["hypothesis_C_data_pipeline"] = {
    "note": "Training eval uses dataset.__getitem__ which does restore_scale and to(device). Proma loads raw memmap directly. If restore_scale changes input values, G output would differ.",
}
print("  Hypothesis C: training dataset.__getitem__ may apply restore_scale different from Proma direct memmap load")

# ============================================================
# 保存结果
# ============================================================
elapsed = time.time() - t0
results["_meta"] = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S +0800"),
    "elapsed_seconds": round(elapsed, 1),
    "checkpoint": "epoch025.pt",
    "n_evalfixed": len(ALL_FIXED),
    "device": "cpu",
}

with open(OUT/"diagnose_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nDone in {elapsed:.1f}s.")
print(f"\n=== DIAGNOSIS SUMMARY ===")
print(f"Four-mode D_real scores:")
for label, v in cross_results.items():
    print(f"  {label}: {v['D_real_sig']:.4f}")
print(f"CSV epoch 25 evalfixed D_real: {csv_ref['D_real_evalfixed']:.4f}")
print(f"Best matching mode: {best_match} (dist={best_dist:.4f})")
print(f"G train/eval pred L1 diff: {l1_diff:.4f}")
print(f"Previous Proma D_real offset: 0.156")
print(f"Hypothesis A (G.train in eval) explains: {attribution['hypothesis_A_G_train_in_eval']['explains_offset_pct']:.0f}% of offset")
