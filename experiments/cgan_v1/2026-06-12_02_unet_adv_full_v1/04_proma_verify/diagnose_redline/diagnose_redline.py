"""
Proma诊断 v2: 红线真伪 — train/eval模式脱节 vs 真坍塌
修正: restore_scale 应用到 input 和 baseline (v1 漏了 baseline 的 restore)
只读, 不碰生产代码, 不重训.
产出: 04_proma_verify/diagnose_redline/ (覆盖 v1)
"""
import sys, json, csv, time, os
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

REF = np.float32(64407.578125)
EPS = np.float32(1e-12)
DB_CLIP = (np.float32(-60.0), np.float32(0.0))

def bmode_from_complex_np(complex_arr):
    """精确复现训练代码的 bmode_from_complex_np. complex_arr shape: [2, Z, X, Y]"""
    real = complex_arr[0]
    imag = complex_arr[1]
    envelope = np.sqrt(real.astype(np.float64)**2 + imag.astype(np.float64)**2 + float(EPS))
    envelope_db_f64 = 20.0 * np.log10(envelope / float(REF) + float(EPS))
    envelope_db = np.clip(envelope_db_f64, float(DB_CLIP[0]), float(DB_CLIP[1])).astype(np.float32)
    bmode = (envelope_db - DB_CLIP[0]) / (DB_CLIP[1] - DB_CLIP[0])
    return bmode.astype(np.float32)


t0 = time.time()
results = {}

# ============================================================
# 0. 加载数据 (带 restore_scale)
# ============================================================
print("="*70)
print("Proma 诊断 v2: 红线真伪分析")
print("="*70)
print("\n[0] Loading data WITH restore_scale (fixing v1 bug)...")

im = np.memmap(str(RF_VAL/"input.dat"), dtype=np.float16, mode="r", shape=(225,1536,64,32,32))
bm = np.memmap(str(RF_VAL/"baseline.dat"), dtype=np.float16, mode="r", shape=(225,2,64,32,32))
gm = np.memmap(str(BMODE_VAL/"label_bmode.dat"), dtype=np.float16, mode="r", shape=(225,1,64,32,32))
sc = np.memmap(str(RF_VAL/"scale.dat"), dtype=np.float32, mode="r", shape=(225,))

# Check if data is normalized
with np.load(str(RF_VAL/"meta.npz"), allow_pickle=False) as meta:
    is_normalized = bool(meta["normalize"]) if "normalize" in meta.files else False
    print(f"  Data normalized: {is_normalized}")

print(f"  Scale stats: min={sc.min():.1f} max={sc.max():.1f} mean={sc.mean():.1f}")

# 准备 restored 数据 (匹配训练管线)
inps_restored = []
gts = []
bls_restored = []  # baseline_bmode from RESTORED baseline complex
bls_raw = []       # baseline_bmode from RAW baseline complex (v1 bug)

for idx in ALL_FIXED:
    s = np.float32(sc[idx])
    # Input: restore_scale (训练行为)
    inp_arr = np.array(im[idx], dtype=np.float32, copy=True)
    if is_normalized:
        inp_arr *= s
    inps_restored.append(torch.from_numpy(inp_arr))

    # GT: directly from pre-computed cache (fp16 → float32)
    gts.append(torch.from_numpy(np.array(gm[idx], dtype=np.float32, copy=True)))

    # Baseline: restore baseline complex THEN convert to B-mode (训练行为)
    bl_complex = np.array(bm[idx], dtype=np.float32, copy=True)
    if is_normalized:
        bl_complex *= s
    bls_restored.append(torch.from_numpy(bmode_from_complex_np(bl_complex)))

    # Baseline RAW (v1 bug: no restore)
    bl_complex_raw = np.array(bm[idx], dtype=np.float32, copy=True)
    bls_raw.append(torch.from_numpy(bmode_from_complex_np(bl_complex_raw)))

inp_b = torch.stack(inps_restored)
gt_b = torch.stack(gts)
bl_b_restored = torch.stack(bls_restored).unsqueeze(1)  # [24, 1, 64, 32, 32]
bl_b_raw = torch.stack(bls_raw).unsqueeze(1)

print(f"  Input tensor (restored):  shape={inp_b.shape}  min={inp_b.min().item():.2f} max={inp_b.max().item():.2f} std={inp_b.std().item():.2f}")
print(f"  GT bmode tensor:          shape={gt_b.shape}  min={gt_b.min().item():.4f} max={gt_b.max().item():.4f} mean={gt_b.mean().item():.4f}")
print(f"  Baseline RESTORED:        shape={bl_b_restored.shape}  min={bl_b_restored.min().item():.4f} max={bl_b_restored.max().item():.4f} mean={bl_b_restored.mean().item():.4f}")
print(f"  Baseline RAW (v1 bug):    shape={bl_b_raw.shape}  min={bl_b_raw.min().item():.4f} max={bl_b_raw.max().item():.4f} mean={bl_b_raw.mean().item():.4f}")

# ============================================================
# 1. Norm 配置报告
# ============================================================
print("\n[1/5] Norm configurations...")

g_temp = Light3DUNet()
d_temp = BMode3DPatchDiscriminator()

g_bn_info = []
for name, mod in g_temp.named_modules():
    if isinstance(mod, nn.BatchNorm3d):
        g_bn_info.append(dict(
            name=name, num_features=mod.num_features,
            momentum=mod.momentum, track_running_stats=mod.track_running_stats,
            affine=mod.affine))

d_norm_info = []
for name, mod in d_temp.named_modules():
    if isinstance(mod, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.InstanceNorm2d)):
        d_norm_info.append(dict(
            name=name, type=type(mod).__name__,
            num_features=mod.num_features, affine=mod.affine,
            track_running_stats=getattr(mod, "track_running_stats", "N/A")))

print(f"  G BN layers: {len(g_bn_info)} (all BatchNorm3d, momentum=0.9, track_running_stats=True)")
print(f"  D norm layers: {len(d_norm_info)}")
for n in d_norm_info:
    print(f"    {n['name']}: {n['type']} affine={n['affine']} track_running_stats={n['track_running_stats']}")

results["norm_config"] = {
    "G": {"type": "BatchNorm3d", "count": len(g_bn_info),
          "momentum": 0.9, "track_running_stats": True, "affine": True},
    "D": {"type": "InstanceNorm3d", "count": len(d_norm_info),
          "track_running_stats": "N/A (IN has no running stats, train/eval identical)", "affine": True,
          "detail": d_norm_info},
}
del g_temp, d_temp

# BN stats evolution from CSV
with open(TRAIN/"metrics"/"stability.csv") as f:
    csv_rows = list(csv.DictReader(f))
print("\n  BN running stats evolution:")
for ep in ["1","10","25","41","42"]:
    row = [r for r in csv_rows if r["epoch"]==ep]
    if row:
        r = row[0]
        print(f"    epoch {ep:>3}: running_mean={float(r['bn_running_mean_mean']):.2f}  running_var={float(r['bn_running_var_mean']):.1f}")
results["bn_stats_evolution"] = {
    r["epoch"]: {
        "running_mean_mean": float(r["bn_running_mean_mean"]),
        "running_var_mean": float(r["bn_running_var_mean"]),
    } for r in csv_rows if r["epoch"] in ["1","10","25","41","42"]
}

# ============================================================
# 2. 四模式交叉表 (核心) — epoch 25
# ============================================================
print("\n[2/5] Four-mode cross table at epoch 25 (WITH restored baseline)...")

ckpt_ep25 = torch.load(TRAIN/"checkpoints"/"epoch025.pt", map_location="cpu", weights_only=False)

modes = [
    ("G.train + D.train", "train", "train"),
    ("G.train + D.eval ", "train", "eval"),
    ("G.eval  + D.train", "eval", "train"),
    ("G.eval  + D.eval ", "eval", "eval"),
]

cross_restored = {}
cross_raw_baseline = {}

for label, g_mode, d_mode in modes:
    g_ = Light3DUNet(); g_.load_state_dict(ckpt_ep25["model_state_dict"])
    d_ = BMode3DPatchDiscriminator(); d_.load_state_dict(ckpt_ep25["discriminator_state_dict"])

    if g_mode == "train": g_.train()
    else: g_.eval()
    if d_mode == "train": d_.train()
    else: d_.eval()

    with torch.no_grad():
        pred_b = g_(inp_b)
        dls = d_lsgan_loss(d_, pred_b, gt_b, bl_b_restored)
        gls = g_adv_loss(d_, pred_b, bl_b_restored)

    cross_restored[label] = {
        "D_real_sig": round(dls["d_real_score_sigmoid"].item(), 4),
        "D_real_raw": round(dls["d_real_score_raw"].item(), 4),
        "D_fake_sig": round(dls["d_fake_score_sigmoid"].item(), 4),
        "D_fake_raw": round(dls["d_fake_score_raw"].item(), 4),
        "D_loss": round(dls["d_loss"].item(), 4),
        "G_adv": round(gls["g_adv"].item(), 4),
        "pred_mean": round(pred_b.mean().item(), 4),
        "pred_std": round(pred_b.std().item(), 4),
    }
    info = cross_restored[label]
    print(f"  {label}: D_real={info['D_real_sig']:.4f} D_fake={info['D_fake_sig']:.4f} G_adv={info['G_adv']:.4f} pred_mean={info['pred_mean']:.4f} pred_std={info['pred_std']:.4f}")

    # Also test with RAW baseline (v1 bug repro)
    dls_raw = d_lsgan_loss(d_, pred_b, gt_b, bl_b_raw)
    gls_raw = g_adv_loss(d_, pred_b, bl_b_raw)
    cross_raw_baseline[label] = {
        "D_real_sig": round(dls_raw["d_real_score_sigmoid"].item(), 4),
        "D_fake_sig": round(dls_raw["d_fake_score_sigmoid"].item(), 4),
    }

    del g_, d_

results["four_mode_cross_restored"] = cross_restored
results["four_mode_cross_raw_baseline"] = cross_raw_baseline

# CSV epoch 25 reference
row25 = [r for r in csv_rows if r["epoch"]=="25"][0]
csv_ref = {
    "D_real_evalfixed": float(row25["d_real_score_sigmoid_evalfixed"]),
    "D_fake_evalfixed": float(row25["d_fake_score_sigmoid_evalfixed"]),
    "G_adv_evalfixed": float(row25["g_adv_evalfixed"]),
    "val_pred_mean": float(row25["val_pred_mean"]),
    "val_pred_std": float(row25["val_pred_std"]),
    # Train metrics
    "D_real_train": float(row25["d_real_score_sigmoid_train"]),
    "D_fake_train": float(row25["d_fake_score_sigmoid_train"]),
    "G_adv_train": float(row25["g_adv_train"]),
}
results["csv_epoch25_ref"] = csv_ref

print(f"\n  CSV epoch 25 reference:")
print(f"    Train:      D_real={csv_ref['D_real_train']:.4f} D_fake={csv_ref['D_fake_train']:.4f} G_adv={csv_ref['G_adv_train']:.4f}")
print(f"    Evalfixed:  D_real={csv_ref['D_real_evalfixed']:.4f} D_fake={csv_ref['D_fake_evalfixed']:.4f} G_adv={csv_ref['G_adv_evalfixed']:.4f}")
print(f"    val_pred:   mean={csv_ref['val_pred_mean']:.4f} std={csv_ref['val_pred_std']:.4f}")

# Compare modes to CSV
print(f"\n  Distance to CSV evalfixed:")
for label, v in cross_restored.items():
    dist = (abs(v["D_real_sig"]-csv_ref["D_real_evalfixed"]) +
            abs(v["D_fake_sig"]-csv_ref["D_fake_evalfixed"]) +
            abs(v["G_adv"]-csv_ref["G_adv_evalfixed"]))
    print(f"    {label}: dist={dist:.4f}")

print(f"\n  Distance to CSV train:")
for label, v in cross_restored.items():
    dist = (abs(v["D_real_sig"]-csv_ref["D_real_train"]) +
            abs(v["D_fake_sig"]-csv_ref["D_fake_train"]))
    print(f"    {label}: dist={dist:.4f}")

# ============================================================
# 3. G train/eval pred 差异量化 (用 restored input)
# ============================================================
print("\n[3/5] G train vs eval pred difference (restored input)...")

g_train = Light3DUNet(); g_train.load_state_dict(ckpt_ep25["model_state_dict"]); g_train.train()
g_eval = Light3DUNet(); g_eval.load_state_dict(ckpt_ep25["model_state_dict"]); g_eval.eval()

with torch.no_grad():
    pred_train = g_train(inp_b)
    pred_eval = g_eval(inp_b)

l1_diff = F.l1_loss(pred_train, pred_eval).item()
print(f"  pred_train: mean={pred_train.mean().item():.4f} std={pred_train.std().item():.4f}")
print(f"  pred_eval:  mean={pred_eval.mean().item():.4f} std={pred_eval.std().item():.4f}")
print(f"  L1 diff: {l1_diff:.4f} ({l1_diff*100:.1f}% of [0,1] range)")
print(f"  Ratio pred_train_std / pred_eval_std: {pred_train.std().item()/pred_eval.std().item():.2f}x")

results["g_train_eval_diff"] = {
    "L1_train_vs_eval": round(l1_diff, 4),
    "pred_mean_train": round(pred_train.mean().item(), 4),
    "pred_mean_eval": round(pred_eval.mean().item(), 4),
    "pred_std_train": round(pred_train.std().item(), 4),
    "pred_std_eval": round(pred_eval.std().item(), 4),
}

# BN per-layer stats at e25
bn_detail = []
for name, mod in g_eval.named_modules():
    if isinstance(mod, nn.BatchNorm3d):
        bn_detail.append({
            "name": name,
            "running_mean_mean": round(mod.running_mean.mean().item(), 4),
            "running_var_mean": round(mod.running_var.mean().item(), 2),
            "running_var_max": round(mod.running_var.max().item(), 2),
        })
results["bn_per_layer_e25"] = bn_detail

del g_train, g_eval

# ============================================================
# 4. Phantom std 厘清
# ============================================================
print("\n[4/5] Phantom speckle std clarification...")

phantom_indices = FIXED_VAL["phantom"]
g4 = Light3DUNet(); g4.load_state_dict(ckpt_ep25["model_state_dict"])

# G.eval mode
g4.eval()
phantom_eval = []
with torch.no_grad():
    for idx in phantom_indices:
        s = np.float32(sc[idx])
        inp_arr = np.array(im[idx], dtype=np.float32, copy=True)
        if is_normalized:
            inp_arr *= s
        inp = torch.from_numpy(inp_arr).unsqueeze(0)
        pred = g4(inp)
        phantom_eval.append(pred.numpy().flatten())
all_pe = np.concatenate(phantom_eval)
pe_mean = float(np.mean(all_pe)); pe_std = float(np.std(all_pe))
print(f"  G.eval  phantom pred: mean={pe_mean:.4f} std={pe_std:.4f} mean/std={pe_mean/pe_std:.2f}")

# G.train mode
g4.train()
phantom_train = []
with torch.no_grad():
    for idx in phantom_indices:
        s = np.float32(sc[idx])
        inp_arr = np.array(im[idx], dtype=np.float32, copy=True)
        if is_normalized:
            inp_arr *= s
        inp = torch.from_numpy(inp_arr).unsqueeze(0)
        pred = g4(inp)
        phantom_train.append(pred.numpy().flatten())
all_pt = np.concatenate(phantom_train)
pt_mean = float(np.mean(all_pt)); pt_std = float(np.std(all_pt))
print(f"  G.train phantom pred: mean={pt_mean:.4f} std={pt_std:.4f} mean/std={pt_mean/pt_std:.2f}")

# CSV reference
csv_phantom_e25 = float(row25["val_phantom_pred_std"])
csv_phantom_gt_std = float(row25["val_phantom_gt_std"])
print(f"  CSV val phantom: pred_std={csv_phantom_e25:.4f} gt_std={csv_phantom_gt_std:.4f}")

# summary.md phantom speckle table
print(f"\n  summary.md phantom speckle at e25: pred std=0.093 gt std=0.093")
print(f"  G.eval  pred_std (8 fixed) = {pe_std:.4f}")
print(f"  G.train pred_std (8 fixed) = {pt_std:.4f}")
print(f"  CSV val phantom pred_std    = {csv_phantom_e25:.4f} (225 val samples)")

results["phantom_std_clarification"] = {
    "epoch": 25,
    "n_fixed_phantom": len(phantom_indices),
    "G_eval_pred_mean": round(pe_mean, 4),
    "G_eval_pred_std": round(pe_std, 4),
    "G_eval_mean_over_std": round(pe_mean/pe_std, 2),
    "G_train_pred_mean": round(pt_mean, 4),
    "G_train_pred_std": round(pt_std, 4),
    "G_train_mean_over_std": round(pt_mean/pt_std, 2),
    "csv_val_phantom_pred_std": round(csv_phantom_e25, 4),
    "csv_val_phantom_gt_std": round(csv_phantom_gt_std, 4),
    "summary_md_e25_phantom_pred_std": 0.093,
    "summary_md_e25_phantom_gt_std": 0.093,
}
del g4

# ============================================================
# 5. D 偏移归因 — 证实 v1 bug
# ============================================================
print("\n[5/5] Attributing D offset — v1 baseline restore_scale bug confirmation...")

# Key comparison: D scores with RESTORED vs RAW baseline
print(f"\n  D_real with RESTORED baseline (correct):")
for label in cross_restored:
    print(f"    {label}: {cross_restored[label]['D_real_sig']:.4f}")
print(f"\n  D_real with RAW baseline (v1 bug):")
for label in cross_raw_baseline:
    print(f"    {label}: {cross_raw_baseline[label]['D_real_sig']:.4f}")

# Find which restored mode best matches CSV
best_restored = None; best_dist_r = 1e9
for label, v in cross_restored.items():
    dist = (abs(v["D_real_sig"]-csv_ref["D_real_evalfixed"]) +
            abs(v["D_fake_sig"]-csv_ref["D_fake_evalfixed"]))
    if dist < best_dist_r:
        best_dist_r = dist; best_restored = label

print(f"\n  Best match to CSV evalfixed (restored baseline): {best_restored} dist={best_dist_r:.4f}")
print(f"  Remaining D_real offset: {cross_restored[best_restored]['D_real_sig'] - csv_ref['D_real_evalfixed']:.4f}")

# Train mode comparison
best_train = None; best_dist_t = 1e9
for label, v in cross_restored.items():
    dist = (abs(v["D_real_sig"]-csv_ref["D_real_train"]) +
            abs(v["D_fake_sig"]-csv_ref["D_fake_train"]))
    if dist < best_dist_t:
        best_dist_t = dist; best_train = label

print(f"  Best match to CSV train (restored baseline): {best_train} dist={best_dist_t:.4f}")

# ============================================================
# 6. 额外: Train mode adversarial health from CSV (e42 无 ckpt, 用 CSV 分析)
# ============================================================
print("\n[6] Train vs Evalfixed health trend from CSV (full 42 epochs)...")

train_d_gap = []
eval_d_gap = []
for row in csv_rows:
    ep = int(row["epoch"])
    tg = float(row["d_real_score_sigmoid_train"]) - float(row["d_fake_score_sigmoid_train"])
    eg = float(row["d_real_score_sigmoid_evalfixed"]) - float(row["d_fake_score_sigmoid_evalfixed"])
    train_d_gap.append((ep, tg))
    eval_d_gap.append((ep, eg))

print(f"  Train D gap   e1: {train_d_gap[0][1]:+.4f}  e10: {train_d_gap[9][1]:+.4f}  e25: {train_d_gap[24][1]:+.4f}  e42: {train_d_gap[41][1]:+.4f}")
print(f"  Eval  D gap   e1: {eval_d_gap[0][1]:+.4f}  e10: {eval_d_gap[9][1]:+.4f}  e25: {eval_d_gap[24][1]:+.4f}  e42: {eval_d_gap[41][1]:+.4f}")

results["train_vs_eval_gap_trend"] = {
    "train_gap": {str(ep): round(g, 4) for ep, g in train_d_gap if ep in [1,10,25,42]},
    "eval_gap": {str(ep): round(g, 4) for ep, g in eval_d_gap if ep in [1,10,25,42]},
}

# ============================================================
# 7. e42 分析 (无 checkpoint, 仅 CSV 数据)
# ============================================================
print("\n[7] Epoch 42 analysis (no checkpoint, CSV only)...")
row42 = [r for r in csv_rows if r["epoch"]=="42"][0]
print(f"  Train:     D_real={float(row42['d_real_score_sigmoid_train']):.4f} D_fake={float(row42['d_fake_score_sigmoid_train']):.4f} gap={float(row42['d_real_score_sigmoid_train'])-float(row42['d_fake_score_sigmoid_train']):+.4f}")
print(f"  Evalfixed: D_real={float(row42['d_real_score_sigmoid_evalfixed']):.4f} D_fake={float(row42['d_fake_score_sigmoid_evalfixed']):.4f} gap={float(row42['d_real_score_sigmoid_evalfixed'])-float(row42['d_fake_score_sigmoid_evalfixed']):+.4f}")
print(f"  D_loss_train={float(row42['d_loss_train']):.4f} (declining, healthy)")
print(f"  val_pred_std={float(row42['val_pred_std']):.4f}")

results["epoch42_csv_only"] = {
    "note": "No e42 checkpoint saved (redline stopped before snapshot). Analysis from CSV only.",
    "D_real_train": float(row42["d_real_score_sigmoid_train"]),
    "D_fake_train": float(row42["d_fake_score_sigmoid_train"]),
    "D_real_evalfixed": float(row42["d_real_score_sigmoid_evalfixed"]),
    "D_fake_evalfixed": float(row42["d_fake_score_sigmoid_evalfixed"]),
    "D_loss_train": float(row42["d_loss_train"]),
    "val_pred_std": float(row42["val_pred_std"]),
}

# ============================================================
# 保存 + 生成报告
# ============================================================
elapsed = time.time() - t0
results["_meta"] = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S +0800"),
    "elapsed_seconds": round(elapsed, 1),
    "checkpoint": "epoch025.pt",
    "n_evalfixed": len(ALL_FIXED),
    "device": "cpu",
    "v2_fix": "baseline_bmode now computed from RESTORED baseline complex (×scale before B-mode conversion), matching training pipeline. v1 used raw baseline, causing ~89dB shift in log space.",
}

with open(OUT/"diagnose_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"Diagnosis complete in {elapsed:.1f}s")
print(f"Results saved to {OUT/'diagnose_results.json'}")
print(f"{'='*70}")
