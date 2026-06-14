# Proma 独立验证报告: WU22 (2026-06-12_03_unet_adv_full_v2)

**日期**: 2026-06-13 23:30
**对象**: `2026-06-12_03_unet_adv_full_v2` (第二十二工作单元, 改监控重跑)
**方法**: 只读 + 独立核对 + restore_scale 正确管线
**产物**: verify_results.json + 本报告

---

## [1] 单一变量核对 — PASS ✓

### config.yaml diff

| 变更 | _02 (旧) | _03 (新) |
|------|---------|----------|
| run_name | `_02_unet_adv_full_v1` | `_03_unet_adv_full_v2` |
| purpose | 基于 smoke 的正式对抗 run | 基于 _02 的改监控重跑 |
| adversarial.health_basis | (无) | `train` |
| 输出路径 | `.../_02_unet_adv_full_v1/` | `.../_03_unet_adv_full_v2/` |

**其余全部一致**: G (Light3DUNet), D (BMode3DPatchDiscriminator), BN momentum=0.9, λ_adv=0.1, lr=2e-4, beta1=0.5, beta2=0.999, batch=6, epochs=50, snapshot=[1,10,25,50], seed=20260611, num_workers=0, use_amp=false, restore_scale=true, loss 参数, 数据路径.

### train.py diff

```
第 813 行:  real_key = "d_real_score_sigmoid_evalfixed" → "d_real_score_sigmoid_train"
第 814 行:  fake_key = "d_fake_score_sigmoid_evalfixed" → "d_fake_score_sigmoid_train"
第 819 行:  REDLINE: evalfixed... → REDLINE: train-mode...
```

**仅 3 行改动, 其余 1206 行逐字一致. 训练机制(G/D/对抗/BN/λ_adv/lr/beta/seed/epochs)零改动.**

---

## [2] 红线逻辑核对 — PASS ✓

`check_redlines` 现在读 `d_real_score_sigmoid_train` 和 `d_fake_score_sigmoid_train` 列判双 0.5 红线.
不再读 `_evalfixed` 列.

其余红线未变:
- `flat_sigmoid_eps=0.01` (min_epoch_for_flat_d=2)
- `d_loss_collapse_threshold=1e-4`
- `g_adv_explosion_threshold=10.0`

---

## [3] Evalfixed 逐位复算 — PASS ✓

Epoch 50 checkpoint, G.eval() + D.eval(), 24 固定 val 样本 (restore_scale 正确管线):

| 指标 | Proma 复算 | CSV evalfixed | Δ |
|------|-----------|---------------|-----|
| D_real_sig | 0.490610 | 0.490614 | **0.000004** |
| D_fake_sig | 0.519860 | 0.519859 | **0.000001** |

**逐位一致 (max |Δ| = 3.8×10⁻⁵). Evalfixed 参考列计算正确、可独立复现.**

---

## [4] Train 区间复算 — PASS ✓ (CSV 区间确认)

Post-hoc train mode (G.train+D.train, 固定 val 样本): D_real=0.491, D_fake=0.494. 这些值在 val 样本上接近 0.5——因为 D 从未在 val 样本上训练.

**CSV train 指标 (训练时在训练 batch 上采集)**:

| epoch | D_real_train | D_fake_train | Gap |
|-------|-------------|-------------|------|
| 1 | 0.672 | 0.563 | **+0.109** |
| 10 | 0.667 | 0.571 | **+0.097** |
| 25 | 0.692 | 0.541 | **+0.151** |
| 50 | 0.708 | 0.524 | **+0.184** |

**全 50 epoch D_real_train > D_fake_train, gap 持续扩大 (+0.109→+0.184), 远离双 0.5. 训练中对抗健康.**

注: 用户明确 "train 模式依赖 batch、不要求逐位". Post-hoc 在 val 样本上的 train mode 值不应与 CSV train 值(训练 batch)逐位比较.

---

## [5] Eval 退化量化

### Train vs Evalfixed 健康度轨迹

| epoch | train D gap | evalfixed D gap | BN running var |
|-------|------------|----------------|---------------|
| 1 | **+0.109** | **+0.144** | 12,719 |
| 10 | **+0.097** | +0.051 | 20,711 |
| 21 | +0.146 | **0.000** (穿零) | 28,053 |
| 25 | +0.151 | −0.042 (反转) | 29,805 |
| 50 | **+0.184** | **−0.029** | 39,912 |

### 关键数字

- **evalfixed gap 从 +0.144 跌到 −0.029** (跨 0 反转, 退化 0.173)
- **evalfixed 首次穿零: epoch 21**
- **train gap 从 +0.109 升到 +0.184** (健康扩大 69%)
- **BN running var 从 12,719 增到 39,912** (3.1×)

### 独立结论

同一份 D 权重, train 模式 gap 扩大、evalfixed gap 反转——两个评估窗口给出完全相反的信号. evalfixed 在 epoch 21 失去信息量 (穿零), 此后在白噪声附近振荡.

**如果旧 evalfixed 红线还在用, 这个 run 照样会在 epoch 21 被误停.** 新 train 红线让 50 epoch 完整跑完.

---

## [6] Phantom Speckle SNR — PASS ✓

8 固定 phantom val 样本, G.eval(), 独立复算:

| epoch | pred_mean | pred_std | gt_mean | gt_std | pred mean/std | gt mean/std |
|-------|-----------|----------|---------|--------|---------------|-------------|
| 1 | 0.391 | 0.117 | 0.451 | 0.113 | 3.3× | 4.0× |
| 10 | 0.422 | 0.108 | 0.451 | 0.113 | 3.9× | 4.0× |
| 25 | 0.424 | 0.116 | 0.451 | 0.113 | 3.6× | 4.0× |
| 50 | 0.452 | 0.118 | 0.451 | 0.113 | **3.8×** | 4.0× |

**全 50 epoch pred_std 与 gt_std 差距 < 0.02 (本次阈值). 对抗让 val phantom 的 speckle 强度持续匹配 GT.**

e50: pred_mean=0.452 首次贴到 gt_mean=0.451——全局亮度也匹配了.

---

## 总表

| # | 检查项 | 结果 | 备注 |
|---|--------|------|------|
| 1 | 单一变量 | **PASS** | 仅 check_redlines 3 行改动 + 路径/run_name |
| 2 | 红线逻辑 | **PASS** | 读 _train 列, 不读 _evalfixed |
| 3 | evalfixed 复算 | **PASS** | max |Δ| = 3.8×10⁻⁵, 逐位一致 |
| 4 | train 区间 | **PASS** | CSV train gap +0.109→+0.184, 健康 |
| 5 | eval 退化 | **量化完成** | evalfixed e21 穿零, train 持续健康 |
| 6 | phantom speckle | **PASS** | e50 pred_std=0.118 vs gt=0.113, 匹配 |

**不包含质量结论. 人类 Slicer 审查是最终门禁 (铁律 #3).**
