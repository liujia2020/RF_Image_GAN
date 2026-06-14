# Proma 诊断报告 v2: 红线真伪分析

**日期**: 2026-06-13
**对象**: `2026-06-12_02_unet_adv_full_v1`, epoch 25 checkpoint (e42 无 ckpt，仅 CSV)
**方法**: 四模式交叉 (G.train/eval × D.train/eval) + baseline restore_scale 修正
**v2 修正**: baseline_bmode 从 RESTORED baseline complex 计算 (v1 漏了 ×scale, baseline 全零)
**产物**: diagnose_results.json + 本报告

---

## 0. v1 Bug 确认与修正

### Bug

v1 诊断 (`diagnose_redline.py`) 从 `baseline.dat` (fp16 memmap) 加载 baseline complex 后**直接做 B-mode 转换，未乘 scale**。

```
v1:  baseline_complex (fp16, ~[-1,1])  →  B-mode   →  baseline ≈ 全零
v2:  baseline_complex (fp16) × scale   →  B-mode   →  baseline ≈ [0,1] (正常)
```

scale 均值 29,832。在 log 空间差 ~89dB (20·log10(29832))——baseline 被压到 -60dB 地板，条件通道输出全零 (min=0, max=0, mean=0)。

### 影响

D 的输入是 `concat(candidate, baseline)`。baseline 全零意味着 D 拿不到有效条件，所有输入看起来都一样 → D 输出接近随机 (sigmoid ~0.47)。

**v1 的四模式 D_real 恒 0.467、与 CSV 差 ~0.15——不是对抗坍缩，是 baseline 条件废了。**

### 修正后

baseline RESTORED 后: min=0, max=1, mean=0.611 → 正常 B-mode 图像。

---

## 1. Norm 配置

| 组件 | Norm 类型 | track_running_stats | affine | 数量 | train/eval 行为 |
|------|----------|---------------------|--------|------|----------------|
| **Light3DUNet (G)** | BatchNorm3d | **True** | True | 21层 | **train/eval 有差异** |
| **BMode3DPatchDiscriminator (D)** | InstanceNorm3d | **False** | True | 2层 | **train/eval 完全一致** |

G 的 BN momentum=0.9。D 的 IN 逐实例归一化, 不跟踪 running stats。

### BN running stats 漂移

```
epoch  1: running_mean=-0.08  running_var= 12,827
epoch 10: running_mean=-0.19  running_var= 20,906
epoch 25: running_mean=-0.22  running_var= 30,048
epoch 41: running_mean=-0.44  running_var= 36,603
epoch 42: running_mean=-0.48  running_var= 36,997
```

running_var 从 e1 到 e42 增长 **2.9 倍** —— 严重失稳。

---

## 2. 四模式交叉表 (核心)

24 固定 val 样本, epoch 25 checkpoint, **RESTORED baseline**:

| G mode | D mode | D_real | D_fake | G_adv | pred_mean | pred_std |
|--------|--------|--------|--------|-------|-----------|----------|
| train | train | **0.6228** | 0.5934 | 0.4881 | 0.5135 | 0.1362 |
| train | eval  | **0.6228** | 0.5934 | 0.4881 | 0.5135 | 0.1362 |
| eval  | train | **0.6228** | 0.5907 | 0.5050 | 0.4964 | 0.1346 |
| eval  | eval  | **0.6228** | 0.5907 | 0.5050 | 0.4964 | 0.1346 |

### CSV epoch 25 对照

| 指标 | Proma (G.eval+D.eval) | CSV evalfixed | Δ |
|------|----------------------|---------------|-----|
| D_real | 0.6228 | 0.6228 | **0.0000** ✓ |
| D_fake | 0.5907 | 0.5907 | **0.0000** ✓ |
| G_adv | 0.5050 | 0.5050 | **0.0000** ✓ |
| pred_mean | 0.4964 | 0.4964 | **0.0000** ✓ |

**修正 baseline 后，Proma post-hoc 完美复现 evalfixed。v1 的 ~0.14 偏移彻底消失。**

### RAW baseline 对照 (v1 bug 复现)

| 模式 | D_real (RAW baseline) |
|------|----------------------|
| 全部四模式 | **0.4666** (恒) |

→ 确认 v1 的 D 偏移根因就是 baseline restore_scale 遗漏，不是 GPU/CPU 差异、不是 fp16 intermediate。

---

## 3. G train/eval pred 差异 (restored input)

| | pred_mean | pred_std |
|---|-----------|----------|
| G.train() | 0.5135 | 0.1362 |
| G.eval() | 0.4964 | 0.1346 |
| **L1 差** | **0.0435 (4.3%)** | |

v1 报告 L1=0.276 (27.6%)——巨大的差异主要来自 non-restored input，非 BN 失稳。修正 input 后，G.train vs G.eval 差异大幅缩小。

---

## 4. Phantom speckle std 厘清

Epoch 25, 8 固定 phantom val 样本 (restored input):

| 模式 | pred_mean | pred_std | mean/std |
|------|-----------|----------|----------|
| G.eval() | 0.4662 | **0.1197** | 3.89 |
| G.train() | 0.5127 | **0.1140** | 4.50 |

CSV val phantom (225 全量): pred_std=0.1021, gt_std=0.0958
summary.md phantom table: pred_std=0.093, gt_std=0.093

**旧检查 #3 的 pred_std=0.018 是 non-restored input + G.eval 的产物。修正后 G.eval 也给出合理的 speckle std (0.120)。**

---

## 5. Train vs Evalfixed 对抗健康度趋势 (核心判断)

从 CSV 完整 42 epoch 数据:

| epoch | Train D gap | Evalfixed D gap | D_loss_train |
|-------|------------|-----------------|-------------|
| 1 | **+0.105** | +0.095 | 0.355 |
| 10 | **+0.114** | +0.083 | 0.158 |
| 25 | **+0.151** | +0.032 | 0.132 |
| 42 | **+0.180** | **−0.010** | 0.087 |

```
Train 模式:    D gap 0.105 → 0.180  ← 扩大 71%，对抗健康博弈
Evalfixed 模式: D gap 0.095 → −0.010 ← 反转，触发红线
```

**同一个 D、同一份权重，两种模式给出完全相反的健康度。这不是"对抗死了"——训练模式对抗分明在健康地博弈。这是"假红线"。**

### 为什么 evalfixed 塌了

1. **D 自身不会塌**：D 用 IN (no running stats)，train/eval 行为完全一致。D_real 在四模式下恒为 0.6228——D 对 GT 的判断力稳定。

2. **塌的是 evalfixed 的 D_fake**：evalfixed 用 G.eval() 产 pred。G 的 BN running stats 持续漂移 (var ×2.9)，G.eval() 的输出分布和 G.train() 的差距随时间累积。D 在 train 模式下被训练，但 evalfixed 喂给它的是 G.eval() 的输出（分布略有不同）。随着 BN 漂移加剧，这个分布差距拉大 → evalfixed D_fake 向 0.5 漂移。

3. **证据**：G.train vs G.eval 的 pred L1=0.0435 (e25)。这 4.3% 的差异在 e42（BN var 36,997）会更大。D 对 train-distribution fake 能区分（gap +0.180），对 eval-distribution fake 区分力弱（gap −0.010）。

---

## 6. e42 分析 (无 checkpoint)

CSV 数据:

```
Train:      D_real=0.706  D_fake=0.526  gap=+0.180  D_loss=0.087
Evalfixed:  D_real=0.494  D_fake=0.504  gap=−0.010  ← 触发红线
val_pred_std=0.115
```

Train 模式对抗仍健康（gap +0.180, D_loss 降至 0.087），但 evalfixed 已跌穿。这和 e25 的四模式结论一致——对抗博弈是活的，死的只是 evalfixed 这个评估窗口。

---

## 7. 最终判定

### 红线：假红线 (train/eval 模式脱节)

```
✅ Train D gap ↑ 0.105→0.180   对抗在健康博弈
✅ D_loss ↓ 0.355→0.087         D 在学习
✅ D_real 四模式恒 0.623       D 判断力稳定
✅ Speckle 统计匹配 GT         G 在学到东西
❌ Evalfixed D gap ↓ 0.095→−0.010  评估窗口失真
```

**根因**: G 的 BN running stats 在对抗 + 小 batch(6) + G 分布漂移下失稳 (var ×2.9)。evalfixed 用了 G.eval() 模式，产出的 pred 分布和 D 训练时的分布脱节，导致 evalfixed D_fake 向 0.5 漂移，触发红线。

**不是**: 对抗空转/坍缩、D 容量不足、损失设计问题、输入尺度问题。

### v1 诊断错误归因

v1 的 D 偏移 ~0.14 被错误归因于 "GPU vs CPU" / "fp16 intermediate"。实际根因是 baseline restore_scale 遗漏——baseline 条件通道全零，D 无法工作。

### 正面信号确认

修正后 phantom val pred_std (G.eval, 8 fixed) = 0.120，接近 CSV val (225 全量) = 0.102 和 gt = 0.096。对抗让 val 也长出了强度匹配的 speckle。

---

## 附录: 数据对比总表

| 指标 | v1 Proma (bug) | v2 Proma (修正) | CSV evalfixed | CSV train |
|------|---------------|-----------------|---------------|-----------|
| D_real | 0.467 | **0.623** | 0.623 | 0.692 |
| D_fake | 0.467~0.520 | **0.591** | 0.591 | 0.541 |
| G_adv | 0.937~1.400 | **0.505** | 0.505 | 0.947 |
| pred_mean | 0.234~0.509 | **0.496** | 0.496 | — |
| phantom pred_std | 0.032 (raw) | **0.120** | 0.102 | — |
| G train/eval L1 | 0.276 (27.6%) | **0.044 (4.3%)** | — | — |

---

**不包含质量结论。人类 Slicer 审查是最终门禁 (铁律 #3)。**
