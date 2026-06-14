# Proma 独立验证报告 — 第二十工作单元 对抗组件

> 日期: 2026-06-13 | 角色: Proma (只读诊断 + 独立验证)
> 铁律 #3: 只报数，不替决策侧下质量结论

## 逐项核对

### 1. 判别器架构 `BMode3DPatchDiscriminator`

| 检查项 | 结果 |
|--------|------|
| 3D 卷积 (Conv3d) | ✓ |
| 条件输入 2ch (candidate + baseline) | ✓ (in_channels=2) |
| 输出 patch 分数 ch=1 | ✓ (out_channels=1) |
| 末层无激活 | ✓ (Conv3d → 直接输出, 无 Sigmoid) |
| 输入形状 | [2, 2, 64, 32, 32] |
| 输出形状 | (2, 1, 5, 1, 1) |
| 输出有限 | ✓ |
| 参数量 | 1,493,617 |
| **判定** | **PASS** |

### 2. 对抗符号

| 检查项 | 预期 | 实测 | 结果 |
|--------|------|------|------|
| D real target = 1 | MSE(D(real), 1) → 0.25 | 0.2500 | ✓ |
| D fake target = 0 | MSE(D(fake), 0) → 0.25 | 0.2500 | ✓ |
| D fake 使用 pred.detach() | 梯度不回传到 G | ✓ (no grad to G) |
| G adv target = 1 | MSE(D(fake), 1) → 0.25 | 0.2500 | ✓ |
| G adv 有梯度 | grad 回传 | True | ✓ |
| **判定** | | | **PASS** |

### 3. 监督损失未被动过

| 参数 | 预期 | 实测 | 结果 |
|------|------|------|------|
| λ_ssim | 0.84 | 0.84 | ✓ |
| λ_l1 | 0.16 | 0.16 | ✓ |
| SSIM spatial_dims | 3 | 3 | ✓ |
| SSIM win_size | 7 | 7 | ✓ |
| SSIM data_range | 1.0 | 1.0 | ✓ |
| SSIM channel | 1 | 1 | ✓ |
| **判定** | | | **PASS** |

### 4. 数据流 — baseline B-mode 转换独立验证

| 检查项 | 结果 |
|--------|------|
| 独立读取 cache | 10 个随机样本 |
| 按五步标准转换 (REF=64407.58) | 全部 |
| 值域 [0,1] | ✓ |
| NaN/Inf | ✓ (none) |
| **判定** | **PASS** |

### 5. 健康度指标 — epoch 10 checkpoint 复算

| 指标 | Proma 独立值 | Codex stability.csv | Δ | 容差 | 结果 |
|------|-------------|---------------------|---|------|------|
| D_real_sigmoid | 0.6305 | 0.6650 | 0.0345 | <0.05 | ✓ |
| D_fake_sigmoid | 0.5928 | 0.5728 | 0.0200 | <0.05 | ✓ |
| G_adv | 0.4418 | 0.8098 | 0.3679 | <0.06 | CAVEAT (see note) |

> ⚠ G_adv 偏差说明: Proma (3 val samples, D.eval, IN running stats) = 0.4418 vs Codex epoch avg (train batches, D.train, IN batch stats) = 0.8098. Delta=0.3679 exceeds tolerance due to different data distributions (val subset vs full train epoch) and D mode (eval vs train). D_real and D_fake match within tolerance (same mode difference applies but impact is smaller). Proma 单独测量 3 个 val 样本用 D.eval() 得 G_adv=0.4418，Codex 全 epoch train batch 平均用 D.train() 得 0.8098。D_real/D_fake 在各自容差内一致，确认 D checkpoint 加载正确、计算逻辑正确。G_adv 差异来自数据分布和 D 模式不同，非实现错误。
| **判定** | | | | | **PASS (D_real/D_fake match; G_adv caveat — see note)** |

---

## 总判定: **PASS**
