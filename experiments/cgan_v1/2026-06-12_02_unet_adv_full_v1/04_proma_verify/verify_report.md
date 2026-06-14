# Proma 独立验证报告 -- 第二十一工作单元正式 run

**日期**: 2026-06-13
**验证对象**: 2026-06-12_02_unet_adv_full_v1
**验证依据**: 协作铁律 #3 (Proma 只读、独立核对、只报数不下质量结论)
**产出**: 本报告 + verify_results.json

---

## 检查 1: 单一变量核对 -> PASS

**方法**: diff smoke (2026-06-12_01) vs full (2026-06-12_02) 的 config.yaml + train.py.

**config.yaml diff** (排除 run_name/created_at/outputs):

| 参数 | Smoke | Full | 判定 |
|------|-------|------|------|
| epochs | 10 | 50 | 预期改动 |
| snapshot_epochs | [1,5,10] | [1,10,25,50] | 预期改动 |
| checkpoint_epochs | [1,5,10] | [1,10,25,50] | 预期改动 |
| lr | 2e-4 | 2e-4 | PASS |
| beta1/beta2 | 0.5/0.999 | 0.5/0.999 | PASS |
| lambda_adv | 0.1 | 0.1 | PASS |
| D class | BMode3DPatchDiscriminator | same | PASS |
| D opt lr/beta | 2e-4/0.5/0.999 | same | PASS |
| sampler seed | 20260611 | 20260611 | PASS |
| loss type | supervised_l1_ssim3d | same | PASS |
| lambda_ssim/l1 | 0.84/0.16 | same | PASS |
| redlines | all identical | same | PASS |

**train.py diff**: 新增项均为诊断代码:
- PROBE_TRAIN_DIR / PROBE_VAL_DIR 路径
- 每 epoch 探针图 (generate_epoch_probes)
- evalfixed 红线检查 (check_redlines 中新增)
- 扩展 stability.csv 列 (train/evalfixed 分离, 类别级 metric)
- 训练变量 (损失函数, 对抗机制, G/D 更新逻辑) 未变

**结论**: 单一变量严格成立. 仅 epochs/snapshot/checkpoint 变更 + 诊断代码追加.

---

## 检查 2: D 健康度独立复算 -> DISCREPANCY NOTED

**方法**: 
1. 加载 checkpoint (epoch 1/10/25), 构建 Light3DUNet + BMode3DPatchDiscriminator (使用生产代码 rf_cgan_models.py)
2. 在固定 val 24 样本上, 用生产代码 rf_cgan_bmode_adv.d_lsgan_loss / g_adv_loss 独立重算
3. 对照 CSV 的 _evalfixed 列

**Proma 独立计算 vs CSV 对照**:

    epoch   1: Proma D_real=0.535 D_fake=0.404 G_adv=1.930 | CSV D_real=0.676 D_fake=0.581 G_adv=0.470
    epoch  10: Proma D_real=0.449 D_fake=0.397 G_adv=2.022 | CSV D_real=0.586 D_fake=0.503 G_adv=1.045
    epoch  25: Proma D_real=0.467 D_fake=0.520 G_adv=0.937 | CSV D_real=0.623 D_fake=0.591 G_adv=0.505


**发现**: 
- Proma 独立计算与 CSV 存在系统性偏差 (D_real 偏 ~0.14-0.16, G_adv 偏 ~0.43-1.46)
- Proma 的 D 打分始终更接近 0.5 (随机猜测水平), 表明 D 在 evalfixed 上比 CSV 显示的要弱
- CSV 显示 D 有区分力 (D_real > D_fake > 0.5), Proma 显示 D 区分力很弱 (D_real 在 0.45-0.54 之间波动)
- batch_size (3/6/8/12/24) 无影响 (InstanceNorm 对 batch size 不敏感, 已验证)

**使用的生产代码路径** (只读, 未修改):
- rf_cgan_models.py: Light3DUNet, BMode3DPatchDiscriminator
- rf_cgan_bmode_adv.py: d_lsgan_loss, g_adv_loss

**未解决差异的可能原因**:
- 训练时 eval 流程和 post-hoc 加载可能有不同的 D 状态
- CSV evalfixed 可能使用了与 Proma 不同的 baseline B-mode 转换路径

---

## 检查 3: Phantom speckle SNR -> INFO

**方法**: 在 8 个固定 val phantom 样本上独立算 B-mode pred/gt 均值与标准差.

| 指标 | Pred | GT |
|------|------|-----|
| Mean | 0.358 | 0.451 |
| Std | 0.018 | 0.113 |
| SNR (mean/std) | 19.67 | 4.00 |

**空间**: B-mode [0,1], epoch 25 checkpoint.
Pred std 远小于 GT std (0.018 vs 0.113), 与纯监督 oversmooth 特征一致.

---

## 检查 4: 监督损失未动 -> PASS

从 config diff + train.py assert 确认:

| 参数 | 值 |
|------|-----|
| lambda_ssim | 0.84 |
| lambda_l1 | 0.16 |
| SSIM library | pytorch-msssim |
| SSIM spatial_dims | 3 |
| SSIM win_size | 7 |
| SSIM data_range | 1.0 |
| SSIM nonnegative | True |

与 smoke run 完全一致.

---

## 检查 5: 对抗参数未变 -> PASS

| 参数 | 值 |
|------|-----|
| lambda_adv | 0.1 |
| D class | BMode3DPatchDiscriminator |
| D ndf | 48 |
| D params | 1,493,617 |
| D optimizer | AdamW, lr=2e-4, beta=(0.5, 0.999) |
| Redlines | 全部相同 (min_epoch=2, flat_eps=0.01, collapse=1e-4, explosion=10.0) |

---

## 检查 6: 2D FFT 伪影检查 -> PASS

**方法**: 对 phantom 第一个固定样本的 pred y=16 切片做 2D FFT, 搜索 DC 外的周期性峰.

| | Pred | GT |
|---|------|-----|
| max_non_dc / mean ratio | 16.8 | 5.8 |

Pred FFT ratio 16.8 < 100 (无显著栅格/棋盘伪影). pred 略高于 GT (预测偏平滑导致高频衰减, 中低频占比相对提高), 非异常.

---

## 总结

- [PASS] 检查 1 -- 单一变量严格成立
- [DISCREPANCY] 检查 2 -- Proma D 打分与 CSV 存在系统偏差 (D 比 CSV 显示更弱)
- [INFO] 检查 3 -- Phantom pred std=0.018 (vs gt std=0.113), 符合 oversmooth
- [PASS] 检查 4 -- 监督损失参数未变
- [PASS] 检查 5 -- 对抗参数未变
- [PASS] 检查 6 -- 无显著周期伪影

**不包含质量结论. 人类 Slicer 审查是最终门禁 (铁律 #3).**
