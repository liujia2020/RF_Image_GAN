# 第十五工作单元提前停止摘要

时间：2026-06-08 16:25 +0800

- 状态：用户要求提前停止
- 最后完整 epoch：25
- 原计划 epoch：50
- NaN/Inf：未出现
- carrier：已归一化，复用 struct 的同一个 label 派生尺度 `s`
- `G_carrier_raw`：归一化后 O(1)
- `G_carrier_unnormalized_raw`：未归一化复数 L1，仅用于记录
- 已产出：epoch10、epoch25 三类 triplet 图；checkpoint epoch10、20

## 最后 epoch 关键指标

```text
D_fake_score = 0.50008
D_real_score = 0.73097
G_adv_weighted = 1.00012
G_struct_weighted = 1.07906
G_carrier_weighted = 0.36411
G_total = 2.44329
pred_max_abs = 280509.56
```

## 观察边界

本摘要只记录训练健康事实，不做图像质量结论。质量仍需后续 NIfTI 与用户 3D Slicer 签收。
