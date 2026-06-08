# 2026-06-05_pilot_phase1

创建时间：2026-06-05 +08:00

状态：`phase2a_training_enabled`

## 目的

这是 cGAN pilot 的 Phase 2a 正式 pilot run。目标只做 loss 设计体检，不做 Slicer，不下质量结论。

## 单变量假设

将强 voxel-wise complex L1 fidelity 替换为结构低通 envelope fidelity + 弱 carrier fidelity，可以减少回归到均值的风险，同时保留 DAS 的确定性结构。

## 当前限制

- 等待训练或训练中。
- 未产出 checkpoint。
- 未触发验证关卡。
- `validation_rois` 仍为空，占位等待人工 Slicer ROI 冻结。
- 本 run 不导出整卷 NIfTI，不宣布质量。

## 诊断输出

Phase 1 / Phase 2a 诊断输出位于本 run 的 `logs/` 目录。
