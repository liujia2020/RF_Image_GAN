# 2026-06-12_02_unet_adv_full_v1

第二十一工作单元：B-mode 对抗正式 50-epoch run。

## 来源

本 run 基于 `2026-06-12_01_unet_adv_smoke_v1` 复制。对抗机制、数据、sampler、seed、监督损失、D/G 优化器与 smoke 保持一致。

## 唯一训练变量

- `epochs`: `10 -> 50`
- `snapshot_epochs`: `[1, 10, 25, 50]`
- `checkpoint_epochs`: `[1, 10, 25, 50]`

## 新增诊断

- 每 epoch 固定 val 集 `D.eval()` 对账健康度，列名后缀 `_evalfixed`
- 训练内 `D.train()` 健康度保留为参考，列名后缀 `_train`
- 每 epoch train/val 合并探针图：`02_train/probes/train` 与 `02_train/probes/val`
- snapshot epoch 的 phantom B-mode 空间 mean/std 表
- epoch 1 与 epoch 50 的 phantom/muscle pred 切片 2D FFT 频谱图
- 固定 train/val 每类 loss/SSIM/L1/pred_std 曲线

## 质量边界

本 run 只报数和产物，不做质量结论。正式质量判断留给 `03_validate` 和人类 Slicer 审查。
