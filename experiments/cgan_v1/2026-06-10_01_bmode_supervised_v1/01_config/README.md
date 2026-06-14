# 2026-06-10_01_bmode_supervised_v1

第十八工作单元：网络输出单通道 B-mode，使用纯监督 `0.84*(1-SSIM3D)+0.16*L1` 基线。

## 数据

- 输入 RF cache：`/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607`
- B-mode GT：`/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_bmode_gt_ref64407p58_20260610`
- B-mode 全局参照：`REF=64407.58`

## 训练约束

- 无对抗、无判别器、无 carrier。
- 模型输出单通道 `[B,1,64,32,32]`，末端 sigmoid，值域 `[0,1]`。
- baseline 仅转换为 B-mode 用于三联图对照，不参与损失。
- SSIM 使用 `pytorch-msssim` 的单尺度 3D SSIM，不手写。

## 质量边界

本 run 只报告训练事实和代理指标，不做图像质量结论。质量结论仍需后续 3D Slicer 签收。
