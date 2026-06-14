# 2026-06-11_02_unet_supervised_v2

第十九工作单元：用 Light3DUNet 替换 TinyBModeRFNet，继续 B-mode 纯监督路线。

## 目的

上一轮 TinyBModeRFNet 失败，核心问题是感受野约 5 体素、无编码-解码、无多尺度特征。本 run 的唯一目标是验证轻量 3D U-Net 架构能否让预测脱离灰雾，并改善 train/val SSIM 和 L1 趋势。

## 数据

- 输入 RF cache: `/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607`
- B-mode GT: `/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_bmode_gt_ref64407p58_20260610`
- B-mode 全局参照: `REF=64407.58`

## 训练约束

- 模型：`Light3DUNet`，约 2.11M 参数，3 级各向异性 stride-conv 下采样 + trilinear 上采样 + skip。
- batch：`samples_per_category=2`，有效 batch size 6。
- 损失：`0.84*(1-SSIM3D)+0.16*L1`，pytorch-msssim 单尺度 3D SSIM，window=7。
- AMP 关闭，`num_workers=0`，无对抗、无判别器、无 carrier。
- stability.csv 记录 loss/SSIM/L1、pred mean/std、固定 val 指标和 BN running stats 汇总。

## 质量边界

本 run 只报告训练事实、代理指标和三联图，不做图像质量结论。质量结论仍需后续 3D Slicer 人类审查。
