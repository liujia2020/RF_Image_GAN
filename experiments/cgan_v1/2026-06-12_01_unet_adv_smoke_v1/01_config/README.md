# 2026-06-12_01_unet_adv_smoke_v1

第二十工作单元：在 Light3DUNet B-mode 监督基线上加入轻量 3D 条件判别器，做 10 epoch smoke。

## 唯一新增变量

相对 `2026-06-11_02_unet_supervised_v2`，本 run 冻结生成器架构/初始化、监督损失、优化器、cache、batch 和 seed，只新增：

- `BMode3DPatchDiscriminator`
- LSGAN adversarial loss
- `lambda_adv=0.1`

## 冻结项

- G：`Light3DUNet`
- G optimizer：Adam, lr=`2e-4`, betas=`(0.5, 0.999)`
- 监督损失：`0.84*(1-SSIM3D)+0.16*L1`
- `samples_per_category=2`，有效 batch size 6
- AMP 关闭，`num_workers=0`
- RF cache 和 B-mode GT 路径不变

## Smoke 红线

命中以下任一项立即停止并报回：

- NaN/Inf
- VRAM 超过 6GB
- sigmoid(D_fake) 恒定约 0.5 且 D_real 也约 0.5
- D_loss 塌到 0 且 G_adv 爆涨

## 质量边界

本 run 只报告对抗健康度、代理指标、分类别 pred/GT std 和三联图，不做图像质量结论。
