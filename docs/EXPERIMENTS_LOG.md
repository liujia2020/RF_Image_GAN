# 实验日志

最后更新：2026-06-05 01:57:34 +08:00

本文件记录决策级实验结果，不保存完整训练日志。完整日志、CSV、图和 NIfTI 应放在对应 run folder 或旧仓库产物目录中。

## 1. 早期回归 baseline

### `tiny_nonpoint_full_32`

- 数据：32×16×16 nonpoint full split，cache 训练。
- 模型：TinyResidualRFNet。
- 结果：test complex improvement 约 47.11%，abs improvement 约 49.62%。
- 作用：32 patch 普通 Tiny 对照，也是早期 dense32 full-volume 拼图来源。

### `tiny_random64_pilot80`

- 数据：64×32×32 pilot，train/val/test=57/12/12，未用 cache。
- 模型：TinyResidualRFNet。
- 结果：test complex improvement 约 51.39%，abs improvement 约 62.73%。
- 作用：验证 64×32×32 patch 方向可行。

### `tiny_random64_full1500`

- 数据：64×32×32 full1500，300 files × 5 patches，file-level 70/15/15 split。
- 模型：TinyResidualRFNet + InstanceNorm。
- 训练：曾因电脑重启中断，后从 checkpoint 继续。
- 结果：test complex improvement 47.32%，abs improvement 52.50%。
- 问题：full-volume seam 约 1.46x label，拼接仍有明显周期缝。

## 2. BN 修复缝根因 #1

### `tiny_bn_random64_full1500`

- 数据：同 full1500。
- 改动：TinyResidualRFNet 中 InstanceNorm3d 改为 BatchNorm3d。
- 结果：test complex improvement 56.85%，abs improvement 51.85%。
- full-volume：Carotid_012 上 seam 从 IN 约 1.46x 降到 BN 约 0.535x。
- 13 卷诊断：carotid5 / muscle4 / phantom4 上 BN seam 全面压低。
- 判决：BatchNorm 修复 InstanceNorm per-patch 统计不连续，是范式无关的有效工程结论。

phantom 备注：

- BN 在 phantom 上有一致的轻微细节损失。
- 当时判为可接受 trade-off。
- 后续回归范式被 Slicer 否决后，此结论只作为 BN 行为记录，不作为最终质量结论。

## 3. UNet 对照

### `unet_random64_full1500`

- 数据：同 full1500。
- 模型：UNetResidualRFNet，约 9.08M 参数。
- 结果：test complex improvement 约 45.34%，abs improvement 约 51.09%。
- 判决：未胜 Tiny。patch 指标和 full-volume seam 都没有改善。
- 教训：更大感受野/更复杂网络不自动解决拼接质量。

## 4. 动态范围线程

### abs_weight sweep

- 扫描：`abs_weight = 0.1, 0.5, 1.0, 2.0`。
- 观察：幅度欠射没有被有效修复，slope 基本钉死。
- 视觉：部分高频/detail 指标像是被细麻点纹理刷分，图像反而更平。
- 判决：在 L1 家族里加大幅度惩罚不是根本解。

### SSIM 损失

- 损失：complex L1 + 0.1 envelope L1 + 1.0 envelope SSIM。
- 结果：首次明显撬动 slope、abs_std、p99 等动态范围代理指标。
- 反作弊检查：2D hist 中 slope 变陡且 Pearson r 不降反升，误差自相关未变白。
- 当时判读：不是简单灌噪声。

### WideDeep + SSIM

- 模型：WideDeepResidualRFNet，约 3.74M 参数。
- 效果：卷级 p99/deep p99 代理指标继续改善。
- 问题：深模型带来 patch 边缘 zero-padding 污染，seam 回退。
- 修复：crop8 / overlap / halo 在推理协议层面可压 seam，且不牺牲动态范围代理指标。

### NIfTI + Slicer 人眼否决

- 导出：未压缩 `.nii`，spacing z=0.0362 mm，x/y=0.2 mm。
- 用户在 3D Slicer 人眼检查。
- 发现：
  - phantom 线靶在 pred 中丢失，label 有而 pred 无。
  - 组织 speckle 形态差，呈均值回归 mush。
- 判决：
  - 推翻“动态范围线程已收官”。
  - 回归 L1/SSIM 只改善代理指标，不能达到 DAS 真实 speckle。
  - 根因是回归逼近条件均值，随机 speckle 被平均。
  - 转向 conditional GAN。

## 5. cGAN 新路线

### `2026-06-05_smoke`

- 仓库：`RF_Image_GAN`。
- 分支：`codex/cgan-v1`。
- run folder：`experiments/cgan_v1/runs/2026-06-05_smoke/`。
- 数据：旧仓库 `Data_cache_random64_full1500/train` 前 100 个 patch。
- 注意：这 100 个 patch 当前全是 carotid；仅用于 smoke，不是正式均衡训练。
- Generator：`TinyResidualRFNet`，随机初始化，不加载旧 checkpoint。
- Discriminator：`Envelope2DPatchDiscriminator`，2D envelope PatchGAN。
- Loss：LSGAN + `10 * complex_L1`。
- 训练：20 epoch，batch_size=2，AMP 开启。
- 结果：`SMOKE_STATUS: PASS`。
- 作用：证明最小 cGAN 训练环路可跑，无 NaN/OOM/单边崩溃。
- 限制：不评价图像质量，不触发验证关卡。

## 6. 当前开放问题

1. cGAN pilot 的类别均衡采样如何设计。
2. fidelity decomposition 如何避免再次把 speckle 拉成均值。
3. 判别器看 2D envelope 是否足够，何时升级到 3D 或多尺度。
4. 如何在防幻觉和真实 speckle 之间平衡。
5. 正式 run 的 ROI、NIfTI 和 Slicer 验证材料如何提前冻结。
