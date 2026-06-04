# cGAN 方案计划

最后更新：2026-06-05 01:57:34 +08:00

## 1. 当前判决

原来的回归路线已经冻结为负结果。WideDeep + SSIM + crop8 虽然改善了 slope、p99、abs_std、SSIM 等代理指标，但导出 NIfTI 后经 3D Slicer 人眼检查失败：phantom 线靶在预测中丢失，组织 speckle 形态变成均值回归式的 mush。

因此项目从“纯回归拟合 DAS”转向“成对 conditional GAN”。本文只是计划文档，不代表已经允许正式训练。

## 2. 最终目标

目标不是刷代理指标，而是让神经网络从 delay-aligned 多角度 RF tensor 直接重建出质量对标 DAS 的复数 RF volume。输出必须同时满足：

- 组织和仿体 speckle 形态接近 DAS。
- 血管壁、组织界面、仿体结构等相干结构不能丢。
- 点靶/线靶如果在 label 中可见，预测中也不能消失。
- 完整 volume 拼接后不能出现 patch-grid 周期缝。

任何“质量达标”结论都必须走 `docs/VALIDATION_PROTOCOL.md` 的验证关卡。Codex 只能产出材料和数字，不能替代用户做 Slicer 视觉签收。

## 3. 为什么是成对 conditional GAN

我们有成对数据：RF 输入和对应 DAS label。因此正确的生成式家族是 paired conditional GAN：

- 生成器 G 以 RF input 作为条件，必要时也接收 baseline。
- 判别器 D 判断“在同一个条件下，这个输出像不像 DAS”。
- 保真项约束 G 不要脱离当前 RF frame，避免幻觉。
- 对抗项负责逼出真实 speckle 分布。

不用无条件 GAN，因为它只会生成“看起来像 DAS”的随机样本，不绑定这份 RF。也不用 CycleGAN，因为我们有配对监督，没必要丢掉最强信号。

## 4. 第一阶段范围

第一阶段只做最小可行 cGAN 路线：

- 主攻 tissue/phantom 的真实 speckle。
- 保持 paired RF-to-DAS 重建设定。
- 保持完整 volume stitch 可验证。
- 每个正式 run 必须产出 NIfTI、标准指标、固定对比图、verdict。

暂不作为第一阶段主目标：

- `simu_point` 点靶数据作为主要训练类。
- 扩散模型。原因是 3D volume + 本地 8GB GPU 目前不现实。
- 任何不经 Slicer 的质量结论。

点靶仍是 sanity check 和后续分支，不是第一轮 GAN 的主攻对象。

## 5. 从回归线继承的可复用资产

继续复用：

- `RFLearningDataset`、`RFCachedDataset`、cache builder 思路。
- BatchNorm 修复缝根因 #1 的经验。
- startup check 和配置自检纪律。
- crop8/halo、coverage check、full-volume stitch 管线。
- NIfTI 导出和验证关卡。
- 回归 checkpoint 作为 baseline 和负结果证据。

不要继续作为最终目标：

- 纯 L1/SSIM 回归。
- 只看 slope、p99、SSIM、abs_std 等代理指标就判质量。

## 6. 架构候选

### 生成器 G

第一版先保守：

- 输入：`input [B,1536,Z,X,Y]`。
- 可选条件：`baseline [B,2,Z,X,Y]`。
- 输出：`pred [B,2,Z,X,Y]`。
- 本地先用小 batch + AMP 做 smoke test。

后续可以尝试更大 generator，但必须先过 smoke，再做 pilot。

### 判别器 D

第一版为了省显存，先用 2D envelope PatchGAN：

- 从复数 volume 取 envelope。
- 取中间 y 切片，得到 `[B,1,Z,X]`。
- D 接收 candidate envelope 和 baseline envelope 拼接后的 `[B,2,Z,X]`。

如果 2D D 稳定，再考虑：

- 多切片 D。
- 3D patch D。
- 多尺度 D。

## 7. 损失设计原则

第一版 smoke test 使用：

```text
G_loss = adversarial_loss + lambda_fidelity * complex_L1
```

注意：正式方案不能简单把 voxel-wise L1/SSIM 加大。原因是：

- 随机 speckle 无法逐 voxel 精确预测。
- 对随机 speckle 做强 L1，会把 G 拉向条件均值，重新产生 mush。
- SSIM 的协方差项要求 pred 与 label speckle 相关，这对随机 speckle 不合理，也会把结果拉向均值。

正确方向是分解保真目标：

- 对确定性/可学习部分施加强保真：血管壁、组织界面、仿体结构、大尺度 envelope trend、点/线靶位置。
- 对 fine-scale 随机 speckle 主要交给对抗项匹配分布。

因此，lambda 只是次要变量；“保真项到底作用在什么成分上”才是核心设计变量。

## 8. 风险清单

按当前 cGAN 设定，最可能的问题依次是：

1. 保真项太强，输出回到 mush。
   - 信号：G_adv 不动，speckle SNR 仍偏离 label，histogram/KS 不改善。

2. 生成真实 speckle 但结构错位或幻觉。
   - 信号：low-pass envelope 与 label 的结构相关性低于 BN baseline；label 中可见结构在 pred 中消失。

3. speckle 变真实，但结构保真下降。
   - 信号：speckle 指标接近 label，但 vessel wall/phantom structure 被破坏。

4. patch 级看起来正常，full-volume stitch 失败。
   - 信号：完整 volume 上 amplitude seam 或 texture seam 变差。

5. 代理指标改善但人眼失败。
   - 这是回归线已经发生过的事故。验证关卡是唯一仲裁。

每个正式 run 的 `verdict.md` 必须逐条记录这些风险的检测结果。

## 9. 当前 smoke test

当前已建立最小 smoke run：

```text
experiments/cgan_v1/runs/2026-06-05_smoke/
```

目的只是一件事：确认 cGAN 训练环路在本地 GPU 上能跑通，不 NaN、不 OOM、不发生单边崩溃。

它不是正式实验，不评价图像质量，也不触发验证关卡。

## 10. 下一步

1. 根据 smoke test 结果决定是否进入 pilot。
2. pilot 前先制定类别均衡采样策略。
3. 设计正式 generator/discriminator/loss 版本。
4. 每个正式 run 都必须按 `RUNBOOK.md` 建 run folder，冻结 config。
5. 每个正式质量结论都必须按 `VALIDATION_PROTOCOL.md` 输出 NIfTI 和标准指标，并由用户做 Slicer 签收。
