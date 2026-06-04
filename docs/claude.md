# RF-to-Volume 项目：协作方式与决策纪律

最后更新：2026-06-05 01:57:34 +08:00

这份文档写给新接手的 Claude / Claude Code / Codex。它不重复所有实验细节，而是记录协作分工、思维纪律和关键决策链。

## 1. 三方角色

### 用户

- 项目负责人。
- 唯一能做最终人眼视觉签收的人。
- 在 Claude 和 Codex 之间传递关键判断。
- 当代理指标和图像观感冲突时，用户的 Slicer 人眼检查是最终否决权。

### Claude / Claude Code

- 决策者和审核者。
- 负责方案设计、风险判断、实验优先级、结果解释。
- 可以给 Codex 下明确执行指令。
- 不直接修改项目文件，除非用户明确改变分工。

### Codex

- 执行者。
- 负责写代码、跑实验、导出 CSV/图/NIfTI、更新文档、commit。
- 可以指出风险和事实矛盾。
- 不独立宣布图像质量通过。

## 2. 协作原则

1. 决策和执行分离。
   - Claude 做策略判断。
   - Codex 做工程执行。
   - 用户做最终审核。

2. 文件是共同事实源。
   - 重要结论必须写进 `docs/`。
   - 每次正式训练必须有 run folder。
   - 不靠聊天记忆维持项目状态。

3. 失败必须被记录。
   - 失败不是丢脸，是路线判断依据。
   - 不要把失败写成“还需优化”。

## 3. 必守思维纪律

1. 一次只改一个主变量。
   - 换 patch size 时不要同时换网络。
   - 换 loss 时不要同时换数据。

2. 先做廉价诊断，再做高成本训练。
   - 先跑 small split / smoke / dry-run。
   - 再决定是否跑长训练。

3. patch 级指标不能判断 full-volume 拼接质量。
   - block seam 必须在完整 volume 上看。
   - seam 指标和 Slicer 图都要有。

4. Jupyter 状态残留是高风险源。
   - 正式训练必须 Restart & Run All。
   - startup check 必须打印真实模型类名、experiment_name、checkpoint 路径、参数量。

5. 视觉会被显示参数欺骗。
   - dB ref、db_min、aspect 都会改变观感。
   - 动态范围要看线性图、p99、slope、abs_std。
   - 几何比例必须使用真实 spacing。

6. 代理指标可能被刷分。
   - slope、SSIM、HF、detail_top10 都可能“看起来进步”但图像失败。
   - 必须用不可作弊指标和人眼图像交叉仲裁。

7. 质量结论必须过验证关卡。
   - NIfTI / 3D Slicer 人眼签收是硬门槛。
   - 人眼 fail，则该 run fail；指标不能翻盘。
   - Codex 只产出材料，不判质量。

8. 兴奋时先验证。
   - 指标突然很好，先怀疑是否被显示、归一化、ROI、代理指标骗了。
   - 先做反作弊检查，再写“收官”。

## 4. 缝根因决策链

历史上 sliding-window full-volume 出现明显 block seam。最终拆成两个根因：

### 根因 #1：InstanceNorm per-patch 统计不连续

- InstanceNorm3d 在每个 patch 内独立统计。
- patch 与 patch 之间归一化基准不同。
- 换成 BatchNorm3d 后，使用训练集 running stats，跨 patch 统计一致。

结果：

- `tiny_bn_random64_full1500` 显著降低 seam。
- 13 卷 dense64 诊断中 carotid/muscle/phantom seam 全面压低。
- 这是范式无关的有效工程结论，GAN 后续仍应保留 BN 经验。

### 根因 #2：数据 / baseline 本身空间不连续

- baseline 在 patch 边界处仍有约 2.8-3.0x label 的跳变。
- 但 BN 后 pred 层面没有明显传染到最终输出。
- 因此 #2 在 BN-L1 下被降级。

### 深模型边缘污染

WideDeep + SSIM 后发现新的 seam：

- jump profile 在 x=32/64/96 有周期峰。
- 原因符合深模型 zero-padding 边缘污染。
- crop8 / overlap / halo 能在推理协议层面修复。

这个经验对 GAN 仍重要：正式 full-volume 推理要考虑 halo/crop。

## 5. 动态范围线程最终判决

回归路线曾尝试：

- abs_weight sweep。
- envelope SSIM。
- WideDeep 增加容量。
- crop8/halo 修复 deep model seam。

代理指标一度改善：

- slope 变陡。
- p99 接近 label。
- abs_std 接近 label。
- deep z 动态范围改善。

但 NIfTI + 3D Slicer 人眼检查否决：

- phantom 线靶丢失。
- 组织 speckle mush。

最终判断：

- 纯回归 L1/SSIM 拟合 DAS 已到范式天花板。
- 回归逼近条件均值，无法生成真实随机 speckle。
- 该路线归类为负结果，作为转 cGAN 的证据。

## 6. 为什么转 paired conditional GAN

目标是对标 DAS，而 DAS speckle 是随机纹理分布问题。纯回归会平均随机成分，生成 mush。

我们有成对数据：

```text
RF input <-> DAS label
```

所以应使用 paired conditional GAN：

- 对抗项：逼近 DAS speckle 分布。
- 保真项：锁住当前 RF frame 的结构，防止幻觉。

不用无条件 GAN，因为它不是重建。不用 CycleGAN，因为我们不缺配对数据。

## 7. cGAN 设计特别警惕

最大风险不是 mode collapse，而是 fidelity 太强，把输出重新拉回 mush。

因此正式方案必须区分：

- 确定性结构：血管壁、界面、仿体结构、大尺度 envelope trend，需要保真。
- 随机 speckle：不能逐 voxel 强行 L1，对抗项负责分布匹配。

如果保真项设计错误，GAN 也会退化成回归。

## 8. 当前工程状态

新仓库：

```text
/home/liujia/RF_Image_GAN
```

分支：

```text
codex/cgan-v1
```

已完成：

- cGAN 项目结构。
- 中文文档。
- `Envelope2DPatchDiscriminator`。
- smoke run `2026-06-05_smoke`。
- smoke test 已跑通：`SMOKE_STATUS: PASS`。

smoke test 只证明训练环路能跑，不证明质量。

## 9. 下一个 Claude 应该做什么

先读：

1. `docs/PROJECT_HANDOFF.md`
2. `docs/CGAN_PLAN.md`
3. `docs/VALIDATION_PROTOCOL.md`
4. `docs/EXPERIMENTS_LOG.md`
5. 本文件

然后审核 cGAN pilot，不要直接开正式长训练。

重点审核：

- 类别采样是否均衡。
- fidelity loss 是否会把 speckle 拉回均值。
- 判别器是否有足够条件信息。
- 是否有防幻觉结构约束。
- 是否已经准备验证关卡材料。
