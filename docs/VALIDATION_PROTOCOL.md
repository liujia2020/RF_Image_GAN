# 验证协议

最后更新：2026-06-05 01:57:34 +08:00

## 1. 总规则

任何图像质量结论都必须通过验证关卡。代理指标不能单独证明质量达标。

如果 NIfTI / 3D Slicer 人眼检查失败，则该 run 失败。指标不能推翻人眼失败。

职责边界：

- Codex 负责产出 NIfTI、图、CSV、日志和摘要。
- Claude / Claude Code 负责策略审核和解释风险。
- 用户负责最终视觉签收。

## 2. 正式 run 必须产出的材料

每个正式 run 至少要有：

- label、baseline、candidate 的线性 envelope NIfTI。
- 同一 volume 下共享 label reference 的 dB NIfTI。
- 固定切片对比 PNG，便于快速浏览。
- 标准指标 CSV。
- `verdict.md`，记录人眼状态、指标状态、最终决定。

NIfTI spacing 必须正确：

```text
z = 0.0362 mm
x = 0.2 mm
y = 0.2 mm
```

默认数组轴顺序为 `[z,x,y]`。如果某个导出不是这个顺序，必须在导出脚本和 `verdict.md` 里写清楚。

## 3. Slicer 人眼检查清单

每个 volume 都要在同一窗位下并排检查 candidate 和 DAS label：

- speckle 形态：应该是 DAS 式颗粒感，不能是糊状 mush。
- 相干结构：血管壁、条带、组织界面、仿体结构必须保留。
- 点靶/线靶：label 中可见的目标不能在 pred 中消失。
- 幻觉：不能出现 label/RF 不支持的新结构。
- 拼接缝：不能有 patch-grid 周期亮线；也不能在边界处出现 speckle 粒径、方向、纹理突变。
- 深部区域：不能塌陷，也不能无控制过冲。

人眼状态只允许：

- `PASS`
- `FAIL`
- `INCONCLUSIVE`

`INCONCLUSIVE` 必须有后续动作，不能用于质量 claim。

## 4. 标准指标

尽量按类别分别报告：carotid、muscle、phantom。

必需指标：

- complex L1 以及相对 baseline 的 improvement。
- envelope L1 以及相对 baseline 的 improvement。
- PSNR，必须写明是在 envelope 还是 dB envelope 上计算。
- speckle SNR：在预先冻结的 homogeneous ROI 内计算。阈值不是固定 1.91，而是同一 ROI 中 label 的实测值；Rayleigh 约 1.91 只作为 sanity range。
- envelope histogram 距离或 KS statistic。
- 有合法 ROI 时报告 gCNR。
- stitched volume 的 seam 指标。
- homogeneous ROI 内 2D speckle autocorrelation：比较 pred 和 label 的半高宽，过平滑会使 autocorrelation peak 变宽，噪声幻觉可能表现为过尖。

点靶/线靶适用时还要报告：

- FWHM。
- peak preservation。
- 如果 ROI 定义充分，报告 side-lobe 或 contrast。

如果缺 ROI，不准硬算或假装通过，标记为 `MISSING_ROI`。

## 5. ROI 预冻结协议

speckle SNR、histogram、autocorrelation 等 ROI 指标必须在看模型输出之前定义。

步骤：

1. 在 3D Slicer 打开 label NIfTI。
2. 手动选择 homogeneous ROI，避开边界、界面、强反射体。
3. 每个被评审 volume 至少记录一个对应组织类别的 ROI。
4. 把 ROI 坐标写入 run 的 `config.yaml`，键名为 `validation_rois`。
5. ROI 一旦冻结，同一 volume 后续不能移动。
6. 缺 ROI 时，相关指标写 `MISSING_ROI`。

## 6. GAN 训练稳定性快照

任何 GAN run 都必须在训练过程中记录以下值。默认在 epoch 10、25、50 记录；短 run 则取最近完成 epoch。

- `D_real`：判别器对 label patch 的平均分数。
- `D_fake`：判别器对生成 patch 的平均分数。
- `G_adv`：生成器对抗损失。
- `G_fidelity`：生成器保真损失。

提前停止条件：

- `D_fake` 接近 0：判别器彻底赢，生成器学不到。
- `D_real` 和 `D_fake` 都塌到接近 0，且 `G_adv` 不下降：判别器失去信息量。
- 任一关键 loss 出现 NaN。

这些快照必须存进 run 的 `logs/`。缺失快照的正式 run 不能进入质量评审。

## 7. 接受标准

一个 run 只有同时满足以下条件，才能成为候选：

- Slicer 人眼状态为 `PASS`。
- label 中可见的点靶/线靶没有消失；FWHM 不超过 label 的 2 倍。
- 大尺度 echo 结构保留；low-pass envelope correlation 不低于 BN regression baseline。
- 没有 stitch seam 回归。
- speckle 统计合理并有记录。
- 所有材料归档到 run folder，并在 `EXPERIMENTS_LOG.md` 追加记录。

注意：如果 speckle 从 mush 变真实，voxel-wise L1 可能比回归 baseline 差。这不自动构成失败。失败与否看结构、speckle、缝、人眼签收。

## 8. 拒绝标准

出现以下任一情况，run 必须拒绝或暂缓：

- 人眼看到 label 中存在的结构在 pred 中丢失。
- speckle 明显 mush 或明显幻觉。
- 点靶/线靶消失。
- 只改善代理指标，未通过 Slicer。
- full-volume 出现 patch-grid amplitude seam 或 texture seam。

## 9. 为什么这份协议必须严格

WideDeep + SSIM 回归线曾经在代理指标上看起来很有希望，并被文档写得过于乐观。但用户用 3D Slicer 人眼检查后发现 phantom 线靶丢失、组织 speckle mush，直接否决了该路线。

这份协议就是为了防止“指标好看但图像失败”的事故再次发生。
