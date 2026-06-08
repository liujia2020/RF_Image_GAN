# 给 Claude 的第十四工作单元中止汇报

时间：2026-06-08 12:30 +0800

## 本轮目标

第十四工作单元为 `adv + normalized struct` 两损失重训：

- 去掉 carrier：`lambda_carrier=0.0`
- 保留 adversarial loss：`lambda_adv=1.0`
- 保留上一轮 normalized struct：`lambda_struct=1.0`
- 不改架构、不改判别器、不改 LP sigma、不启用 AMP
- 目录：`experiments/cgan_v1/2026-06-08_02_phase2a_advstruct_v1/`

## 实现事实

`rf_cgan_losses.py` 已实现：

- `lambda_carrier == 0.0` 时直接跳过 `F.l1_loss(pred, label)`
- carrier 不参与前向、不进 `G_total`、不产生梯度
- `lambda_carrier != 0.0` 时保留原 carrier 逻辑

小样本自检通过：

```text
G_adv_weighted = 1.49114
G_struct_weighted = 0.81067
G_carrier_weighted = 0
G_carrier_skipped = 1
G_total = 2.30181
G_total - (adv + struct) = 0
NaN/Inf = false
```

## 训练状态

正式训练已启动后由用户要求中止。

- 中止前最后完整 epoch：19
- 每 epoch batch：350
- 无 NaN/Inf
- 峰值显存：约 2.61 GB
- 平均每 epoch：约 459.8 秒
- 已产出 epoch10 三类 triplet 图

## 关键指标

epoch 1:

```text
D_fake_score = 0.50875
G_adv_weighted = 0.96107
G_struct_weighted = 1.10383
G_carrier_weighted = 0
G_total = 2.06491
pred_max_abs = 280800.875
```

epoch 19:

```text
D_fake_score = 0.50005
D_real_score = 0.73097
G_adv_weighted = 1.00004
G_struct_weighted = 1.08838
G_carrier_weighted = 0
G_carrier_skipped = 1
G_total = 2.08842
pred_max_abs = 280758.469
```

## 用户观察

用户查看当前训练图后认为：

```text
去掉 carrier 后，预测基本仍与 baseline 相似，没有明显脱离 3 角度低质量输入。
```

因此用户决定提前停止本轮，不继续跑满 50 epoch。

## 当前判断边界

这里只报告训练事实与用户观察，不做最终图像质量结论。质量仍需后续 NIfTI 与用户 3D Slicer 签收。

## 给决策侧的问题

本轮已经排除 carrier 主导，但 `adv + normalized struct` 仍未在早期表现出明显脱离 baseline 的趋势。需要决策侧判断下一步是否应转向：

- 调整判别器/对抗目标，使 D 不再快速贴近 `D_fake_score≈0.5`
- 改变 generator 输出路径或约束，避免 residual/identity 捷径
- 增加更直接区分 baseline 与 label 的结构/频谱/散斑目标
- 重新审视输入、label、baseline 的任务定义与可学习信号
