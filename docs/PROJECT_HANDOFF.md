# 项目交接文档

最后更新：2026-06-05 01:57:34 +08:00

## 1. 一句话目标

用神经网络把 delay-aligned 多角度 RF tensor 直接重建成质量对标 DAS 的复数 RF volume，并能在完整 volume 上可靠拼接和验证。

## 2. 当前总状态

项目已经从 `RF_Image` 的回归路线切到新的 `RF_Image_GAN` 仓库。

当前分支：

```text
codex/cgan-v1
```

当前战略判决：

- 纯回归路线冻结为负结果。
- WideDeep + SSIM + crop8 只改善代理指标，未通过 NIfTI + 3D Slicer 人眼验证。
- 失败现象：phantom 线靶丢失，组织 speckle 形态 mush。
- 范式级原因：回归倾向条件均值，无法生成真实随机 speckle。
- 新方向：paired conditional GAN，用对抗项逼真实 speckle，用保真项锁结构防幻觉。

## 3. 新仓库结构

```text
RF_Image_GAN/
  docs/
    CGAN_PLAN.md
    VALIDATION_PROTOCOL.md
    RUNBOOK.md
    PROJECT_HANDOFF.md
    EXPERIMENTS_LOG.md
    claude.md
  experiments/
    cgan_v1/
      runs/
        2026-06-05_smoke/
          README.md
          config.yaml
          train.ipynb
  rf_cgan_models.py
```

旧仓库 `RF_Image` 仍保留完整历史、数据管线、cache、checkpoint、stitch 输出和负结果证据。不要删除。

## 4. 数据链路

数据主链路如下：

```text
原始 RF / MATLAB .mat
  -> MATLAB delay/alignment/patch 生成
  -> patch HDF5
  -> RFLearningDataset 解析
  -> rf_cache_builder.py 生成 memmap cache
  -> RFCachedDataset 训练读取
```

当前 cGAN smoke test 复用旧仓库 cache：

```text
/home/liujia/RF_Image/Data_cache_random64_full1500
```

该 cache 来自 64×32×32 full1500 数据：

- train：1050 patch
- val：225 patch
- test：225 patch
- input shape：`[1536,64,32,32]`
- label shape：`[2,64,32,32]`
- baseline shape：`[2,64,32,32]`

其中 label/baseline 的 2 个通道是 real/imag。

## 5. 回归路线关键结果

这些是负结果和 baseline 证据，保留供后续对照。

| 实验 | 数据 | 模型 | 关键结果 |
|---|---|---|---|
| `tiny_nonpoint_full_32` | 32×16×16 nonpoint | TinyResidualRFNet | test complex improvement 约 47.11%，abs 约 49.62% |
| `tiny_random64_full1500` | 64×32×32 full1500 | TinyResidualRFNet + IN | test complex improvement 47.32%，abs 52.50%；full-volume seam 约 1.46x |
| `tiny_bn_random64_full1500` | 同上 | TinyResidualRFNet + BN | test complex improvement 56.85%，abs 51.85%；13 卷 seam 全面降低，成为强 BN-L1 baseline |
| `unet_random64_full1500` | 同上 | UNetResidualRFNet | patch/test 和 stitch 均未胜 Tiny |
| `wide_ssim_random64_full1500` | 同上 | WideDeep + SSIM | 代理指标改善，但 Slicer 人眼失败；归类为负结果 |

## 6. 缝根因结论

缝根因 #1：InstanceNorm per-patch 统计不连续。

- 已由 BatchNorm3d 重训解决。
- 13 卷 dense64 诊断中，BN seam 三类均压低。
- carotid/muscle 非过平滑；phantom 有轻微细节损失但当时可接受。

缝根因 #2：baseline / 数据本身空间不连续。

- baseline seam 仍约 2.8-3.0x。
- 但 BN 后 pred 层面没有显著传染，优先级降级。
- 对深模型产生的边缘 padding 污染，crop8/halo 在推理协议层面有效。

这些结论与回归/GAN 范式无关，仍然可复用。

## 7. 动态范围线程最终判决

曾经的链条：

```text
L1 残差诊断
-> 发现幅度欠射/动态范围压缩
-> abs_weight sweep 无效
-> SSIM 撬动 slope/p99/abs_std
-> WideDeep + SSIM 进一步改善代理指标
-> crop8/halo 修复深模型 seam
-> NIfTI + 3D Slicer 人眼验证否决
```

最终结论：

- 回归范式不是交付解。
- 代理指标提升不能等同于 DAS 质量。
- Slicer 人眼验证拥有最终否决权。
- 该线作为“为什么必须转 GAN”的负结果证据保留。

## 8. 当前 cGAN smoke test

run 路径：

```text
experiments/cgan_v1/runs/2026-06-05_smoke/
```

配置：

- 数据：`Data_cache_random64_full1500/train` 前 100 个 patch。
- 当前这 100 个 patch 全是 carotid，只用于 smoke，不是正式训练。
- Generator：旧仓库 `TinyResidualRFNet`，随机初始化，不加载回归 checkpoint。
- Discriminator：`Envelope2DPatchDiscriminator`，2D envelope PatchGAN。
- 损失：LSGAN + `10 * complex_L1`。
- epoch：20。
- batch size：2。
- AMP：开启。

已运行结果：

- `SMOKE_STATUS: PASS`
- 无 NaN。
- 无 OOM。
- D/G loss 有非平凡变化。
- 判别器没有一边倒把生成器打死。

这只说明训练环路能跑，不说明成像质量好。

## 9. 当前新增代码

`rf_cgan_models.py`：

- `Envelope2DPatchDiscriminator`：2D PatchGAN 判别器。
- `extract_envelope_slice`：从 `[B,2,Z,X,Y]` 复数 volume 取 envelope 中间 y 切片。
- `count_trainable_params`：参数量统计。

smoke notebook 为了保持新仓库干净，临时从旧仓库导入：

- `TinyResidualRFNet`
- `RFCachedDataset`

后续正式阶段可考虑把必要模块复制或重构到新仓库，但不要急。

## 10. 下一步建议

1. 不要把 smoke 当质量实验。
2. 设计 cGAN pilot：
   - 类别均衡采样。
   - 明确 generator / discriminator / fidelity decomposition。
   - 记录 GAN 稳定性快照。
3. 每个正式 run 必须按 `RUNBOOK.md` 建目录。
4. 每个质量结论必须按 `VALIDATION_PROTOCOL.md` 出 NIfTI、指标和 Slicer 材料。
5. 用户负责视觉签收；Codex 只报材料和数字。

## 11. 对下一个 Claude / Claude Code 的提醒

请先读：

1. `docs/claude.md`
2. `docs/PROJECT_HANDOFF.md`
3. `docs/CGAN_PLAN.md`
4. `docs/VALIDATION_PROTOCOL.md`
5. `docs/EXPERIMENTS_LOG.md`

不要直接让 Codex 开正式训练。先审 cGAN pilot 方案，尤其是：

- 保真项是否又把 speckle 拉回均值。
- 判别器看到的条件是否足够。
- 是否有防幻觉结构约束。
- 是否提前定义 ROI 和验证材料。
