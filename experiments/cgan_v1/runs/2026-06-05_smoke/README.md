# 2026-06-05_smoke

创建时间：2026-06-05 01:20:00 +08:00

最近更新：2026-06-05 01:57:34 +08:00

## 目的

这是 cGAN 新路线的最小 smoke test。它只验证训练环路是否稳定，不评价图像质量。

具体检查：

- 是否 OOM。
- 是否 NaN。
- 判别器是否一上来把生成器打死。
- G/D loss 是否有非平凡变化。
- AMP、optimizer、dataset、loss 是否能接起来。

## 配置摘要

- 数据：`/home/liujia/RF_Image/Data_cache_random64_full1500/train`
- 样本：前 100 个 patch。
- 当前这 100 个样本全是 carotid，因此只适合 smoke，不适合正式质量实验。
- patch：`input [1536,64,32,32]`，`label/baseline [2,64,32,32]`。
- Generator：`TinyResidualRFNet`，随机初始化，不加载回归 checkpoint。
- Discriminator：`Envelope2DPatchDiscriminator`，2D envelope PatchGAN。
- Loss：LSGAN + `10 * complex_L1`。
- epoch：20。
- batch size：2。
- AMP：开启。

## 运行结果

状态：`PASS`

运行时间：2026-06-05 01:46:59 至约 01:56:51 +08:00。

关键输出：

```text
SMOKE_STATUS: PASS
G: TinyResidualRFNet, trainable_params=264,738
D: Envelope2DPatchDiscriminator, trainable_params=662,721
full_samples=1050, smoke_samples=100, batches_per_epoch=50
```

最终 epoch：

```text
epoch 20
d_real=0.675289
d_fake=0.562523
d_loss=0.144111
g_adv=0.715592
g_fid=0.161143675
g_loss=2.327029
```

## 判读

通过 smoke test 表示可以进入 cGAN pilot 设计。

它不表示：

- 图像质量变好。
- speckle 已经真实。
- 可以跳过验证关卡。
- 可以直接写论文结果。

下一步应该是设计类别均衡的 pilot run，并按 `docs/RUNBOOK.md` 建正式 run folder。
