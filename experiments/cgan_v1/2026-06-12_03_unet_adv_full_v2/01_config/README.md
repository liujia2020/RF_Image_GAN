# 2026-06-12_03_unet_adv_full_v2

第二十二工作单元：修红线基准后重跑 B-mode 对抗正式 50-epoch run。

## 来源

本 run 从 `2026-06-12_02_unet_adv_full_v1` 的代码、config 与目录结构复制。

## 本轮唯一改动

只修监控红线的判据基准：

- `_02`：`sigmoid(D_real)` 与 `sigmoid(D_fake)` 同时接近 0.5 的红线看 `_evalfixed` 列
- `_03`：该红线改为看 train-mode 列，即 `_train`

对应 config 中记录：

```yaml
adversarial:
  redlines:
    health_basis: train
```

## 保持不动

训练机制全部与 `_02` 保持一致：

- `Light3DUNet`
- `BMode3DPatchDiscriminator`
- `lambda_adv=0.1`
- `BN momentum=0.9`
- Adam `lr=2e-4`, `betas=(0.5,0.999)`
- D -> G 交替训练顺序
- 数据、sampler、seed
- `epochs=50`
- snapshot/checkpoint epochs `[1, 10, 25, 50]`

`_evalfixed` 健康度继续计算、继续写入 `stability.csv`、继续画入
`adversarial_health_curves.png`，但只作为参考列，不再触发“双 0.5”红线。

## 质量边界

本 run 只报数字和产物，不做人眼质量结论。正式质量判断留给 `03_validate`
和 3D Slicer 审查。
