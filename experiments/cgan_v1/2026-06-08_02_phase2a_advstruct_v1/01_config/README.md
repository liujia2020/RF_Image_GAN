# Phase2a 正式训练：adv + normalized struct v1

创建时间：2026-06-08 09:51:00 +0800

## 目的

去掉 carrier，只保留 adversarial loss 与归一化 structural loss，观察对抗项是否能真正参与训练。

本运行只做损失设计健康检查，不做图像质量结论。图像质量结论必须等待后续 NIfTI 导出与用户 3D Slicer 签收。

## 关键前提

- cache：`/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607/`
- reader 读 normalized cache 后执行 `fp16 -> fp32 -> 乘 scale`，进入网络前是真实幅值。
- AMP 全程关闭：`use_amp=false`。
- `num_workers=0`。
- 损失权重：`lambda_adv=1.0`，`lambda_struct=1.0`，`lambda_carrier=0.0`。
- carrier 在 `lambda_carrier == 0` 时跳过计算，不参与前向/反向。
- struct 尺度：`s = mean(abs(env(label))) + 1e-8`，pred 与 label 同除这个 label 派生标量。

## 运行结构

```text
2026-06-08_02_phase2a_advstruct_v1/
├── 01_config/
├── 02_train/
├── 03_validate/
└── 04_proma_verify/
```

## 执行入口

正式结果必须由 notebook 一次性 Run All 产出：

```bash
PYTHONNOUSERSITE=1 /home/liujia/miniconda3/envs/rf-cgan-clean/bin/python -m nbconvert \
  --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=rf-cgan-clean \
  experiments/cgan_v1/2026-06-08_02_phase2a_advstruct_v1/02_train/train.ipynb
```
