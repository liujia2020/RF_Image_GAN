# Phase2a 正式训练：normalized struct v1

创建时间：2026-06-08 00:00:00 +0800

## 目的

使用已全量验证通过的新 fp16 cache，运行 Phase2a 正式训练，观察“label 均值包络尺度归一化 struct 损失”后的训练健康事实。

本运行只做损失设计健康检查，不做图像质量结论。图像质量结论必须等待后续 NIfTI 导出与用户 3D Slicer 签收。

## 关键前提

- cache：`/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607/`
- reader 读 normalized cache 后执行 `fp16 -> fp32 -> 乘 scale`，进入网络前是真实幅值。
- AMP 全程关闭：`use_amp=false`。
- `num_workers=0`。
- 损失权重：`lambda_adv=1.0`，`lambda_struct=1.0`，`lambda_carrier=0.5`。
- struct 尺度：`s = mean(abs(env(label))) + 1e-8`，pred 与 label 同除这个 label 派生标量。

## 运行结构

```text
2026-06-08_01_phase2a_normstruct_v1/
├── 01_config/      配置、manifest
├── 02_train/       train.py、train.ipynb、日志、metrics、figures、checkpoints
└── 03_validate/    本单元不做质量验证，仅放占位说明
```

## 执行入口

正式结果必须由 notebook 一次性 Run All 产出：

```bash
PYTHONNOUSERSITE=1 /home/liujia/miniconda3/envs/rf-cgan-clean/bin/python -m nbconvert \
  --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=rf-cgan-clean \
  experiments/cgan_v1/runs/2026-06-08_01_phase2a_normstruct_v1/02_train/train.ipynb
```
