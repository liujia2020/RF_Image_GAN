# Phase2a 正式训练：newcache 损失设计健康检查

创建时间：2026-06-07 14:02:18 +0800

## 目的

使用已全量验证通过的新 fp16 cache，运行 Phase2a 正式训练，观察当前 frozen 损失设计的稳定性与趋势。

本运行只做损失设计健康检查，不做图像质量结论。图像质量结论必须等待后续 NIfTI 导出与用户 3D Slicer 签收。

## 唯一关键前提

- cache 已验证通过：`/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607/`
- reader 读 normalized cache 后执行 `fp16 -> fp32 -> 乘 scale`，进入网络前是真实幅值。
- AMP 全程关闭：`use_amp=false`。

## 运行结构

```text
2026-06-07_01_phase2a_full_newcache/
├── 01_config/      配置、manifest、给 Claude 的冲突说明
├── 02_train/       训练脚本、日志、metrics、figures、checkpoints
└── 03_validate/    本单元不做质量验证，仅放占位说明
```

## 执行入口

正式结果由 notebook 一次性 Run All 产出：

```bash
PYTHONNOUSERSITE=1 /home/liujia/miniconda3/envs/rf-cgan-clean/bin/python -m nbconvert \
  --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=rf-cgan-clean \
  experiments/cgan_v1/runs/2026-06-07_01_phase2a_full_newcache/02_train/train.ipynb
```
