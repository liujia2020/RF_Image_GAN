# B-mode 可视化检查

目的：从测试集 cache 直接读取 `label` 和 `baseline`，恢复 scale 后生成 B-mode 对比图，用于肉眼确认 33 角度 label 是否明显优于 3 角度 baseline。

## 数据

- cache：`/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607/test`
- 类别：`carotid / muscle / phantom`
- 每类样本数：5
- 切面：`XZ / XY / ZY`

## 显示规则

- envelope：`sqrt(real^2 + imag^2)`
- dB：`20*log10(envelope + 1e-12)`
- label 与 baseline 在每张图内共享同一个 `vmin/vmax`
- 不做滤波、后处理或额外归一化
- aspect ratio 使用物理间距：`Z=0.0362mm, X=Y=0.2mm`

## 输出

- 图像：`figures/`
- 清单：`results/bmode_visual_check_manifest.csv`
- 脚本：`scripts/generate_bmode_visual_check.py`
