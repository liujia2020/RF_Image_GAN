# Phase2a 正式训练自动摘要

生成时间：2026-06-08 06:58:32 +0800

## 配置事实

- cache：`/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607`
- use_amp：`False`
- num_workers：`0`
- epochs 配置：`50`

## 训练健康事实

- 已记录 epoch 数：50
- 最后 epoch D_real：0.0009506612950644922
- 最后 epoch D_fake：0.0009156855266025689
- 最后 epoch D_real_score：0.7308451574189322
- 最后 epoch D_fake_score：0.5002075669595173
- 最后 epoch G_adv_weighted：1.0006948564733777
- 最后 epoch G_struct_weighted：0.9820561853476933
- 最后 epoch G_carrier_weighted：3049.2515522112167
- 最后 epoch G_total：3051.234298793248
- 最后 epoch pred_max_abs：280029.71875
- 最后 epoch has_nan：0
- 最后 epoch has_inf：0
- 最后 epoch available_memory_gb：52.79230499267578

## 最后一次 speckle 固定探针

- epoch 50 carotid: pred_SNR=0.9581451018332265, label_SNR=0.8605585231244595, env_hist_KS=0.23267745971679688
- epoch 50 muscle: pred_SNR=1.0433780318813244, label_SNR=1.0457405854365114, env_hist_KS=0.3522834777832031
- epoch 50 phantom: pred_SNR=1.40701871479066, label_SNR=1.4853961800174242, env_hist_KS=0.37091064453125

## 边界

本摘要只报损失设计健康事实，不做图像质量结论；质量结论需后续 NIfTI 与用户 3D Slicer 签收。