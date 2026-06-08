# Phase2a 正式训练自动摘要

生成时间：2026-06-07 22:47:12 +0800

## 配置事实

- cache：`/home/liujia/rf_training_cache/cgan_phase2a_64x32x32_random_full1500_fp16_cache_20260607`
- use_amp：`False`
- num_workers：`0`
- epochs 配置：`50`

## 训练健康事实

- 已记录 epoch 数：50
- 最后 epoch D_real：0.001432613685465185
- 最后 epoch D_fake：0.0026530648964814777
- 最后 epoch D_real_score：0.7304820035185132
- 最后 epoch D_fake_score：0.5006023549182075
- 最后 epoch G_adv_weighted：0.9993258147580283
- 最后 epoch G_struct_weighted：43135.23649204799
- 最后 epoch G_carrier_weighted：3055.3873911830356
- 最后 epoch G_total：46191.62325753348
- 最后 epoch pred_max_abs：280180.1875
- 最后 epoch has_nan：0
- 最后 epoch has_inf：0
- 最后 epoch available_memory_gb：47.743743896484375

## 最后一次 speckle 固定探针

- epoch 50 carotid: pred_SNR=0.948416037354329, label_SNR=0.8605585231244595, env_hist_KS=0.22900962829589844
- epoch 50 muscle: pred_SNR=1.0328892343393072, label_SNR=1.0457405854365114, env_hist_KS=0.3496856689453125
- epoch 50 phantom: pred_SNR=1.3706125500830166, label_SNR=1.4853961800174242, env_hist_KS=0.3653736114501953

## 边界

本摘要只报损失设计健康事实，不做图像质量结论；质量结论需后续 NIfTI 与用户 3D Slicer 签收。