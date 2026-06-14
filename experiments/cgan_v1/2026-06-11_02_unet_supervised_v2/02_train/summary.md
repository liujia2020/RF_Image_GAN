# 2026-06-11_02_unet_supervised_v2 summary

本 run 为 B-mode 单通道纯监督基线，不做质量结论。

- metrics rows: 50
- SSIM library: pytorch-msssim, 3D single-scale, win_size=7
- first loss/ssim/l1: 0.7826884341239929 / 0.1073749488379274 / 0.2055213509287153
- last loss/ssim/l1: 0.5175830612863813 / 0.39952005692890713 / 0.08237458659069878
- first pred_baseline_l1: 0.3108324202469417
- last pred_baseline_l1: 0.14292690570865357
- first pred_std: 0.10072640748960632
- last pred_std: 0.1428107436639922
- last fixed-val pred_std: 0.09149209689348936
- last BN running mean/var mean: -0.522735174922716 / 34047.5252693707
