# 2026-06-12_01_unet_adv_smoke_v1 summary

本 run 为 B-mode 单通道纯监督基线，不做质量结论。

- metrics rows: 10
- lambda_adv: 0.1
- SSIM library: pytorch-msssim, 3D single-scale, win_size=7
- first total/sup/ssim/l1: 0.8858460235595703 / 0.7950602262360709 / 0.09213313959538937 / 0.2028255272763116
- last total/sup/ssim/l1: 0.833356854234423 / 0.7523816858019148 / 0.12605486550501416 / 0.11417372844048909
- first D real/fake sigmoid: 0.6656943556240627 / 0.569825564793178
- last D real/fake sigmoid: 0.6649650274004255 / 0.5727582378046853
- first/last G_adv: 0.9078579645710332 / 0.809751677257674
- first pred_baseline_l1: 0.2891080353089741
- last pred_baseline_l1: 0.15186864009925297
- first pred_std: 0.1527080652117729
- last pred_std: 0.15779583854334694
- last fixed-val pred_std: 0.11873533576726913
- last BN running mean/var mean: -0.2036845109735926 / 20999.276940430915
- last val carotid pred_std/gt_std: 0.09601116459816694 / 0.09632845595479012
- last val muscle pred_std/gt_std: 0.10317014623433352 / 0.09984172787517309
- last val phantom pred_std/gt_std: 0.09839103929698467 / 0.09576937090605497

红线状态：若本文件生成，训练未触发脚本内红线异常。
质量结论：无；需要后续 3D Slicer 人类审查。
