# 2026-06-12_02_unet_adv_full_v1 summary

Formal B-mode adversarial run stopped by the configured redline before epoch 50.
This file reports numbers only; image quality judgment is reserved for later 3D
Slicer review.

- status: REDLINE_STOPPED
- stop point: epoch 42, after writing `metrics/stability.csv`, before epoch-42 probes/checkpoint
- redline: `sigmoid(D_real_evalfixed)` and `sigmoid(D_fake_evalfixed)` both near 0.5
- redline values: real=0.4942, fake=0.5045
- completed metric rows: 42 / 50
- checkpoints written: epoch001, epoch010, epoch025
- combined probes written: train/val epoch001 through epoch041
- missing by design after stop: epoch050 checkpoint, epoch050 FFT, epoch050 phantom SNR, final normal `cell8` summary

Cell1 self-check:

- G model: Light3DUNet
- D model: BMode3DPatchDiscriminator
- G params: 2,113,889
- D params: 1,493,617
- batch=6 peak VRAM in Cell1: 4.062 GB
- Cell1 pred std: 0.160631
- status: self_check_passed

Training and fixed validation:

| epoch | train loss | train SSIM | train L1 | train pred_std | val loss | val SSIM | val L1 | val pred_std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.884442 | 0.094261 | 0.185134 | 0.154790 | 0.750444 | 0.123146 | 0.086795 | 0.052019 |
| 10 | 0.836967 | 0.127712 | 0.115571 | 0.154261 | 0.794354 | 0.075034 | 0.108639 | 0.118501 |
| 25 | 0.823395 | 0.153413 | 0.109543 | 0.154727 | 0.797319 | 0.071545 | 0.108857 | 0.120512 |
| 42 | 0.824662 | 0.164698 | 0.107843 | 0.158766 | 0.793797 | 0.075243 | 0.106259 | 0.114530 |

Adversarial health:

| epoch | D_loss_train | D_real_train | D_fake_train | G_adv_train | D_loss_evalfixed | D_real_evalfixed | D_fake_evalfixed | G_adv_evalfixed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.354810 | 0.669678 | 0.565152 | 0.939998 | 0.106299 | 0.675681 | 0.580497 | 0.470420 |
| 10 | 0.157756 | 0.675453 | 0.561881 | 0.857535 | 0.269022 | 0.585725 | 0.502586 | 1.045089 |
| 25 | 0.131815 | 0.692176 | 0.540759 | 0.947352 | 0.301337 | 0.622778 | 0.590668 | 0.505046 |
| 42 | 0.086979 | 0.706251 | 0.526426 | 1.057532 | 0.557481 | 0.494174 | 0.504478 | 0.993349 |

Per-category fixed curves at epoch 42:

| category | trainfixed loss | trainfixed SSIM | trainfixed L1 | trainfixed pred_std | val loss | val SSIM | val L1 | val pred_std | val gt_std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carotid | 0.715749 | 0.167498 | 0.102798 | 0.098278 | 0.792953 | 0.076029 | 0.105107 | 0.091867 | 0.096328 |
| muscle | 0.718077 | 0.164695 | 0.102635 | 0.098746 | 0.790986 | 0.078745 | 0.107072 | 0.098606 | 0.099842 |
| phantom | 0.720505 | 0.161874 | 0.102999 | 0.099092 | 0.797452 | 0.070956 | 0.106598 | 0.099437 | 0.095769 |

Phantom speckle SNR in B-mode log-compressed [0,1] space:

| epoch | pred mean | pred std | pred mean/std | gt mean | gt std | gt mean/std |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.401214 | 0.015427 | 26.0067 | 0.377056 | 0.093382 | 4.03777 |
| 10 | 0.375050 | 0.080168 | 4.67833 | 0.377056 | 0.093382 | 4.03777 |
| 25 | 0.377395 | 0.092992 | 4.05836 | 0.377056 | 0.093382 | 4.03777 |

Note: this is B-mode log-compressed-space mean/std, comparing pred vs gt. It is
not a linear Rayleigh speckle SNR comparison.

FFT artifact screen:

- epoch 1 phantom: non-DC peak / median = 32.4235
- epoch 1 muscle: non-DC peak / median = 34.7463
- epoch 50 FFT was not produced because the run stopped at epoch 42
- Discrete periodic peak judgment requires visual review of the saved FFT figures; no quality conclusion is made here.

BN trend:

- BN running mean mean: -0.083609 at epoch 1 to -0.482770 at epoch 42
- BN running var mean: 12,826.985 at epoch 1 to 36,997.089 at epoch 42

Other diagnostics:

- carotid/muscle gCNR: MISSING_ROI in `stability.csv`
- loss curves: `figures/light3dunet_bmode_supervised_loss_curves.png`
- adversarial health curves: `figures/adversarial_health_curves.png`
- category train/val curves: `figures/category_train_val_curves.png`
- combined probes: `probes/train/` and `probes/val/`, epoch001 through epoch041
