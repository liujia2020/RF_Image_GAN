# 2026-06-12_03_unet_adv_full_v2 summary

Formal B-mode adversarial 50-epoch rerun after changing only the flat-D redline
basis from evalfixed to train-mode. This summary reports numbers only; quality
judgment is reserved for later 3D Slicer review.

- status: COMPLETED_50_EPOCHS
- redline status: no scripted redline exception stopped training
- health basis for flat-D redline: train-mode columns
- evalfixed health: still computed, logged, and plotted as reference only
- metrics rows: 50
- total train-loop seconds: 25,595.43 s
- peak CUDA memory reported by script: 4.0719 GB
- checkpoints: epoch001, epoch010, epoch025, epoch050
- combined probes: 50 train images and 50 val images

Cell1 self-check:

- G model: Light3DUNet
- D model: BMode3DPatchDiscriminator
- G params: 2,113,889
- D params: 1,493,617
- batch=6 peak VRAM in Cell1: 4.062 GB
- Cell1 pred std: 0.160631
- status: self_check_passed

Training and fixed validation:

| epoch | train loss | sup loss | train SSIM | train L1 | train pred_std | val loss | val SSIM | val L1 | val pred_std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.898324 | 0.800353 | 0.084510 | 0.195882 | 0.183545 | 0.768828 | 0.108317 | 0.123841 | 0.108972 |
| 10 | 0.832800 | 0.752524 | 0.125980 | 0.114670 | 0.156708 | 0.800987 | 0.068051 | 0.113438 | 0.116470 |
| 25 | 0.828833 | 0.730567 | 0.151118 | 0.109419 | 0.155367 | 0.802804 | 0.065566 | 0.111743 | 0.120106 |
| 50 | 0.821776 | 0.715931 | 0.167976 | 0.106446 | 0.158277 | 0.795749 | 0.073173 | 0.107590 | 0.121125 |

Train-mode adversarial health:

| epoch | D_loss_train | D_real_train | D_fake_train | G_adv_train |
|---:|---:|---:|---:|---:|
| 1 | 0.349805 | 0.671900 | 0.562883 | 0.979715 |
| 10 | 0.171547 | 0.667371 | 0.570511 | 0.802758 |
| 25 | 0.126112 | 0.691974 | 0.540998 | 0.982654 |
| 50 | 0.087076 | 0.708070 | 0.523754 | 1.058451 |

Evalfixed reference health:

| epoch | D_loss_evalfixed | D_real_evalfixed | D_fake_evalfixed | G_adv_evalfixed |
|---:|---:|---:|---:|---:|
| 1 | 0.079238 | 0.723461 | 0.579028 | 0.493291 |
| 10 | 0.311348 | 0.567433 | 0.516325 | 0.921541 |
| 25 | 0.448572 | 0.598619 | 0.640291 | 0.256256 |
| 50 | 0.599832 | 0.490614 | 0.519859 | 0.906286 |

Per-category fixed curves at epoch 50:

| category | trainfixed loss | trainfixed SSIM | trainfixed L1 | trainfixed pred_std | val loss | val SSIM | val L1 | val pred_std | val gt_std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carotid | 0.715891 | 0.167790 | 0.105221 | 0.099768 | 0.787355 | 0.082700 | 0.105144 | 0.093319 | 0.096328 |
| muscle | 0.720357 | 0.162978 | 0.107865 | 0.100463 | 0.797825 | 0.070924 | 0.108758 | 0.101789 | 0.099842 |
| phantom | 0.744563 | 0.133455 | 0.104154 | 0.101088 | 0.802067 | 0.065895 | 0.108869 | 0.101894 | 0.095769 |

Phantom speckle SNR in B-mode log-compressed [0,1] space:

| epoch | pred mean | pred std | pred mean/std | gt mean | gt std | gt mean/std |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.323920 | 0.143243 | 2.26133 | 0.377056 | 0.093382 | 4.03777 |
| 10 | 0.366703 | 0.089558 | 4.09459 | 0.377056 | 0.093382 | 4.03777 |
| 25 | 0.363549 | 0.100211 | 3.62784 | 0.377056 | 0.093382 | 4.03777 |
| 50 | 0.386769 | 0.097952 | 3.94856 | 0.377056 | 0.093382 | 4.03777 |

Note: this is B-mode log-compressed-space mean/std, comparing pred vs gt. It is
not a linear Rayleigh speckle SNR comparison.

FFT artifact screen:

| epoch | category | non-DC peak / median |
|---:|---|---:|
| 1 | phantom | 5.76988 |
| 1 | muscle | 16.55486 |
| 50 | phantom | 5.22598 |
| 50 | muscle | 4.09804 |

Discrete periodic peak judgment requires visual review of the saved FFT figures;
no quality conclusion is made here.

BN trend:

- BN running mean mean: -0.074867 at epoch 1 to -0.551611 at epoch 50
- BN running var mean: 12,719.056 at epoch 1 to 39,912.361 at epoch 50

Other diagnostics:

- carotid/muscle gCNR: MISSING_ROI in `stability.csv`
- stability table: `metrics/stability.csv`
- phantom SNR table: `metrics/phantom_bmode_snr.csv`
- FFT records: `metrics/fft_artifact_screen.json`
- loss curves: `figures/light3dunet_bmode_supervised_loss_curves.png`
- adversarial health curves: `figures/adversarial_health_curves.png`
- category train/val curves: `figures/category_train_val_curves.png`
- combined probes: `probes/train/` and `probes/val/`, epoch001 through epoch050
