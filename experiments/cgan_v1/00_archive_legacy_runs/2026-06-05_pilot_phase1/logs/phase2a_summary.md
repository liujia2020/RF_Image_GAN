# Phase 2a summary

Generated: 2026-06-05 10:58:09 +0800

## 1. NaN / collapse

- aborted: None
- completed_epochs: 1 / 1

## 2. Batch / memory / timing

- actual_batch_size: 6
- dry_run_peak_mem_GB: 3.378
- first_epoch_seconds: 279.398
- first_epoch_peak_mem_GB: 5.648
- total_seconds: 285.004

## 3. Final fixed-val speckle check

- carotid: pred_SNR=1.5447, label_SNR=1.6018, KS=0.0108
- muscle: pred_SNR=1.5524, label_SNR=1.4842, KS=0.0351
- phantom: pred_SNR=1.6411, label_SNR=1.7127, KS=0.0118

## 4. Objective observation only

This run is not a quality conclusion. Inspect the saved PNGs only as a loss-design health check.
If pred_SNR stays far above label_SNR, the output is still smoother than label speckle. If it approaches label_SNR while KS decreases, the adversarial term is moving in the intended direction.

## 5. Stitch capability report

Full-volume stitch needs dense coverage patches, not random64 training patches. Random64_full1500 samples are sparse random positions and cannot tile a whole volume. Old RF_Image has rf_stitch.py, rf_stitch_vis.py, rf_export_stitched_to_nii.py, and dense64 generation/stitch artifacts. For cGAN full-volume validation, migrate or wrap the old dense64 extraction + rf_stitch.py path; do not attempt stitch from random64 cache.
