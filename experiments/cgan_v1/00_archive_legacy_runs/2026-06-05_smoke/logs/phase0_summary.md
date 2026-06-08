# Phase 0 diagnostics summary

Generated: 2026-06-05 02:40:50 +0800

## A. Cache category counts

Category source: `meta.npz['category']` in each cache split. The path field is retained for traceability, but category counts are not inferred from filename or index ranges.

| split | category | patch_count | fraction | total_split_patches |
| --- | --- | --- | --- | --- |
| train | carotid | 350 | 0.333333 | 1050 |
| train | muscle | 350 | 0.333333 | 1050 |
| train | phantom | 350 | 0.333333 | 1050 |
| train | other | 0 | 0 | 1050 |
| val | carotid | 75 | 0.333333 | 225 |
| val | muscle | 75 | 0.333333 | 225 |
| val | phantom | 75 | 0.333333 | 225 |
| val | other | 0 | 0 | 225 |
| test | carotid | 75 | 0.333333 | 225 |
| test | muscle | 75 | 0.333333 | 225 |
| test | phantom | 75 | 0.333333 | 225 |
| test | other | 0 | 0 | 225 |

## B. Automated homogeneous ROI speckle diagnostics

ROI method: deterministic automatic proxy ROIs on the label middle-y XZ slice. Candidate windows are ranked by low normalized gradient, low strong-reflector occupancy, and low p99/mean excess. This is not a human-frozen Slicer ROI set.

| category | n_rois | speckle_snr_mean_over_std_mean | autocorr_fwhm_z_vox_mean | autocorr_fwhm_x_vox_mean | autocorr_fwhm_z_mm_mean | autocorr_fwhm_x_mm_mean |
| --- | --- | --- | --- | --- | --- | --- |
| carotid | 12 | 2.2496 | 6.43098 | 3.20983 | 0.232802 | 0.641966 |
| muscle | 12 | 2.04111 | 6.65233 | 3.82635 | 0.240814 | 0.765269 |
| phantom | 12 | 2.14793 | 6.01636 | 2.75736 | 0.217792 | 0.551472 |

Detailed ROI rows: `phase0_speckle_rois.csv`.

## C. Current implementation facts

```json
{
  "batchnorm_available_in_source": true,
  "batchnorm_confirmed_runtime": true,
  "complex_phase_constraint": "no explicit carrier/phase loss or phase consistency term; only complex L1 in smoke G fidelity",
  "d_condition": "baseline envelope is concatenated with candidate/label envelope as channel 1",
  "d_loss_formula": "0.5 * (MSE(D(label_env+baseline_env), 1) + MSE(D(pred_env_detached+baseline_env), 0))",
  "d_slice": "extract_envelope_slice(..., y_idx=None), therefore middle y slice only in current notebook",
  "extract_envelope_slice_y_support": "supports arbitrary single y_idx; no multi-slice helper yet",
  "extractor_has_y_idx_arg": true,
  "first_norm_class_runtime": "BatchNorm3d",
  "g_adv": "MSELoss(D(cat([pred_env, baseline_env])), ones)",
  "g_fid": "torch.mean(torch.abs(pred - label))",
  "g_loss_formula": "g_loss = g_adv + lambda_fid * g_fid",
  "g_norm": "TinyResidualRFNet(use_batch_norm=True); rf_models.py uses BatchNorm3d when use_batch_norm=True",
  "instance_norm_in_g_current_path": "current TinyResidualRFNet path uses BatchNorm3d, not InstanceNorm3d",
  "lambda_fidelity": 10.0,
  "no_training_or_loss_change": true,
  "notebook_concats_baseline_to_d": true,
  "notebook_contains_complex_l1": true
}
```
