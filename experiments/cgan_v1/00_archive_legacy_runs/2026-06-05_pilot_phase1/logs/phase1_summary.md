# Phase 1 diagnostics summary

Generated: 2026-06-05 09:31:02 +0800

## Loss probe
| batch_categories | y_idx | g_adv_raw | g_struct_raw | g_carrier_raw | g_adv_weighted | g_struct_weighted | g_carrier_weighted | g_total | first_norm_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| carotid,muscle,phantom | 8 | 0.826563 | 0.240835 | 0.323408 | 0.826563 | 2.40835 | 0.323408 | 3.55832 | BatchNorm3d |

## Y-direction speckle FWHM
| category | n_rois | speckle_snr_mean_over_std_mean | autocorr_fwhm_y_vox_mean | autocorr_fwhm_y_mm_mean |
| --- | --- | --- | --- | --- |
| carotid | 12 | 2.10651 | 3.04182 | 0.608363 |
| muscle | 12 | 2.07205 | 2.98932 | 0.597864 |
| phantom | 12 | 2.12076 | 3.1129 | 0.62258 |

## Low-pass visualizations
| category | sample_idx | png | nifti_written | residual_abs_over_env_mean |
| --- | --- | --- | --- | --- |
| carotid | 0 | /home/liujia/RF_Image_GAN/experiments/cgan_v1/runs/2026-06-05_pilot_phase1/figures/lowpass/carotid_env_lowpass_residual_xz_y16.png | True | 0.45063 |
| muscle | 350 | /home/liujia/RF_Image_GAN/experiments/cgan_v1/runs/2026-06-05_pilot_phase1/figures/lowpass/muscle_env_lowpass_residual_xz_y16.png | True | 0.446174 |
| phantom | 700 | /home/liujia/RF_Image_GAN/experiments/cgan_v1/runs/2026-06-05_pilot_phase1/figures/lowpass/phantom_env_lowpass_residual_xz_y16.png | True | 0.45245 |

## Stitch capability
| repo | matching_file_count | files |
| --- | --- | --- |
| RF_Image_GAN | 9 | experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/carotid_env_label.nii; experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/carotid_env_minus_lp.nii; experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/carotid_lp_env_label.nii; experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/muscle_env_label.nii; experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/muscle_env_minus_lp.nii; experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/muscle_lp_env_label.nii; experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/phantom_env_label.nii; experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/phantom_env_minus_lp.nii; experiments/cgan_v1/runs/2026-06-05_pilot_phase1/nii/lowpass/phantom_lp_env_label.nii |
| RF_Image | 32 | __pycache__/rf_stitch.cpython-310.pyc; __pycache__/rf_stitch.cpython-312.pyc; __pycache__/rf_stitch_vis.cpython-310.pyc; outputs/nii_for_slicer/Carotid_008_bnl1_db.nii; outputs/nii_for_slicer/Carotid_008_bnl1_env.nii; outputs/nii_for_slicer/Carotid_008_label_db.nii; outputs/nii_for_slicer/Carotid_008_label_env.nii; outputs/nii_for_slicer/Carotid_008_widessim_db.nii; outputs/nii_for_slicer/Carotid_008_widessim_env.nii; outputs/nii_for_slicer/Carotid_073_bnl1_db.nii; outputs/nii_for_slicer/Carotid_073_bnl1_env.nii; outputs/nii_for_slicer/Carotid_073_label_db.nii; outputs/nii_for_slicer/Carotid_073_label_env.nii; outputs/nii_for_slicer/Carotid_073_widessim_db.nii; outputs/nii_for_slicer/Carotid_073_widessim_env.nii; outputs/nii_for_slicer/Muscle_056_bnl1_db.nii; outputs/nii_for_slicer/Muscle_056_bnl1_env.nii; outputs/nii_for_slicer/Muscle_056_label_db.nii; outputs/nii_for_slicer/Muscle_056_label_env.nii; outputs/nii_for_slicer/Muscle_056_widessim_db.nii; outputs/nii_for_slicer/Muscle_056_widessim_env.nii; outputs/nii_for_slicer/Phantom_036_bnl1_db.nii; outputs/nii_for_slicer/Phantom_036_bnl1_env.nii; outputs/nii_for_slicer/Phantom_036_label_db.nii; outputs/nii_for_slicer/Phantom_036_label_env.nii; outputs/nii_for_slicer/Phantom_036_widessim_db.nii; outputs/nii_for_slicer/Phantom_036_widessim_env.nii; rf_export_stitched_to_nii.py; rf_stitch.py; rf_stitch_vis.py |
