# Regression Legacy Archive

Last updated: 2026-06-04 21:32:24 +08:00

The regression line is frozen, not deleted.

Purpose of this archive:

- Preserve evidence for why the project pivoted to conditional GAN.
- Keep BN-L1 and Wide+SSIM/crop8 baselines available for comparison.
- Separate one-off regression diagnostics from the new cGAN workspace.

Current status:

- Files have not been moved yet.
- This directory records the planned archive boundary.
- Move files only with `git mv` after checking import/runtime impact.

Reusable assets are not archived here:

- dataset/cache code
- core model registry
- evaluation/visualization/stitch utilities
- validation protocol

Those remain project-level assets.
