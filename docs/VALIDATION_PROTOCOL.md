# VALIDATION_PROTOCOL

Last updated: 2026-06-05 01:19:34 +08:00

## 1) Rule

Quality claims require the validation gate. Proxy metrics alone are never enough.

If human NIfTI/Slicer review fails, the run fails. Metrics cannot overturn a human failure.

Codex produces materials and numbers. The user signs off visual quality.

## 2) Required Artifacts For A Formal Run

Each formal run must provide:

- Linear-envelope NIfTI for `label`, baseline comparison, and candidate model.
- dB NIfTI with a shared label reference for the same volume.
- Fixed comparison PNGs for quick review.
- Standard metrics CSV.
- `verdict.md` with human review status.

NIfTI spacing must be:

- z: `0.0362 mm`
- x: `0.2 mm`
- y: `0.2 mm`

The expected NIfTI array order is `[z,x,y]` unless explicitly documented.

## 3) Human Slicer Checklist

For each reviewed volume, inspect candidate vs DAS label at the same window/level:

- Speckle morphology: granular DAS-like texture, not mush.
- Coherent structures: vessel wall, bands, interfaces, and phantom structures preserved.
- Point/line targets when present: label-visible targets must not disappear.
- Hallucination: no new structures unsupported by label/RF.
- Stitch seams: no patch-grid periodic lines in amplitude AND no texture discontinuity at patch boundaries (speckle grain size or orientation should not visibly change at x=32/64/96 boundaries).
- Deep field: no collapse or uncontrolled over-amplification.

Human status values:

- `PASS`
- `FAIL`
- `INCONCLUSIVE`

`INCONCLUSIVE` requires a follow-up action before a quality claim.

## 4) Standard Metrics

Report by category where possible: carotid, muscle, phantom.

Required:

- complex L1 and improvement against baseline.
- envelope L1 and improvement against baseline.
- PSNR on envelope or dB envelope, with exact definition stated.
- speckle SNR in homogeneous regions. Reference target: label-measured SNR in the same pre-frozen ROI (see Section 7 below). The Rayleigh value ~1.91 is a sanity range, not the pass/fail threshold; the actual DAS label may differ. Report pred SNR, label SNR, and pred/label for each ROI.
- histogram distance or KS statistic for envelope distribution.
- gCNR when valid ROIs are available.
- seam metrics for stitched volumes.
- 2D speckle autocorrelation in homogeneous ROIs: compute 2D autocorrelation of the linear envelope patch, compare width to label autocorrelation in the same ROI. Over-smoothed outputs (mush) show wider autocorrelation peaks; uncorrelated hallucinations show delta-like peaks. Report pred vs label autocorrelation half-width in axial and lateral directions.

Point/line target metrics when applicable:

- FWHM.
- peak preservation.
- side-lobe or contrast measurement if an ROI is defined.

If ROI definitions are missing, mark the metric as `MISSING_ROI`, not as pass/fail.

## 5) Acceptance Policy

A run can be considered a candidate only when all are true:

- Human Slicer status is `PASS`.
- No gross degradation in structural fidelity: point/line targets present where label shows them (FWHM within 2x label), large-scale echo structure preserved (low-pass-filtered envelope correlation vs label >= BN regression baseline). Note: voxel-wise complex L1 is expected to be worse than the mush-producing BN regression baseline when speckle becomes realistic; voxel-wise L1 degradation does NOT constitute a fidelity failure.
- No stitch seam regression.
- Speckle statistics are plausible and documented.
- Artifacts are archived in the run folder and summarized in `EXPERIMENTS_LOG.md`.

A run must be rejected or held when:

- Human review finds missing DAS-visible structures.
- Speckle appears mush-like or hallucinated.
- Point/line targets disappear where label shows them.
- The result only improves proxy metrics.

## 6) Lessons Embedded In This Protocol

The WideDeep+SSIM regression line looked promising by proxy metrics and was documented too optimistically. Human Slicer review later rejected it. This protocol exists to prevent that failure mode from recurring.

## 7) ROI Pre-Freeze Protocol

ROIs for speckle metrics (speckle SNR, histogram, autocorrelation) must be defined before any GAN run starts. They must not be selected after inspecting model outputs.

Required steps:

1. Open label NIfTI volumes in 3D Slicer.
2. Manually segment homogeneous soft-tissue ROIs away from boundaries, interfaces, and strong reflectors. One ROI per tissue class (carotid soft tissue, muscle, phantom homogeneous zone) per reviewed volume.
3. Record ROI coordinates (center z/x/y, radius or bounding box) in the run's `config.yaml` under the key `validation_rois`.
4. Freeze: once recorded, ROIs must not be modified for that volume. New volumes may add new ROIs; existing ROIs must not move.
5. All speckle metrics are computed inside these frozen ROIs. If an ROI definition is missing, mark the metric `MISSING_ROI` per Section 4.

## 8) GAN Training Stability Snapshots

For any GAN run, the training notebook must record the following at epochs 10, 25, and 50 (or the nearest completed epoch):

- D_real: mean discriminator score on label patches.
- D_fake: mean discriminator score on generator patches.
- G_adv: generator adversarial loss value.
- G_fidelity: generator fidelity (L1/complex) loss value.

Abort criteria (record as EARLY_FAIL in `verdict.md` and stop training):

- D_real and D_fake both converge to the same value near 0 while G_adv is not decreasing: discriminator is no longer informative.
- D_fake converges to 0 (discriminator wins completely): generator is not learning.
- G_adv or G_fidelity contains NaN.

These snapshots are required artifacts in the run's `logs/`. Absence of snapshots means the run cannot be formally reviewed.
