# VALIDATION_PROTOCOL

Last updated: 2026-06-04 21:32:24 +08:00

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
- Stitch seams: no patch-grid periodic lines.
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
- speckle SNR in homogeneous regions, target reference around Rayleigh speckle SNR `~1.91` where applicable.
- histogram distance or KS statistic for envelope distribution.
- gCNR when valid ROIs are available.
- seam metrics for stitched volumes.

Point/line target metrics when applicable:

- FWHM.
- peak preservation.
- side-lobe or contrast measurement if an ROI is defined.

If ROI definitions are missing, mark the metric as `MISSING_ROI`, not as pass/fail.

## 5) Acceptance Policy

A run can be considered a candidate only when all are true:

- Human Slicer status is `PASS`.
- No gross degradation in fidelity metrics.
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
