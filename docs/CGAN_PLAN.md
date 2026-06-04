# CGAN_PLAN

Last updated: 2026-06-05 01:17:55 +08:00

## 1) Current Verdict

The regression line is frozen as a negative result. WideDeep+SSIM+crop8 improved proxy metrics, but NIfTI + 3D Slicer human review rejected it: phantom line targets were lost and tissue speckle became mush-like. The project now pivots to a paired conditional GAN family.

This document is a planning record only. It does not authorize training by itself.

## 2) Goal

Use a neural network to map delay-aligned multi-angle RF tensors directly to an image-quality output comparable to DAS, including realistic tissue/phantom speckle and preserved coherent structures.

The target is not "better proxy metrics"; the target is visual and quantitative agreement with DAS under the validation gate in `docs/VALIDATION_PROTOCOL.md`. Required speckle statistics are defined in VALIDATION_PROTOCOL Section 4: speckle SNR (vs label-measured in same ROI), envelope histogram KS statistic, and 2D speckle autocorrelation. These are mandatory output artifacts for every formal run.

## 3) Why Conditional Paired GAN

The dataset is paired: RF input corresponds to a DAS label. That makes a conditional paired GAN the appropriate family:

- The generator is conditioned on RF input and optionally baseline.
- The discriminator judges whether an output is DAS-like under the same condition.
- Fidelity losses keep the output bound to this RF frame and prevent hallucination.

Unconditional GAN is not reconstruction. CycleGAN discards paired supervision and is only a fallback when paired data do not exist.

## 4) First-Phase Scope

Primary scope:

- Tissue and phantom realistic speckle.
- Paired RF-to-DAS reconstruction.
- Full-volume stitch compatibility.
- Validation-gate artifacts: NIfTI, standard metrics, fixed comparison figures.

Out of first-phase scope:

- `simu_point` as a main training class.
- Diffusion models, because 3D volume training on local 8GB GPU is not practical now.
- Any claim of quality pass without human Slicer review.

Point targets remain a sanity check and later branch, not the first GAN objective.

## 5) Reusable Assets From Regression Line

Keep and reuse:

- `RFLearningDataset`, `RFCachedDataset`, and cache builder logic.
- BN normalization lesson and startup self-check discipline.
- Stitching, crop8/halo, coverage checks.
- NIfTI export and validation-gate discipline.
- Regression checkpoints as baselines and negative-result evidence.

Do not reuse as final objective:

- Pure L1/SSIM regression as the main training paradigm.
- Proxy-metric-only acceptance.

## 6) Candidate Architecture Questions

Generator candidates:

- Start from a conservative residual RF-to-complex generator.
- Keep shape contract: `input [B,1536,Z,X,Y]`, `baseline [B,2,Z,X,Y]`, output `[B,2,Z,X,Y]`.
- Use AMP and small batch first; local smoke tests only.

Discriminator candidates:

- PatchGAN-style discriminator on envelope or dB envelope.
- Conditional discriminator that receives baseline/envelope or low-channel condition, not full 1536 RF at first.
- Consider 2D-slice discriminator first for memory, then 3D patch discriminator if feasible.

Loss candidates:

- Fidelity: complex L1 and/or envelope L1. Do NOT add SSIM. SSIM's covariance term requires pred and label speckle patterns to correlate, which is impossible for random speckle; it pulls G toward the conditional mean exactly like L1 regression.

- Fidelity target decomposition (critical): voxel-wise L1/complex constrains every voxel including the random speckle component, which G cannot predict exactly; this forces G toward the conditional mean (mush). The correct design is to restrict fidelity to deterministic/learnable components only:
  - Coherent structures: vessel walls, tissue interfaces, phantom structures.
  - Low-frequency envelope trend (large-scale echo level).
  - Point/line target positions when present.
  Fine-scale random speckle must be left to the adversarial term, not constrained by fidelity. Tuning lambda alone cannot resolve this; the decomposition of what fidelity acts on is the primary design variable.

- Adversarial: hinge GAN or LSGAN, chosen for stability.
- Feature matching if discriminator instability appears.
- No quality claim until validation gate passes.

## 7) Training Risk Register

Risks are ordered by probability for a paired conditional GAN with fidelity loss. Mode collapse probability is low because paired supervision prevents G from ignoring the RF condition.

1. Fidelity too strong -> mush (highest risk).
   Detection: G_adv loss stagnates near its initial value and G has no incentive to improve adversarially; speckle SNR stays well above label-measured ROI SNR; envelope histogram KS does not decrease over training.

2. Hallucination: realistic speckle but wrong coherent structure.
   Detection: structural correlation (pred vs label on low-pass envelope) degrades relative to BN regression baseline; point/line targets visible in label disappear in pred.

3. Speckle realism improves while structural fidelity falls.
   Detection: speckle SNR approaches label value but voxel-wise complex L1 vs baseline degrades beyond acceptable structural loss (VALIDATION_PROTOCOL acceptance criteria). Note: some voxel-wise L1 degradation relative to the mush baseline is expected and acceptable when speckle becomes realistic.

4. Patch-level smoke-test success but stitch/full-volume failure.
   Detection: seam metrics (amplitude and texture) must be measured on a stitched volume, not on patch-level metrics alone.

5. Proxy metrics improve while human review fails (inherited from regression line).
   Detection: validation gate in VALIDATION_PROTOCOL is the only arbiter.

Every run must record detection signals per risk in its `verdict.md`.

## 8) Next Actions Before Any Training

1. Finalize `docs/RUNBOOK.md`.
2. Create the cGAN experiment folder and templates.
3. Audit current code for reusable modules and files that must not be moved.
4. Only after review, write a minimal cGAN smoke-test implementation.
5. Smoke test is not a formal experiment and must not be interpreted as quality progress. Smoke test has exactly one pass criterion: the training loop runs without NaN, OOM, or D_loss collapse AND both G_adv and D losses are non-trivially moving in the first 20 epochs (neither single-sided convergence to zero). All other observations during smoke test are informational only. Smoke test result must not trigger the validation gate.
