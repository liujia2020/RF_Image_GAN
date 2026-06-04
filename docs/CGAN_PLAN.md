# CGAN_PLAN

Last updated: 2026-06-04 21:32:24 +08:00

## 1) Current Verdict

The regression line is frozen as a negative result. WideDeep+SSIM+crop8 improved proxy metrics, but NIfTI + 3D Slicer human review rejected it: phantom line targets were lost and tissue speckle became mush-like. The project now pivots to a paired conditional GAN family.

This document is a planning record only. It does not authorize training by itself.

## 2) Goal

Use a neural network to map delay-aligned multi-angle RF tensors directly to an image-quality output comparable to DAS, including realistic tissue/phantom speckle and preserved coherent structures.

The target is not "better proxy metrics"; the target is visual and quantitative agreement with DAS under the validation gate in `docs/VALIDATION_PROTOCOL.md`.

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

- Fidelity: complex L1, envelope L1, possibly SSIM with small weight.
- Adversarial: hinge GAN or LSGAN, chosen for stability.
- Feature matching if discriminator instability appears.
- No quality claim until validation gate passes.

## 7) Training Risk Register

- Mode collapse: discriminator wins or generator produces repeated texture.
- Hallucination: realistic speckle but wrong coherent structure.
- Speckle realism improves while DAS fidelity falls.
- Patch-level success but stitch/full-volume failure.
- Proxy metrics improve while human review fails.

Every run must record these risks in its `verdict.md`.

## 8) Next Actions Before Any Training

1. Finalize `docs/RUNBOOK.md`.
2. Create the cGAN experiment folder and templates.
3. Audit current code for reusable modules and files that must not be moved.
4. Only after review, write a minimal cGAN smoke-test implementation.
5. Smoke test is not a formal experiment and must not be interpreted as quality progress.
