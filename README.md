# RF_Image_GAN

Created: 2026-06-04 21:32:24 +08:00

This repository is the clean workspace for the RF-to-DAS conditional GAN direction.

## Current Project Verdict

The previous pure-regression line is frozen as a negative result. It improved proxy metrics but failed NIfTI + 3D Slicer human review: phantom line targets were lost and tissue speckle became mush-like.

The active direction is paired conditional GAN:

- condition: RF input and related low-channel reconstruction context
- target: DAS label
- adversarial objective: realistic DAS-like speckle distribution
- fidelity objective: preserve frame-specific coherent structure and prevent hallucination

## What This Repository Contains Now

```text
docs/
  CGAN_PLAN.md
  VALIDATION_PROTOCOL.md
  RUNBOOK.md
  PROJECT_HANDOFF.md
  EXPERIMENTS_LOG.md
  claude.md

experiments/
  cgan_v1/
    configs/
    notebooks/
    scripts/
    templates/
    runs/

legacy/
  regression/
```

No training code has been added yet. No training should start before a run folder and frozen config exist.

## Hard Rule

Quality claims require the validation gate:

- human NIfTI / 3D Slicer review
- standard metrics
- run-level `verdict.md`

Proxy metrics alone are not a pass.

## Relationship To `RF_Image`

`RF_Image` remains the full historical project with regression scripts, outputs, caches, and checkpoints. This repository starts the clean cGAN line and should not receive large generated data.
