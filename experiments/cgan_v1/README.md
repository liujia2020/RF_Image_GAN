# cGAN v1 Experiment Track

Last updated: 2026-06-04 21:32:24 +08:00

This folder is the clean workspace for the conditional GAN direction.

Current status:

- Regression is frozen as a negative result for DAS-like speckle quality.
- This track is not yet implemented.
- No training should run before `docs/CGAN_PLAN.md`, `docs/VALIDATION_PROTOCOL.md`, and `docs/RUNBOOK.md` are accepted.

## Folder Layout

```text
experiments/cgan_v1/
  configs/
  notebooks/
  scripts/
  templates/
  runs/
```

## Rules

- New cGAN files go here, not in the project root.
- Every formal run gets a folder under `runs/`.
- Large generated artifacts should be referenced by path and normally stay out of git.
- Human Slicer review is required before quality claims.

## First Milestone

Create a minimal smoke-test implementation after design review:

- generator/discriminator skeleton
- one tiny train loop
- no quality claim
- no full training without explicit user confirmation
