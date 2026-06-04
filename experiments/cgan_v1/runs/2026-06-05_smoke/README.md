# 2026-06-05_smoke

Created: 2026-06-05 01:20:00 +08:00

## Why

Verify that the minimal cGAN training loop runs on the local 8GB GPU without NaN, OOM, or immediate single-sided collapse. This run makes no image-quality claim.

## Hypothesis

Over the first 20 epochs, G and D both show non-trivial movement without NaN/OOM/single-sided convergence.

## Expected Outcome

PASS means the project may proceed to a pilot lambda scan.

EARLY_ABORT means diagnose the printed reason and fix the loop before any pilot training.

## Status

pending
