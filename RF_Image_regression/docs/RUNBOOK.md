# RUNBOOK

Last updated: 2026-06-04 21:32:24 +08:00

## 1) Purpose

This runbook defines how RF-to-volume experiments are created, recorded, reviewed, and closed. It is designed for both human review and AI-assisted project continuity.

No formal training run should start without a run folder and frozen config.

## 2) Run Folder Standard

Each formal run gets one folder:

```text
experiments/<track>/runs/<YYYY-MM-DD_short_name>/
  README.md
  config.yaml
  run_manifest.json
  train.ipynb
  validate.ipynb
  metrics/
  figures/
  nii/
  logs/
  checkpoints.txt
  verdict.md
```

Large binary artifacts may live outside git, but paths must be recorded.

## 3) Required Files

`README.md`:

- Why this run exists.
- One-variable hypothesis.
- Expected outcome.
- Final status.

`config.yaml`:

- The frozen source of truth for training.
- If it changes after training starts, create a new run.

`run_manifest.json`:

- timestamp
- task name
- git commit
- dirty git status
- config hash
- dataset paths
- checkpoint paths
- Python/CUDA/GPU environment

`train.ipynb`:

- Startup check.
- Config printout.
- Loss curves.
- Notes on interruptions/OOM/restarts.

`validate.ipynb`:

- NIfTI export.
- Metric generation.
- Figure generation.
- Pointers to Slicer review materials.

`verdict.md`:

- Human visual status.
- Metric status.
- Final decision.
- What failed, if anything.
- Next action.

## 4) Lifecycle

1. Plan: write the run purpose and hypothesis.
2. Freeze config: write `config.yaml`; record git state.
3. Train: run only from the frozen config.
4. Export materials: NIfTI, figures, metrics.
5. Human review: user checks Slicer materials.
6. Close: write `verdict.md`, update `EXPERIMENTS_LOG.md`, then commit.

## 5) Timestamp Rule

Every run and diagnostic output begins with:

```text
YYYY-MM-DD HH:MM:SS +08:00 | TASK: <task name>
```

The same timestamp or generated timestamp must appear in `run_manifest.json`.

## 6) Failure Recording Rule

Failures are first-class results. Record:

- What was expected.
- What actually happened.
- How it was detected.
- Whether it was metric-only, visual-only, or both.
- Whether the result invalidates a hypothesis.

Do not dilute failures into vague "needs improvement" language.

## 7) AI Role Boundary

Codex may:

- Generate code, configs, metrics, figures, NIfTI files, and summaries.
- Point out risks or inconsistencies.
- Propose interpretations with evidence.

Codex may not:

- Declare image quality pass without user Slicer review.
- Treat proxy metrics as final quality.
- Start a new formal training run without a run folder and frozen config.

## 8) Commit Rule

One meaningful action, one commit. Commit messages must describe the actual change. Do not mix training code, docs, and large-output cleanup in one commit unless explicitly planned.
