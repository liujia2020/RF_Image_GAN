# cGAN Runs

Each formal training run gets one subfolder:

```text
YYYY-MM-DD_short_name/
```

Required run files:

- `README.md`
- `config.yaml`
- `run_manifest.json`
- `train.ipynb`
- `validate.ipynb`
- `checkpoints.txt`
- `verdict.md`

Suggested artifact folders:

- `metrics/`
- `figures/`
- `nii/`
- `logs/`

Large artifacts may be ignored by git, but paths must be recorded.
