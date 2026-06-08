# RF_Image

RF residual-learning experiments for RCA-OPW beamforming reconstruction.

## Project Docs

- [Project handoff](docs/PROJECT_HANDOFF.md)
- [Experiments log](docs/EXPERIMENTS_LOG.md)
- [Collaboration notes](docs/claude.md)

## Source Layout

- `rf_*.py`: dataset, model, training, evaluation, cache, stitching, and visualization utilities.
- `*.ipynb`: training and evaluation notebooks for Tiny and UNet experiments.
- `build_RF_*.m`: MATLAB sample generation scripts.
- `MATLAB_dense_generation/`: dense sliding-window generation scripts.

Large datasets, cache files, checkpoints, metrics, logs, and visual outputs are intentionally excluded from git.
