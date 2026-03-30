## nmlg_proj1

Project code for running architecture sweeps and post-hoc analysis of training dynamics
(e.g., gradient-norm patterns and “accuracy boost” style events). This repo is in-progress.

### Repository overview

- `main.py`: entry point to train a model from a JSON config.
- `train.py`: training loop used by `main.py`.
- `nets.py`: model definitions + `build_model(...)`.
- `load_data.py`: dataset loading (MNIST-style).
- `run_sweep.py`: run a folder of configs as a sweep.
- `generate_sweep_*.py`: scripts that generate config JSONs for sweeps.
- `analyze_results.py`: aggregates runs and computes metrics from `training_history.json`.
- `gradient_analysis.py`: defines gradient metrics/pattern logic used in analysis/plots.
- `plot_results.py`: plotting utilities for aggregated results and example runs.
