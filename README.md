## nmlg_proj1
In-progress research code for running small neural-network architecture sweeps and analyzing training dynamics (especially gradient-norm patterns and “accuracy boost” style events). 

## Repository overview
- `main.py`: train a model from a JSON config and write run artifacts to `outputs/`
- `train.py`: training loop used by `main.py`
- `nets.py`: model definitions + `build_model(...)`
- `load_data.py`: dataset loading (MNIST-style)
- `run_sweep.py`: run many configs in a sweep
- `generate_sweep_*.py`: generate config JSONs for sweeps
- `analyze_results.py`: aggregate runs and compute metrics from saved training history
- `gradient_analysis.py`: gradient metrics / pattern logic
- `plot_results.py`: plotting utilities for aggregated results and example runs

## Quickstart
Create a virtual environment and install dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy matplotlib pandas
```

### Run a single experiment
```
python main.py --config configs/<subfolder>/<config>.json
```
Outputs are written under outputs/ (ignored by git).

### Run a sweep
Generate configs (example):
```
python generate_sweep_three_layer_skip_conv_uniform_lr.py --subfolder three_layer_skip_conv_uniform_lr
```

Run the sweep:
```
python run_sweep.py --subfolder three_layer_skip_conv_uniform_lr --output-subfolder three_layer_skip_conv_uniform_lr
```

### Analysis / plots
Aggregate results and compute gradient-pattern metrics:
```
python analyze_results.py
```
Plot utilities live in plot_results.py (the exact entrypoints vary as the project evolves).

### Config notes (current)
Common keys used across configs:
- architecture (e.g. three_layer_skip)
- input_size (e.g. 784), output_size (e.g. 10)
- hidden_sizes (e.g. [h1, h2, h3])
- activation (e.g. relu)
- optimizer (e.g. Adam)
- ln_rate (default learning rate)
- layer_lns (per-layer LR overrides)

**Optional** (currently only supported for three_layer_skip):
- layer_types (choose conv/linear for early layers), e.g. { "layer1": "conv", "layer2": "linear" }

## What is not tracked
This repo ignores generated artifacts and local environment folders:

ignored: outputs/, results/, configs/, data/, .venv/, __pycache__/
