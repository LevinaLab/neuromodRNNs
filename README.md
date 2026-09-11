# Diffusion of neuromodulators for temporal credit assignment

This repository contains the code for the paper:
**Diffusion of neuromodulators for temporal credit assignment**
[João Barretto-Bittar, Anna Levina, Emmanouil Giannakakis, and Roxana Zeraati]


If you have any questions, please contact us through GitHub.

## Environment setup

Python 3.12.3 is required. Install dependencies using either conda or a plain virtual environment.

**With conda:**

```bash
conda create -n modRNN python=3.12.3
conda activate modRNN
pip install -r requirements.txt
```

**With venv:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### JAX with GPU support

The `requirements.txt` installs the CPU-only build of JAX. To run on a GPU, reinstall JAX with the appropriate CUDA wheel **after** the step above — it will replace the CPU build:

```bash
# CUDA 12 (most common on modern clusters)
pip install -U "jax[cuda12]"
```

Check the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) if you need a different CUDA version. Verify that JAX sees the GPU before launching experiments:

```python
python -c "import jax; print(jax.devices())"
# Expected (GPU): [CudaDevice(id=0)]
```

## Project structure

```
.
├── main.py           # Entry point — composes config and dispatches to the right experiment
├── launcher.sh       # Submits one SLURM job per seed
├── worker.sh         # SLURM job script (do not run directly)
├── conf/
│   ├── config.yaml   # Root Hydra config
│   ├── task/         # One YAML per task (e.g. pattern_generation.yaml)
│   └── experiment/   # One YAML per learning rule / experiment variant
├── figures/
│   └── fig2/
│       ├── fig2.ipynb      # Notebook to reproduce Figure 2
│       ├── plot_utils.py   # Auxiliary plotting helpers used by the notebook
│       └── results/        # Pre-computed experiment outputs (committed to git)
├── seeds/            # Seed files used in the paper (one integer per line)
│   ├── seeds.txt     # Default seed list used across all paper experiments
│   └── ...           # Additional per-experiment seed files if applicable
└── src/
    ├── config/       # Config dataclasses
    |── modRNN/       # Learning rules, network, tasks, training loop
    └── train/        # Task-specific training entry points
```

## Running a single experiment

For a quick local run with one seed, call `main.py` directly via Hydra's CLI — no SLURM needed.

```bash
python main.py \
    task=<task> \
    +experiment=<experiment> \
    net_params.seed=<seed> \
    task.seed=<seed> \
    save_paths.experiment_name="<experiment>" \
    save_paths.condition="seed_<seed>"
```

### Arguments

**`task`** — selects a config file from `conf/task/`. The three implemented tasks are:

| Value | Config file |
|---|---|
| `cue_accumulation` | `conf/task/cue_accumulation.yaml` |
| `delayed_match` | `conf/task/delayed_match.yaml` |
| `pattern_generation` | `conf/task/pattern_generation.yaml` |

**`+experiment`** — selects a config file from `conf/experiment/`, layered on top of the task config. The `+` prefix tells Hydra this group is not in the defaults list and is being appended at the CLI. 

**`net_params.seed` / `task.seed`** — integer random seed, passed twice to seed both the network and the task data generation.

### Example

```bash
python main.py \
    task=cue_accumulation \
    +experiment=diffusion \
    net_params.seed=42 \
    task.seed=42 \
    save_paths.experiment_name="diffusion" \
    save_paths.condition="seed_42"
```

### How the config is composed

Hydra merges configuration in the following order (later layers override earlier ones):

1. **`base_config`** — default values from the `ConfigTrain` dataclass in `main.py`
2. **`conf/task/<task>.yaml`** — task-specific hyperparameters (e.g. learning rate, network architecture)
3. **`conf/experiment/<experiment>.yaml`** — experiment/learning-rule overrides (e.g. connectivity, learning rule)
4. **`conf/config.yaml` `_self_` block** — repo-level defaults for `save_paths`
5. **CLI key-value overrides** — per-run values like `net_params.seed=42`

The resulting config is validated against the `ConfigTrain` dataclass and dispatched to the appropriate training function in `src/train/`.

---

## Reproducing paper results

### Experiment

The table below lists the experiments used to produce the results reported in the paper, which were combined with all the three tasks.
| Type | Experiments |
|---|---|
| Learning Curves |`BPTT`, `diffusion`, `e_prop_hardcoded`, `fixed_suffled_diffusion.yaml`|
| Gradient Alignment |`align_local_connectivity_diffusion_aligned`, `align_local_connectivity_diffusion_fixed` |
| nn-connectivity |`diffusion_nn`, `fixed_shuffle_diffusion_nn`, `align_nn_connectivity_diffusion_aligned`, `align_nn_connectivity_diffusion_fixed.yaml`|

### Figures

Figure 2 can be reproduced directly from the notebook:

```
figures/fig2/fig2.ipynb
```

Pre-computed results for all experiments shown in Figure 2 are committed to the repository under `figures/fig2/results/`, so the plot can be generated without re-running the full training sweep. Simply open the notebook and run all cells — it will load the saved results by default.

To instead regenerate the results from scratch, run the relevant experiments (see the [Experiment](#experiment) table above) and point the notebook to your output directory before running.

The auxiliary script `figures/fig2/plot_utils.py` is imported by the notebook and contains the plotting helpers; it does not need to be run directly.

### Seed files

Seed files are stored under `seeds/` at the repository root. Each file is a plain text list with one integer seed per line (blank lines and lines starting with `#` are ignored). The corresponding files follow the naming convention `seeds/seeds_<task>_<experiment>.txt`.

---

## Running multiple seeds on a SLURM cluster

For large sweeps, use `launcher.sh` to submit one job per seed:

```bash
bash launcher.sh <seeds_file> <task> <experiment>
```

**`seeds_file`** is a plain text file with one integer seed per line (comments starting with `#` and blank lines are ignored).

```bash
# seeds.txt
0
1
42
```

```bash
bash launcher.sh seeds/pattern_generation_seeds.txt pattern_generation diffusion
```
