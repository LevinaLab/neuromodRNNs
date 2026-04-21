"""
Shared end-of-training plotting utilities.

Plots that are identical across all tasks:
  * training / evaluation metric curves
  * learned weights (hist + spatial layout)
  * firing-rate stats (logged, not plotted)

Per-task example figures (spikes + task inputs + outputs) are per-task
and dispatched through `TaskSpec.plot_examples`.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
from jax import numpy as jnp

from src.modRNN import learning_utils
from src.modRNN import plots
from src.modRNN.training.task_spec import TaskSpec


def plot_training_curves(
    history: Dict[str, List],
    spec: TaskSpec,
    output_dir: str,
) -> None:
    """
    Plot train/eval curves for each metric in spec.metric_names.

    Produces a 2 x len(metrics) grid: top row = training, bottom row = eval.
    Saved to `<output_dir>/figures/training.svg`.
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    n_metrics = len(spec.metric_names)
    fig, axs = plt.subplots(
        2, n_metrics,
        figsize=(6 * n_metrics, 8),
        squeeze=False,
    )

    iterations = history["iterations"]
    for col, name in enumerate(spec.metric_names):
        label = name.replace("_", " ").title()

        axs[0, col].plot(iterations, history[f"{name}_training"],
                         label=f'Training {label}')
        axs[0, col].set_title(f'Training {label}')
        axs[0, col].set_xlabel('Iterations')
        axs[0, col].set_ylabel(label)
        axs[0, col].legend()

        axs[1, col].plot(iterations, history[f"{name}_eval"],
                         label=f'Evaluation {label}', color='tab:red')
        axs[1, col].set_title(f'Evaluation {label}')
        axs[1, col].set_xlabel('Iterations')
        axs[1, col].set_ylabel(label)
        axs[1, col].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'training.svg'), format='svg')
    plt.close(fig)


def plot_final_weights(state, cfg, output_dir: str) -> None:
    """Weight histograms and spatially-indexed weight layouts."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    layer_names = ["Input layer", "Recurrent layer", "Readout layer"]
    plots.plot_LSNN_weights(
        state,
        layer_names=layer_names,
        save_path=os.path.join(figures_dir, "weights.svg"),
    )
    plots.plot_weights_spatially_indexed(
        state, cfg.net_arch.gridshape,
        os.path.join(figures_dir, "spatially_weights.svg"),
    )


def log_firing_rate_stats(state, sample_batch, logger: logging.Logger) -> None:
    """Log mean/max/min firing rates on a sample batch (in Hz)."""
    variables = {
        'params': state.params,
        'eligibility params': state.eligibility_params,
        'spatial params': state.spatial_params,
    }
    recurrent_carries, _ = state.apply_fn(variables, sample_batch['input'])
    _, _, _, z, _ = recurrent_carries
    firing_rates = 1000 * learning_utils.compute_firing_rate(
        z, sample_batch["trial_duration"]
    )
    logger.info(
        'firing rate eval set  average %.1f max %.1f min %.1f',
        jnp.mean(firing_rates), jnp.max(firing_rates), jnp.min(firing_rates),
    )