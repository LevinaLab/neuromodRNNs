"""Train the pattern generation task (regression)."""
 
from __future__ import annotations
 
import os
from typing import Optional
 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from flax import struct
from jax import numpy as jnp
from optax import losses
 
from src.modRNN import learning_utils, plots, tasks
from src.modRNN.training import TaskSpec, train_and_evaluate
from flax.typing import Array
 
 
# =============================================================================
# Task-specific pieces
# =============================================================================
 
def _input_dim(cfg) -> int:
    # Single input population emits Poisson noise; signal emerges from network.
    return cfg.net_arch.n_neurons_channel
 
 
def _make_batch(cfg, *, n_batches: int, batch_size: int, seed: int):
    """Thin wrapper that pulls task params out of cfg."""
    return tasks.pattern_generation(
        n_batches=n_batches, batch_size=batch_size, seed=seed,
        frequencies=cfg.task.frequencies,
        n_population=cfg.net_arch.n_neurons_channel,
        f_input=cfg.task.f_input, trial_dur=cfg.task.trial_dur,
    )
 
 
def _optimization_loss(logits, labels, z, c_reg, f_target, trial_length):
    """MSE task loss + firing-rate regularization.
 
    `logits` here is the raw readout (no softmax, because this is regression).
    """
    if labels.ndim == 2:
        labels = jnp.expand_dims(labels, axis=-1)
    task_loss = 0.5 * jnp.mean(losses.squared_error(targets=labels, predictions=logits))
    av_f_rate = learning_utils.compute_firing_rate(z=z, trial_length=trial_length)
    f_target_per_ms = f_target / 1000
    reg_loss = 0.5 * c_reg * jnp.sum(
        jnp.mean(jnp.square(av_f_rate - f_target_per_ms), axis=0)
    )
    return task_loss + reg_loss
 
 
class Metrics(struct.PyTreeNode):
    """Pattern-generation metrics: MSE (`loss`) and target-normalized MSE."""
    loss: float
    normalized_loss: Optional[float] = None
    count: Optional[int] = None
 
 
def _compute_metrics(*, labels: Array, predictions: Array) -> Metrics:
    """MSE and target-normalized MSE, summed over batches.
 
    `labels` here are the regression targets (not class labels). The arg name
    stays `labels` for uniformity with the shared `train_step` signature.
    """
    targets = labels
    if targets.ndim == 2:
        targets = jnp.expand_dims(targets, axis=-1)
 
    sq_err = losses.squared_error(targets=targets, predictions=predictions)
    # Normalize each trial by the target's squared-sum (targets are zero-mean).
    squared_sum_targets = jnp.sum(jnp.square(targets), axis=1)
    normalized = sq_err / squared_sum_targets[:, None, :]
 
    loss = sq_err.mean(axis=1)
    return Metrics(
        loss=jnp.sum(loss),
        normalized_loss=jnp.sum(normalized),
        count=targets.shape[0],
    )
 
 
def _plot_examples(*, cfg, state, eval_batch, output_dir: str, n_examples: int) -> None:
    """Plot example trials: Poisson inputs, recurrent spikes, prediction vs target."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
 
    batch = eval_batch[0]
    variables = {
        'params': state.params,
        'eligibility params': state.eligibility_params,
        'spatial params': state.spatial_params,
    }
    recurrent_carries, y = state.apply_fn(variables, batch['input'])
    _, _, _, z, _ = recurrent_carries
 
    n_to_plot = min(n_examples, batch['input'].shape[0])
    for i in range(n_to_plot):
        fig = plt.figure(figsize=(8, 9))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 2.5, 4])
        ax_in, ax_lif, ax_out = [fig.add_subplot(gs[k]) for k in range(3)]
 
        plots.plot_pattern_generation_inputs(batch['input'][i], ax=ax_in)
        plots.plot_recurrent(
            z[i], n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF,
            neuron_type="LIF", ax=ax_lif,
        )
        plots.plot_pattern_generation_prediction(
            y[i], targets=batch["label"][i], ax=ax_out,
        )
 
        fig.suptitle(f"Example {i + 1}: {cfg.save_paths.condition}")
        fig.tight_layout()
        fig.savefig(
            os.path.join(figures_dir, f"example_{i + 1}.png"), format="png",
        )
        plt.close(fig)
 
 
# =============================================================================
# Task spec + entry point
# =============================================================================
SPEC = TaskSpec(
    name="pattern_generation",
    task_type="regression",
    input_dim_from_cfg=_input_dim,
    make_batch=_make_batch,
    optimization_loss=_optimization_loss,
    metrics_class=Metrics,
    compute_metrics=_compute_metrics,
    metric_names=("loss", "normalized_loss"),
    early_stop_metric="normalized_loss",
    early_stop_better="lower",
    log_format="%s epoch %03d MSE %.4f nMSE %.4f",
    log_scale=(1.0, 1.0),
    plot_examples=_plot_examples,
    plot_example_count=3,
)
 
 
def train_and_evaluate_entry(cfg):
    """Entry point called from main.py."""
    return train_and_evaluate(cfg, SPEC)