"""Train the pattern generation task (regression).
 
Design note — training data is fixed
-------------------------------------
Unlike the classification tasks (cue_accumulation, delayed_match) which
generate fresh training batches every epoch, pattern_generation uses a
*single fixed realization* throughout training. The target pattern and
the Poisson input realization are determined once by `cfg.task.seed` and
replayed identically every epoch. Different `cfg.task.seed` values give
different realizations, but within a single run, every epoch sees the
same data.

All three batch-generator methods below consequently use `cfg.task.seed`
rather than deriving per-epoch seeds.
"""
 
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
# Shared task-specific helpers
# =============================================================================
 
def _input_dim(cfg) -> int:
    return cfg.net_arch.n_neurons_channel
 
 
def _generate_batches(cfg, *, n_batches: int, batch_size: int):
    """
    Single source-of-truth wrapper around tasks.pattern_generation.
 
    Note: seed is always `cfg.task.seed`. See module docstring for rationale.
    """
    return tasks.pattern_generation(
        n_batches=n_batches, batch_size=batch_size,
        seed=cfg.task.seed,
        frequencies=cfg.task.frequencies,
        n_population=cfg.net_arch.n_neurons_channel,
        f_input=cfg.task.f_input, trial_dur=cfg.task.trial_dur,
    )
 
 
# =============================================================================
# Batch generators (train / eval / test) — all share the same realization
# =============================================================================
 
def _make_train_batch(cfg, *, epoch: int):
    """Training batches are identical across epochs by design."""
    del epoch  # intentionally unused
    return _generate_batches(
        cfg,
        n_batches=cfg.train_params.train_batch_size,
        batch_size=cfg.train_params.train_mini_batch_size,
    )
 
 
def _make_eval_batch(cfg):
    """Evaluation batch — same realization as training (intentional)."""
    return _generate_batches(
        cfg,
        n_batches=cfg.train_params.test_batch_size,
        batch_size=cfg.train_params.test_mini_batch_size,
    )
 
 
def _make_test_batch(cfg, *, offset: int):
    """Early-stopping test batch — same realization regardless of offset."""
    del offset  # intentionally unused
    return _generate_batches(
        cfg,
        n_batches=cfg.train_params.test_batch_size,
        batch_size=cfg.train_params.test_mini_batch_size,
    )
 
 
# =============================================================================
# Loss / metrics
# =============================================================================
 
def _optimization_loss(logits, labels, z, c_reg, f_target, trial_length):
    """MSE task loss + firing-rate regularization."""
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
    """
    MSE and target-normalized MSE, summed over batches.
 
    `labels` are regression targets here; the arg name matches the shared
    `train_step` signature for uniformity with classification tasks.
    """
    targets = labels
    if targets.ndim == 2:
        targets = jnp.expand_dims(targets, axis=-1)
 
    sq_err = losses.squared_error(targets=targets, predictions=predictions)
    squared_sum_targets = jnp.sum(jnp.square(targets), axis=1)
    normalized = sq_err / squared_sum_targets[:, None, :]
 
    loss = sq_err.mean(axis=1)
    return Metrics(
        loss=jnp.sum(loss),
        normalized_loss=jnp.sum(normalized),
        count=targets.shape[0],
    )
 
 
# =============================================================================
# Example plots
# =============================================================================
 
def _plot_examples(*, cfg, state, eval_batch, output_dir: str, n_examples: int) -> None:
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
            os.path.join(figures_dir, f"example_{i + 1}.svg"), format="svg",
        )
        plt.close(fig)
 
 
# =============================================================================
# Task spec + entry point
# =============================================================================
SPEC = TaskSpec(
    name="pattern_generation",
    task_type="regression",
    input_dim_from_cfg=_input_dim,
    make_train_batch=_make_train_batch,
    make_eval_batch=_make_eval_batch,
    make_test_batch=_make_test_batch,
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
    return train_and_evaluate(cfg, SPEC)
 