"""Train the cue accumulation task."""
 
from __future__ import annotations
 
import os
from functools import lru_cache
from typing import Optional
 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from flax import struct
from flax.linen import softmax
from jax import numpy as jnp
from optax import losses
 
from src.modRNN import learning_utils, plots, tasks
from src.modRNN.training import TaskSpec, train_and_evaluate
from src.modRNN.training.common import epoch_seed_sequence
from flax.typing import Array
 
 
# =============================================================================
# Shared task-specific helpers
# =============================================================================
 
def _input_dim(cfg) -> int:
    """Number of input channels depends on cfg.task.input_mode."""
    if cfg.task.input_mode == "original":
        return 4 * cfg.net_arch.n_neurons_channel
    if cfg.task.input_mode == "modified":
        return 3 * cfg.net_arch.n_neurons_channel
    raise ValueError(f"Unknown input_mode: {cfg.task.input_mode!r}")
 
 
def _generate_batches(cfg, *, n_batches: int, batch_size: int, seed: int):
    """Single source-of-truth wrapper around tasks.cue_accumulation_task."""
    return tasks.cue_accumulation_task(
        n_batches=n_batches, batch_size=batch_size, seed=seed,
        n_cues=cfg.task.n_cues,
        min_delay=cfg.task.min_delay, max_delay=cfg.task.max_delay,
        n_population=cfg.net_arch.n_neurons_channel,
        f_input=cfg.task.f_input, f_background=cfg.task.f_background,
        t_cue=cfg.task.t_cue, t_cue_spacing=cfg.task.t_cue_spacing,
        p=cfg.task.p, input_mode=cfg.task.input_mode, dt=cfg.task.dt,
    )
 
 
# =============================================================================
# Batch generators (train / eval / test)
# =============================================================================
# Cache the per-epoch seed sequence so it's computed once per (seed, n_iter)
# pair. 
@lru_cache(maxsize=8)
def _cached_epoch_seeds(master_seed: int, n_epochs: int):
    return epoch_seed_sequence(master_seed, n_epochs)
 
 
def _make_train_batch(cfg, *, epoch: int):
    """Training batch varies across epochs, deterministic from cfg.task.seed."""
    seeds = _cached_epoch_seeds(cfg.task.seed, cfg.train_params.iterations)
    train_seed = int(seeds[epoch - 1])
    return _generate_batches(
        cfg,
        n_batches=cfg.train_params.train_batch_size,
        batch_size=cfg.train_params.train_mini_batch_size,
        seed=train_seed,
    )
 
 
def _make_eval_batch(cfg):
    """Fixed evaluation set, seeded from cfg.task.seed."""
    return _generate_batches(
        cfg,
        n_batches=cfg.train_params.test_batch_size,
        batch_size=cfg.train_params.test_mini_batch_size,
        seed=cfg.task.seed,
    )
 
 
def _make_test_batch(cfg, *, offset: int):
    """Early-stopping confirmation batch; uses a distinct seed per offset."""
    return _generate_batches(
        cfg,
        n_batches=cfg.train_params.test_batch_size,
        batch_size=cfg.train_params.test_mini_batch_size,
        seed=cfg.task.seed + offset + 1,
    )
 
 
# =============================================================================
# Loss / metrics
# =============================================================================
 
def _optimization_loss(logits, labels, z, c_reg, f_target, trial_length):
    """Cross-entropy task loss + firing-rate regularization."""
    task_loss = jnp.mean(losses.softmax_cross_entropy(logits=logits, labels=labels))
    av_f_rate = learning_utils.compute_firing_rate(z=z, trial_length=trial_length)
    f_target_per_ms = f_target / 1000
    reg_loss = 0.5 * c_reg * jnp.sum(
        jnp.mean(jnp.square(av_f_rate - f_target_per_ms), axis=0)
    )
    return task_loss + reg_loss
 
 
class Metrics(struct.PyTreeNode):
    """Cue-accumulation metrics: cross-entropy loss + binary accuracy."""
    loss: float
    accuracy: Optional[float] = None
    count: Optional[int] = None
 
 
def _compute_metrics(*, labels: Array, predictions: Array) -> Metrics:
    loss = losses.softmax_cross_entropy(labels=labels, logits=predictions)
    loss = jnp.mean(loss, axis=-1)
    inference = jnp.argmax(jnp.sum(predictions, axis=1), axis=-1)
    label = jnp.argmax(labels[:, 0, :], axis=-1)
    correct = jnp.equal(inference, label)
    return Metrics(
        loss=jnp.sum(loss),
        accuracy=jnp.sum(correct),
        count=predictions.shape[0],
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
    recurrent_carries, logits = state.apply_fn(variables, batch['input'])
    y = softmax(logits)
    _, _, _, z, _ = recurrent_carries
 
    n_to_plot = min(n_examples, batch['input'].shape[0])
    for i in range(n_to_plot):
        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 2.5, 2.5, 2.5])
        ax_in, ax_lif, ax_alif, ax_out = [fig.add_subplot(gs[k]) for k in range(4)]
 
        plots.plot_cue_accumulation_inputs(
            batch['input'][i], n_population=cfg.net_arch.n_neurons_channel,
            input_mode=cfg.task.input_mode, ax=ax_in,
        )
        plots.plot_recurrent(
            z[i], n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF,
            neuron_type="LIF", ax=ax_lif,
        )
        plots.plot_recurrent(
            z[i], n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF,
            neuron_type="ALIF", ax=ax_alif,
        )
        plots.plot_softmax_output(y[i, :, 0], ax=ax_out)
 
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
    name="cue_accumulation",
    task_type="classification",
    input_dim_from_cfg=_input_dim,
    make_train_batch=_make_train_batch,
    make_eval_batch=_make_eval_batch,
    make_test_batch=_make_test_batch,
    optimization_loss=_optimization_loss,
    metrics_class=Metrics,
    compute_metrics=_compute_metrics,
    metric_names=("loss", "accuracy"),
    early_stop_metric="accuracy",
    early_stop_better="higher",
    log_format="%s epoch %03d loss %.4f accuracy %.2f%%",
    log_scale=(1.0, 100.0),
    plot_examples=_plot_examples,
    plot_example_count=3,
)
 
 
def train_and_evaluate_entry(cfg):
    return train_and_evaluate(cfg, SPEC)