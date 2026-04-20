"""Train the delayed match-to-sample task."""
 
from __future__ import annotations
 
import os
from typing import Optional
 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from flax import struct
from flax.linen import softmax
from jax import numpy as jnp
from optax import losses
 
from .. import learning_utils, plots, tasks
from ..training import TaskSpec, train_and_evaluate
from flax.typing import Array
 
 
# =============================================================================
# Task-specific pieces
# =============================================================================
 
def _input_dim(cfg) -> int:
    # 4 populations: two cues (which themselves use 2 populations each),
    # plus fixation and background channels.
    return 4 * cfg.net_arch.n_neurons_channel
 
 
def _make_batch(cfg, *, n_batches: int, batch_size: int, seed: int):
    """Thin wrapper that pulls task params out of cfg."""
    return tasks.delayed_match_task(
        n_batches=n_batches, batch_size=batch_size, seed=seed,
        n_population=cfg.net_arch.n_neurons_channel,
        f_background=cfg.task.f_background, f_input=cfg.task.f_input,
        p=cfg.task.p,
        fixation_time=cfg.task.fixation_time,
        cue_time=cfg.task.cue_time,
        cue_delay_time=cfg.task.cue_delay_time,
        decision_delay=cfg.task.decision_delay,
        LS_avail=cfg.task.LS_avail,
    )
 
 
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
    """Delayed-match metrics: cross-entropy loss + binary accuracy."""
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
 
 
def _plot_examples(*, cfg, state, eval_batch, output_dir: str, n_examples: int) -> None:
    """Plot example trials showing inputs, spikes, and softmax output."""
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
 
        plots.plot_delayed_match_inputs(
            batch['input'][i], n_population=cfg.net_arch.n_neurons_channel, ax=ax_in,
        )
        plots.plot_recurrent(
            z[i], n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF,
            neuron_type="LIF", ax=ax_lif,
        )
        plots.plot_recurrent(
            z[i], n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF,
            neuron_type="ALIF", ax=ax_alif,
        )
        plots.plot_softmax_output(
            y[i, :, 1], ax=ax_out,
            label="Probability of 1",
            title="Softmax Output: neuron coding 1",
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
    name="delayed_match",
    task_type="classification",
    input_dim_from_cfg=_input_dim,
    make_batch=_make_batch,
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
    """Entry point called from main.py."""
    return train_and_evaluate(cfg, SPEC)