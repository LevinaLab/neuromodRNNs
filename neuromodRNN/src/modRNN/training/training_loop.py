"""
Standard training experiment: train + periodic evaluation + early stopping.
 
`train_and_evaluate(cfg, spec)` is the entry point used by the per-task
train scripts (`src/train/train_*.py`). It composes the shared
primitives (model build, state, train_step) from `primitives.py` with
the bookkeeping specific to this experiment type:
 
  * Periodic evaluation every 25 epochs.
  * Early stopping on a configurable metric criterion (3-batch
    confirmation).
  * Per-epoch metric history accumulation.
  * End-of-run plotting and history pickling.
"""
 
from __future__ import annotations
 
import logging
import os
import pickle
from functools import partial
from typing import Any, Callable, Dict, Iterable, List
 
import hydra
import jax
from jax import random
 
from src.modRNN.training.primitives import (
    TrainStateEProp,
    create_train_state,
    epoch_seed_sequence,
    model_from_config,
    normalize_batch_metrics,
    train_step,
)
from src.modRNN.training.task_spec import TaskSpec
from src.modRNN.training.plotting import (
    log_firing_rate_stats,
    plot_final_weights,
    plot_training_curves,
)
 
from flax.training.train_state import TrainState
from flax.typing import Array, PRNGKey
 
logger = logging.getLogger(__name__)
 
 
# =============================================================================
# Per-epoch helpers (training-experiment-specific)
# =============================================================================
def _format_log_line(phase: str, epoch: int, metrics, spec: TaskSpec) -> str:
    """Format a log line using spec.log_format and spec.log_scale."""
    values = [
        getattr(metrics, name) * scale
        for name, scale in zip(spec.metric_names, spec.log_scale)
    ]
    return spec.log_format % (phase, epoch, *values)
 
 
def _train_epoch(
    train_step_fn: Callable,
    state: TrainState,
    train_batches: Iterable,
    epoch: int,
    optimization_loss_fn: Callable,
    LS_avail: int,
    f_target: float,
    c_reg: float,
    learning_rule: str,
    task: str,
    diffusion_mode: str,
    key: PRNGKey,
    spec: TaskSpec,
):
    """Run one training epoch."""
    batch_metrics: List[Any] = []
    accumulated_grads = jax.tree_util.tree_map(lambda x: x * 0, state.params)
 
    for batch in train_batches:
        state, metrics, grads = train_step_fn(
            state=state, batch=batch,
            optimization_loss_fn=optimization_loss_fn,
            LS_avail=LS_avail, f_target=f_target, c_reg=c_reg,
            learning_rule=learning_rule, task=task,
            diffusion_mode=diffusion_mode, key=key,
        )
        batch_metrics.append(metrics)
        accumulated_grads = jax.tree_util.tree_map(
            lambda a, g: a + g, accumulated_grads, grads
        )
 
    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics, spec.metric_names)
    logger.info(_format_log_line('train', epoch, metrics, spec))
    return state, metrics, accumulated_grads
 
 
def _eval_step(
    state: TrainState,
    batch: Dict[str, Array],
    LS_avail: int,
    compute_metrics_fn: Callable,
):
    """Evaluate on a single micro-batch."""
    variables = {
        'params': state.params,
        'eligibility params': state.eligibility_params,
        'spatial params': state.spatial_params,
    }
    _, y = state.apply_fn(variables, batch['input'])
    return compute_metrics_fn(
        labels=batch['label'][:, -LS_avail:],
        predictions=y[:, -LS_avail:, :],
    )
 
 
def _evaluate_model(
    eval_step_fn: Callable,
    state: TrainState,
    batches: Iterable,
    epoch: int,
    LS_avail: int,
    spec: TaskSpec,
):
    """Evaluate on an iterable of batches."""
    batch_metrics = [eval_step_fn(state, b, LS_avail) for b in batches]
    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics, spec.metric_names)
    logger.info(_format_log_line('eval ', epoch, metrics, spec))
    return metrics
 
 
def _early_stop_confirmed(
    spec: TaskSpec, cfg, state, eval_step_fn, epoch: int
) -> bool:
    """
    Confirm early stopping by checking `stop_criteria` on 3 test batches.
 
    The task decides whether these three batches are different from
    each other (e.g. classification tasks use three different seeds)
    or the same (pattern_generation).
    """
    for i in range(3):
        test_batch = list(spec.make_test_batch(cfg, offset=i))
        test_metrics = _evaluate_model(
            eval_step_fn, state, test_batch, epoch,
            LS_avail=cfg.task.LS_avail, spec=spec,
        )
        value = getattr(test_metrics, spec.early_stop_metric)
        if not spec.early_stop_satisfied(value, cfg.train_params.stop_criteria):
            return False
    return True
 
 
def _save_history(history: Dict[str, List], output_dir: str) -> None:
    """Pickle each metric history to `output_dir/train_info/<key>.pkl`."""
    train_info_dir = os.path.join(output_dir, 'train_info')
    os.makedirs(train_info_dir, exist_ok=True)
    for history_key, values in history.items():
        with open(os.path.join(train_info_dir, f'{history_key}.pkl'), 'wb') as f:
            pickle.dump(values, f, pickle.HIGHEST_PROTOCOL)
 
 
# =============================================================================
# Main training loop
# =============================================================================
def train_and_evaluate(cfg, spec: TaskSpec) -> TrainState:
    """
    Run the full training + evaluation loop for a task described by `spec`.
    """
    # -- Setup -----------------------------------------------------------
    n_in = spec.input_dim_from_cfg(cfg)
 
    master_key = random.key(cfg.net_params.seed)
    rule_key, state_key, _ = random.split(master_key, 3)
 
    model = model_from_config(cfg)
    state = create_train_state(
        rng=state_key,
        learning_rate=cfg.train_params.lr,
        model=model,
        input_shape=(cfg.train_params.train_mini_batch_size, n_in),
        batch_size=cfg.train_params.train_batch_size,
        mini_batch_size=cfg.train_params.train_mini_batch_size,
    )
 
    train_step_fn = jax.jit(
        partial(train_step, compute_metrics_fn=spec.compute_metrics),
        static_argnames=["LS_avail", "learning_rule", "task", "diffusion_mode"],
    )
    eval_step_fn = jax.jit(
        partial(_eval_step, compute_metrics_fn=spec.compute_metrics),
        static_argnames=["LS_avail"],
    )
 
    closure = jax.tree_util.Partial(spec.optimization_loss)
 
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
 
    # -- Evaluation set is fixed for the whole run ----------------------
    eval_batch = list(spec.make_eval_batch(cfg))
 
    # -- Metric histories -----------------------------------------------
    history: Dict[str, List] = {f"{name}_training": [] for name in spec.metric_names}
    history.update({f"{name}_eval": [] for name in spec.metric_names})
    history["iterations"] = []
 
    # -- Training loop --------------------------------------------------
    logger.info('Starting training...')
 
    final_epoch = 0
    stopped_early = False
    for epoch in range(1, cfg.train_params.iterations + 1):
        sub_rule_key, rule_key = random.split(rule_key)
 
        train_batches = spec.make_train_batch(cfg, epoch=epoch)
 
        logger.info("\t Starting Epoch: %d", epoch)
        state, train_metrics, _ = _train_epoch(
            train_step_fn=train_step_fn,
            state=state,
            train_batches=train_batches,
            epoch=epoch,
            optimization_loss_fn=closure,
            LS_avail=cfg.task.LS_avail,
            f_target=cfg.train_params.f_target,
            c_reg=cfg.train_params.c_reg,
            learning_rule=cfg.train_params.learning_rule,
            task=cfg.task.task_type,
            diffusion_mode=cfg.train_params.diffusion_mode,
            key=sub_rule_key,
            spec=spec,
        )
        final_epoch = epoch
 
        # -- Periodic evaluation + early-stopping check -----------------
        if (epoch - 1) % 25 == 0:
            eval_metrics = _evaluate_model(
                eval_step_fn, state, eval_batch, epoch,
                LS_avail=cfg.task.LS_avail, spec=spec,
            )
            for name in spec.metric_names:
                history[f"{name}_training"].append(getattr(train_metrics, name))
                history[f"{name}_eval"].append(getattr(eval_metrics, name))
            history["iterations"].append(epoch - 1)
 
            value = getattr(eval_metrics, spec.early_stop_metric)
            if spec.early_stop_satisfied(value, cfg.train_params.stop_criteria):
                if _early_stop_confirmed(
                    spec, cfg, state, eval_step_fn, epoch
                ):
                    logger.info(
                        'Met early stopping criteria, breaking at epoch %d',
                        epoch,
                    )
                    stopped_early = True
                    break
 
    # -- Final evaluation after last epoch ------------------------------
    eval_metrics = _evaluate_model(
        eval_step_fn, state, eval_batch, final_epoch,
        LS_avail=cfg.task.LS_avail, spec=spec,
    )
    for name in spec.metric_names:
        history[f"{name}_training"].append(getattr(train_metrics, name))
        history[f"{name}_eval"].append(getattr(eval_metrics, name))
    history["iterations"].append(final_epoch)
 
    # -- Save + plot ----------------------------------------------------
    _save_history(history, output_dir)
    plot_training_curves(history, spec, output_dir)
    plot_final_weights(state, cfg, output_dir)
    log_firing_rate_stats(state, eval_batch[0], logger)
    spec.plot_examples(
        cfg=cfg, state=state, eval_batch=eval_batch,
        output_dir=output_dir, n_examples=spec.plot_example_count,
    )
 
    if stopped_early:
        logger.info("Training stopped early at epoch %d.", final_epoch)
    else:
        logger.info("Training completed all %d iterations.", final_epoch)
 
    return state