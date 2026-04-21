"""
Shared training pipeline for all LSSN tasks.

This module contains everything that is the *same* between the three tasks
(cue_accumulation, delayed_match, pattern_generation):

  * model construction from a Hydra config
  * TrainState setup and optimizer config (including grad accumulation)
  * train_step / train_epoch / eval_step / evaluate_model
  * the main `train_and_evaluate` loop, including early stopping and
    final plot/save bookkeeping

The pieces that *do* differ between tasks — batch generators, loss functions,
metric definitions, and visualization — are passed in via a `TaskSpec` object
(see `task_spec.py`). The training loop itself is agnostic to the task.
"""

from __future__ import annotations

import logging
import os
import pickle
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import hydra
import jax
import numpy as np
import optax
from flax.training import train_state
from jax import numpy as jnp
from jax import random

from src.modRNN.models import LSSN
from src.modRNN import learning_rules
from src.modRNN.training.task_spec import TaskSpec
from src.modRNN.training.plotting import (
    plot_training_curves,
    plot_final_weights,
    log_firing_rate_stats,
)

from flax.typing import Array, PRNGKey

TrainState = train_state.TrainState
logger = logging.getLogger(__name__)


# =============================================================================
# Model construction
# =============================================================================
def model_from_config(cfg) -> LSSN:
    """
    Build an LSSN model from a Hydra config.

    Note: `b_out` is not currently threaded through because their
    functionality is only partially implemented and they only work correctly
    at their default values. Weight and carry initializers can be modified on
    the returned model before training begins, if needed.
    """
    # Derive per-component seeds from the master seed, so changing
    # cfg.net_params.seed changes *all* network-side RNGs coherently.
    master_key = random.PRNGKey(cfg.net_params.seed)
    subkey, _ = random.split(master_key)
    (feedback_seed,
     rec_connectivity_seed,
     diff_kernel_seed,
     cell_loc_seed,
     input_sparsity_seed,
     readout_sparsity_seed) = random.randint(subkey, (6,), 10_000, 10_000_000)

    return LSSN(
        # architecture
        n_ALIF=cfg.net_arch.n_ALIF,
        n_LIF=cfg.net_arch.n_LIF,
        n_out=cfg.net_arch.n_out,
        sigma=cfg.net_arch.sigma,
        gridshape=cfg.net_arch.gridshape,
        n_neuromodulators=cfg.net_arch.n_neuromodulators,
        sparse_input=cfg.net_arch.sparse_input,
        sparse_readout=cfg.net_arch.sparse_readout,
        connectivity_rec_layer=cfg.net_arch.connectivity_rec_layer,
        feedback=cfg.net_arch.feedback,
        # RSNN neurons params
        thr=cfg.net_params.thr,
        tau_m=cfg.net_params.tau_m,
        tau_adaptation=cfg.net_params.tau_adaptation,
        beta=cfg.net_params.beta,
        gamma=cfg.net_params.gamma,
        refractory_period=cfg.net_params.refractory_period,
        k=cfg.net_params.k,
        radius=cfg.net_params.radius,
        input_sparsity=cfg.net_params.input_sparsity,
        readout_sparsity=cfg.net_params.readout_sparsity,
        recurrent_sparsity=cfg.net_params.recurrent_sparsity,
        # readout params
        tau_out=cfg.net_params.tau_out,
        # learning rule (used only to decide whether to stop_gradient on z)
        learning_rule=cfg.train_params.learning_rule,
        # seeds
        input_sparsity_seed=input_sparsity_seed,
        readout_sparsity_seed=readout_sparsity_seed,
        FeedBack_seed=feedback_seed,
        rec_connectivity_seed=rec_connectivity_seed,
        diff_kernel_seed=diff_kernel_seed,
        cell_loc_seed=cell_loc_seed,
        # weight init
        gain=cfg.net_params.w_init_gain,
        # Simulation
        dt=cfg.net_params.dt,
    )


# =============================================================================
# TrainState
# =============================================================================
class TrainStateEProp(TrainState):
    """TrainState extended with e-prop-specific fields.

    Besides the standard (params, tx, opt_state), we carry:
      * eligibility_params: scalars/arrays needed to compute eligibility traces
        (alpha, rho, kappa, betas, thr, gamma, feedback weights...).
      * spatial_params: connectivity/sparsity masks and diffusion kernel. Not
        learned; kept in the state so they're pytree-mapped alongside params.
      * init_eligibility_carries: initial eligibility vectors,
        used as the starting carry of the eligibility-trace scan each batch.
      * init_error_grid: initial error grid for the diffusion rule.
    """
    eligibility_params: Dict[str, Array]
    spatial_params: Dict[str, Array]
    init_eligibility_carries: Dict[str, Array]
    init_error_grid: Array


def _get_initial_variables(
    rng: PRNGKey, model: LSSN, input_shape: Tuple[int, ...]
) -> Tuple[Dict, Dict, Dict]:
    """Run model.init with a dummy input to materialize all variable collections."""
    dummy_x = jnp.ones(input_shape)
    variables = model.init(rng, dummy_x)
    return (variables['params'],
            variables['eligibility params'],
            variables['spatial params'])


def create_train_state(
    rng: PRNGKey,
    learning_rate: float,
    model: LSSN,
    input_shape: Tuple[int, ...],
    batch_size: int,
    mini_batch_size: int,
) -> TrainStateEProp:
    """
    Create the initial TrainState.

    If batch_size > mini_batch_size, wrap the optimizer in `optax.MultiSteps`
    so that gradients are accumulated across (batch_size / mini_batch_size)
    micro-batches before each weight update. This lets us use large effective
    batch sizes without running out of memory per step.
    """
    key_params, key_elig, key_grid = random.split(rng, 3)
    params, eligibility_params, spatial_params = _get_initial_variables(
        key_params, model, input_shape
    )

    # By default, both are initialized as 0-valued arrays
    init_eligibility_carries = model.initialize_eligibility_carry(key_elig, input_shape)
    init_error_grid = model.initialize_grid(rng=key_grid, input_shape=input_shape)

    tx = optax.adam(learning_rate=learning_rate)
    grad_accum_steps = int(batch_size / mini_batch_size)
    if grad_accum_steps > 1:
        tx = optax.MultiSteps(
            opt=tx, every_k_schedule=grad_accum_steps, use_grad_mean=True
        )

    return TrainStateEProp.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        eligibility_params=eligibility_params,
        spatial_params=spatial_params,
        init_eligibility_carries=init_eligibility_carries,
        init_error_grid=init_error_grid,
    )


# =============================================================================
# Metric aggregation (task-independent)
# =============================================================================
def normalize_batch_metrics(batch_metrics: Sequence[Any], metric_names: Sequence[str]):
    """
    Sum each metric across batches and divide by total count.

    Works for any `Metrics` PyTreeNode class whose fields are in `metric_names`
    plus a `count` field. This is why we require `metric_names[0] == "loss"`
    in TaskSpec — having a convention lets us generalize.
    """
    metric_cls = type(batch_metrics[0])
    total_count = np.sum([m.count for m in batch_metrics])
    kwargs = {}
    for name in metric_names:
        total = np.sum([getattr(m, name) for m in batch_metrics])
        # .item() to convert np scalar -> python scalar for clean logging
        kwargs[name] = total.item() / total_count
    return metric_cls(**kwargs)


def _format_log_line(phase: str, epoch: int, metrics, spec: TaskSpec) -> str:
    """Format a log line using spec.log_format and spec.log_scale."""
    values = [
        getattr(metrics, name) * scale
        for name, scale in zip(spec.metric_names, spec.log_scale)
    ]
    return spec.log_format % (phase, epoch, *values)


# =============================================================================
# Train / eval steps
# =============================================================================
def train_step(
    state: TrainState,
    batch: Dict[str, Array],
    optimization_loss_fn: Callable,
    LS_avail: int,
    f_target: float,
    c_reg: float,
    learning_rule: str,
    task: str,
    shuffle: bool,
    shuffle_key: PRNGKey,
    compute_metrics_fn: Callable,
) -> Tuple[TrainState, Any, Dict]:
    """Train for a single micro-batch step.

    Returns (new_state, metrics, grads). Grads are returned only for
    diagnostic plotting downstream — the actual weight update has already
    happened via state.apply_gradients.
    """
    # LS_avail gates which time steps contribute to the task loss.
    y, grads = learning_rules.compute_grads(
        batch=batch,
        state=state,
        optimization_loss_fn=optimization_loss_fn,
        LS_avail=LS_avail,
        f_target=f_target,
        c_reg=c_reg,
        learning_rule=learning_rule,
        task=task,
        shuffle=shuffle,
        key=shuffle_key,
    )
    new_state = state.apply_gradients(grads=grads)

    # Metrics over the LS-available window only — matches the training signal.
    metrics = compute_metrics_fn(
        labels=batch['label'][:, -LS_avail:],
        predictions=y[:, -LS_avail:, :],
    )
    return new_state, metrics, grads


def train_epoch(
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
    shuffle: bool,
    shuffle_key: PRNGKey,
    spec: TaskSpec,
) -> Tuple[TrainState, Any, Dict]:
    """Run one training epoch. Accumulates grads for diagnostic plotting."""
    batch_metrics: List[Any] = []
    accumulated_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)

    for batch in train_batches:
        state, metrics, grads = train_step_fn(
            state=state, batch=batch,
            optimization_loss_fn=optimization_loss_fn,
            LS_avail=LS_avail, f_target=f_target, c_reg=c_reg,
            learning_rule=learning_rule, task=task,
            shuffle=shuffle, shuffle_key=shuffle_key,
        )
        batch_metrics.append(metrics)
        # Note: actual parameter update is already handled by optax.MultiSteps
        # if grad_accum is enabled. This accumulation is purely for plotting.
        accumulated_grads = jax.tree_util.tree_map(
            lambda a, g: a + g, accumulated_grads, grads
        )

    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics, spec.metric_names)
    logger.info(_format_log_line('train', epoch, metrics, spec))
    return state, metrics, accumulated_grads


def eval_step(
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


def evaluate_model(
    eval_step_fn: Callable,
    state: TrainState,
    batches: Iterable,
    epoch: int,
    LS_avail: int,
    spec: TaskSpec,
):
    """Evaluate on a dataset (an iterable of batches)."""
    batch_metrics = [eval_step_fn(state, b, LS_avail) for b in batches]
    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics, spec.metric_names)
    logger.info(_format_log_line('eval ', epoch, metrics, spec))
    return metrics


# =============================================================================
# Main training loop
# =============================================================================
def train_and_evaluate(cfg, spec: TaskSpec) -> TrainState:
    """
    Run the full training + evaluation loop for a task described by `spec`.

    The loop:
      1. Builds the model + train state.
      2. Generates a fixed evaluation set (same seed every time).
      3. At each epoch: generates a fresh training set, runs `train_epoch`.
      4. Every 25 epochs: evaluates, records metrics, checks early-stop
         (three consecutive passes required).
      5. At the end: saves per-epoch metric histories as pickles, saves
         weight figures, training-curve figures, and task-specific example
         figures.

    Returns the final TrainState.
    """
    # -- Setup -------------------------------------------------------------
    n_in = spec.input_dim_from_cfg(cfg)

    master_key = random.key(cfg.net_params.seed)
    shuffle_key, state_key, _ = random.split(master_key, 3)

    model = model_from_config(cfg)
    state = create_train_state(
        rng=state_key,
        learning_rate=cfg.train_params.lr,
        model=model,
        input_shape=(cfg.train_params.train_mini_batch_size, n_in),
        batch_size=cfg.train_params.train_batch_size,
        mini_batch_size=cfg.train_params.train_mini_batch_size,
    )

    # jit the step functions once. `compute_metrics_fn` is bound up-front via
    # functools.partial so it disappears from the jitted call signature —
    # otherwise jax.jit can't hash it, and static_argnames can't find its
    # target argument names on a `**kwargs` lambda.
    train_step_fn = jax.jit(
        partial(train_step, compute_metrics_fn=spec.compute_metrics),
        static_argnames=["LS_avail", "learning_rule", "task", "shuffle"],
    )
    eval_step_fn = jax.jit(
        partial(eval_step, compute_metrics_fn=spec.compute_metrics),
        static_argnames=["LS_avail"],
    )

    # `tree_util.Partial` makes the loss function a pytree leaf, which is
    # needed to pass a Callable into a jitted function.
    closure = jax.tree_util.Partial(spec.optimization_loss)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # -- Evaluation set is fixed for the whole run ------------------------
    eval_batch = list(spec.make_batch(
        cfg,
        n_batches=cfg.train_params.test_batch_size,
        batch_size=cfg.train_params.test_mini_batch_size,
        seed=cfg.task.seed,
    ))

    # -- Metric histories (one list per metric, plus iterations) ----------
    history: Dict[str, List] = {name: [] for name in spec.metric_names}
    history.update({f"{name}_eval": [] for name in spec.metric_names})
    history["iterations"] = []

    # -- Training loop ----------------------------------------------------
    logger.info('Starting training...')
    # Pre-generate per-epoch seeds so training data is reproducible.
    train_task_seeds = random.randint(
        random.PRNGKey(cfg.task.seed),
        (cfg.train_params.iterations,),
        10_000, 10_000_000,
    )

    final_epoch = 0
    stopped_early = False
    for epoch, train_seed in zip(
        range(1, cfg.train_params.iterations + 1), train_task_seeds
    ):
        sub_shuffle_key, shuffle_key = random.split(shuffle_key)

        train_batches = spec.make_batch(
            cfg,
            n_batches=cfg.train_params.train_batch_size,
            batch_size=cfg.train_params.train_mini_batch_size,
            seed=int(train_seed),
        )

        logger.info("\t Starting Epoch: %d", epoch)
        state, train_metrics, _ = train_epoch(
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
            shuffle=cfg.train_params.shuffle,
            shuffle_key=sub_shuffle_key,
            spec=spec,
        )
        final_epoch = epoch

        # -- Periodic evaluation + early-stopping check -------------------
        if (epoch - 1) % 25 == 0:
            eval_metrics = evaluate_model(
                eval_step_fn, state, eval_batch, epoch,
                LS_avail=cfg.task.LS_avail, spec=spec,
            )

            # metric names are "loss" and "display_metric", where display_metric can be accuray, or MSE for example
            for name in spec.metric_names:
                history[name].append(getattr(train_metrics, name))
                history[f"{name}_eval"].append(getattr(eval_metrics, name))
            history["iterations"].append(epoch - 1)

            # Early stopping: require 3 consecutive test batches to pass.
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

    # -- Final evaluation after last epoch --------------------------------
    eval_metrics = evaluate_model(
        eval_step_fn, state, eval_batch, final_epoch,
        LS_avail=cfg.task.LS_avail, spec=spec,
    )
    for name in spec.metric_names:
        history[f"{name}_training"].append(getattr(train_metrics, name))
        history[f"{name}_eval"].append(getattr(eval_metrics, name))
    history["iterations"].append(final_epoch)

    # -- Save + plot ------------------------------------------------------
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


# =============================================================================
# Early-stopping confirmation (requires 3 consecutive passing test sets)
# =============================================================================
def _early_stop_confirmed(
    spec: TaskSpec, cfg, state, eval_step_fn, epoch: int
) -> bool:
    """
    Confirm early stopping by checking `stop_criteria` on 3 fresh test batches.
    """
    for i in range(3):
        test_batch = list(spec.make_batch(
            cfg,
            n_batches=cfg.train_params.test_batch_size,
            batch_size=cfg.train_params.test_mini_batch_size,
            seed=cfg.task.seed + i + 1,
        ))
        test_metrics = evaluate_model(
            eval_step_fn, state, test_batch, epoch,
            LS_avail=cfg.task.LS_avail, spec=spec,
        )
        value = getattr(test_metrics, spec.early_stop_metric)
        if not spec.early_stop_satisfied(value, cfg.train_params.stop_criteria):
            return False
    return True


# =============================================================================
# History saving
# =============================================================================
def _save_history(history: Dict[str, List], output_dir: str) -> None:
    """Pickle each metric history to `output_dir/train_info/<name>.pkl`."""
    train_info_dir = os.path.join(output_dir, 'train_info')
    os.makedirs(train_info_dir, exist_ok=True)
    for name, values in history.items():
        with open(os.path.join(train_info_dir, f'{name}'), 'wb') as f:
            pickle.dump(values, f, pickle.HIGHEST_PROTOCOL)