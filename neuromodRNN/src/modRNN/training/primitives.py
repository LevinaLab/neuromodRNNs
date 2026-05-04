"""
Shared training primitives.
 
These are the building blocks that any experiment type (training,
alignment, future variants) needs:
 
  * Model construction from a Hydra config.
  * `TrainStateEProp` — the e-prop-aware extension of Flax's TrainState.
  * `create_train_state` — set up the model + optimizer + state.
  * `train_step` — one micro-batch step (compute grads, apply, return metrics).
  * `normalize_batch_metrics` — generic batch-metric reduction.
  * `epoch_seed_sequence` — deterministic per-epoch seed derivation.
 
Anything specific to a particular experiment (the orchestration loop,
periodic evaluation, plotting, history saving) lives in the
experiment-type's own module — currently `training_loop.py` for the
standard training experiment.
"""
 
from __future__ import annotations
 
from typing import Any, Callable, Dict, Sequence, Tuple
 
import jax
import numpy as np
import optax
from flax.training import train_state
from jax import numpy as jnp
from jax import random
 
from src.modRNN.models import LSSN
from src.modRNN import learning_rules
from flax.typing import Array, PRNGKey
 
TrainState = train_state.TrainState
 
 
# =============================================================================
# Model construction
# =============================================================================
def model_from_config(cfg) -> LSSN:
    """
    Build an LSSN model from a Hydra config.
 
    Note: `b_out` is not threaded through because its functionality is
    only partially implemented and works correctly only at its default
    value of 0.
    """
    master_key = random.PRNGKey(cfg.net_params.seed)
    subkey, _ = random.split(master_key)
    (feedback_seed,
     rec_connectivity_seed,
     diff_kernel_seed,
     cell_loc_seed,
     shuffle_permutation_seed,
     input_sparsity_seed,
     readout_sparsity_seed) = random.randint(subkey, (7,), 10_000, 10_000_000)

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
        # ALIF params
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
        shuffle_permutation_seed=shuffle_permutation_seed,
        FeedBack_seed=feedback_seed,
        rec_connectivity_seed=rec_connectivity_seed,
        diff_kernel_seed=diff_kernel_seed,
        cell_loc_seed=cell_loc_seed,
        # weight init
        gain=cfg.net_params.w_init_gain,
        dt=cfg.net_params.dt,
    )
 
# =============================================================================
# TrainState
# =============================================================================
class TrainStateEProp(TrainState):
    """TrainState extended with e-prop-specific fields."""
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
    so gradients are accumulated across (batch_size / mini_batch_size)
    micro-batches before each weight update.
    """
    key_params, key_elig, key_grid = random.split(rng, 3)
    params, eligibility_params, spatial_params = _get_initial_variables(
        key_params, model, input_shape
    )
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
# Metric aggregation
# =============================================================================
def normalize_batch_metrics(batch_metrics: Sequence[Any], metric_names: Sequence[str]):
    """Sum each metric across batches and divide by total count."""
    metric_cls = type(batch_metrics[0])
    total_count = np.sum([m.count for m in batch_metrics])
    kwargs = {}
    for name in metric_names:
        total = np.sum([getattr(m, name) for m in batch_metrics])
        kwargs[name] = total.item() / total_count
    return metric_cls(**kwargs)
 
 
# =============================================================================
# Per-epoch seed derivation
# =============================================================================
def epoch_seed_sequence(master_seed: int, n_epochs: int):
    """
    Deterministically derive a sequence of per-epoch seeds from a master seed.

    Returns a jnp array of shape (n_epochs,). Cast individual values
    with `int(seq[i])` before passing to numpy-based task generators.
    """
    return random.randint(
        random.PRNGKey(master_seed),
        (n_epochs,),
        10_000, 10_000_000,
    )
 
 
# =============================================================================
# One micro-batch training step
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
    diffusion_mode: str,
    key: PRNGKey,
    compute_metrics_fn: Callable,
) -> Tuple[TrainState, Any, Dict]:
    """
    Compute gradients on one micro-batch, apply them, return updated
    state plus per-batch metrics and grads.
 
    The grads are returned (in addition to being applied to state) so
    callers that want to inspect or accumulate them across batches can
    do so. 
    """
    y, grads = learning_rules.compute_grads(
        batch=batch,
        state=state,
        optimization_loss_fn=optimization_loss_fn,
        LS_avail=LS_avail,
        f_target=f_target,
        c_reg=c_reg,
        learning_rule=learning_rule,
        task=task,
        diffusion_mode=diffusion_mode,
        key=key,
    )
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics_fn(
        labels=batch['label'][:, -LS_avail:],
        predictions=y[:, -LS_avail:, :],
    )
    return new_state, metrics, grads