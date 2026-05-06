"""
Alignment experiment: train with BPTT, periodically measure how well
alternative learning rules' gradients align with the BPTT gradient.
 
This experiment type is parallel to the standard training loop in
`training_loop.py`. It shares all primitives (model, state,
train_step) but its orchestration is different:
 
  * Training is always BPTT — it's the "ground truth" gradient.
  * No periodic evaluation, no early stopping, no example plots:
    the experiment's purpose is gradient comparison, not task
    performance.
 
Configuration: the user specifies a list of "comparisons" in
`cfg.alignment.comparisons`. Each comparison has a tag (used for
output filenames), a rule name, and any rule-specific kwargs (notably
`diffusion_mode` for diffusion). This lets a single run measure
alignment of multiple variants — e.g., e_prop and three diffusion
modes — all against the same BPTT training.
 
Alignment measurement details
-----------------------------
At measurement epochs (every cfg.alignment.interval epochs):
 
  1. Snapshot the network state BEFORE this epoch's training step
     (call it state_t). Under optax.MultiSteps with grad accumulation,
     state_t is also the state the optimizer's BPTT gradients are
     computed against — params don't update until the Nth
     micro-batch.
 
  2. The BPTT step accumulates per-micro-batch gradients during this
     epoch's training pass. Mean of those = the iteration-level BPTT
     gradient that the optimizer just used.
 
  3. For each comparison, replay every micro-batch in this epoch
     through that comparison's rule, at state_t. Mean of those = the
     iteration-level gradient that comparison would have produced.
 
  4. Compute per-layer cosine similarity / angle / norms between the
     two iteration-level gradients.
 
This setup matches the structure of training: the optimizer compares
mean-of-N BPTT gradients with mean-of-N alternative-rule gradients,
both at the same state.
 
Output structure (under hydra.run.dir):
  train_info/         BPTT training metrics, identical to standard run.
  align_info/         Alignment-experiment outputs:
    iterations.pkl                     measurement epochs.
    <tag>_cosine_per_layer.pkl         {layer: cos history}.
    <tag>_angle_per_layer.pkl          {layer: angle (rad) history}.
    <tag>_gradnorm_bptt.pkl            {layer: ||g_bptt|| history}.
    <tag>_gradnorm_rule.pkl            {layer: ||g_rule|| history}.
 
`<tag>` is the user-specified `tag` field of each comparison.
"""
 
from __future__ import annotations
 
import logging
import os
import pickle
from functools import partial
from typing import Any, Callable, Dict, List
 
import hydra
import jax
import numpy as np
from jax import numpy as jnp
from jax import random
 
from src.modRNN import learning_rules
from src.modRNN.training.primitives import (
    create_train_state,
    model_from_config,
    normalize_batch_metrics,
    train_step,
)
from src.modRNN.training.task_spec import TaskSpec
 
from flax.training.train_state import TrainState
from flax.typing import Array, PRNGKey
 
logger = logging.getLogger(__name__)
 
 
# Layer paths in the gradient pytree. Each path is reported as a
# separate per-layer alignment time series.
_LAYER_PATHS = (
    ("ALIFCell_0", "input_weights"),
    ("ALIFCell_0", "recurrent_weights"),
    ("ReadOut_0", "readout_weights"),
)
 
 
# =============================================================================
# Alignment metrics
# =============================================================================
def _flatten(grad_layer: Array) -> np.ndarray:
    """Convert a layer gradient to a flat numpy 1-D array."""
    return np.asarray(grad_layer).reshape(-1)
 
 
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors. Returns NaN if either is zero."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    # Clip to handle the rare floating-point case where the dot product
    # exceeds 1 by a tiny bit, which would make arccos NaN later.
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
 
 
def _angle_rad(cos_value: float) -> float:
    """Convert cosine similarity to angle in degrees."""
    if np.isnan(cos_value):
        return float("nan")
    return float(np.arccos(cos_value))
 
 
def _layer_alignment(
    bptt_grads: Dict, rule_grads: Dict
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-layer alignment metrics between two gradient pytrees.
 
    Returns a dict with four keys:
      cosine_per_layer    {layer_name: float}
      angle_per_layer     {layer_name: float (radians)}
      gradnorm_bptt       {layer_name: float}
      gradnorm_rule       {layer_name: float}
    """
    cosine_per_layer = {}
    angle_per_layer = {}
    gradnorm_bptt = {}
    gradnorm_rule = {}
 
    for module_name, layer_name in _LAYER_PATHS:
        b = _flatten(bptt_grads[module_name][layer_name])
        r = _flatten(rule_grads[module_name][layer_name])
        key = f"{module_name}.{layer_name}"
 
        cos = _cosine(b, r)
        cosine_per_layer[key] = cos
        angle_per_layer[key] = _angle_rad(cos)
        gradnorm_bptt[key] = float(np.linalg.norm(b))
        gradnorm_rule[key] = float(np.linalg.norm(r))
 
    return {
        "cosine_per_layer": cosine_per_layer,
        "angle_per_layer": angle_per_layer,
        "gradnorm_bptt": gradnorm_bptt,
        "gradnorm_rule": gradnorm_rule,
    }
 
 
def _grad_pytree_zeros_like(grads):
    """Build a zeros pytree matching the structure of `grads`."""
    return jax.tree_util.tree_map(jnp.zeros_like, grads)
 
 
def _accumulate(acc, new):
    """Sum two gradient pytrees leaf-wise."""
    return jax.tree_util.tree_map(lambda a, n: a + n, acc, new)
 
 
def _scale(tree, factor):
    """Multiply every leaf by a scalar."""
    return jax.tree_util.tree_map(lambda x: x * factor, tree)
 
 
# =============================================================================
# Compute-grads wrappers, jitted per comparison
# =============================================================================
def _build_grad_fn_for_comparison(comparison, cfg, spec: TaskSpec) -> Callable:
    """
    Return a jitted function (state, batch, key) -> grads for the given
    comparison.
 
    Each comparison's rule and rule-specific kwargs (currently
    `diffusion_mode`) are baked in via partial closure, so the JIT
    cache key is per-comparison.
    """
    closure = jax.tree_util.Partial(spec.optimization_loss)
    rule = comparison.rule
    diffusion_mode = comparison.diffusion_mode
 
    def call(state, batch, key):
        _, grads = learning_rules.compute_grads(
            batch=batch,
            state=state,
            optimization_loss_fn=closure,
            LS_avail=cfg.task.LS_avail,
            f_target=cfg.train_params.f_target,
            c_reg=cfg.train_params.c_reg,
            learning_rule=rule,
            task=cfg.task.task_type,
            diffusion_mode=diffusion_mode,
            key=key,
        )
        return grads
 
    return jax.jit(call)
 
 
# =============================================================================
# Alignment history accumulator
# =============================================================================
class _AlignmentHistory:
    """Collects per-iteration alignment measurements for one comparison."""
 
    def __init__(self):
        self.cosine_per_layer: Dict[str, List[float]] = {}
        self.angle_per_layer: Dict[str, List[float]] = {}
        self.gradnorm_bptt: Dict[str, List[float]] = {}
        self.gradnorm_rule: Dict[str, List[float]] = {}
 
    def append(self, measurements: Dict[str, Dict[str, float]]):
        for key, value in measurements["cosine_per_layer"].items():
            self.cosine_per_layer.setdefault(key, []).append(value)
        for key, value in measurements["angle_per_layer"].items():
            self.angle_per_layer.setdefault(key, []).append(value)
        for key, value in measurements["gradnorm_bptt"].items():
            self.gradnorm_bptt.setdefault(key, []).append(value)
        for key, value in measurements["gradnorm_rule"].items():
            self.gradnorm_rule.setdefault(key, []).append(value)
 
    def save(self, output_dir: str, tag: str):
        """Pickle the histories under <output_dir>/align_info/<tag>_*.pkl."""
        align_dir = os.path.join(output_dir, "align_info")
        os.makedirs(align_dir, exist_ok=True)
        for suffix, value in (
            ("cosine_per_layer", self.cosine_per_layer),
            ("angle_per_layer", self.angle_per_layer),
            ("gradnorm_bptt", self.gradnorm_bptt),
            ("gradnorm_rule", self.gradnorm_rule),
        ):
            path = os.path.join(align_dir, f"{tag}_{suffix}.pkl")
            with open(path, "wb") as f:
                pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
 
 
# =============================================================================
# Main alignment loop
# =============================================================================
def train_and_compute_alignment(cfg, spec: TaskSpec) -> TrainState:
    """
    Train with BPTT; periodically measure alignment between BPTT and
    each comparison in cfg.alignment.comparisons.
 
    Saves both standard training metrics (under train_info/) and
    alignment metrics (under align_info/).
    """
    # -- Sanity check on comparison tags --------------------------------
    tags = [c.tag for c in cfg.alignment.comparisons]
    if len(set(tags)) != len(tags):
        raise ValueError(
            f"Duplicate tags in cfg.alignment.comparisons: {tags}. "
            "Each comparison's tag must be unique — it's used as the "
            "filename prefix for that comparison's output."
        )
 
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
 
    # Training step always uses BPTT.
    train_step_fn = jax.jit(
        partial(train_step, compute_metrics_fn=spec.compute_metrics),
        static_argnames=["LS_avail", "learning_rule", "task", "diffusion_mode"],
    )
 
    closure = jax.tree_util.Partial(spec.optimization_loss)
 
    # Pre-build per-comparison grad functions. JIT cache key is
    # per-comparison, so each (rule, diffusion_mode) combination
    # compiles once.
    comparison_grad_fns = {
        c.tag: _build_grad_fn_for_comparison(c, cfg, spec)
        for c in cfg.alignment.comparisons
    }
 
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
 
    # -- Histories -------------------------------------------------------
    train_history: Dict[str, List] = {
        f"{name}_training": [] for name in spec.metric_names
    }
    train_history["iterations"] = []
 
    alignment_histories = {
        c.tag: _AlignmentHistory() for c in cfg.alignment.comparisons
    }
    alignment_iterations: List[int] = []
 
    # -- Training loop --------------------------------------------------
    logger.info(
        "Starting alignment experiment: training with BPTT, measuring "
        "alignment of %s every %d epochs.",
        [c.tag for c in cfg.alignment.comparisons],
        cfg.alignment.interval,
    )
 
    for epoch in range(1, cfg.train_params.iterations + 1):
        sub_rule_key, rule_key = random.split(rule_key)
        is_align_epoch = (epoch - 1) % cfg.alignment.interval == 0
 
        # Snapshot the pre-training state if this is an alignment epoch.
        # Under optax.MultiSteps with grad accumulation, params don't
        # actually change until the Nth call to apply_gradients within
        # the iteration, so all BPTT grads in this iteration are
        # computed at this state. JAX arrays are immutable, so this is
        # a real snapshot, not a reference that the training step will
        # mutate.
        state_for_alignment = state if is_align_epoch else None
        microbatches_seen: List[Dict[str, Array]] = []
        bptt_grad_sum = None
 
        # -- BPTT training pass ------------------------------------
        train_batches = spec.make_train_batch(cfg, epoch=epoch)
        batch_metrics = []
        for batch in train_batches:
            state, metrics, grads = train_step_fn(
                state=state, batch=batch,
                optimization_loss_fn=closure,
                LS_avail=cfg.task.LS_avail,
                f_target=cfg.train_params.f_target,
                c_reg=cfg.train_params.c_reg,
                learning_rule="BPTT",
                task=cfg.task.task_type,
                diffusion_mode="aligned",   # ignored for BPTT
                key=sub_rule_key,
            )
            batch_metrics.append(metrics)
 
            if is_align_epoch:
                microbatches_seen.append(batch)
                if bptt_grad_sum is None:
                    bptt_grad_sum = _grad_pytree_zeros_like(grads)
                bptt_grad_sum = _accumulate(bptt_grad_sum, grads)
 
        batch_metrics = jax.device_get(batch_metrics)
        train_metrics = normalize_batch_metrics(batch_metrics, spec.metric_names)
        for name in spec.metric_names:
            train_history[f"{name}_training"].append(getattr(train_metrics, name))
        train_history["iterations"].append(epoch)
 
        if epoch % 25 == 0 or epoch == 1:
            logger.info("epoch %d  BPTT train loss=%.4f",
                        epoch, train_metrics.loss)
 
        # -- Periodic alignment measurement -------------------------
        if is_align_epoch:
            alignment_iterations.append(epoch)
            n_microbatches = len(microbatches_seen)
            bptt_grad_mean = _scale(bptt_grad_sum, 1.0 / n_microbatches)
 
            # Each comparison gets its own subkey, split from a
            # per-epoch key. Different comparisons are then independent
            # in any randomness their rule introduces (e.g. shuffled
            # diffusion modes get independent permutations across
            # comparisons).
            sub_align_key, rule_key = random.split(rule_key)
 
            for c in cfg.alignment.comparisons:
                comp_key, sub_align_key = random.split(sub_align_key)
 
                # Replay every micro-batch in this iteration through
                # this comparison's rule, at the snapshotted state.
                # Sum, then divide by N — same averaging the optimizer
                # would apply.
                rule_grad_sum = None
                for batch in microbatches_seen:
                    rule_grads = comparison_grad_fns[c.tag](
                        state_for_alignment, batch, comp_key,
                    )
                    if rule_grad_sum is None:
                        rule_grad_sum = _grad_pytree_zeros_like(rule_grads)
                    rule_grad_sum = _accumulate(rule_grad_sum, rule_grads)
                rule_grad_mean = _scale(rule_grad_sum, 1.0 / n_microbatches)
 
                m = _layer_alignment(bptt_grad_mean, rule_grad_mean)
                alignment_histories[c.tag].append(m)
 
                logger.info(
                    "  align %s @ epoch %d: cos %s",
                    c.tag, epoch,
                    {k: f"{v:.3f}" for k, v in m["angle_per_layer"].items()},
                )
 
    # -- Save histories --------------------------------------------------
    _save_train_history(train_history, output_dir)
    _save_alignment_iterations(alignment_iterations, output_dir)
    for tag, history in alignment_histories.items():
        history.save(output_dir, tag)
 
    logger.info("Alignment experiment finished. Output dir: %s", output_dir)
    return state
 
 
def _save_train_history(history: Dict[str, List], output_dir: str) -> None:
    """Pickle BPTT training metrics alongside alignment metrics."""
    train_info_dir = os.path.join(output_dir, "train_info")
    os.makedirs(train_info_dir, exist_ok=True)
    for key, values in history.items():
        with open(os.path.join(train_info_dir, f"{key}.pkl"), "wb") as f:
            pickle.dump(values, f, pickle.HIGHEST_PROTOCOL)
 
 
def _save_alignment_iterations(iterations: List[int], output_dir: str) -> None:
    """Pickle the iteration list — the time axis for all alignment metrics."""
    align_dir = os.path.join(output_dir, "align_info")
    os.makedirs(align_dir, exist_ok=True)
    with open(os.path.join(align_dir, "iterations.pkl"), "wb") as f:
        pickle.dump(iterations, f, pickle.HIGHEST_PROTOCOL)