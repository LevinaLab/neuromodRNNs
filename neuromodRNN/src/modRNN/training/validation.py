"""
Validation utilities for debugging the hardcoded e-prop implementation.
 
`test_e_prop_grads` compares gradients computed by the hardcoded e-prop
against gradients computed by JAX autodiff (with the same pseudo-derivative
via the custom VJP on `spike`). They should agree up to numerical precision
for e_prop_autodiff, so large disagreement signals a bug in the hardcoded
pipeline.

"""
 
from __future__ import annotations
 
from typing import Callable, Tuple
 
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from optax import losses
 
from .. import learning_rules
from flax.typing import Array, PRNGKey
 
 
def cosine_similarity_degrees(grad_a: Array, grad_b: Array) -> Array:
    """
    Angle (in degrees) between the flattened gradient vectors.
 
    Returns 0° for identical directions and 180° for opposite directions.
    Useful for "are the hardcoded grads pointing the same way as autodiff?"
    """
    flat_a = grad_a.reshape(-1)
    flat_b = grad_b.reshape(-1)
    cos_sim = losses.cosine_similarity(flat_a, flat_b)
    return jnp.degrees(jnp.arccos(cos_sim))
 
 
def test_e_prop_grads(
    state,
    train_batches,
    optimization_loss_fn: Callable,
    LS_avail: int,
    f_target: float,
    c_reg: float,
    task: str,
    shuffle: bool,
    shuffle_key: PRNGKey,
) -> Tuple[Array, Array, Array, Array]:
    """
    Accumulate autodiff and hardcoded e-prop grads across a batch, then return:
      * cosine-similarity angle for recurrent grads (degrees)
      * cosine-similarity angle for input grads (degrees)
      * max |autodiff - hardcoded| for recurrent grads
      * max |autodiff - hardcoded| for input grads
    """
    autodiff_acc = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    hardcoded_acc = jax.tree_util.tree_map(jnp.zeros_like, state.params)
 
    for batch in train_batches:
        _, autodiff_grads = learning_rules.compute_grads(
            batch=batch, state=state,
            optimization_loss_fn=optimization_loss_fn,
            LS_avail=LS_avail, f_target=f_target, c_reg=c_reg,
            learning_rule="e_prop_autodiff",
            task=task, shuffle=shuffle, key=shuffle_key,
        )
        autodiff_acc = jax.tree_util.tree_map(
            lambda a, g: a + g, autodiff_acc, autodiff_grads
        )
 
        _, hardcoded_grads = learning_rules.compute_grads(
            batch=batch, state=state,
            optimization_loss_fn=optimization_loss_fn,
            LS_avail=LS_avail, f_target=f_target, c_reg=c_reg,
            learning_rule="e_prop_hardcoded",
            task=task, shuffle=shuffle, key=shuffle_key,
        )
        hardcoded_acc = jax.tree_util.tree_map(
            lambda a, g: a + g, hardcoded_acc, hardcoded_grads
        )
 
    auto_rec = autodiff_acc['ALIFCell_0']['recurrent_weights']
    auto_in = autodiff_acc['ALIFCell_0']['input_weights']
    hard_rec = hardcoded_acc['ALIFCell_0']['recurrent_weights']
    hard_in = hardcoded_acc['ALIFCell_0']['input_weights']
 
    return (
        cosine_similarity_degrees(auto_rec, hard_rec),
        cosine_similarity_degrees(auto_in, hard_in),
        jnp.max(jnp.abs(auto_rec - hard_rec)),
        jnp.max(jnp.abs(auto_in - hard_in)),
    )
 
 
def plot_grad_comparison(
    iterations,
    recurrent_cos_sim,
    input_cos_sim,
    max_recurrent,
    max_input,
    save_path: str,
) -> None:
    """Side-by-side plot of grad agreement metrics across training."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
 
    axs[0].plot(iterations, recurrent_cos_sim, label="Recurrent grads")
    axs[0].plot(iterations, input_cos_sim, label="Input grads")
    axs[0].set_title('Angle between hardcoded and autodiff e-prop grads')
    axs[0].set_ylabel("Degrees")
    axs[0].set_xlabel('Iterations')
    axs[0].legend()
 
    axs[1].plot(iterations, max_recurrent, label="Recurrent grads")
    axs[1].plot(iterations, max_input, label="Input grads")
    axs[1].set_yscale("log")
    axs[1].set_title('Max |autodiff - hardcoded|')
    axs[1].set_ylabel("Max absolute difference")
    axs[1].set_xlabel('Iterations')
    axs[1].legend()
 
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)