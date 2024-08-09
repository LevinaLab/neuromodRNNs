from jax import numpy as jnp
def compute_firing_rate(z, trial_length):
    return jnp.sum(z, axis=1) / trial_length[:, None]
    #z shape (n_batches, n_rec)