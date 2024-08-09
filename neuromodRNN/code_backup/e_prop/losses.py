from optax.losses import softmax_cross_entropy
from jax.nn import one_hot
from jax import random, numpy as jnp
from flax import struct
from typing import (
  Any,
  Callable,
  Dict,
  List,
  Optional,
  Sequence,
  Tuple,
  Iterable  
 )

Array = jnp.ndarray

class Metrics(struct.PyTreeNode):
  """Computed metrics."""

  loss: float
  accuracy: Optional[float] = None
  count: Optional[int] = None

def compute_binary_classification_metrics(*, labels: Array, logits: Array) -> Metrics:
  """Computes the metrics, summed across the batch if a batch is provided.
     Normalization across the batches is done in a different function. 
     Notes:
     For classification tasks, we assume 
  """


  # softmax_cross_entropy expects labels to be one-hot encoded
  loss = softmax_cross_entropy(labels=one_hot(labels, 2), logits=logits) 
  
  # Compute accuracy:
  # Inference: Although in my opnion somehow contradictory, inference is considered the cummulative
  # evidence during period where learning signal is available. Kept this way, but I'm passing as default
  # LS_avail as 1, so that anyways the decision is taken only looking at value of outputs at last time step.
  # In this case, sum is only getting rid of time dimension, which will have size 1. But code still
  # prepared to handle the scenario described in Bellec 2020
  inference = jnp.argmax(jnp.sum(logits, axis=1), axis=-1) #  jnp.argmax(jnp.sum(logits, axis=1), axis=-1) # sum the the output overtime, generate cummulative evidence. Select the one with higher evidence. (n_batches,)
  
 
  # Labels are either 0 or 1 tensors, with shape (n_batch, n_t), but n_t is redundant, since all entries are equal over this dimension, therefore select only one element.
  # It has this shape for cases where L_avail is larger than 1   
  binary_accuracy = jnp.equal(inference, labels[:,0]) 
  
  # metrics are summed over batches, counts are stored to normalize it later. This is important if paralellizing through multiple devices
  return Metrics(
      loss=jnp.sum(loss),
      accuracy= jnp.sum(binary_accuracy),
      count = logits.shape[0] # number of batches basically
       )