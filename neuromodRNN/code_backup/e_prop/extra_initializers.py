
from jax import numpy as jnp
from jax import random
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API

import sys
import os

# Get the current directory of this file (which is 'project/src/general_src')
file_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the root 'project' directory
sys.path.append(file_dir + "/..")
from . import spatial_embedings
  
# Just for typing the type of inputs of functions


from typing import (
  Any,
  Callable,
  Tuple, 
  Union, 
 )

from flax.typing import (
  Array,
  PRNGKey,
  Dtype,  
  
)

DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex




def generalized_initializer(init_fn:Callable, gain:float=1.0, avoid_self_recurrence:bool=False, local_connectivity:Union[None, Array]=None)-> Callable:
    """
    Creates a new initializer function that modifies the output of a given initialization function.

    This function generates an initializer which first uses a provided initialization function (`init_fn`)
    to initialize the weights. The weights are then scaled by a specified gain factor. Optionally, 
    the function can also modify the weights to avoid self-recurrence by subtracting the identity matrix 
    from the initialized weights, applicable only for square matrices (e.g., recurrent layers).

    Parameters
    ----------
    init_fn : callable
        The initialization function to be wrapped. This function should take the arguments
        (key, shape, dtype) and return an array of the specified shape and dtype.
    gain : float, optional
        A scaling factor to apply to the initialized weights. Default is 1.0.
    avoid_self_recurrence : bool, optional
        If True, the initializer will modify the weights to avoid self-recurrence by subtracting
        the identity matrix from the weights. This is applicable only if the shape of the weights
        is square (i.e., shape[-2] == shape[-1]). Default is False.
    local_connectivity: Union[None, Array], optional
        If None, simply ignores. If an array, masks the initialized weight with the desired connectivity pattern,
        given by the provided array.
     
    
    Returns
    -------
    callable
        A new initializer function that takes the arguments (key, shape, dtype) and returns an
        array of initialized weights, optionally modified as described.

    Raises
    ------
    ValueError
        If `avoid_self_recurrence` is True and the shape of the weights is not square, an error
        is raised indicating that the axes -2 and -1 must have the same size.

    Examples
    --------
    >>> import jax.random as random
    >>> import jax.numpy as jnp
    >>> import flax.linen.initializers as initializers
    >>> key = random.PRNGKey(0)
    >>> init_fn = initializers.lecun_normal()
    >>> initializer = generalized_initializer(init_fn, gain=1.0, avoid_self_recurrence=True)
    >>> weights = initializer(key, (3, 3), jnp.float32)
    >>> print(weights)
    [[ 0.          -0.11211079 -0.06463727]
    [-0.00396822  0.          -0.0031225 ]
    [ 0.11873566 -0.06857998  0.        ]]
    """

    def new_initializer(key:PRNGKey, shape:Tuple[int], dtype:Dtype =jnp.float32):
        # Use the provided initializer function to initialize the weights (all the linen.initializers use this initialization structure)
        w = init_fn(key, shape, dtype)
        
        # Apply the gain scaling
        w *= gain
        if local_connectivity is not None:
            w = w * local_connectivity
        # Subtract the identity matrix if required
        if avoid_self_recurrence:
            if shape[-2] == shape[-1]:
                identity = jnp.eye(shape[-1], dtype=dtype)
                w = w * (1 - identity)
            else:
            #TODO: change this error message
                raise ValueError("Axis -2 with size {} doesn`t match size of axis -1 {}. avoid_self_recurrence is thought to be applied to recurrent layer with square matrix connectivity".format(shape[-2], shape[-1]))
        return w

    return new_initializer

# So far harcoding the spatial_embeding function. If we want to play around with different ones, need to change it to be an argument of initializer
def initialize_connectivity_mask(local_connectivity, key:PRNGKey, n_rec:int, sigma: float, dtype:Dtype =jnp.float32):
    
    def initializer(key=key, shape=(n_rec, n_rec), dtype=jnp.float32):
        # we are using in a context of recurrent connections, where all neurons can be pre or post, so pre and post should share
        # the same keys
        key_x,key_y, M_key = random.split((key), 3)
        
        if local_connectivity:
            return spatial_embedings.twodMatrix(random.uniform(key_x, shape=(n_rec,)), 
                                                random.uniform(key_y, shape=(n_rec,)),
                                                random.uniform(key_x, shape=(n_rec,)),
                                                random.uniform(key_y, shape=(n_rec,)), 
                                                sigma =sigma, 
                                                key=M_key)    
        else:
            return  nn.initializers.ones(key=key, shape=(n_rec, n_rec), dtype=jnp.float32)

    return initializer


# TODO: for symmetric, init_fn is useless, make it optional
# TODO: change so that if not Random or Symmetric, raises error. COuld also include adaptative
def feedback_weights_initializer(init_fn: Callable,key:PRNGKey, shape:Tuple[int, ...], weights_out: Array, feedback: bool, gain:float=1.0) -> Callable:
  """
    Creates an initializer function for feedback weights based on the specified feedback type.

    This function generates an initializer that either returns the given output weights directly
    (in the case of symmetric feedback) or initializes new weights using the provided initializer 
    function and scales them by a specified gain factor.

    Parameters
    ----------
    init_fn : callable
        The initialization function to be used when the feedback type is not 'Symmetric'.
        This function should take the arguments (key, shape, dtype) and return an array of
        the specified shape and dtype.
    key : PRNGKey
        A PRNG key used for random number generation in the initializer function.
    shape : tuple of int
        The shape of the weights to be initialized.
    weights_out : array-like
        The output weights to be used directly if the feedback type is 'Symmetric'.
    feedback : str 
        The type of feedback. If 'Symmetric', the output weights are returned directly.
        Otherwise, new weights are initialized using `init_fn`.
    gain : float, optional
        A scaling factor to apply to the initialized weights when the feedback type is not
        'Symmetric'. Default is 1.0.

    Returns
    -------
    callable
        An initializer function that takes optional arguments (key, shape, dtype) and returns
        an array of weights based on the specified feedback type.

    Raises
    ------
    ValueError
        If any required argument is missing or incorrectly specified.

    Examples
    --------
    >>> import jax.random as random
    >>> import jax.numpy as jnp
    >>> import flax.linen.initializers as initializers
    >>> key = random.PRNGKey(0)
    >>> init_fn = initializers.lecun_normal()
    >>> weights_out = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    >>> feedback = 'Symmetric'
    >>> initializer = feedback_weights_initializer(init_fn, key, (2, 2), weights_out, feedback, gain=1.0)
    >>> weights = initializer()
    >>> print(weights)
    [[0.1 0.2]
     [0.3 0.4]]
    
    >>> feedback = 'Random'
    >>> initializer = feedback_weights_initializer(init_fn, key, (2, 2), weights_out, feedback, gain=1.0)
    >>> weights = initializer()
    >>> print(weights)
    [[-0.05423447  0.10352807]
     [ 0.08480088 -0.03604186]]
    """
  def initializer(key=key, shape=shape, dtype=jnp.float32):

    if feedback == 'Symmetric':
      return weights_out
    else:
      w = init_fn(key, shape, dtype)
      return  w * gain
  
  return initializer