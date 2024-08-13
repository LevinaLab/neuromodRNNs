#TODO: so far, all neuromodulator diffuse equally. Might be an additional thing to allow them to have differen diffusion params


""" Build personalized initializers used in models"""


from jax import random, numpy as jnp
from flax import linen as nn 

import sys
import os

# Get the current directory of this file (which is 'project/src/general_src')
file_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the `src` directory
sys.path.append(file_dir + "/..")
from modRNN import spatial_embedings
  
# Just for typing the type of inputs of functions
from typing import (
  Callable,
  Tuple, 
  Union, 
 )

from flax.typing import (
  Array,
  PRNGKey,
  Dtype,    
)




def generalized_initializer(init_fn:Callable, gain:float=1.0, avoid_self_recurrence:bool=False, mask_connectivity:Union[None, Array]=None)-> Callable:
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

    """

    def initializer(key:PRNGKey, shape:Tuple[int], dtype:Dtype =jnp.float32):
        # Use the provided initializer function to initialize the weights (all the linen.initializers use this initialization structure)
        w = init_fn(key, shape, dtype)
        
        # Apply the gain scaling
        w *= gain

        # If applicable, apply connectivity Mask
        if mask_connectivity is not None:
            w = w * mask_connectivity

        # Subtract the identity matrix if required
        if avoid_self_recurrence:
            if shape[-2] == shape[-1]:
                identity = jnp.eye(shape[-1], dtype=dtype)
                w = w * (1 - identity)
            else:
            #TODO: change this error message
                raise ValueError("Axis -2 with size {} doesn`t match size of axis -1 {}. avoid_self_recurrence is thought to be applied to recurrent layer with square matrix connectivity".format(shape[-2], shape[-1]))
        return w

    return initializer

# So far harcoding the spatial_embeding function. If we want to play around with different ones, need to change it to be an argument of initializer
def initialize_connectivity_mask(local_connectivity:bool, gridshape:Tuple[int, int], neuron_indices:Array, key:PRNGKey,
                                  n_rec:int, sigma: float, dtype:Dtype =jnp.float32):
    """
    Creates a new initializer function for initializing connectivity mask.

    If local_connectivity is required, generates mask array to guarantee this requirement. Mask consists of 0s and 1s, with 0 indicating no connection between the respectively pre and post synaptic neurons, and 1 a possible connection.
    Otherwise, returns array of ones (which is a sort of identity mask).
    In the local connectivity setting, the probability of a pre-synaptic neuron connecting to a post-synaptic neuron is given as function of the distance between the two neurons and the parameter sigma (For more details, look spatial_embeddings.py).    
    For determining distances, receives position of neurons relative to a grid and information to reconstruct the grid.

    Parameters
    ----------
    local_connectivity : bool
        Boolean value indicating if layer has local connectivity pattern or not
    gridshape: Tuple[w, h], 
        Tuple containing pair of int which indicate gridshape --> w (width, or number of columns) and h (height, number of rows)
    neuron_indices: Array (n_neurons, 2)
        Array with 2-D coding of neurons position in the grid. Each row represent a cell, the columns indicate the row and column, respectively, of the cell in the grid
    key: PRNGKey
        PRNGkey for random number generator
    n_rec: int
        Number of recurrent neurons in the layer
    sigma: float
        Width of gaussian controlling propability of connection given distance
    dtype: Dtype, default is float32
        Dtype of mask

     
    
    Returns
    -------
    callable
        A new initializer function that takes the arguments (key, shape, dtype) and returns a mask array as described.

    """
    def initializer(key=key, shape=(n_rec, n_rec), dtype=dtype):

        # if local_connectivity True, build mask according to spatial embedding        
        if local_connectivity:
            
            w, h = gridshape # unpack gridshape, where w is the width (col number) of grid and h is height
            
            # create grid
            grid = spatial_embedings.twod_grid(w, h) # look spatial embedding for information
            
            # convert 2d index coding into 1d (flatten grid, since twodMatrix expects this type of position encoding)
            row_indices = neuron_indices[:, 0]
            col_indices = neuron_indices[:, 1]
            oned_indices = row_indices * w + col_indices

            # get selected cells in grid
            selected_cells = grid[oned_indices, :]

            # Get x and y positions
            x = selected_cells[:, 0]
            y = selected_cells[:, 1]

            # for recurrent connection, all cells can be both pre and post depending on the connection, so therefore same locations for pre and post
            return spatial_embedings.twodMatrix(Pre_x=x, Pre_y=y, Post_x=x, Post_y=y, sigma=sigma, key=key)
                                        
        # If local_connectivivity is False, mask is just ones, so that it does`t change the weights                             
        else:
            return  nn.initializers.ones(key=key, shape=shape, dtype=dtype)

    return initializer


def initialize_sparsity_mask(sparse_connectivity:bool, shape:Tuple[int, ...], key:PRNGKey, sparsity:float, dtype:Dtype =jnp.float32):
    """
    Creates a new initializer function for initializing output sparsity mask.

    If sparsity is required, generates mask array to guarantee this requirement. Mask consists of 0s and 1s, with 0 indicating no connection between the respectively pre and post synaptic neurons, and 1 a possible connection.
    Otherwise, returns array of ones (which is a sort of identity mask).
    The position of the 1 entries is chosen randomly, to achieve ~ sparsity% of connections
    Parameters
    ----------
    sparse_connectivity : bool
        Boolean value indicating if layer has random sparse pattern
    shape: Tuple[int,], 
        Tuple containing shape of mask (should be same as shape of weights to me masked)
    key: PRNGKey
        PRNGkey for random number generator
    sparsity: float
        Float between 0. and 1. Indicates the desire percentage of exiting connections in the layer

    dtype: Dtype, default is float32
        Dtype of mask

     
    
    Returns
    -------
    callable
        A new initializer function that takes the arguments (key, shape, dtype) and returns a mask array as described.

    """
    def initializer(key=key, shape=shape, dtype=dtype):

        # if local_connectivity True, build mask according to spatial embedding        
        if sparse_connectivity:
            
            # Create a zeros array with desired shape
            mask = jnp.zeros(shape, dtype=dtype)
            
            # Calculate the total number of elements and the number of 1s needed
            rows, cols = shape
            total_elements = rows * cols
            num_ones = int(total_elements * sparsity)   # ~ sparsity of the total elements
            
            # Randomly select active connections (no replacement, so that no conneciton is chosen twice)
            flat_indices = random.choice(key, total_elements, shape=(num_ones,), replace=False)
    
            # Convert flat indices to 2D indices
            row_indices = flat_indices // cols
            col_indices = flat_indices % cols

            # Introduce 1s at chosen postions
            mask = mask.at[row_indices, col_indices].set(1)

            # for recurrent connection, all cells can be both pre and post depending on the connection, so therefore same locations for pre and post
            return mask
                                        
        # If local_connectivivity is False, mask is just ones, so that it does`t change the weights                             
        else:
            
            return  nn.initializers.ones(key=key, shape=shape, dtype=dtype)

    return initializer



def initialize_neurons_position(gridshape:Tuple[int, int], key: PRNGKey, n_rec: int, dtype:Dtype =jnp.float32):
    """
    Creates a new initializer function for initializing positions of neuron in a 2D grid.

    Neurons are randomly assigned to positions in a 2D grid, without repetition. 

    Parameters
    ----------
    gridshape: Tuple[w, h], 
        Tuple containing pair of int which indicate gridshape --> w (width, or number of columns) and h (height, number of rows)

    key: PRNGKey
        PRNGkey for random number generator
    n_rec: int
        Number of recurrent neurons in the layer

    dtype: Dtype, default is float32
        Ignored

     
    
    Returns
    -------
    callable
        A new initializer function that takes the arguments (key, shape, dtype) and returns an array with 2D coding of neurons 
        position in the given grid. Each row represent a cell, the columns indicate the row and column, respectively, of the cell in the grid .

    """  
    
    def initializer(key=key, shape=(n_rec, n_rec), dtype=jnp.float32):
        w, h = gridshape
        cells_indices = spatial_embedings.cell_to_twod_grid(w=w, h=h, n_cells=n_rec, key=key)
        return cells_indices
    
    return initializer





def feedback_weights_initializer(init_fn: Callable,key:PRNGKey, shape:Tuple[int, ...], weights_out: Array, sparsity_mask:Array, feedback: str, gain:float=1.0) -> Callable:
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
    shape : tuple of int (n_pre, n_post)
        The shape of the weights to be initialized.
    weights_out : Array (n_pre, n_post)
        The output weights to be used directly if the feedback type is 'Symmetric'.
    feedback : str ()
        The type of feedback. If 'Symmetric', the output weights are returned directly.
        If 'Random', feedback wieght are initialize using provided `init_fn`
        Otherwise, raises error
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
        If requested feedback method is not .
    """

    def initializer(key=key, shape=shape, dtype=jnp.float32):

        if feedback == 'Symmetric':
            return weights_out
        elif feedback== 'Random':
            w = init_fn(key, shape, dtype)
            return  w * gain * sparsity_mask
        else:
            raise NotImplementedError("The requested feedback mode `{}` has not been implemented yet".format(feedback))
    return initializer

def k_initializer(k, shape) -> Callable:
    """
    Creates an initializer function for neuromodulators diffusion kernels.
    Parameters
    ----------
    k: diffusion decay parameter. Determines percentage of signal that is destroyed at every time step
    key : PRNGKey
        A PRNG key used for random number generation in the initializer function.
    shape : tuple of int (n_modulators, 1, kernel_height, kernel_width)
        The shape of the kernels to be initialized. The output channel correspond to number of neuromodulators, while input channel is 1,
        since it is assumed that each neuromodulator diffuses independently

    Returns
    -------
    callable
        An initializer function that takes optional arguments (key, shape, dtype), and returns diffusion kernel with shape (n_neuromodulators, 1, kernel_height, kernel_width)     
        an array of weights based on the specified feedback type.


    """    
    def initializer(key=random.key(0), shape=shape, dtype=jnp.float32):
        
        kernel = jnp.ones(shape=shape) # 3 output channels, 1 input channel, height 3, width 3
        kernel = kernel / (jnp.sum(kernel, axis=(2,3))[:,:,None,None]) # normalize kernel so that it sums up to 1
        kernel = k * kernel # apply decay
        
        return kernel
    return initializer