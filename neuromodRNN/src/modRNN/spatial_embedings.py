
import numpy as np
import numpy.matlib
from jax import random, numpy as jnp
import matplotlib.pyplot as plt
from flax.typing import (PRNGKey)

# twod_grid
def twod_grid(w: int, h: int):
    """ 
    Generate a grid in the [0,1] x [0, 1] space , with h equally spaced positions along the x-axis (rows)
    and w equally spaced positions along the y-axis (columns). The returned grid positions are encoded in a 2D code with shape (w*h, 2), where the first 
    columns contains the x coordinate and the second y coordinate and each row a position. Grid positions are organized such that rows are first created:
    first element (0,0), second element (1/h, 0) 
    
    Inputs:
    -------
    w : int 
        width of grid
    h : int
        height of grid

    Return:
    -------
    grid_positions : Array (w*h, 2)
        Positions in the grid, where first columns gives x coordinate and second y coordinate.

    """
        
    x = jnp.tile(jnp.arange(w), h) / w
    y = jnp.repeat(jnp.arange(h), w) / h    
    grid_positions = jnp.column_stack((x, y))    
    return grid_positions # shape 


# TODO: Need to adapt this for when I have more than 1 neurotransmitter
def cell_to_twod_grid(w:int, h:int, n_cells:int, key:PRNGKey):
    """
     Randomly select, without repetition, locations in a grid with h equally spaced rows and w equally spaced columns
    in the square [0, 1] x [0,1], where the cells will be located

    Inputs:
    -------
    w : int 
        width of grid
    h : int
        height of grid
    n_cells: int
        number of cells to be allocated in grid. n_cells should be smaller than w * h
    key: PRNG key
        key of random generator

    Return:
    -------
    selected positions : Array (n_cells, 2)
        Array containing indices localizing the position of n_cells in a w*h grid. Each position is encoded by 
        a tuple containing the row and column index respectively.

    """
            
    total_positions = w * h
    if n_cells > total_positions:
        raise ValueError("n_cells cannot be greater than the total number of cells in the grid.")
    selected_indices = random.choice(key, total_positions, (n_cells,), replace=False) # randomly select without repetition position in the grid
    
    # Convert linear indices to 2D indices
    selected_rows = selected_indices // w 
    selected_cols = selected_indices % w    
    return jnp.column_stack((selected_rows, selected_cols))
   



# 2D matrix
def sq_distance(x1:float, y1:float, x2:float, y2:float): 
    """Compute the Squared Euclidean distance between two points given their coordinates x1,y1 and x2,y2. Note that distances are computed in a torus (circular borders)"""
    dx = jnp.abs(x2 - x1)
    dy = jnp.abs(y2 - y1)
 
    dx = jnp.minimum(1.0 - dx, dx)
    dy = jnp.minimum(1.0 - dy, dy)

    return (dx * dx + dy * dy)
     
def twodMatrix(Pre_x, Pre_y, Post_x, Post_y, key, sigma=0.001, dtype=jnp.float32):
    """
    Given coordinates of pre and post synaptic neurons, generates local connectivity mask.
    The probability of a pre synaptic neuron connecting to a post synaptic neuron depends 
    on their distance in the grid (Manhattan distance in a torus topology) and the parameter sigma
    """
    
    l1 = len(Pre_x)
    l2 = len(Post_x)
    
    M = jnp.zeros((l2, l1), dtype=dtype)  
    #subkey, key = random.split(random.PRNGKey(0))
    pre_x = jnp.transpose(jnp.ones((l2, l1)) * Pre_x / jnp.max(Pre_x))
    pre_y = jnp.transpose(jnp.ones((l2, l1)) * Pre_y / jnp.max(Pre_y))
    post_x = jnp.ones((l1, l2)) * Post_x / jnp.max(Post_x)
    post_y = jnp.ones((l1, l2)) * Post_y / jnp.max(Post_y)
    
    dis = jnp.transpose(sq_distance(pre_x, pre_y, post_x, post_y)).astype(dtype)
    I = random.uniform(key, (l2, l1)) < 1 / (1 + jnp.exp(dis / (4 * sigma)))
    M = M.at[I].set(1)
    
    return M.astype(dtype)


