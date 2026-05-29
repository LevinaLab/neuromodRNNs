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
     
def nearest_neighbors_mask(neuron_indices: jnp.ndarray, w: int, h: int, dtype=jnp.float32) -> jnp.ndarray:
    """
    Build a deterministic binary connectivity mask connecting each neuron to its
    occupied Moore neighbours (8-connectivity) in a torus-wrapped grid.
 
    For every neuron i at grid position (r, c), all 8 positions reachable by
    offsets dr, dc in {-1, 0, +1} x {-1, 0, +1} minus {(0,0)} are examined after
    torus wrapping ((r+dr) % h, (c+dc) % w).  If a neighbour slot is occupied
    by another neuron j, the entry M[j, i] is set to 1 (post-synaptic neuron
    indexes rows, pre-synaptic neuron indexes columns, consistent with
    twodMatrix).  The diagonal is explicitly zeroed to prevent self-connections.
 
    Inputs:
    -------
    neuron_indices : Array (n_rec, 2)
        Row/column positions of each neuron in the grid, as produced by
        cell_to_twod_grid.  neuron_indices[i] = (row_i, col_i).
    w : int
        Width of the grid (number of columns).
    h : int
        Height of the grid (number of rows).
    dtype : jnp dtype, optional
        Data type of the returned mask.  Default is jnp.float32.
 
    Return:
    -------
    M : Array (n_rec, n_rec)
        Binary connectivity mask.  M[post, pre] = 1 if neuron pre is a
        nearest neighbour of neuron post in the torus grid, else 0.
        Diagonal entries are always 0 (no self-connections).
    """
    # Convert to plain numpy for static index arithmetic (called at init time,
    # not inside a JIT-traced function, so this is safe and avoids concretisation
    # errors that would arise from tracing through Python dicts).
    # not JAX-traceable, so jnp operations would cause concretisation errors.
    neuron_indices_np = np.array(neuron_indices, dtype=np.int32)
    n_rec = neuron_indices_np.shape[0]
 
    # Build a lookup: grid position (row, col) -> neuron index, or -1 if empty.
    pos_to_neuron = np.full((h, w), fill_value=-1, dtype=np.int32)
    for i in range(n_rec):
        r, c = int(neuron_indices_np[i, 0]), int(neuron_indices_np[i, 1])
        pos_to_neuron[r, c] = i
 
    # Accumulate connections using boolean array; convert to jnp at the end.
    M = np.zeros((n_rec, n_rec), dtype=np.float32)
 
    # Moore neighbourhood offsets (8 directions, excluding (0, 0))
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]
 
    for i in range(n_rec):
        r, c = int(neuron_indices_np[i, 0]), int(neuron_indices_np[i, 1])
        for dr, dc in offsets:
            nr = (r + dr) % h
            nc = (c + dc) % w
            j = pos_to_neuron[nr, nc]
            if j >= 0:           # neighbour slot is occupied
                M[j, i] = 1.0   # j is post, i is pre
 
    # Zero the diagonal (no self-connections)
    np.fill_diagonal(M, 0.0)
 
    return jnp.array(M, dtype=dtype)
 
 
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