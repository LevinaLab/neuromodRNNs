
import numpy as np
import numpy.matlib
from jax import random, numpy as jnp
import matplotlib.pyplot as plt




# 2D matrix
def sq_distance(x1, y1, x2, y2): 
    dx = jnp.abs(x2 - x1)
    dy = jnp.abs(y2 - y1)
 
    dx = jnp.minimum(1.0 - dx, dx)
    dy = jnp.minimum(1.0 - dy, dy)

    return (dx * dx + dy * dy)
     
def twodMatrix(Pre_x, Pre_y, Post_x, Post_y, sigma=0.001, key=random.PRNGKey(0), dtype=jnp.float32):
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
    
    return M .astype(dtype)




"""

# 1D Matrix

def onedmatrix(Pre, Post, sigma = 0.02):
    
    l1 = len(Pre)
    l2 = len(Post)
    
    M = np.zeros((l2, l1)) 

    pre = np.transpose(np.ones((l2, l1))*Pre/np.max(Pre))
    post = np.ones((l1, l2))*Post/np.max(Post)
    
    Abs_dif = np.abs(pre - post)
    dis = np.transpose(np.minimum(Abs_dif, 1 - Abs_dif))
    
    I = np.random.rand(l2, l1) < 1/(1 + np.exp(dis**2/sigma))
    M[I] = 1
    
    return M

def OneD_Matrix(Pre, Post, sigma = 0.02):  
    pre = np.arange(Pre)
    post = np.arange(Post)
    return onedmatrix(pre, post, sigma)

def OneD_EI_Matrix(Pre, Post, E_frac = 0.8, sigma_e = 0.01, sigma_i = 0.03):
    
    NE = int(Pre*E_frac)
    NI = Pre - NE
    
    E = OneD_Matrix(NE, Post, sigma = sigma_e)
    I = OneD_Matrix(NI, Post, sigma = sigma_i)
    
    M = np.zeros((Post, Pre))
    M[:, :NE] = E
    M[:, NE:] = -I
    
    return M



# 2D Matrix

def twod_grid(N):
    s = int(np.ceil(np.sqrt(N)))
    x = np.tile(np.arange(s), s)[:N]/s
    y = np.repeat(np.arange(s), s)[:N]/s
    return x, y
    
def sq_distance(x1, y1, x2, y2): 
    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
 
    dx = np.minimum(1.0 - dx, dx)
    dy = np.minimum(1.0 - dy, dy)

        
    return (dx*dx + dy*dy)
    
    

def twodMatrix(Pre_x, Pre_y, Post_x, Post_y, sigma = 0.001):
    
    l1 = len(Pre_x)
    l2 = len(Post_x)
    
    M = np.zeros((l2, l1))  
    
    pre_x = np.transpose(np.ones((l2, l1))*Pre_x/np.max(Pre_x))
    pre_y = np.transpose(np.ones((l2, l1))*Pre_y/np.max(Pre_y))
    post_x = np.ones((l1, l2))*Post_x/np.max(Post_x)
    post_y = np.ones((l1, l2))*Post_y/np.max(Post_y)
    
    dis = np.transpose(sq_distance(pre_x, pre_y, post_x, post_y))
    I = np.random.rand(l2, l1) < 1/(1 + np.exp(dis/(4*sigma)))
    M[I] = 1
    
    return M


def TwoD_Matrix(Pre, Post, sigma = 0.001):
    
    Pre_x, Pre_y = twod_grid(Pre)
    Post_x, Post_y = twod_grid(Post)
    
    return twodMatrix(Pre_x, Pre_y, Post_x, Post_y, sigma)


def TwoD_EI_Matrix(Pre, Post, E_frac = 0.8, sigma_e = 0.01, sigma_i = 0.03):
    
    NE = int(Pre*E_frac)
    NI = Pre - NE
    
    E = TwoD_Matrix(NE, Post, sigma = sigma_e)
    I = TwoD_Matrix(NI, Post, sigma = sigma_i)
    
    M = np.zeros((Post, Pre))
    M[:, :NE] = E
    M[:, NE:] = -I
    
    return M
"""