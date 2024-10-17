"""utils for learning rules"""

from jax import lax, numpy as jnp
from jax import vmap
import jax
# Just for typing the type of inputs of functions
from typing import (
  Dict,
  Tuple,
 )

from flax.typing import Array



# Name of variables try to follow the notation from --> Bellec et al. 2020: A solution to the learning dilemma for recurrent networks of spiking neurons
##################################################################################################
# Functions to compute eligibility trace: updated of input and recurrent weights 
##################################################################################################


# These auxiliary functions are coded to handle single examples, batches are dealed with vmap at main function

def pseudo_derivative(v: Array, A_thr: Array,r: Array, gamma:float, thr:float) -> Array: 
    """ Compute pseudo derivative
        gamma, thr --> scalars; v, A_thr,r --> (n_rec,)"""
    # if neuron is refractory period, the pseudo derivative shoould be 0    
    no_refractory = (r ==0)
    return no_refractory * (gamma/thr) * jnp.maximum(jnp.zeros_like(v), (1 - jnp.abs((v - A_thr) / thr))) #(n_rec,)

def update_v_eligibility_vector(v_vector: Array, inputs: Array, alpha: float) -> Array: # seems to be working
    """  Update voltage eligiblity vector
        alpha --> scalar; v_vector, inputs --> (n_pre,). Note that inputs here can be the inputs to the network (x) or the input from the other recurrent neurons (z), depending on which weight is being updated"""
    return alpha * v_vector + (1-alpha) * inputs  # (n_pre,) --> it only depends o pre-synaptic neuron, so can be store like this and broadcasted to (n_rec, n_pre) where needed
  
def update_a_eligibility_vector(a_vector: Array,v_vector: Array, psi: Array, betas: Array, rho: float) -> Array:
    """ Update adaptation eligiblity vector
        rho --> scalar,  v_vector --> (n_pre); psi,  betas --> (n_post); a_vector --> (n_post, n_pre)"""  
    return   v_vector[None, ...] * psi[..., None] * (1-rho)  + (rho -  (1-rho) * betas[...,None] * psi[...,None]) * a_vector #(n_post, n_pre)


def eligibility_trace(v_eligibility_vector: Array, a_eligibility_vector: Array, psi:Array, betas: Array) -> Array:
    """ Compute eligibility trace at time t, given all the necessary values at time t:
        psi,  betas --> (n_post);  v_eligibility_vector --> (n_pre); a_eligibility_vector --> (n_post, n_pre) """
    return psi[..., None] * (v_eligibility_vector[None, ...] - betas[..., None] * a_eligibility_vector) #n_post, n_pre


    

def compute_eligibitity_trace(old_eligibility_carries:Dict[str,Array], eligibility_input: Tuple[Array, Array, Array], eligibility_params: Dict[str,Dict]) -> Tuple[Dict[str,Array], Array]:
    """
    Compute the eligibility trace and update the eligibility vectors (carries) for one time step.

    Parameters
    ----------
    old_eligibility_carries : Dict of arrays
        Dictionary containing the previous eligibility vectors and traces (time t-1):
        - 'v_eligibility_vector' : Array (n_pre,)
            Previous eligibility membrane potential vectors.
        - 'a_eligibility_vector' :Array (n_post, n_pre)
            Previous adaptation eligibility vectors .
        - 'low_pass_eligibility_trace' : Array (n_post, n_pre)
            Previous low pass eligibility traces. ()
        - 'psi' : ndarray (n_rec,)
            Previous pseudo derivative.

    eligibility_input : Tuple of arrays
        Tuple containing the current values needed for eligibility updates:
        - v :Array (n_rec,)
            Membrane potentials of recurrent neurons.
        - A_thr : Array
            Adaptive thresholds of recurrent neurons.
        - inputs : Array
            Inputs to the network (either external inputs or incoming recurrent spikes, depending on which update).

    eligibility_params : Nested Dict
        Dictionary containing the parameters needed for the update:
        - 'ALIFCell_0' : Dict
            Parameters related to the ALIF cell:
            - 'thr' : float
                Initial firing threshold value.
            - 'gamma' : float
                Dampening factor for the pseudo derivative.
            - 'alpha' : float
                Decay rate for the membrane potential eligibility vector.
            - 'rho' : float
                Decay rate for the adaptation eligibility vector.
            - 'betas' : Array (n_rec, )
                Scaling factor for the eligibility trace.
        - 'ReadOut_0' : Dict
            Parameters related to the readout layer:
            - 'kappa' : float
                Decay rate for the low pass trace.

    Returns
    -------
    eligibility_carries : dict
        Updated (current) eligibility vectors and traces (time t):
        - 'v_eligibility_vector' : Array
            Updated membrane potential eligibility vector.
        - 'a_eligibility_vector' : Array
            Updated adaptation eligibility vector.
        - 'psi' : Array
            Updated pseudo derivative.
        - 'low_pass_eligibility_trace' :Array
            Updated low pass eligibility trace.

    trace : numpy.Array
        The current eligibility trace at time t.
    """
    
    
    
    thr = eligibility_params['ALIFCell_0']['thr'] # scalar
    gamma = eligibility_params['ALIFCell_0']['gamma'] # scalar
    alpha = eligibility_params['ALIFCell_0']['alpha'] # scalar
    rho = eligibility_params['ALIFCell_0']['rho'] # scalar
    betas = eligibility_params['ALIFCell_0']['betas'] # scalar
    kappa = eligibility_params['ReadOut_0']['kappa'] # scalar
    
    # Unpack the eligibility input (ALIF hidden variables and inputs (both network input x and recurret spikes z))
    v, A_thr, r, inputs = eligibility_input
    
    # Unpack initial eligilibility vectors (or in this code also called eligibility carries): old because they are at timte t-1    
    old_v_vector = old_eligibility_carries['v_eligibility_vector']
    old_a_vector = old_eligibility_carries['a_eligibility_vector']
    old_low_pass_trace = old_eligibility_carries['low_pass_eligibility_trace'] # for offline version not necessary, but for online it is. So need to be returned. For offline, it is computed separately.
    old_psi = old_eligibility_carries['psi']

        
    # update eligibility vectors to time t (or in this code also called eligibility carries)
    v_vector = update_v_eligibility_vector(old_v_vector, inputs, alpha) # (n_pre,)
    a_vector = update_a_eligibility_vector(old_a_vector, old_v_vector, old_psi,betas, rho) # (n_post, n_pre)
    psi = pseudo_derivative(v, A_thr,r, gamma, thr) # (n_post,)
    trace =  eligibility_trace(v_vector, a_vector, psi, betas)  # (n_post, n_pre)
   

    
    eligibility_carries = {'v_eligibility_vector':v_vector, 
                           'a_eligibility_vector':a_vector, 
                           'psi':psi,                           
                           'low_pass_eligibility_trace':old_low_pass_trace}      
    return eligibility_carries ,  trace

# Uses vmap to vectorize compute_eligibility_trace to correct handle batches
def batched_eligibitity_trace(eligibility_carries:Dict[str, Array,], eligibility_input: Tuple[Array, Array, Array], eligibility_params: Dict[str,Dict]) -> Tuple[Dict[str,Array], Array]:
    """Batched version of compute_eligibility_trace"""
    batched_func = vmap(compute_eligibitity_trace, in_axes=(0, 0, None))
    return batched_func(eligibility_carries, eligibility_input, eligibility_params)

def low_pass_eligibility_trace(low_pass_trace:Array, eligibility_trace:Array, eligibility_params:Dict[str, Array]):
    """
    Computes low-pass filtered version of eligibility trace. Uses decay factor kappa, of output neurons memmbrane potential.
    Carry and output are the low-pass filtered version of the eligibility trace at time t.
    """
    kappa = eligibility_params['ReadOut_0']['kappa'] 
    return (1-kappa) * eligibility_trace + kappa * low_pass_trace,  (1-kappa) * eligibility_trace + kappa * low_pass_trace

def batched_low_pass_eligibility_trace(low_pass_trace:Array, eligibility_trace:Array, eligibility_params:Dict[str, Array]):
    """Batched version of compute_eligibility_trace"""
    batched_func = vmap(low_pass_eligibility_trace, in_axes=(0, 0, None))
    return batched_func(low_pass_trace, eligibility_trace, eligibility_params)


# Equivalent eligibility trace for readout layer
###################################################################################################################################
def readout_eligibility_vector(readout_vector: Array, z: Array, kappa: float) -> Tuple[Array, Array]:
    """ 
    Compute equivalent version of eligibility vector for readout weights: low-pass filtered spikes from recurrent layer
    kappa --> scalar; v_vector, z --> (n_rec,)
    """
    trace = kappa * readout_vector + (1-kappa) *z 
    return trace, trace # (n_rec,)

# vmap readout_eligibility_vector to handle batches
def batch_readout_eligibility_vector(readout_vector, z, kappa):
    """Batched version of readout_eligibility_vector"""
    batched_func = vmap(readout_eligibility_vector, in_axes=(0, 0, None))
    return batched_func(readout_vector, z, kappa)

###################################################################################################################################
def error_signal(y: Array, true_y: Array) -> Array:
    """Computes error signal for task. For classification, assumes that y is already computed using softmax"""
    err = y -true_y # shape (n_b, n_t, n_out)
    return err

def learning_signal(error_signal: Array, kernel: Array) -> Array:
    """Compute learning signal"""
    return jnp.dot(error_signal, jnp.transpose((kernel), (1,0))) # kernel shape (n_rec, n_out) for consistency with others, so tranpose to have it (n_out, n_rec)

def batched_learning_signal(error_signal: Array, kernel: Array) -> Array:
    """Batched version ofl earning_signal"""
    batched_func = vmap(learning_signal, in_axes=(0, None))
    return batched_func(error_signal, kernel)

##################################################################################################################################

# CA utilities for diffusing errors
def circular_pad_channel(channel:Array, radius:int) -> Array:
    """
    Given a channel and a single input example, pad the input circularly, so that circular convolution
    can be performed.
    """
    return jnp.pad(channel, pad_width=((radius, radius), (radius, radius)), mode='wrap')

def vmap_circular_pad(channel:Array, radius:int) -> Array:
    """
    Perform the circular padding in each channel for each batch independently
    """
    vmaped_function = vmap(vmap(circular_pad_channel, in_axes=(0,None)), in_axes=(0,None)) # ugly, but I want to pad each channel in each batch independently. vmpa only maps to one dimension, so need to nest it to apply for channels and batches
    return vmaped_function(channel, radius)


def error_diffusion(carry:Array, inputs:Array, params:Tuple[int, Array]) -> Array:
    """
    Diffuse the error signal for one time step over the different batches and channels
    
    Parameters:
    ----------
    carry: Array (n_b, n_channels, h, w)
        Current error grids.
    inputs: Array 
        Not used. Kept in the case want to adapt for input being update inside of this function
    params: Tuple [int, Array]
        radius: radius for CA. Setting it to 1 guarantees that error signal is only propagated to Moore neighbourhood of cell (square neighbourhood)
        kernel: Array (n_neuromodulators, 1,H, W) it assumes that each neurotransmitter difusses independently. So that each kernel has only one input channel

    Return
    ------
    output: Array (n_b, n_channels, h, w)
        New error grid after diffusion
    """
    radius, kernel = params
    # the  number of output channels of kernel is equal to number of neuromodulators, since each neuromod diffuses independently
    n_neuro_modulators= kernel.shape[0]
    
        
    error_grid = carry # the previous grid of diffused errors
    error_grid_pad = vmap_circular_pad(error_grid, radius)
    new_error_grid = lax.conv_general_dilated(lhs = error_grid_pad,
                                         rhs = kernel,
                                         window_strides=(1,1),
                                         padding='VALID', # valid means without padding, have alredy added the necessary circular padding
                                         dimension_numbers=('NCHW', 'OIHW', 'NCHW'), # dimensions of image, kernel, output
                                         feature_group_count=n_neuro_modulators # Guarantee that error grids for different neuromodulators are updated independently
    )
    
    return new_error_grid
###################################################################################################################################

#Firing Rate regularization
###################################################################################################
def compute_firing_rate(z:Array, trial_length:Array) -> Array:
    """
    Compute firing rate of each recurrent neuron for all the trials in a batch,
    given the spike trains and trial length of each trial in batch
    """
    return jnp.sum(z, axis=1) / trial_length[:, None]
    #z shape (n_batches, n_rec)

def firing_rate_error(z:Array, trial_length:Array, f_target:float)->Array:
    """
    Compute firing rate error, compared to firing rate target. Uses offline version with 
    average firing rate computed at end of trial.
    """
    z = jnp.transpose(z, (1,0,2)) # transpose to (n_batch, n_t, n_rec), since all other functions expect ( n_t, n_batch, n_rec)
    # Not doing online, but using batch average
    firing_rates = compute_firing_rate(z, trial_length) #(n_batches, n_rec)
    error =  firing_rates - (f_target /1000)  #  divide by 1000, because all units are in ms, but f_target should be passed in Hz
    return error # (n_batches, n_rec)

#############################################################################################

def shift_one_time_step_back(array):
    """Shifts an array one step back on time. New array has entry 0 at time 0"""
    # Create a new array initialized to zeros
    shifted_array = jnp.zeros_like(array)
    
    # Set new_array[:, 1:, :] to array[:, :-1, :]
    shifted_array = shifted_array.at[1:, :, :].set(array[:-1, :, :])    
    return shifted_array





def shuffle_error_grid(key, error_grid):
    """Shuffle the error grid within the same batch and within same neurotransmitter"""
    
    n_batches, _, h, w = error_grid.shape
    
    # Create a random permutation of indices for each batch
    batch_indices = jnp.arange(h * w)
    
    #Shuffle each batch independently
    shuffled_indices = jax.vmap(lambda k: jax.random.permutation(k, batch_indices, independent=True))(jax.random.split(key, n_batches))

    # Flatten the last two dimensions for easy reshaping
    flat_grid = error_grid.reshape(n_batches, 1, h * w)
    
    # Apply the shuffled indices to the last two dimensions for each batch
    shuffled_grid = jax.vmap(lambda x_b, idx: x_b[:, idx])(flat_grid, shuffled_indices)
    
    # Reshape back to the original shape
    shuffled_grid = shuffled_grid.reshape(n_batches, 1, h, w)
    
    return shuffled_grid