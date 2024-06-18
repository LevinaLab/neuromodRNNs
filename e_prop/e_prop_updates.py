

from jax import lax, numpy as jnp
from jax import vmap


# Just for typing the type of inputs of functions

from typing import (
  Any,  
  Dict,
  Tuple,
 )

from flax.typing import Array

DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex


# TODO: Classification tasks are so far hardcoded, no update for regression
# TODO: maybe refactor how ugrad are computed to offline implementation, it is more compatible with friring rate regularization
# TODO: Firing rate regularization
# TODO: No update for bias term of readout neurons
# TODO: see if want to use the traces for plotting. So far, not returning them, only low pass version



# Name of variables try to follow the notation from --> Bellec et al. 2020: A solution to the learning dilemma for recurrent networks of spiking neurons
##################################################################################################
# Functions to compute low pass filtered eligibility trace: updated of input and recurrent weights 
##################################################################################################


# These auxiliary functions are coded to handle single examples, batches are dealed with vmap at main function

def pseudo_derivative(v: Array, A_thr: Array, gamma:float, thr:float) -> Array: 
    """ Compute pseudo derivative
        gamma, thr --> scalars; v, A_thr --> (n_rec,)"""
    return (gamma / thr) * jnp.maximum(jnp.zeros_like(v), (1 - jnp.abs((v - A_thr) / thr))) #(n_rec,)

def update_v_eligibility_vector(v_vector: Array, inputs: Array, alpha: float) -> Array: # seems to be working
    """  Update voltage eligiblity vector
        alpha --> scalar; v_vector, inputs --> (n_pre,). Note that inputs here can be the inputs to the network (x) or the input from the other recurrent neurons (z), depending on which weight is being updated"""
    return alpha * v_vector + inputs # (n_pre,) --> it only depends o pre-synaptic neuron, so can be store like this and broadcasted to (n_rec, n_pre) when needed
  
def update_a_eligibility_vector(a_vector: Array,v_vector: Array, psi: Array, betas: Array, rho: float) -> Array:
    """ Update adaptation eligiblity vector
        rho --> scalar,  v_vector --> (n_pre); psi,  betas --> (n_post); a_vector --> (n_post, n_pre)"""  
    return  v_vector[None, ...] * psi[..., None] + (rho -  betas[...,None] * psi[...,None]) * a_vector #(n_post, n_pre)


def eligibility_trace(v_eligibility_vector: Array, a_eligibility_vector: Array, psi:Array, betas: Array) -> Array:
    """ Compute eligibility trace at time t, given all the necessary values at time t:
        psi,  betas --> (n_post);  v_eligibility_vector --> (n_pre); a_eligibility_vector --> (n_post, n_pre) """
    return psi[..., None] * (v_eligibility_vector[None, ...] - betas[..., None] * a_eligibility_vector) #n_post, n_pre


    
# TODO: Probably it will be better to refactor the whole function. Right now, it computes the low-pass filtered eligibility trace, which is actually
# the one used for the online implementation of e-prop updates. However, not compatible with other losses as for firing regularization, so it could be reasonable
# to change for only eligibility trace and use offline implementation (which is mathematically the same, but probably I can implement it more efficiently with combination of other losses)

def low_pass_eligibitity_trace(old_eligibility_carries:Dict[str,Array], eligibility_input: Tuple[Array, Array, Array], eligibility_params: Dict[str,Dict]) -> Tuple[Dict[str,Array], Array]:
    """
    Compute the low pass eligibility trace and update the eligibility vectors (carries) for one time step.

    Parameters
    ----------
    old_eligibility_carries : dict
        Dictionary containing the previous eligibility vectors and traces (time t-1):
        - 'v_eligibility_vector' : ndarray (n_pre,)
            Previous eligibility membrane potential vectors.
        - 'a_eligibility_vector' :ndarray (n_post, n_pre)
            Previous adaptation eligibility vectors .
        - 'low_pass_eligibility_trace' : ndarray (n_post, n_pre)
            Previous low pass eligibility traces. ()
        - 'psi' : ndarray (n_rec,)
            Previous pseudo derivative.

    eligibility_input : tuple
        Tuple containing the current values needed for eligibility updates:
        - v :ndarray (n_rec,)
            Membrane potentials of recurrent neurons.
        - A_thr : ndarray
            Adaptive thresholds of recurrent neurons.
        - inputs : ndarray
            Inputs to the network (either external inputs or incoming recurrent spikes, depending on which update).

    eligibility_params : dict
        Dictionary containing the parameters needed for the update:
        - 'ALIFCell_0' : dict
            Parameters related to the ALIF cell:
            - 'thr' : float
                Threshold value.
            - 'gamma' : float
                Dampening factor for the pseudo derivative.
            - 'alpha' : float
                Decay rate for the membrane potential eligibility vector.
            - 'rho' : float
                Decay rate for the adaptation eligibility vector.
            - 'betas' : ndarray (n_rec, )
                Scaling factor for the eligibility trace.
        - 'ReadOut_0' : dict
            Parameters related to the readout layer:
            - 'kappa' : float
                Decay rate for the low pass trace.

    Returns
    -------
    eligibility_carries : dict
        Updated (current) eligibility vectors and traces (time t):
        - 'v_eligibility_vector' : ndarray
            Updated membrane potential eligibility vector.
        - 'a_eligibility_vector' : ndarray
            Updated adaptation eligibility vector.
        - 'psi' : ndarray
            Updated pseudo derivative.
        - 'low_pass_eligibility_trace' :ndarray
            Updated low pass eligibility trace.

    low_pass_trace : numpy.ndarray
        The current low pass eligibility trace.
    """
    
    
    # TODO: initialize eligibility carries properly
    thr = eligibility_params['ALIFCell_0']['thr'] # scalar
    gamma = eligibility_params['ALIFCell_0']['gamma'] # scalar
    alpha = eligibility_params['ALIFCell_0']['alpha'] # scalar
    rho = eligibility_params['ALIFCell_0']['rho'] # scalar
    betas = eligibility_params['ALIFCell_0']['betas'] # scalar
    kappa = eligibility_params['ReadOut_0']['kappa'] # scalar
    
    # Unpack the eligibility input (ALIF hidden variables and inputs (both network input x and recurret spikes z))
    v, A_thr, inputs = eligibility_input
    
    # Unpack initial eligilibility vectors (or in this code also called eligibility carries): old because they are at timte t-1    
    old_v_vector = old_eligibility_carries['v_eligibility_vector']
    old_a_vector = old_eligibility_carries['a_eligibility_vector']
    old_low_pass_trace = old_eligibility_carries['low_pass_eligibility_trace']
    old_psi = old_eligibility_carries['psi']

        
    # update eligibility vectors to time t (or in this code also called eligibility carries)
    v_vector = update_v_eligibility_vector(old_v_vector, inputs, alpha) # (n_pre,)
    a_vector = update_a_eligibility_vector(old_a_vector, old_v_vector, old_psi,betas, rho) # (n_post, n_pre)
    psi = pseudo_derivative(v, A_thr, gamma, thr) # (n_post,)
    low_pass_trace =  eligibility_trace(v_vector, a_vector, psi, betas) + kappa * old_low_pass_trace # (n_post, n_pre)
   

    
    eligibility_carries = {'v_eligibility_vector':v_vector, 
                           'a_eligibility_vector':a_vector, 
                           'psi':psi,
                           'low_pass_eligibility_trace':low_pass_trace
                            }      
    return eligibility_carries ,  low_pass_trace

# Uses vmap to vectorize low_pass_eligibility_trace to correct handle batches
def batched_low_pass_eligibitity_trace(eligibility_carries:Dict[str, Array,], eligibility_input: Tuple[Array, Array, Array], eligibility_params: Dict[str,Dict]) -> Tuple[Dict[str,Array], Array]:
    batched_func = vmap(low_pass_eligibitity_trace, in_axes=(0, 0, None))
    return batched_func(eligibility_carries, eligibility_input, eligibility_params)



# Compute Learning Signal 
####################################################################################################

# TODO: So far it is assuming classification tasks
def learning_signal(y: Array, true_y: Array, kernel: Array) -> Array:
    """computes learning signal for classification task"""
    #y, true_y  --> (n_t, n_out) (in case of single output at end of task, guarantee that time y and true_y were expanded to include the time axis);
    # kernel (n_rec, n_out)
    
    # For classification tasks,the output y of the network is already probabilities computed with softmax    
    def output_error(y, true_y):        
        return y - true_y
    err = output_error(y, true_y) # shape (n_t, n_out)

    return jnp.dot(err, jnp.transpose((kernel), (1,0))) # kernel shape (n_rec, n_out) for consistency with others, so tranpose to have it (n_out, n_rec)

# Batched version
def batched_learning_signal(y_batch: Array, true_y_batch: Array, kernel: Array) -> Array:
    """computes bacthed learning signal for classification task"""
    # Vectorize the learning_signal function over the first dimension (batch dimension)
    batched_func = vmap(learning_signal, in_axes=(0, 0, None))
    return batched_func(y_batch, true_y_batch, kernel)



# Compute Update
###################################################################################################

# TODO: t_crop assumes that learning signal is avaialble only at end, can change to also accept window intervals in the middle of task
def vectorized_grads(batch_init_eligibility_carries:Dict[str,Array], batch_inputs: Tuple[Array, Array,Tuple[Array, Array, Array]], eligibility_params: Dict[str,Dict], t_crop: float=0) -> Tuple[Array, Array]:
    """
    Compute vectorized updates for recurrent or input weights. This form is more efficient specially
    for tasks where learning signal is available only during a particular time window.

    This function performs the batched version of the online updates
    based on a batch of inputs and initial eligibility carries. It uses a feedback 
    kernel to compute the learning signal and applies a low pass filter to the 
    eligibility traces.
    
    The inputs should be time major, since lax.scan assumes time as 

    Parameters
    ----------
    batch_init_eligibility_carries : dict
        Dictionary containing the initial eligibility vectors and traces for the batch:
        - 'v_eligibility_vector' : ndarray (n_batches, n_pre)
            Initial membrane potential eligibility vectors for the batch.
        - 'a_eligibility_vector' : ndarray (n_batches, n_post, n_pre)
            Initial adaptation eligibility vectors for the batch.
        - 'low_pass_eligibility_trace' : ndarray (n_batches, n_post, n_pre)
            Initial low pass eligibility traces for the batch.
        - 'psi' : ndarray (n_batches, n_rec)
            Initial pseudo derivatives for the batch.

    batch_inputs : tuple
        Tuple containing the batched update input values:
        - y_batch : ndarray (n_t, n_batches, n_out)
            Batched output values of the network.
        - true_y_batch : ndarray (n_t, n_batches, n_out)
            Batched true output values (ground truth).
        - eligibility_input : tuple
          Tuple containing the current values needed for eligibility updates:
            - v :ndarray (n_t, n_batches, n_rec)
            Membrane potentials of recurrent neurons.
            - A_thr : ndarray (n_t, n_batches, n_rec)
            Adaptive thresholds of recurrent neurons.
            - inputs : ndarray (n_t, n_batches, n_pre) --. Inputs to the post-synaptic neuron (either external inputs or incoming recurrent spikes, depending on which update).
            

    eligibility_params : dict
        Dictionary containing the parameters needed for the update:
        - 'ReadOut_0' : dict
            Parameters related to the readout layer:
            - 'feedback_weights' : ndarray (n_rec, n_out)
                Feedback weights used to compute the learning signal.
        - Other parameters necessary for the eligibility trace update (see the low_pass_eligibitity_trace function).

    t_crop : int (optional) default = 0, signal available throughout whole task
        Time for which learning signal is available at end of task. If it is valid throught out the whole taks, use default value 0

    Returns
    -------
    eligibility_traces : ndarray
        Updated eligibility traces for the batch.
        
    update : ndarray
        Computed updates based on the cropped eligibility traces and learning signals.

    Notes
    -----
    The function uses a scan operation to iterate through time steps, updating the 
    eligibility traces, and then computes the final updates by performing an element-wise 
    multiplication and summation over the batch and time dimensions.
    
    See TODO note of low_pass_eligibility_trace for reasons to refactor this function as well.
    """
    
    kernel = eligibility_params['ReadOut_0']['feedback_weights']   
    y_batch, true_y_batch, eligibility_input = batch_inputs


    if true_y_batch.ndim < 3:  # Guarantee that Labels have time dimension, so that are compatible with operation.
        true_y_batch = jnp.expand_dims(true_y_batch, axis=1)

    # Compute learning signal for time where it is available
    L = batched_learning_signal(y_batch[:,-t_crop:,:],true_y_batch[:,-t_crop:,:],kernel) #n_batch, n_t, n_rec 
    
   
    # Evolve through time the low pass eligibility trace, given the history of eligibility inputs and the initial carries
    _, low_pass_eligibility_traces = lax.scan(
        lambda carry, input: batched_low_pass_eligibitity_trace(carry, input, eligibility_params),
        batch_init_eligibility_carries,
        eligibility_input
    )
    
    trace= low_pass_eligibility_traces # trace (n_t,n_b,n_post,n_pre)
    # crop it for only time where learning signal is available
    crop_trace = trace[-t_crop:,:,:,:] #(n_t,n_b,n_rec,n_in)
    
    # perform the necessary element-wise multiplication and sum over batches and time dimensions    
    update = jnp.einsum('btri,tbri->ir', jnp.expand_dims(L,axis=3), crop_trace)
    return low_pass_eligibility_traces, update

###################################################################################################

# Readout layer
###################################################################################################

# Equivalent eligibility trace for readout layer
###################################################################################################
def readout_eligibility_vector(readout_vector: Array, z: Array, kappa: float) -> Tuple[Array, Array]: # seems to be working
    """ Compute equivalent version of eligibility vector for readout weights
        kappa --> scalar; v_vector, z --> (n_rec,)"""
    trace = kappa * readout_vector + z 
    return trace, trace # (n_rec,)

# vmap readout_eligibility_vector to handle batches
def batch_readout_eligibility_vector(readout_vector, z, kappa):
    batched_func = vmap(readout_eligibility_vector, in_axes=(0, 0, None))
    return batched_func(readout_vector, z, kappa)

#Update
###################################################################################################


def output_grads(init_batch_trace: Dict[str,Array], batch_inputs: Tuple[Array, Array, Array], params: Dict[str,Dict], t_crop:float=0) ->Tuple[Array, Array]:
    """
    Compute updates for output. It doesn`t need e-prop theory, but similar nomenclature was used for consistency.

    This function computes the updates for output weights by processing a batch of 
    inputs and initial eligibility traces.
    v_eligibility_vector_out is actually the low pass filtered spikes z from recurrent layer

    Parameters
    ----------
    init_batch_trace : dict
        Dictionary containing the initial eligibility traces for the batch:
        - 'v_eligibility_vector_out' : ndarray (n_batch, n_rec)
            Initial carry for the "equivalent trace" for output layer. 

    batch_inputs : tuple
        Tuple containing the batched input values:
        - y_batch : ndarray
            Batched predicted outputs of the network (n_batch, n_time, n_out).
        - true_y_batch : ndarray
            Batched true output values (ground truth) (n_batch, n_time, n_out).
        - z : ndarray
            Network activity (spikes) (n_time, n_batch, n_out). For scan, time dimension needs to be leading axis.

    params : dict
        Dictionary containing the parameters needed for the update:
        - 'ReadOut_0' : dict
            Parameters related to the readout layer:
            - 'kappa' : float
                Decay factor of leaky readout neurons (used to for low pass filtered spikes z).

    t_crop : int (optional) default = 0, signal available throughout whole task
        Time for which learning signal is available at end of task. If it is valid throught out the whole taks, use default value 0

    Returns
    -------
    traces : ndarray (n_t, n_batch, n_out, n_rec)
        History of the low-pass filtered recurrent spikes z.
        
    grads : ndarray
        Computed grads for the output weights based on the cropped eligibility traces and errors.

    Notes
    -----
    This function processes the batch of eligibility traces and inputs, applying the 
    learning signal and eligibility trace updates in a vectorized manner for efficiency. 
    The function uses a scan operation to iterate through time steps, updating the 
    eligibility traces, and then computes the final updates by performing an element-wise 
    multiplication and summation over the batch and time dimensions.
    """
    kappa = params['ReadOut_0']['kappa'] 
    y_batch, true_y_batch, z = batch_inputs # (n_batch, n_time, n_out)
    init_carry= init_batch_trace["v_eligibility_vector_out"]
    
    if true_y_batch.ndim < 3:  # Guarantee that Labels have time dimension, so that are compatible with operation.
        true_y_batch = jnp.expand_dims(true_y_batch, axis=1)

    err = y_batch[:,-t_crop:,:] - true_y_batch[:,-t_crop:,:]  #(n_batch, n_time, n_out)
    
    
    # Scan over time to get history of low pass filtered z 
    _, traces = lax.scan(
        lambda carry, input: batch_readout_eligibility_vector(carry, input, kappa),
        init_carry,
        z # for scan it needs to be time major
    )
       
    crop_trace = traces[-t_crop:,:,:] #(n_t,n_b,n_rec)
      

    # perform element-wise multiplication and sum over batches and time dimensions    
    grads = jnp.einsum('btor,tbor->ro', jnp.expand_dims(err,3), jnp.expand_dims(crop_trace,2)) # weights have shape (pre, post), so grad shoulg have same shape --> ro)
    
    return traces, grads

def e_prop_grads(eligibility_inputs: Tuple[Array, Array, Array, Array], state, local_connectivity: bool, t_crop: int) ->Dict[str, Dict]:
    """
    Compute the e-prop gradients for a given batch of eligibility inputs and state.

    This function computes the gradients for input, recurrent, and output weights 
    using the e-prop algorithm. The gradients are calculated 
    by processing a batch of inputs and initial eligibility traces.
    
    Parameters
    ----------
    eligibility_inputs : tuple
        Tuple containing the batched input values:
        - y_batch : ndarray
            Batched predicted outputs of the network (n_batch, n_time, n_out).
        - true_y_batch : ndarray
            Batched true output values (ground truth) (n_batch, n_time, n_out).
        - recurrent_carries : tuple
            Tuple containing the recurrent state variables:
            - v : ndarray 
                Membrane potentials of the neurons (n_batch, n_time, n_rec).
            - a : ndarray
                not used.
            - A_thr : ndarray
                Adaptive thresholds of the neurons (n_batch, n_time, n_rec).
            - z : ndarray
                Recurrent spikes (n_batch, n_time, n_rec).
        - x : ndarray
            Inputs to the network (n_batch, n_time, n_in).

    state : object
        State object containing the following attributes:
        - eligibility_params : dict
            Parameters needed for the eligibility trace updates. (see the low_pass_eligibitity_trace function)
        - init_eligibility_carries : dict
            Initial eligibility vectors and traces for the batch:
            - 'inputs' : dict
                Initial eligibility carries for the input weights. 
            - 'rec' : dict
                Initial eligibility carries for the recurrent weights.
            - 'out' : dict
                Initial eligibility carries for the output weights.

    t_crop: int (optional) default = 0, signal available throughout whole task
        Time for which learning signal is available at end of task. If it is valid throught out the whole taks, use default value 0

    Returns
    -------
    grads : dict
        Dictionary containing the computed gradients for the network weights:
        - 'ALIFCell_0' : dict
            Gradients for the ALIF cell:
            - 'input_weights' : ndarray 
                Gradients for the input weights. (n_in, n_rec)
            - 'recurrent_weights' : ndarray
                Gradients for the recurrent weights (n_rec, n_rec)
        - 'ReadOut_0' : dict
            Gradients for the readout layer:
            - 'readout_weights' : numpy.ndarray
                Gradients for the readout weights. (n_rec, n_out)

    Notes
    -----
    This function processes the batch of eligibility traces and inputs, applying the 
    e-prop algorithm to compute the gradients for input, recurrent, and output weights.
    It ensures that self-recurrence is not learned by zeroing out the diagonal of the 
    recurrent weight gradients. Also guarantee self connectivity if model was initialized with it.
    """
    

    
    # Inputs
    y_batch, true_y_batch,recurrent_carries, x = eligibility_inputs 

    v,_, A_thr , z = recurrent_carries # _ is a, which is not used 
    v = jnp.transpose(v, (1,0,2)) # for the scan needs to be time major (n_t,n_batches, n_rec)
    A_thr = jnp.transpose(A_thr, (1,0,2)) # for the scan needs to be time major
    z = jnp.transpose(z, (1,0,2)) # for the scan needs to be time major (n_t,n_batches, n_rec)
    x = jnp.transpose(x, (1,0,2)) # for the scan needs to be time major (n_t,n_batches, n_in)
    inputs_in = (y_batch, true_y_batch, (v, A_thr,x))
    inputs_rec = (y_batch, true_y_batch, (v, A_thr,z))
    inputs_out = (y_batch, true_y_batch, z)
    

    # State
    eligibility_params = state.eligibility_params
    init_e_carries = state.init_eligibility_carries  
 
    
   
        
    grads = {'ALIFCell_0':{}, 'ReadOut_0':{}}
    
    # Input Grads
    input_traces, grads['ALIFCell_0']['input_weights'] = vectorized_grads(init_e_carries['inputs'],
                                                              inputs_in, eligibility_params,
                                                              t_crop = t_crop)
    
    # Recurrent Grads
    rec_traces, grads['ALIFCell_0']['recurrent_weights'] = vectorized_grads(init_e_carries['rec'],
                                                                  inputs_rec, eligibility_params,
                                                                  t_crop =t_crop)
    
    n_rec = jnp.shape(grads['ALIFCell_0']['recurrent_weights'])[0]
    identity = jnp.eye(n_rec, dtype=grads['ALIFCell_0']['recurrent_weights'].dtype)
    grads['ALIFCell_0']['recurrent_weights'] = grads['ALIFCell_0']['recurrent_weights'] * (jnp.array(1) - identity) # guarantee that no self recurrence is learned 
    
    # Guarantee that local connectivity is kept (otherwise, e-prop will lead to growth of new synapses)
    if local_connectivity:
        grads['ALIFCell_0']['recurrent_weights'] *= state.connectivity_mask['ALIFCell_0']['M']
   
    # Output Grads
    out_traces, grads['ReadOut_0']['readout_weights'] = output_grads(init_e_carries['out'], 
                                                           inputs_out, eligibility_params, 
                                                           t_crop)
                                                          
    return grads