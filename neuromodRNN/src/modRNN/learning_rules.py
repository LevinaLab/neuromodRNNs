# TODO: adapt the diffusion update once I define how to initialize everything
# TODO: adapt the diffusion update once have more than 1 neuromodulator
# TODO: adapt for when I have sparse connectivity to output (use a mask to remove the errors)

"""Library file used learning rules: e-pro;, moddiffusion"""

from jax import lax, numpy as jnp
from jax import value_and_grad
from jax.nn import softmax
import jax
import os
import sys

# Get the current directory of this file 
file_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to 'src' directory
sys.path.append(file_dir + "/..")
from modRNN import learning_utils

# Just for typing the type of inputs of functions
from typing import (
  Dict,
  Tuple,
  Callable
 )
from flax.typing import Array




def e_prop_vectorized(batch_init_carries:Tuple[Dict[str,Array], Array], 
                           batch_inputs: Tuple[Array, Array,Tuple[Array, Array, Array]], 
                           params: Tuple[Dict[str,Dict],Dict[str,Dict]], 
                           LS_avail:int, z:Array, trial_length:Array, f_target: float, c_reg: float                         
    ):
    """
    Implements vectorized version of e-prop, where update computation is done "offline".It is more efficient for tasks where learning signal is avaialble only at a few number of time steps since it then allows
    learning signals and gradients to be computed specifically at this points. For tasks with high number of time steps when learning signal available,  more memory intensive. 

    Returns hardcoded e-prop gradients (either recurrent or input layer) for a given batch.

    Note: the "online" implementation also applies updates only after a batch, but each single update is computed online and accumulated for applying at the end of batch processing
    """
    


    #unpack init carries(check compute_grads function for detailing on batch_init_carries )
    batch_init_eligibility_carries, _ = batch_init_carries # second element is init_grid_error, which is not used here

    # unpack params (check compute_grads function for detailing on params)
    eligibility_params, _ = params # second value is spatial params, which are not used here
    
    feedback_kernel = eligibility_params['ReadOut_0']['feedback_weights']       
    
    y_batch, true_y_batch, eligibility_input = batch_inputs

    if true_y_batch.ndim == 2:  # Guarantee that Labels have n_out dimension, so that are compatible with operation.
        true_y_batch = jnp.expand_dims(true_y_batch, axis=-1)

    # Compute task learning signal from time when it starts to be available
    task_error = learning_utils.error_signal(y=y_batch[:,-LS_avail:,:], true_y=true_y_batch[:,-LS_avail:,:]) # (n_batch, n_t, n_out)
    
    L = learning_utils.batched_learning_signal(task_error,feedback_kernel) # (n_batch, n_t, n_rec )

    # compute firing rate learning signal. Using traditional offline computation with average firing rate    
    f_error = learning_utils.firing_rate_error(z=z, trial_length=trial_length, f_target=f_target) #(n_batch, n_rec)
    
    
    # Evolve through time the low pass eligibility trace, given the history of eligibility inputs and the initial carries
    # eligibility_trace (n_t, n_b, n_post, n_pre)
    _, eligibility_traces = lax.scan(
        lambda carry, input: learning_utils.batched_eligibitity_trace(carry, input, eligibility_params),
        batch_init_eligibility_carries,
        eligibility_input
    )
    
   
     # compute low_pass eligibility traces, used for the update
     # low_eligibility_traces (n_t, n_b, n_post, n_pre)
    _, low_eligibility_traces = lax.scan(
        lambda carry, input:learning_utils.batched_low_pass_eligibility_trace(carry, input, eligibility_params),
        batch_init_eligibility_carries["low_pass_eligibility_trace"],
        eligibility_traces
    )
   
    # crop it for only time where learning signal is available
    crop_low_trace = low_eligibility_traces[-LS_avail:,:,:,:] #(n_t,n_b,n_rec,n_in)
    
    # compute task loss gradient
    task_update = jnp.einsum('btri,tbri->ir', jnp.expand_dims(L,axis=3), crop_low_trace)
    
    

    # compute firing rate regularization gradient
    acum_eligibility_trace = jnp.sum(eligibility_traces, axis=0) # (n_batches,n_post, n_pre)
    reg_update = (c_reg / trial_length[:,None,None]) * f_error[:,:, None] * acum_eligibility_trace # (n_post, n_pre)
    reg_update = jnp.transpose(jnp.mean(reg_update, axis=0),(1,0))   # (n_pre, n_post)
    return task_update + reg_update


def e_prop_online(batch_init_carries:Tuple[Dict[str,Array], Array],
                batch_inputs: Tuple[Array, Array,Tuple[Array, Array, Array]], 
                params: Tuple[Dict[str,Dict],Dict[str,Dict]], 
                LS_avail:int, z:Array, trial_length:Array, f_target: float, c_reg: float
):
    """
    Implements "online" version of e-prop, where updates are compute "online", but applied only after end of batch. it requires less memory usage and therefore more suitable
    for tasks where learning signal is available through out many time points of task 

    Returns hardcoded e-prop gradients (either recurrent or input layer) for a given batch.
    """
    #unpack init carries (check compute_grads function for detailing on batch_init_carries )
    batch_init_eligibility_carries, _ = batch_init_carries # second element is init_grid_error, which is not used here

    # unpack params (check compute_grads function for detailing on params )
    eligibility_params, _ = params # second value is spatial params, which are not used here

    f_error = learning_utils.firing_rate_error(z, trial_length, f_target) #(n_batch, n_rec)
    
    kernel = eligibility_params['ReadOut_0']['feedback_weights']    
    y_batch, true_y_batch, eligibility_input = batch_inputs
    
    if true_y_batch.ndim == 2:  # Guarantee that Labels have n_out dimension, so that are compatible with operation.
        true_y_batch = jnp.expand_dims(true_y_batch, axis=-1)
    
    
    y_batch = jnp.transpose(y_batch, (1,0,2)) # time need to be leading axis (n_t,n_b, n_out)
    true_y_batch = jnp.transpose(true_y_batch, (1,0,2)) # time need to be leading axis (n_t,n_b, n_out)
    
    def one_step_gradient(eligibility_carries, inputs, eligibility_params):
        y_batch_step, true_y_batch_step, eligibility_input_step = inputs
        task_error = learning_utils.error_signal(y=y_batch_step, true_y=true_y_batch_step) #(n_b, n_out)
        L = learning_utils.batched_learning_signal(error_signal=task_error, kernel=kernel) #(n_b,n_rec)
        new_eligibility_carries, eligibility_trace = learning_utils.batched_eligibitity_trace(eligibility_carries,eligibility_input_step, 
                                                                                               eligibility_params
        ) #(n_b, n_post, n_pre)
        
        new_eligibility_carries['low_pass_eligibility_trace'], low_pass_trace = learning_utils.batched_low_pass_eligibility_trace(new_eligibility_carries['low_pass_eligibility_trace'],eligibility_trace,eligibility_params)
        
        task_update = jnp.einsum('bri,bri->ir', jnp.expand_dims(L,axis=2), low_pass_trace) #n_pre, n_post
        
        reg_update = (c_reg/trial_length[:,None,None]) * f_error[:,:,None] * eligibility_trace # n_post, n_pre
        reg_update = jnp.transpose(jnp.mean(reg_update, axis=0),(1,0)) # mean over batches and transpose to have shape of weights n_pre, n_post
        return new_eligibility_carries, (task_update,reg_update)
    

    # scan over time to get updates (updates (n_t, n_pre, n_post))
    _, updates = lax.scan(
        lambda carry, input:one_step_gradient(carry, input, eligibility_params),
        batch_init_eligibility_carries,
        (y_batch, true_y_batch,eligibility_input )
    )
    task_updates, reg_updates = updates
    
    return jnp.sum(task_updates[-LS_avail:,:,:], axis=0) + jnp.sum(reg_updates, axis=0)



def neuromod_online(batch_init_carries:Tuple[Dict[str,Array], Array],
                    batch_inputs: Tuple[Array, Array,Tuple[Array, Array, Array]],                     
                    params: Tuple[Dict[str,Dict],Dict[str,Dict]],                     
                    LS_avail:int, z:Array, trial_length:Array, f_target: float, c_reg: float
):
    """
    Implements "online" version of neuromodulator diffusion updated, where updates are compute "online", but applied only after end of batch. 
    Returns hardcoded neuromodulator diffusion gradients(either recurrent or input layer) for a given batch.
    """
    # unpack values eligibility and diffusion params params (check compute_grads function for detailing on params)   
    eligibility_params, spatial_params = params

    # collect spatial params    
    diff_K = spatial_params['ALIFCell_0']['diff_K'] # diffusion kernel

    #radius = (jnp.shape(diff_K)[2] -1) // 2 # radius is not direclty saved as a params, but can be easily recovered since shape of kernel depends on it. k(n_neurotransmitter,1, 2*radius +1, 2*radius +1)
    radius=1
    cells_loc = spatial_params['ALIFCell_0']['cells_loc'] # array containig indices localizing position of the cells in the grid


    # firing regularization error
    f_error = learning_utils.firing_rate_error(z, trial_length, f_target) #(n_batch, n_rec)
    
    feedback_kernel = eligibility_params['ReadOut_0']['feedback_weights']    
       
    # modify shape of y_batch and true_y_batch for operation compatibility
    y_batch, true_y_batch, eligibility_input = batch_inputs    
    
    if true_y_batch.ndim == 2:  # Guarantee that Labels have n_out dimension, so that are compatible with operation.
        true_y_batch = jnp.expand_dims(true_y_batch, axis=-1)
        
    y_batch = jnp.transpose(y_batch, (1,0,2)) # time need to be leading axis (n_t,n_b, n_out)
    true_y_batch = jnp.transpose(true_y_batch, (1,0,2)) # time need to be leading axis (n_t,n_b, n_out)

        

        
    # define update function for single time step
    def one_step_gradient(carries, inputs):
        y_batch_step, true_y_batch_step, eligibility_input_step = inputs
        
        task_error = learning_utils.error_signal(y=y_batch_step, true_y=true_y_batch_step) #(n_b, n_out)
        # compute new learning signal arriving at current time step
        incoming_L = learning_utils.batched_learning_signal(task_error,feedback_kernel) # (n_batch, n_rec)
        
        
        # unpack carries
        eligibility_carries, error_grid = carries
        
        # Diffuses previous error using CA. Grid has shape (n_b, n_neuromodulators, h, w).
        diffused_error_grid = learning_utils.error_diffusion(carry=error_grid, inputs=jnp.zeros(1), params=(radius, diff_K)) # the function doesnt use inputs, it is just there in case decide to scan over it individually
        
        # get indices of where to modify --> add new error signal to locations where they are released        
        cell_rows, cell_cols = cells_loc[:, 0], cells_loc[:, 1]
        # add the current incoming learning signal to grid
        new_error_grid = diffused_error_grid.at[:,0,cell_rows,cell_cols].add(incoming_L)
        # Extract the learning signal available to each cell
        L = new_error_grid[:,0, cell_rows, cell_cols] #(n_b,n_rec)

        # From now on same as e-prop        
        new_eligibility_carries, eligibility_trace = learning_utils.batched_eligibitity_trace(eligibility_carries,eligibility_input_step, 
                                                                                               eligibility_params
        ) #(n_b, n_post, n_pre)
        
        new_eligibility_carries['low_pass_eligibility_trace'], low_pass_trace = learning_utils.batched_low_pass_eligibility_trace(new_eligibility_carries['low_pass_eligibility_trace'],eligibility_trace,eligibility_params)
        
        task_update = jnp.einsum('bri,bri->ir', jnp.expand_dims(L,axis=2), low_pass_trace) #n_pre, n_post
        
        reg_update = (c_reg/trial_length[:,None,None]) * f_error[:,:,None] * eligibility_trace # n_post, n_pre
        reg_update = jnp.transpose(jnp.mean(reg_update, axis=0),(1,0)) # mean over batches and transpose to have shape of weights n_pre, n_post
        return (new_eligibility_carries, new_error_grid), (task_update,reg_update)
    
    
    # scan over time to get updates (updates (n_t, n_pre, n_post))
    _, updates = lax.scan(
        lambda carry, input:one_step_gradient(carry, input),
        batch_init_carries,
        (y_batch, true_y_batch,eligibility_input )
    )

    # unpack different type of updates
    task_updates, reg_updates = updates
    
    return jnp.sum(task_updates[-LS_avail:,:,:], axis=0) + jnp.sum(reg_updates, axis=0) # task regularization only available from LS_avail onwards, firing_rate_reg available every moment
    
    


def output_grads(batch_init_carries: Dict[str,Array], batch_inputs: Tuple[Array, Array, Array], params: Tuple[Dict[str,Dict],Dict[str,Dict]], LS_avail:float) ->Tuple[Array, Array]:
    """
    Compute updates for output. It doesn`t need e-prop theory, but similar nomenclature was used for consistency.
    The same function is used for hardcoded neuromodulator diffusion

    This function computes the updates for output weights by processing a batch of 
    inputs and initial eligibility traces.    


    Notes
    -----
    This function processes the batch of eligibility traces and inputs, applying the 
    learning signal and eligibility trace updates in a vectorized manner for efficiency.
    Once the output dimension is normally low, can still be handled in terms of memory demand. 
    The function uses a scan operation to iterate through time steps, updating the 
    eligibility traces, and then computes the final updates by performing an element-wise 
    multiplication and summation over the batch and time dimensions.
    """
    # unpack params (check compute_grads function for detailing on batch_init_carries )
    eligibility_params, _ = params # second value is spatial params, which are not used here
    kappa = eligibility_params['ReadOut_0']['kappa'] 
    
    y_batch, true_y_batch, z = batch_inputs # Y_batch (n_batch, n_time, n_out)
    
    batch_init_eligibility_carries, _ = batch_init_carries # second element is init_grid_error, which is not used here

    init_carry= batch_init_eligibility_carries["v_eligibility_vector_out"]
    
    if true_y_batch.ndim == 2:  # Guarantee that Labels have w_out dimension, so that are compatible with operation. This is important for when there is only one readout cell
        true_y_batch = jnp.expand_dims(true_y_batch, axis=-1) #

    err = y_batch[:,-LS_avail:,:] - true_y_batch[:,-LS_avail:,:]  #(n_batch, n_time, n_out)
    
    
    # Scan over time to get history of low pass filtered z 
    _, traces = lax.scan(
        lambda carry, input: learning_utils.batch_readout_eligibility_vector(carry, input, kappa),
        init_carry,
        z # for scan it needs to be time major
    )
       
    crop_trace = traces[-LS_avail:,:,:] #(n_t,n_b,n_rec)
     
    # perform element-wise multiplication and sum over batches and time dimensions    
    grads = jnp.einsum('btor,tbor->ro', jnp.expand_dims(err,3), jnp.expand_dims(crop_trace,2)) # weights have shape (pre, post), so grad should have same shape --> ro)
    
    return grads



def autodiff_grads(batch,state, optimization_loss_fn,LS_avail, c_reg, f_target):
    x = batch["input"] # in compute_grads, transpose x to (n_t, n_b, n_in) for the hardcoded versions, so for here need to be transpose back to (n_b, n_t, n_in)
    labels = batch["label"]
    trial_length = batch["trial_duration"]

    if labels.ndim==2: 
        labels = jnp.expand_dims(labels, axis=-1) # this is necessary because target labels might have only (n_batch, n_t) and predictions (n_batch, n_t, n_out=1)

    def loss_fn(params):
        recurrent_carries, y = state.apply_fn({'params': params, 'eligibility params':state.eligibility_params, 'spatial params':state.spatial_params} , x)  
        _,_, _ , z, _ = recurrent_carries # only z is necessary here
        
        loss = optimization_loss_fn(logits=y[:, -LS_avail:, :], labels=labels[:,-LS_avail:, :], z=z,
                                     c_reg=c_reg, f_target=f_target, trial_length=trial_length
        ) # optimization loss will be both task and regultarization, so need to make sure everything is passed
        return loss, (y,z)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, aux_values), grads = grad_fn(state.params)
    y, z = aux_values
    return y, grads  



def compute_grads(batch:Dict[str, Array], state,optimization_loss_fn:Callable, LS_avail: int, 
                  local_connectivity: bool, f_target:float, c_reg:float, learning_rule:str, task:str) ->Dict[str, Dict]:
    """
    Compute grads according to chosen learning rule.

    Parameters
    ----------  
    batch: Dict keys are strings values Arrays
        input: Array (n_b, n_t, n_in)
            input spikes of batch trial
        label: Array (n_b, n_t, n_out)
            correct labels for trials in batch (for classification, should be one-hot encoded)
        trial_length: Array(n_b,)
            length of each trial in batch
    state: TrainState (check train files for more info). Relevant here:
        eligibility_params: Dict of Arraylike
            thr: scalar
                Initial firing threshold of recurrent neurons
            gamma: scalar
                Dampening factor for pseudoderivative
            alpha: scalar
                Decay constant for membrane potential (e^(-delta_t/tau_membrane))
            rho: scalar
                decay constant for adaptation hidden variable (e^(-delta_t/tau_adaptation)) 
            betas: Array (n_rec,)
                Adaptaiton strength parameter. For LIF cells it is 0.
            kappa: scalar
                Decay constant for output cells membrane potential
            feedback_weights: Array (n_rec, n_out)
                Weights used for feedback error signals
        spatial_params: Dict of Arrays
            M: Array (n_rec, n_rec)
                Local connectivity mask
            cells_loc: Array (n_rec, 2)
                Position of cells in the grid
            diff_K: Array (n_neuromodulators, 1, kernel_height, kernel_width)
                Kernels for diffusion simulation through CA
        init_e_carries: Nested dicts with arrays
            init_eligibility_carries_in: Dict of arrays (initial eligibility carries for input weights)
                epsilon_v_in: Array (n_batch, n_in)
                    Initial carry for membrane potential component of eligibility vector
                epsilon_a_in: Array (n_batch,n_rec, n_in)
                    Initial carry for adaptation potential component of eligibility vector
                psi: (n_batch, n_rec)
                    Initial carry for pseudo-derivative
                low_pass_eligibility_trace_in: (n_batch, n_rec, n_in)
                    Initial carry low-pass filtered eligibility trace
            init_eligibility_carries_rec: Dict of arrays (initial eligibility carries for recurrent weights)
                epsilon_v_rec: Array (n_batch, n_rec)
                    Initial carry for membrane potential component of eligibility vector
                epsilon_a_rec: Array (n_batch,n_rec, n_rec)
                    Initial carry for adaptation potential component of eligibility vector
                psi: (n_batch, n_rec)
                    Initial carry for pseudo-derivative
                low_pass_eligibility_trace_rec: (n_batch, n_rec, n_rec)
                    Initial carry low-pass filtered eligibility trace
            init_eligibility_carries_out: Dict of arrays (initial eligibility carries for output weights)
                v_eligibility_vector_out: (n_batch, n_rec)
                    Initial carry for output weight eligibility trace
        init_error_grid: Dict of arrays
            error_grid: Array (h,w)
                Initial carry for error grid
    optimization_loss: callable
        Loss function, possible including both task and firing rate regularization losses, which is used for rules 
        operating with jax autodiff
    LS_avail: int
        Time step from which learning signal is available in the task
    local_connectivity: bool
        True if network has local connectivity
    f_target: float
        Target firing frequency of firing rate regularization loss
    c_reg: float
        Constant regulating weight of firing regularization loss in the optimization problem
    learning_rule: str ["e_prop_hardcoded", "diffusion", "BPTT" ]
        Str representing which learning rule to apply
    task: str
        Str representing nature of loss (for now only accepts "classification" or "regression")

    Returns  
    ----------  
    logits: Array (n_b, n_t, n_out)
        Batch output of model (for classification tasks, returns non-normalized output)
    grads: Pytree (nested dictionaries)     
        Computed gradients according to chosen learning rule for the input, recurrent, and output layers
    """
                       
    # Control flow:
    #   "BBPT" or "e_prop_autodiff" --> use autodiff pipeline
    #   "e_prop_hardcoded" or "diffusion" --> use hardcoded pipeline --> For "e_prop_hardcoded" decide if "online" or "offline" based on value of LS_avail
    #   Any of valid options: raise error


    if (learning_rule == "BPTT") | (learning_rule == "e_prop_autodiff"):
        y, grads = autodiff_grads(batch=batch,state=state, optimization_loss_fn=optimization_loss_fn,
                               LS_avail=LS_avail, c_reg=c_reg, f_target=f_target)
        
        # Guarantee input sparsity is kept        
        grads['ALIFCell_0']['input_weights'] *= state.spatial_params['ALIFCell_0']['sparse_input']
       
       
        # guarantee no autapse
        n_rec = jnp.shape(grads['ALIFCell_0']['recurrent_weights'])[0]
        identity = jnp.eye(n_rec, dtype=grads['ALIFCell_0']['recurrent_weights'].dtype)
        grads['ALIFCell_0']['recurrent_weights'] = grads['ALIFCell_0']['recurrent_weights'] * (jnp.array(1) - identity) # guarantee that no self recurrence is learned
        # Guarantee that local connectivity is kept (otherwise, e-prop will lead to growth of new synapses)
        if local_connectivity:
            grads['ALIFCell_0']['recurrent_weights'] *= state.spatial_params['ALIFCell_0']['M']
        
        # Guarantee readout sparsity is kept        
        grads['ReadOut_0']['readout_weights'] *= state.spatial_params['ReadOut_0']['sparse_readout']
        return y, grads  # returning z only for purpose of plotting average firing rate
    
   
    elif (learning_rule == "e_prop_hardcoded") | (learning_rule == "diffusion"):

        # Forward pass
        variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'spatial params':state.spatial_params}
        recurrent_carries, logits = state.apply_fn(variables, batch['input'])  
        
        # Compute e-prop Updates
        # If task is classification, the eprop update function expects y to be already the assigned probability,
        # but model return logits. 
        if task == "classification":
            y = softmax(logits) # (n_batches, n_t, n_)
        else:
            y = logits # necessary only because want to return logits and not y for classification, easier to compute loss then
        
        # unpack recurrent carries
        v,_, A_thr , z, r = recurrent_carries # _ is a, which is not used 

        # prepare inputs  
        v = jnp.transpose(v, (1,0,2)) # for the scan needs to be time major (n_t,n_batches, n_rec)
        A_thr = jnp.transpose(A_thr, (1,0,2)) # for the scan needs to be time major
        z = jnp.transpose(z, (1,0,2)) # for the scan needs to be time major (n_t,n_batches, n_rec)
        r = jnp.transpose(r, (1,0,2)) # for the scan needs to be time major (n_t,n_batches, n_rec)
        x = jnp.transpose(batch["input"], (1,0,2)) # for the scan needs to be time major (n_t,n_batches, n_in)
        y_true = batch["label"]
        trial_length = batch["trial_duration"]

        # Pack inputs for each layer weight grads
        inputs_in = (y, y_true, (v, A_thr,r, x))
        inputs_rec = (y, y_true, (v, A_thr,r, z))
        inputs_out = (y, y_true, z)
        
        # State
        eligibility_params = state.eligibility_params
        spatial_params = state.spatial_params
        init_e_carries = state.init_eligibility_carries
        init_error_carries = state.init_error_grid
        init_error_grid = init_error_carries["error_grid"]

        #pack params
        params = eligibility_params, spatial_params
          
        if (learning_rule == "e_prop_hardcoded"):
            # For larger values of LS_avail, use online version. Note that when LS_avail is 0, actually means that it is available trought the whole task
            if (LS_avail > 0.) & (LS_avail <10):
                grad_function = e_prop_vectorized
            
            else:
                grad_function = e_prop_online
            
        elif learning_rule == "diffusion":
            grad_function = neuromod_online


        grads = {'ALIFCell_0':{}, 'ReadOut_0':{}}
        
        # Input Grads
        grads['ALIFCell_0']['input_weights'] = grad_function(batch_init_carries=(init_e_carries['inputs'], init_error_grid),
                                                                            batch_inputs= inputs_in, params=params, LS_avail = LS_avail, z=z, 
                                                                            trial_length=trial_length,f_target=f_target, c_reg=c_reg
        )
        
        # Guarantee input sparsity is kept        
        grads['ALIFCell_0']['input_weights'] *= state.spatial_params['ALIFCell_0']['sparse_input']
        
        # Recurrent Grads
        grads['ALIFCell_0']['recurrent_weights'] = grad_function(batch_init_carries=(init_e_carries['rec'],init_error_grid),
                                                                            batch_inputs=inputs_rec, params=params, LS_avail =LS_avail, z=z,
                                                                            trial_length=trial_length,f_target=f_target,
                                                                            c_reg=c_reg
        )
        
            
        # guarantee no autapse and that local connectivity is maintained
        n_rec = jnp.shape(grads['ALIFCell_0']['recurrent_weights'])[0]
        identity = jnp.eye(n_rec, dtype=grads['ALIFCell_0']['recurrent_weights'].dtype)
        grads['ALIFCell_0']['recurrent_weights'] = grads['ALIFCell_0']['recurrent_weights'] * (jnp.array(1) - identity) # guarantee that no self recurrence is learned 
        
        # Guarantee that local connectivity is kept (otherwise, e-prop will lead to growth of new synapses)
        if local_connectivity:
            grads['ALIFCell_0']['recurrent_weights'] *= state.spatial_params['ALIFCell_0']['M']
            
        

        # Output Grads
        grads['ReadOut_0']['readout_weights'] = output_grads(batch_init_carries=(init_e_carries['out'],init_error_grid), 
                                                            batch_inputs=inputs_out, params=params, 
                                                            LS_avail=LS_avail)

        grads['ReadOut_0']['readout_weights'] *= state.spatial_params['ReadOut_0']['sparse_readout']        
        return logits, grads # returning z only for purpose of plotting average firing rate
    
    else:
        raise NotImplementedError("The requested method {} hasn't being implemented. Please provide one of the valid learning rules: 'e_prop_hardcoded', 'e_prop_autodiff', 'diffusion' or 'BPTT'".format(learning_rule))






    