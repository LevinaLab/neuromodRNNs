"""Class and methods necessary tp define LSSN model from Bellec et al. 2020: A solution to the learning dilemma for recurrent networks of spiking neurons"""



from jax import lax, random, custom_vjp, numpy as jnp
from flax import linen as nn 
from flax.linen import initializers
from flax.linen.module import compact, nowrap
from optax import losses
import jax
import sys
import os

# Get the current directory of this file (which is 'project/src/ModRNN')
file_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to 'modRNN' directory
sys.path.append(file_dir + "/..")
from modRNN import  extra_initializers as inits

# Just for typing the type of inputs of functions
from typing import (
  Callable,
  Tuple,   
 )

from flax.typing import (
  Array,
  PRNGKey,
  Dtype,  
  Initializer  
)




# TODO: After finishing it, remove all not used imports
# TODO: check types of eligibility params
# TODO: Random vs Aligned e-prop: how to start the kernel for the random one.
# TODO: raise error if any of allowed feedback; so far, if not Symmetric then it will be random
# TODO: Initialization of carries and eligibility carries: some of them (like A_thr) actually depend on the
# on the values of others, so should be initialized as a funciton of them. Not relevant if want to initialize
# everything as 0, which is probably all the use cases.


# This is only necessary for using autodiff. Allows autodiff to use the pseudo-derivative for computing derivative of hidden state with respect to spike 
@custom_vjp
def spike(v_scaled, gamma, r):
    
    z = (jnp.where(v_scaled > 0, jnp.float64(1),jnp.float64(0)))
    return z
# forwardpass: spike function
def spike_fwd(v_scaled, gamma, r):
    z = spike(v_scaled, gamma, r)
    return z, (v_scaled, gamma, r)
# backpass: pseudoderivative
def spike_bwd(res, g):
    v_scaled, gamma, r = res
    # if neuron is refractory period, pseudo derivative should be 0
    no_refractory = (r == 0)
    pseudo_derivative = no_refractory * gamma * jnp.maximum(jnp.zeros_like(v_scaled), (1 - jnp.abs(v_scaled)))
    return g * pseudo_derivative, None, None # None ensures no gradient is computed for gamma and r

spike.defvjp(spike_fwd, spike_bwd)


class ALIFCell(nn.recurrent.RNNCellBase):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) recurrent neural network cell. 
    Implemented according to equations in 'Bellec et al. 2020: A solution to the learning dilemma for recurrent networks of spiking neurons'

    This cell incorporates both standard Leaky Integrate-and-Fire (LIF) neurons and adaptive LIF (ALIF) neurons
    with dynamic thresholds that adapt over time based on neuron activity. The cell maintains internal
    state variables for membrane potential and adaptation, and are connected recurrently.

   -------
    __call__(carry, x)
        Computes the next hidden and output states given the current carry and input.
    initialize_carry(rng, input_shape)
        Initializes the carry state for the cell.
    num_feature_axes
        Returns the number of feature axes for the cell. 
    """

    # net_arch
    n_ALIF: int = 3 # Number of adaptive neurons ALIF.
    n_LIF: int = 3 # Number of standard LIF neurons.
    connectivity_rec_layer: str = "local" # If or not the recurrent layer presents local connection pattern.
    sigma: float = 0.012 # controls probability of connection in the local connective mode according to distance between neurons.       
    gridshape: Tuple[int, int] = (10, 10) # (h,w) height(n_rows), width (n_cols) of 2D grid used for embedding of recurrent layer.
    n_neuromodulators: int =1 # number of neuromodulators.
    sparse_connectivity: bool = True # if recurrent network is sparsely connected to input
    
    # net_params
    thr: float = 0.6 # Base firing threshold for neurons
    tau_m: float = 20 # Membrane time constant (ms)
    tau_adaptation: float = 2000 # Time constant for adaptation (ms).
    beta: float = 1.7  # Modulator to initialized adaptation strength for ALIF. Notice that this is not the value of the adaptation, see at __call__ how it is used to initialize the value.
    gamma: float = 0.3 # Dampening factor for pseudo-derivative.
    refractory_period: int = 5 # refractory period in ms.
    k: float = 0 # decay rate of diffusion.
    radius:int = 1 # radius of difussion kernel,should probably be kept as one.
    learning_rule:str = "e_prop_hardcoded" # indicate which learning rule is being used. Important to block gradients for e_prop_autodiff. For hardcoded versions doesnt affect.
    input_sparsity: float = 0.1 # between 0 and 1, sparsity of input connections (only used if sparse_connectivity is True)


    # Initializers    
    v_init: Initializer = initializers.zeros_init() # Initializer for recurrent neurons membrane potential at t=0.
    a_init: Initializer = initializers.zeros_init() # Initializer for hidden adaptation variable at t=0.
    A_thr_init: Initializer = initializers.zeros_init() # Initializer for Adapted Threshold at t=0. TODO: deal with this later, should be initialized accordingly to a_init, z and betas.
    z_init: Initializer = initializers.zeros_init() # Initializer for recurrent neurons spike at t=0.
    r_init: Initializer = initializers.zeros_init()# Initializer for refractory period state of recurrent neurons at t=0.
    weights_init: Initializer = initializers.kaiming_normal() # Initializer for input and recurrent weights.
    gain: Tuple[float, float] = (0.5,0.1) # Gain for initialization of input and recurrent weights.
    
    
    # seeds
    local_connectivity_seed: int = 0 # seed for initialize RNG used for local_connectivity mask.
    cell_loc_seed: int = 3 # seed for initialize RNG used for location of cells in the 2D embedding (grid).
    diff_kernel_seed: int = 0 # # seed for initialize RGN for diffusion kernel (not used in the function).
    input_sparsity_seed: int = 3342 # Key for sparsity mask initialization
    # others
    dt: float = 1 # Time step size (ms).
    param_dtype: Dtype = jnp.float32  # dtype of parameters.

    
    @compact
    def __call__(self, carry:Tuple[Array, Array, Array, Array], x:Array) ->Tuple[Array, Tuple[Array, Array]]:
        """
        Compute the next hidden and output states (one time step).

        Parameters
        ----------
        carry : tuple of arrays
            Current state of the cell, containing membrane potential (v), adaptation variable (a),
            adaptive threshold (A_thr), spike output (z) and refractory period state (r).
        x : array
            Input to the cell at the current time step.

        Returns
        -------
        new_carry : tuple of arrays
            Updated state of the cell.
                v: Array (n_batch, n_t, n_rec)
                a: Array (n_batch, n_t, n_rec)
                A_thr: Array (n_batch, n_t, n_rec)
                r: Array (n_batch, n_t, n_rec)
                z: Array (n_batch, n_t, n_rec)
        (new_carry, new_z) : tuple
            Tuple containing the updated state and the spike output for the current time step.
            Returning the tuple with new_carry is important, so that the whole history of carries is outputed
            after the scan with RNN module
        """
    
    
        # initialized parameters for eligibility params (those are the params necessary to compute eligibility trace later)
        thr = self.variable('eligibility params', 'thr', lambda: jnp.array(self.thr, dtype= self.param_dtype))
        gamma = self.variable('eligibility params', 'gamma', lambda: jnp.array(self.gamma, dtype= self.param_dtype))
        alpha = self.variable('eligibility params', 'alpha', lambda: jnp.array(jnp.exp(-self.dt/self.tau_m), dtype= self.param_dtype)) 
        rho = self.variable('eligibility params', 'rho', lambda: jnp.array(jnp.exp(-self.dt/self.tau_adaptation), dtype= self.param_dtype))
        n_rec = self.variable('eligibility params', 'n_rec', lambda: jnp.array(self.n_LIF + self.n_ALIF, dtype= self.param_dtype))
        
        def init_betas(beta= self.beta, rho=rho.value, alpha=alpha.value, n_LIF=self.n_LIF, n_ALIF=self.n_ALIF): 
                """Init betas"""
                return beta * jnp.concatenate((jnp.zeros(n_LIF, dtype= self.param_dtype), jnp.ones(n_ALIF, dtype= self.param_dtype)))  # notice that for LIFs, beta is 0           
                

        # Initialize betas as a fixed variable
        betas = self.variable('eligibility params', 'betas', init_betas) # dim: (n_rec,)
        
        # Initialize position of recurrent cells in 2-D grid
        cells_loc = self.variable('spatial params', 'cells_loc', 
                                inits.initialize_neurons_position(gridshape=self.gridshape, 
                                                                    key=random.key(self.cell_loc_seed),
                                                                    n_rec=self.n_LIF +self.n_ALIF,
                                                                    dtype= self.param_dtype))
        
        # Initialize recurrent connectivity mask --> local connection or 1s, depending on local_connectivity
        M = self.variable('spatial params', 'M',
                        inits.initialize_connectivity_mask(local_connectivity=self.local_connectivity,
                                                            gridshape=self.gridshape,
                                                            neuron_indices=cells_loc.value,
                                                            key=random.key(self.local_connectivity_seed),
                                                            n_rec=self.n_LIF + self.n_ALIF,
                                                            sigma = self.sigma, 
                                                            dtype = self.param_dtype)
        )

        # Initialize input connectivity mask --> mask with desired sparsity, or full of 1s, depending on sparse_connectivity
        input_sparsity_mask = self.variable('spatial params', 'sparse_input',
                        inits.initialize_sparsity_mask(sparse_connectivity=self.sparse_connectivity, shape=(jnp.shape(x)[-1], self.n_LIF + self.n_ALIF),
                                                       key=random.key(self.input_sparsity_seed),sparsity=self.input_sparsity, dtype = self.param_dtype))                                   
         
        
        # Initialize diffusion kernel (not used in forward pass)
        diff_K = self.variable('spatial params', 'diff_K', inits.k_initializer(k=self.k, shape=(self.n_neuromodulators, 1, 2*self.radius+1, 2*self.radius+1)), dtype= self.param_dtype)
            
        # Initialize weights (in @compact method, the init is done inside of __call__) 
        w_in = self.param(
        'input_weights',
        inits.generalized_initializer(self.weights_init,
                                        gain=self.gain[0],
                                        avoid_self_recurrence=False,
                                        mask_connectivity=input_sparsity_mask.value),
                                        (jnp.shape(x)[-1], self.n_LIF + self.n_ALIF), # need to use self.n_LIF + self.n_ALIF instead of n_rec.value, because n_rec.value becomes traceable array, what does not work for jit 
                                        self.param_dtype,
        ) # dim: (n_in, n_rec)
        
        w_rec = self.param(
        'recurrent_weights',      
        inits.generalized_initializer(self.weights_init,
                                        gain=self.gain[1],
                                        avoid_self_recurrence=True, 
                                        mask_connectivity=M.value),
        (self.n_LIF + self.n_ALIF, self.n_LIF + self.n_ALIF),
        self.param_dtype,
        ) # dim: (n_rec, n_rec)    
        
        
        
        # Forward pass - Hidden state:  v: recurrent layer membrane potential a: threshold adaptation
        # Visible state: z: recurrent layer spike output, vo: output layer membrane potential (yo incl. activation function)
        
        v, a, A_thr, z, r = carry  # unpack the hidden stated of the ALIF cells: membrane potential v and adaptation variable a. although in the paper z is the observable state, for a better agreement with the structure of recurrent cells in FLAX, it was also defined here as a carry
        
    
        # Mask with neurons that are not in refractory period
        # update refractory period at time t (looking spike at time t-1)
        new_r = jnp.where(z>0, jnp.ones_like(r) * self.refractory_period, jnp.maximum(0, r-1))    
        no_refractory = (new_r == 0)

        # compute v at time t
        if self.learning_rule == "e_prop_autodiff":
            # for using autodiff with e_prop, need to guarantee that gradient of z doesnt propagate to next steps through the recurrent spike transmissio. Note that e-prop also doesnt count in the effect of the 
            # the reset term in the approximated grads, thus also blocking here.

            local_z = lax.stop_gradient(z) # use for gradients that are not considered in e-prop: spike propagation and reset term
            new_v =  (alpha.value * v +  (1-alpha.value) * ((jnp.dot(local_z, w_rec)) + jnp.dot(x, w_in)) - lax.stop_gradient(z*thr.value)) #important, z and x should have dimension n_b, n_features and w n_features, output dimension
        
        else:
            new_v =   (alpha.value * v + (1-alpha.value) * ((jnp.dot(z, w_rec)) + jnp.dot(x, w_in)) -  v*thr.value) #important, z and x should have dimension n_b, n_features and w n_features, output dimension
        
        # compute a at time t        
        new_a = rho.value * a + (1-rho.value) * z # for adaptation, z is local and doesn`t need to be stopped even for autodiff e-prop
        
        # compute A_thr at time t
        new_A_thr =   thr.value + new_a * betas.value # compute value of adapted threshold at t+1  
        
        # compute z at time t
        new_z = spike((new_v-new_A_thr)/thr.value, gamma.value, new_r)  * no_refractory
        

        
        new_carry =  (new_v, new_a, new_A_thr, new_z, new_r)   
        return new_carry, (new_carry, new_z)      
        
    ##########################################################################

    @nowrap
    def initialize_carry(
    self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        """
        Initialize the carry state for the cell.

        Parameters
        ----------
        rng : PRNGKey
            Random number generator key.
        input_shape : tuple of int
            Shape of the input data, excluding the batch dimension.

        Returns
        -------
        carry : tuple of arrays
            Initialized carry state for the cell.
        """


        batch_dims = input_shape[:-1] # gets batch dimensions (everything except for the last dimension). In this application, input shape is always (n_batch, n_in). So batch_dims = (n_batches,)

        key1, key2, key3, key4, key5 = random.split(rng, 5)
        hidden_shape = batch_dims + (self.n_ALIF + self.n_LIF,) # sum of tuples mean concatenation: (n_batches) + (n_rec) = (n_batches, n_rec)
        v = self.v_init(key1, hidden_shape, self.param_dtype) # (n_batches, n_rec)
        a = self.a_init(key2, hidden_shape, self.param_dtype) # (n_batches, n_rec)
        A_thr = self.A_thr_init(key3, hidden_shape, self.param_dtype) # (n_batches, n_rec)
        z = self.z_init(key4, hidden_shape, self.param_dtype) # (n_batches, n_rec)
        r = self.r_init(key5, hidden_shape, self.param_dtype) # (n_batches, n_rec)
        return (v, a, A_thr, z, r)


    @property
    def num_feature_axes(self) -> int:
        return 1


class ReadOut(nn.recurrent.RNNCellBase):
    """
    Readout layer for a recurrent neural network.

    This layer reads the output from the recurrent layer and processes it through a leaky integrator to produce the final output. It also supports 
    different types of feedback connections, with options for symmetric or random initialized feedback weights for backward pass.

    Methods
    -------

    __call__(carry, z)
        Computes the next output state given the current carry and input from the recurrent layer.
    initialize_carry(rng, input_shape)
        Initializes the carry state for the readout layer.
    num_feature_axes
        Returns the number of feature axes for the readout layer.
    """ 
  
    # net_arch
    n_out: int = 2 # Number of output neurons.   
    sparse_connectivity: bool = True # if recurrent network is sparsely connected to readout

    # net_params
    b_out: Tuple = 0 # Bias for the output neurons. Not implemented yet --> dont change it
    tau_out: float =1 # Time constant for the output layer (ms).
    feedback: str = "Symmetric" # Type of feedback ('Symmetric' or 'Random').
    sparsity: float = 0.1 # between 0 and 1, sparsity of readout connections (only used if sparse_connectivity is True)

    # Initializers
    carry_init: Initializer = initializers.zeros_init() # Initializer for the carry state (output layer membrane potential).
    weights_init: Initializer = initializers.kaiming_normal() # Initializer for the readout weights.
    feedback_init: Initializer = initializers.kaiming_normal() # Initializer for the feedback weights.
    gain: Tuple = (0.5, 0.5) # Gain factors for weight initialization.
    
    
    # seeds
    FeedBack_seed: int = 42 # Key for feedback weights initialization.
    sparsity_seed: int = 3312 # Key for sparsity mask initialization
    # others
    dt: float = 1 # Time step size (ms).
    param_dtype: Dtype = jnp.float32 # Data type for parameters.   

           


    @compact
    def __call__(self, carry:Array, z:Array):   
        """
        Computes the next output state given the current carry and input from the recurrent layer.
        Output and carry are the membrane voltage of the leaky output neurons.
        """
        
        sparsity_mask = self.variable('spatial params', 'sparse_readout',
                        inits.initialize_sparsity_mask(sparse_connectivity=self.sparse_connectivity, shape=(jnp.shape(z)[-1], self.n_out),
                                                       key=random.key(self.sparsity_seed),sparsity=self.sparsity, dtype = self.param_dtype))                                   
                                                        






        # Initialize weights (in @compact method, the init is done inside of __call__)
        w_out = self.param(
            'readout_weights',
            inits.generalized_initializer(self.weights_init,
                                        gain=self.gain[0],
                                        avoid_self_recurrence=False,
                                        mask_connectivity=sparsity_mask.value),
                                        (jnp.shape(z)[-1], self.n_out),
                                        self.param_dtype
        ) # dim: (n_pre, n_post): in the paper architecture this is (n_rec, n_out)

            
        # initialized parameters for eligibility params (those are the params necessary to compute eligibility trace later)

        kappa = self.variable('eligibility params', 'kappa', lambda: jnp.exp(-self.dt/self.tau_out))
        B_out = self.variable('eligibility params', 'feedback_weights', inits.feedback_weights_initializer(self.feedback_init,
                                                                                                            random.key(self.FeedBack_seed), 
                                                                                                            (jnp.shape(z)[-1], self.n_out),
                                                                                                            w_out, sparsity_mask.value, self.feedback, gain=self.gain[1]
                                                                                                            )  #n_rec, n_out
        )            


        # Forward pass - real valued leaky output neurons (output new y, or equivalently to new mebrane voltage)
        y = carry  # unpack the hidden state: for readout it is only the output at previous step (basically the membrane potential, but kept as y for consistency with paper nomenclature)

        new_y = kappa.value * y +  (1 - kappa.value) * jnp.dot(z, w_out)  #TODO: implement bias #(1-kappa.value) *
                    
        return new_y, new_y    
        
    ##########################################################################

    @nowrap
    def initialize_carry(
    self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Array:
        """
        Initialize the carry state for the readout layer.

        Parameters
        ----------
        rng : PRNGKey
            Random number generator key.
        input_shape : tuple of int
            Shape of the input data, at oa single time point.

        Returns
        -------
        y : array
            Initialized carry state for the readout layer.
        """

        batch_dims = input_shape[:-1] # gets batch dimensions (everything except for the last dimension). In this application, input shape is always (n_batch, n_in). So batch_dims = (n_batches,)
        key1, key = random.split(rng)
        hidden_shape = batch_dims + (self.n_out,) # sum of tuples mean concatenation: (n_batches) + (n_rec) = (n_batches, n_rec)
        y = self.carry_init(key1, hidden_shape, self.param_dtype)
        return y


    @property
    def num_feature_axes(self) -> int:
        return 1
  
  
class LSSN(nn.Module):
    """
    Long Short-term Spike Network (LSSN) combining ALIF and LIF cells with a readout layer, as defined in Bellec et al. 2020: A solution to the learning dilemma for recurrent networks of spiking neurons

    This module defines a recurrent neural network with adaptive and leaky integrate-and-fire (ALIF and LIF) cells 
    and a readout layer that processes the output from the recurrent layer. It evolves the recurrent dynamical system on time given the inputs

    

    Methods
    -------
    __call__(x)
        Perform the forward pass through the LSSN.
    initialize_eligibility_carry(rng, input_shape)
        Initialize the eligibility trace carry state.


    
    # net_arch
    n_ALIF: int = 3 # Number of adaptive neurons ALIF.
    n_LIF: int = 3 # Number of standard LIF neurons.
    local_connectivity: bool = True # If or not the recurrent layer presents local connection pattern.
    sigma: float = 0.012 # controls probability of connection in the local connective mode according to distance between neurons.       
    gridshape: Tuple[int, int] = (10, 10) # (w,h) width (n_cols) and height(n_rows) of 2D grid used for embedding of recurrent layer.
    n_neuromodulators: int =1 # number of neuromodulators.
    
    # net_params
    thr: float = 0.6 # Base firing threshold for neurons
    tau_m: float = 20 # Membrane time constant (ms)
    tau_adaptation: float = 2000 # Time constant for adaptation (ms).
    beta: float = 1.7  # Modulator to initialized adaptation strength for ALIF. Notice that this is not the value of the adaptation, see at __call__ how it is used to initialize the value.
    gamma: float = 0.3 # Dampening factor for pseudo-derivative.
    refractory_period: int = 5 # refractory period in ms.
    k: float = 0 # decay rate of diffusion.
    radius:int = 1 # radius of difussion kernel,should probably be kept as one.


    # Initializers    

    weights_init: Initializer = initializers.kaiming_normal() # Initializer for input and recurrent weights.
    gain: Tuple[float, float] = (0.5,0.1) # Gain for initialization of input and recurrent weights.
    
    
    # seeds
    local_connectivity_seed: int = 0 # seed for initialize RNG used for local_connectivity mask.
    cell_loc_seed: int = 3 # seed for initialize RNG used for location of cells in the 2D embedding (grid).
    diff_kernel_seed: int = 0 # # seed for initialize RGN for diffusion kernel (not used in the function).
    
    # others
    dt: float = 1 # Time step size (ms).
    param_dtype: Dtype = jnp.float32  # dtype of parameters. 
    """
                  
    # architecture
    n_ALIF: int = 3 # Number of ALIF neurons.
    n_LIF: int = 3 # Number of LIF neurons.
    n_out: int = 2  # Number of output neurons.
    local_connectivity: bool = True # If or not the recurrent layer presents local connection pattern.
    sigma: float = 0.012 # controls probability of connection in the local connective mode according to distance between neurons.  
    gridshape: Tuple[int, int] = (10, 10) # (w,h) width (n_cols) and height(n_rows) of 2D grid used for embedding of recurrent layer.
    n_neuromodulators: int =1 # number of neuromodulators.
    sparse_input: bool = False # if recurrent network is sparsely connected to external inputs
    sparse_output: bool = False # if recurrent network is sparsely connected to readout
    
    # ALIF params
    thr: float = 0.6 # Base firing threshold for neurons.
    tau_m: float = 20 # Membrane time constant (ms).
    tau_adaptation: float = 2000  # Time constant for adaptation (ms).
    beta: float= 1.7  # Modulator to initialized adaptation strength for ALIF. Notice that this is not the value of the adaptation, see at ALIF module method __call__ how it is used to initialize the value.
    gamma: float = 0.3 # Dampening factor for pseudo-derivative.
    refractory_period: int = 5 # refractory period in ms.
    k: float = 0 # decay rate of diffusion.
    radius:int = 1 # radius of difussion kernel,should probably be kept as one.
    learning_rule:str = "e_prop_hardcoded" # indicate which learning rule is being used. Important to block gradients for e_prop_autodiff. For hardcoded versions doesnt affect.  
    input_sparsity: float = 0.1 # between 0 and 1, sparsity of input connections (only used if sparse_connectivity is True)


    # Readout params
    tau_out: float = 20  # Time constant for the output layer (ms).
    feedback: str = "Symmetric"  # Type of feedback ('Symmetric' or 'Random').
    readout_sparsity: float = 0.1 # between 0 and 1, sparsity of readout connections (only used if sparse_connectivity is True)


    # 
    # Carries initializers
    v_init: Initializer = initializers.zeros_init() # Initializer for recurrent neurons membrane potential at t=0.
    a_init: Initializer = initializers.zeros_init() # Initializer for hidden adaptation variable at t=0.
    A_thr_init: Initializer = initializers.zeros_init() # Initializer for Adapted Threshold at t=0. TODO: deal with this later, should be initialized accordingly to a_init, z and betas.
    z_init: Initializer = initializers.zeros_init() # Initializer for recurrent neurons spike at t=0.
    r_init: Initializer = initializers.zeros_init()# Initializer for refractory period state of recurrent neurons at t=0.
    out_carry_init: Initializer = initializers.zeros_init() # initializer for output carry at t=0 (output neuron voltage membrane)
    e_carry_init: Initializer = initializers.zeros_init()    # intitializer for eligibility carries at t=0
    error_grid_init: Initializer = initializers.zeros_init() # initializer for error grid at t=[]
    
    # weights and biases initializers
    b_out: Tuple = 0 # For now, the bias is not implemented, think about this after training is sucessful
    ALIF_weights_init: Initializer = initializers.kaiming_normal()  # Initializer for input and recurrent weights.
    ReadOut_weights_init: Initializer = initializers.kaiming_normal()  # Initializer for output weights.
    feedback_init: Initializer = initializers.kaiming_normal() # Initializer for the feedback weights.
    gain: Tuple[float, float, float] = (0.5,0.1,0.5, 0.5) # Gain for initialization of input, recurrent, output and feedback weights respectively 
    diff_kernel_init: Initializer = initializers.ones_init()
    
    # seeds
    local_connectivity_seed: int = 0 # seed for initialize RNG used for local_connectivity mask.
    cell_loc_seed: int = 3 # seed for initialize RNG used for location of cells in the 2D embedding (grid).
    diff_kernel_seed: int = 0 # seed for initialize RNG for diffusion kernel (not used in the function).
    FeedBack_seed: int = 42 # seed for initialize RNG for Feedback weights.
    input_sparsity_seed: int = 3342 # Key for input sparsity mask initialization
    readout_sparsity_seed: int = 3312 # Key for readout sparsity mask initialization
    # Others
    dt: float = 1      
    loss: Callable = losses.softmax_cross_entropy   
    param_dtype: Dtype = jnp.float64
    
    
    @compact
    def __call__(self, x):
        """
        Defines the forward pass of the neural network model.

        Parameters:
        x (jax.numpy.ndarray): Input data to the network.

        Returns:
        tuple: A tuple containing:
            - recurrent_carries (Tuple): The recurrent states after processing the input through the RNN .
                containing membrane potential (v), adaptation variable (a), adaptive threshold (A_thr), and spike output (z) of recurrent layer.
            - y (jax.numpy.ndarray): The output of the network after processing through the readout layer. If classification is enabled, this will be the softmax output.

        Description:
        This method implements the forward pass of a recurrent neural network (RNN) with custom ALIF and readout cells.
        The model consists of two main components:
        1. `recurrent`: An RNN layer using `ALIFCell` which processes the input data `x` over time.
        2. `readout`: An RNN layer using `ReadOut` which processes the output from the `recurrent` layer. The Readout layer consits of leaky neurons, with output being the continuous value
            of membrane potential
        """
        
        # Define the layers (nn.RNN does the scan over time)
        recurrent = nn.RNN(ALIFCell(n_ALIF=self.n_ALIF, n_LIF=self.n_LIF, local_connectivity=self.local_connectivity,sparse_connectivity=self.sparse_input, sigma = self.sigma,
                                    gridshape= self.gridshape, n_neuromodulators=self.n_neuromodulators, thr=self.thr, tau_m=self.tau_m, tau_adaptation=self.tau_adaptation,
                                    beta = self.beta, gamma = self.gamma, refractory_period = self.refractory_period, k=self.k, radius=self.radius, learning_rule=self.learning_rule, input_sparsity=self.input_sparsity,
                                    v_init = self.v_init, a_init=self.a_init, A_thr_init=self.A_thr_init, z_init=self.z_init, r_init=self.r_init, weights_init = self.ALIF_weights_init,
                                    gain = (self.gain[0], self.gain[1]), local_connectivity_seed= self.local_connectivity_seed, cell_loc_seed= self.cell_loc_seed,
                                    diff_kernel_seed=self.diff_kernel_seed, input_sparsity_seed=self.input_sparsity_seed, dt=self.dt, param_dtype=self.param_dtype), variable_broadcast=("params", 'eligibility params', 'spatial params'), name="Recurrent"
        )
        
        readout = nn.RNN(ReadOut(n_out=self.n_out, sparse_connectivity=self.sparse_readout, b_out=self.b_out, tau_out=self.tau_out, feedback = self.feedback, sparsity=self.readout_sparsity, 
                                 carry_init= self.out_carry_init, weights_init= self.ReadOut_weights_init,
                                 feedback_init = self.feedback_init, gain = (self.gain[2], self.gain[3]), 
                                 FeedBack_seed=self.FeedBack_seed, sparsity_seed=self.readout_sparsity_seed, dt = self.dt, param_dtype=self.param_dtype), variable_broadcast=("params",'eligibility params', 'spatial params'), name="ReadOut"
        )
        
        # Apply then
        recurrent_carries, z = recurrent(x)
        
        y = readout(z)
        return recurrent_carries, y # recurrent_carries already contains z
        
    @nowrap
    def initialize_eligibility_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        """
        Initializes the eligibility trace carries for the neural network. All the carries are initialized as zeros arrays of respective shape.
        One can pass a different initializer, but it will be shared between all carries. For modifiyng individual carries, the best option is to do
        so after the initialization.

        Parameters:
        rng (PRNGKey): A pseudo-random number generator key.
        input_shape (Tuple[int, ...]): The shape of the input data, expected to be (n_batch, n_in).

        Returns:
        dict: A dictionary containing three dictionaries with initialized eligibility carries for inputs, recurrent, and output:
            - "inputs": A dictionary with keys "v_eligibility_vector", "a_eligibility_vector", "psi", and "low_pass_eligibility_trace".
            - "rec": A dictionary with keys "v_eligibility_vector", "a_eligibility_vector", "psi", and "low_pass_eligibility_trace".
            - "out": A dictionary with key "v_eligibility_vector_out".

    

            Variables:
        - psi: Initial pseudoderivative initialized with `e_carry_init` function for the recurrent layer.
        - epsilon_v_in: Eligibility carry for input weights voltage vector.
        - epsilon_a_in: Eligibility carry for input weights adaptation vector.
        - low_pass_eligibility_trace_in: Low-pass filtered eligibility trace for the input weights.
        - epsilon_v_rec: Eligibility carry for recurrent weights voltage vector
        - epsilon_a_rec: Eligibility carry for recurrent weights adaptation vector.
        - low_pass_eligibility_trace_rec: Low-pass filtered eligibility trace for the recurrent weights.
        - epsilon_v_out: Eligibility carry for the output layer weights.

        Note:
        The eligibility carries are designed to be saved and modified directly in the training state if necessary.
        """
        
        # As a decision, all the eligibility carries will be initialized with the init function. 
        # In the training code, they are saved in the the train_state, so that it can be directly modified there
        # in case there is the necessity to change individual ones
        
        key1, key2, key3, key4, key5, key6, key7 = random.split(rng, 7)
        batch_dims = (input_shape[0],) # gets batch dimensions (everything except for the last dimension). In this application, input shape is always (n_batch, n_in). So batch_dims = (n_batches,)
        hidden_shape = batch_dims + (self.n_ALIF + self.n_LIF,) # sum of tuples mean concatenation: (n_batches) + (n_rec) = (n_batches, n_rec)
        
        # psi
        psi = self.e_carry_init(key1, hidden_shape, self.param_dtype) # shape (n_batches, n_rec)
        
        # input eligibility carries
        epsilon_v_in =  self.e_carry_init(key1, batch_dims + (input_shape[-1],), self.param_dtype) # shape (n_batches, n_in)
        epsilon_a_in =  self.e_carry_init(key2, hidden_shape + input_shape[-1:], self.param_dtype)  # shape (n_batches, n_rec, n_in)
        low_pass_eligibility_trace_in =  self.e_carry_init(key3, hidden_shape + input_shape[-1:], self.param_dtype)  # shape (n_batches, n_rec, n_in)

        # recurrent eligibility carries      
        epsilon_v_rec =  self.e_carry_init(key4, hidden_shape, self.param_dtype)
        epsilon_a_rec =  self.e_carry_init(key5, hidden_shape + (self.n_ALIF + self.n_LIF,), self.param_dtype) # shape (n_batches, n_rec, n_rec)
        low_pass_eligibility_trace_rec = self.e_carry_init(key6, hidden_shape + (self.n_ALIF + self.n_LIF,), self.param_dtype)
        
        # out eligibility carries: theoretically, this value should somehow be coupled to epsilon_v_rec, since
        # both are low-passed filtered version of the ALIF cells' spikes, but with different decay constant
        # but here for simplicity they are treated independently, specially because in most of applications
        # they both will be initialized as zeros
        epsilon_v_out =  self.e_carry_init(key7, hidden_shape, self.param_dtype)
        
        
        init_eligibility_carries_in = {"v_eligibility_vector":epsilon_v_in,
                                        "a_eligibility_vector":epsilon_a_in,
                                        "psi": psi,
                                        "low_pass_eligibility_trace":low_pass_eligibility_trace_in
                                        }

        init_eligibility_carries_rec = {"v_eligibility_vector":epsilon_v_rec,
                                        "a_eligibility_vector":epsilon_a_rec,
                                        "psi": psi,
                                        "low_pass_eligibility_trace":low_pass_eligibility_trace_rec
                                        }
        
        init_eligibility_carries_out = {"v_eligibility_vector_out":epsilon_v_out}
        
        return {"inputs":init_eligibility_carries_in, 
                "rec":init_eligibility_carries_rec,
                "out":init_eligibility_carries_out}
        
    @nowrap
    def initialize_grid(self, rng:PRNGKey, input_shape: Tuple[int, ...]):
       """
       Initialize the error grid wiht shape (n_batch, n_neuromodulators, height, width). Error grid is used in diffusion neuromodulators
       to diffuse the errors transmitted by differente neuromodulators
       """
       batch_dims = (input_shape[0],)
       init_error_grid = self.error_grid_init(rng, batch_dims+(self.n_neuromodulators,) + self.gridshape, self.param_dtype)
       return {"error_grid":init_error_grid}