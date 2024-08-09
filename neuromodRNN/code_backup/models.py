from jax import numpy as jnp
from jax import random, custom_vjp
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API
from flax.linen import initializers
from flax.linen.module import compact, nowrap
from optax import losses

import sys
import os

# Get the current directory of this file (which is 'project/src/general_src')
file_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the root 'project' directory
sys.path.append(file_dir + "/..")
from . import extra_initializers as inits
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
  Initializer  
)

DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex


# TODO: After finishing it, remove all not used imports
# TODO: check types of eligibility params
# TODO: Random vs Aligned e-prop: how to start the kernel for the random one.
# TODO: raise error if any of allowed feedback; so far, if not Symmetric then it will be random
# TODO: Initialization of carries and eligibility carries: some of them (like A_thr) actually depend on the
# on the values of others, so should be initialized as a funciton of them. Not relevant if want to initialize
# everything as 0, which is probably all the use cases.


# This is only necessary for when autodiff e-prop will be implemented
@custom_vjp
def spike(v_scaled, gamma):
    z = jnp.float32(jnp.where(v_scaled > 0, jnp.array(1), jnp.array(0)))
    return z

def spike_fwd(v_scaled, gamma):
    z = spike(v_scaled, gamma)
    return z, (v_scaled, gamma)

def spike_bwd(res, g):
    v_scaled, gamma = res
    pseudo_derivative = gamma * jnp.maximum(jnp.zeros_like(v_scaled), (1 - jnp.abs(v_scaled)))
    return g * pseudo_derivative, None # None ensures no gradient is computed for gamma

spike.defvjp(spike_fwd, spike_bwd)


class ALIFCell(nn.recurrent.RNNCellBase):
  """
    Adaptive Leaky Integrate-and-Fire (ALIF) recurrent neural network cell.

    This cell incorporates both standard Leaky Integrate-and-Fire (LIF) neurons and adaptive LIF (ALIF) neurons
    with dynamic thresholds that adapt over time based on neuron activity. The cell maintains internal
    state variables for membrane potential and adaptation, and supports recurrent connections.

    Attributes
    ----------
    thr : float
        Base firing threshold for neurons.
    tau_m : float
        Membrane time constant (ms).
    tau_adaptation : float
        Time constant for adaptation (ms).
    dt : float
        Time step size (ms).
    n_ALIF : int
        Number of adaptive ALIF neurons.
    n_LIF : int
        Number of standard LIF neurons.
    v_init : Initializer
        Initializer for membrane potential.
    a_init : Initializer
        Initializer for adaptation variable.
    A_thr_init : Initializer
        Initializer for adaptive threshold.
    z_init : Initializer
        Initializer for spike output.
    weights_init : Initializer
        Initializer for weights.
    gain : Tuple[float, float]
        Gain factors for weight initialization.
    param_dtype : Dtype
        Data type for parameters.
    beta : Union(None, list)
        List of beta values for adaptation, or None to compute them automatically.
    gamma : float
        Dampening factor for pseudo-derivative.
    local_connectivity: bool, optional
        If the model will use local_connectivity or not. Notice that so far,
        only the 2D embeding provided is spatial_embedings.py can be used for this.     
    local_connectivity_key: int, optional
        Seed to generate PRGN key for initialization of local_connectivity mask
    sigma: float, optional
        Parameter controling probability of connection in the local connective mode 
        according to distance between neurons.
  """
  """  -------
    __call__(carry, x)
        Computes the next hidden and output states given the current carry and input.
    initialize_carry(rng, input_shape)
        Initializes the carry state for the cell.
    num_feature_axes
        Returns the number of feature axes for the cell.
    """
    
  thr: float = 0.6
  tau_m: float = 20
  tau_adaptation: float = 2000
  dt: float = 1
  n_ALIF: int = 3
  n_LIF: int = 3
  v_init: Initializer = initializers.zeros_init() 
  a_init: Initializer = initializers.zeros_init()
  A_thr_init: Initializer = initializers.zeros_init() # TODO: deal with this later, should be initialized accordingly to a_init, z and betas
  z_init: Initializer = initializers.zeros_init()
  r_init: Initializer = initializers.zeros_init()
  weights_init: Initializer = initializers.kaiming_normal()
  gain: Tuple[float, float] = (0.5,0.1)
  param_dtype: Dtype = jnp.float32 
  beta: float = 1.7  #TODO: implement some test that guarantees that beta has the right shape and right values of 0 on the right positions
  gamma: float = 1
  local_connectivity: bool = True # create locally connection pattern for recurrent neurons
  sigma: float = 0.012 # controls probability of connection in the local connective mode according to distance between neurons
  local_connectivity_key: int = 0
  refractory_period: int = 5
 
  
  @compact
  def __call__(self, carry:Tuple[Array, Array, Array, Array], x:Array) ->Tuple[Array, Tuple[Array, Array]]:
    """
    Compute the next hidden and output states (one time step).

    Parameters
    ----------
    carry : tuple of arrays
        Current state of the cell, containing membrane potential (v), adaptation variable (a),
        adaptive threshold (A_thr), and spike output (z).
    x : array
        Input to the cell at the current time step.

    Returns
    -------
    new_carry : tuple of arrays
        Updated state of the cell.
            v: Array like (n_batch, n_t, n_rec)
            a: Array like (n_batch, n_t, n_rec)
            A_thr: Array like (n_batch, n_t, n_rec)
            z: Array like (n_batch, n_t, n_rec)
    (new_carry, new_z) : tuple
        Tuple containing the updated state and the spike output for the current time step.
        Returning the tuple with new_carry is important, so that the whole history of carries is outputed
        after the scan with RNN module
    """
    
    
    # initialized parameters for eligibility params (those are the params necessary to compute eligibility trace later)
    thr = self.variable('eligibility params', 'thr', lambda: jnp.array(self.thr))
    gamma = self.variable('eligibility params', 'gamma', lambda: jnp.array(self.gamma))
    alpha = self.variable('eligibility params', 'alpha', lambda: jnp.array(jnp.exp(-self.dt/self.tau_m))) 
    rho = self.variable('eligibility params', 'rho', lambda: jnp.array(jnp.exp(-self.dt/self.tau_adaptation)))
    n_rec = self.variable('eligibility params', 'n_rec', lambda: jnp.array(self.n_LIF + self.n_ALIF))
    
    def calculate_betas(beta= self.beta, rho=rho.value, alpha=alpha.value, n_LIF=self.n_LIF, n_ALIF=self.n_ALIF): # since they are constants, should still work fine with jit if I dont pass attributes, but better to guarantee
              return beta * (1 - rho) / (1 - alpha) * jnp.concatenate((jnp.zeros(n_LIF), jnp.ones(n_ALIF)))  # this was taken from paper            
              

    # Initialize betas as a fixed variable
    betas = self.variable('eligibility params', 'betas', calculate_betas) # dim: (n_rec,)
    
    # Initialize connectivity mask. So far, only the 2D embeding provided is spatial_embedings.py can be used for this. In case local_connectivity is False, returns ones matrix (no mask)
    
    M = self.variable('connectivity mask', 'M',
                      inits.initialize_connectivity_mask(self.local_connectivity,
                                                         key=random.key(self.local_connectivity_key),
                                                         n_rec=self.n_LIF + self.n_ALIF,
                                                         sigma = self.sigma, 
                                                         dtype = self.param_dtype)
    )
    
        
    # Initialize weights (in @compact method, the init is done inside of __call__) 
    w_in = self.param(
      'input_weights',
      inits.generalized_initializer(self.weights_init,
                                    gain=self.gain[0],
                                    avoid_self_recurrence=False),
      (jnp.shape(x)[-1], self.n_LIF + self.n_ALIF), # need to use self.n_LIF + self.n_ALIF instead of n_rec.value, because n_rec.value becomes traceable array, what does not work for jit 
      self.param_dtype,
    ) # dim: (n_in, n_rec)
    
    w_rec = self.param(
      'recurrent_weights',      
      inits.generalized_initializer(self.weights_init,
                                    gain=self.gain[1],
                                    avoid_self_recurrence=True, 
                                    local_connectivity=M.value),
      (self.n_LIF + self.n_ALIF, self.n_LIF + self.n_ALIF),
      self.param_dtype,
    ) # dim: (n_rec, n_rec)    
    
    
    
    # Forward pass - Hidden state:  v: recurrent layer membrane potential a: threshold adaptation
    # Visible state: z: recurrent layer spike output, vo: output layer membrane potential (yo incl. activation function)
    
    v, a, A_thr, z, r = carry  # unpack the hidden stated of the ALIF cells: membrane potential v and adaptation variable a. although in the paper z is the observable state, for a better agreement with the structure of recurrent cells in FLAX, it was also defined here as a carry
    
 
        
    no_refractory = (r == 0)
    # For my computer with CPU, this version was running faster, might be different for GPU
    new_v =  (alpha.value * v + jnp.dot(z, w_rec) + jnp.dot(x, w_in) - z * self.thr) #important, z and x should have dimension n_b, n_features and w n_features, output dimension
   
    # Maybe: for efficiency, combine the inputs from input layer and from recurrent layer, to have it a single dot multiplication
    #new_v = self.alpha * v + jnp.dot(jnp.concatenate([inputs,z], axis=-1), jnp.concatenate([w_in,w_rec], axis=-1)) - z * self.thr #important, z and x should have dimension n_b, n_features and w n_features, output dimension
    
    new_a = rho.value * a + z
    
   
    new_A_thr =   self.thr + new_a * betas.value # compute value of adapted threshold at t+1  
    
    
    new_z = no_refractory * spike((new_v-new_A_thr)/self.thr, self.gamma)  

    # update refractory period
    new_r = jnp.where(new_z>0, jnp.ones_like(r) * self.refractory_period, jnp.maximum(0, r-1))
      
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
    v = self.v_init(key1, hidden_shape, self.param_dtype)
    a = self.a_init(key2, hidden_shape, self.param_dtype)
    A_thr = self.A_thr_init(key3, hidden_shape, self.param_dtype)
    z = self.z_init(key4, hidden_shape, self.param_dtype)
    r = self.r_init(key5, hidden_shape, self.param_dtype)
    return (v, a, A_thr, z, r)
  
 
  @property
  def num_feature_axes(self) -> int:
    return 1


class ReadOut(nn.recurrent.RNNCellBase):
  """
  Readout layer for a recurrent neural network.

  This layer reads the output from the recurrent layer and processes it through a leaky integrator to produce the final output. It also supports feedback connections, with options for symmetric or initialized feedback.

  Attributes
  ----------
  n_out : int
      Number of output neurons.
  tau_out : float
      Time constant for the output layer (ms).
  dt : float
      Time step size (ms).
  b_out : Tuple
      Bias for the output neurons. Not implemented yet.
  carry_init : Initializer
      Initializer for the carry state (output layer membrane potential).
  weights_init : Initializer
      Initializer for the readout weights.
  feedback_init : Initializer
      Initializer for the feedback weights.
  gain : Tuple[float, float]
      Gain factors for weight initialization.
  feedback : str
      Type of feedback ('Symmetric' or 'Random').
  param_dtype : Dtype
      Data type for parameters.
  FeedBack_key : int
      Key for feedback weights initialization.

  Methods
  -------
  setup()
      Sets up the feedback weights initializer.
  __call__(carry, z)
      Computes the next output state given the current carry and input from the recurrent layer.
  initialize_carry(rng, input_shape)
      Initializes the carry state for the readout layer.
  num_feature_axes
      Returns the number of feature axes for the readout layer.
  """ 
  
  # TODO: include other parameters  
  n_out: int = 2
  tau_out: float =1
  dt: float = 1
  b_out: Tuple = 0 # For now, the bias is not implemented, think about this after training is sucessful
  carry_init: Initializer = initializers.zeros_init()
  weights_init: Initializer = initializers.kaiming_normal()
  feedback_init: Initializer = initializers.kaiming_normal()
  gain: Tuple = (0.5, 0.5)
  feedback: str = "Symmetric"
  param_dtype: Dtype = jnp.float32
  FeedBack_key: int = 42 
  
#   def setup(self,): # ugly way to make it work as I wanted, probably better solution
#     """
#     Sets up the feedback weights initializer.
#     """
#     feedback_weights_initializer = inits.generalized_initializer(self.feedback_init, gain=self.gain[1])
  @compact
  def __call__(self, carry, z):   
    
    # Initialize weights (in @compact method, the init is done inside of __call__)
    w_out = self.param(
      'readout_weights',
      inits.generalized_initializer(self.weights_init,
                                    gain=self.gain[0],
                                    avoid_self_recurrence=False),
      (jnp.shape(z)[-1], self.n_out),
      self.param_dtype,
    ) # dim: (n_pre, n_post): in the paper architecture this is (n_rec, n_out)
    
     
    # initialized parameters for eligibility params (those are the params necessary to compute eligibility trace later)
    
    kappa = self.variable('eligibility params', 'kappa', lambda: jnp.exp(-self.dt/self.tau_out))
    B_out = self.variable('eligibility params', 'feedback_weights',
                          inits.feedback_weights_initializer(self.feedback_init,
                                                            random.key(self.FeedBack_key), 
                                                            (jnp.shape(z)[-1], self.n_out),
                                                            w_out, self.feedback, gain=self.gain[1]
                                                            )  #n_rec, n_out
    )            
  

    # Forward pass - real valued leaky output neurons (output new y, or equivalently to new mebrane voltage)
    
    
    y = carry  # unpack the hidden state: for readout it is only the output at previous step (basically the membrane potential, but kept as y for consistency with paper nomenclature)
    
    
    new_y = kappa.value * y + jnp.dot(z, w_out) + self.b_out  #TODO: implement bias
              
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
        Shape of the input data, excluding the batch dimension.

    Returns
    -------
    y : array
        Initialized carry state for the readout layer.
    """
    
    batch_dims = input_shape[:-1] # gets batch dimensions (everything except for the last dimension). In this application, input shape is always (n_batch, n_in). So batch_dims = (n_batches,)
    key1 = random.split(rng, 4)
    hidden_shape = batch_dims + (self.n_out,) # sum of tuples mean concatenation: (n_batches) + (n_rec) = (n_batches, n_rec)
    y = self.carry_init(key1, hidden_shape, self.param_dtype)
    return y
  
  
  @property
  def num_feature_axes(self) -> int:
    return 1
  
  
class LSSN(nn.Module):
    """
    Long Short-term Spike Network (LSSN) combining ALIF and LIF cells with a readout layer.

    This module defines a recurrent neural network with adaptive and leaky integrate-and-fire (ALIF and LIF) cells 
    and a readout layer that processes the output from the recurrent layer.

    Attributes
    ----------
    n_ALIF : int
        Number of ALIF neurons.
    n_LIF : int
        Number of LIF neurons.
    n_out : int
        Number of output neurons.
    thr : float
        Threshold for spike generation of recurrent neurons.
    tau_m : float
        Recurren Neurons Membrane time constant.
    tau_adaptation : float
        Adaptation time constant for ALIF neurons.
    beta : None or list
        Adaptation strength for each ALIF neuron.
    gamma : float
        Dampning factor for pseudo derivative.
    tau_out : float
        Time constant for the readout neurons.
    v_init : Initializer
        Initializer for the membrane potential of recurrent neurons.
    a_init : Initializer
        Initializer for the adaptation variable of recurrent neurons.
    A_thr_init : Initializer
        Initializer for the adaptive threshold of recurrent neurons.
    z_init : Initializer
        Initializer for the spike output of recurrent neurons.
    out_carry_init : Initializer
        Initializer for the output layer carry state.
    e_carry_init : Initializer
        Initializer for the eligibility trace carry state.
    b_out : Tuple
        Bias for the output neurons. Not implemented yet.
    ALIF_weights_init : Initializer
        Initializer for the ALIF weights.
    ReadOut_weights_init : Initializer
        Initializer for the readout weights.
    feedback_init : Initializer
        Initializer for the feedback weights.
    gain : Tuple[float, float, float]
        Gain factors for weight initialization.
    dt : float
        Time step size.
    t_crop : int
        Number of time steps to crop from the beginning of the sequence.
    classification : bool
        Whether the task is classification (applies softmax to the output).
    feedback : str
        Type of feedback ('Symmetric' or other).
    param_dtype : Dtype
        Data type for parameters.
    FeedBack_key : int
        Key for feedback weights initialization.
    
    local_connectivity: bool, optional
        If the model will use local_connectivity or not for reccurent layer. Notice that so far,
        only the 2D embeding provided is spatial_embedings.py can be used for this.     
    local_connectivity_key: int, optional
        Seed to generate PRGN key for initialization of local_connectivity mask
    sigma: float, optional
        Parameter controling probability of connection in the local connective mode 
        according to distance between neurons.

    Methods
    -------
    __call__(x)
        Perform the forward pass through the LSSN.
    initialize_eligibility_carry(rng, input_shape)
        Initialize the eligibility trace carry state.
  """
    # architecture
    n_ALIF: int = 3
    n_LIF: int = 3
    n_out: int = 2  

    
    # ALIF params
    thr: float = 0.6
    tau_m: float = 0.8
    tau_adaptation: float = 0.8
    beta: float= 1.7  #TODO: implement some test that guarantees that beta has the right shape and right values of 0 on the right positions
    gamma: float = 1.0
    refractory_period: int = 5
    # Readout params
    tau_out: float = 0.8


    #carries init
    v_init: Initializer = initializers.zeros_init()
    a_init: Initializer = initializers.zeros_init()
    A_thr_init: Initializer = initializers.zeros_init() # TODO: deal with this later, should be initialized accordingly to a_init, z and betas
    z_init: Initializer = initializers.zeros_init()
    r_init: Initializer = initializers.zeros_init()
    out_carry_init: Initializer = initializers.zeros_init()
    e_carry_init: Initializer = initializers.zeros_init()    
    
    # weights and biases
    b_out: Tuple = 0 # For now, the bias is not implemented, think about this after training is sucessful
    ALIF_weights_init: Initializer = initializers.kaiming_normal()
    ReadOut_weights_init: Initializer = initializers.kaiming_normal()
    feedback_init: Initializer = initializers.kaiming_normal()
    gain: Tuple[float, float, float] = (0.5,0.1,0.5, 0.5)
    
    # task attributes
    dt: float = 1  
    
    loss: Callable = losses.softmax_cross_entropy
    
    # Others 
    feedback: str = "Symmetric"
    param_dtype: Dtype = jnp.float32 
    FeedBack_key: int = 42
    local_connectivity: bool = True 
    sigma: float = 0.012  # default values for 100 recurrent neurons, generate network with ~10% connectivity
    local_connectivity_key: int = 0
    
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

        - The `ALIFCell` is initialized with various parameters such as `thr`, `tau_m`, `tau_adaptation`, `n_ALIF`, `n_LIF`, and others, which control its behavior and initialization.
        - The `ReadOut` cell is initialized with parameters like `n_out`, `tau_out`, `weights_init`, `b_out`, and others, which control its behavior and initialization.

        The method performs the following steps:
        1. Processes the input `x` through the `recurrent` RNN layer.
        2. Processes the output of the `recurrent` layer through the `readout` RNN layer.
        3. If the model is set for classification, applies the softmax function to the output.
        4. Returns the recurrent states and the final output.

        Variables:
        - recurrent_carries: Contains the states of the recurrent layer (v, a, A_thr, z).
        - z: The intermediate output after processing through the `recurrent` layer.
        - y: The final output after processing through the `readout` layer, and possibly softmaxed if classification is enabled.
        """
        
        # Define the layers (nn.RNN does the scan over time)
        recurrent = nn.RNN(ALIFCell(thr=self.thr, tau_m=self.tau_m, tau_adaptation=self.tau_adaptation,
                                    n_ALIF=self.n_ALIF, n_LIF=self.n_LIF, weights_init = self.ALIF_weights_init,
                                    v_init = self.v_init, a_init=self.a_init, A_thr_init=self.A_thr_init,
                                    z_init=self.z_init, gain = (self.gain[0], self.gain[1]),
                                    beta = self.beta, gamma = self.gamma,refractory_period = self.refractory_period, dt=self.dt, local_connectivity=self.local_connectivity, sigma = self.sigma,
                                    local_connectivity_key= self.local_connectivity_key, param_dtype=self.param_dtype),
                                    variable_broadcast=("params", 'eligibility params', 'connectivity mask'), name="Recurrent")
        
        readout = nn.RNN(ReadOut(n_out=self.n_out,tau_out=self.tau_out,dt = self.dt,
                                    weights_init= self.ReadOut_weights_init, b_out=self.b_out, 
                                    carry_init= self.out_carry_init, gain = (self.gain[2], self.gain[3]), 
                                    feedback = self.feedback, feedback_init = self.feedback_init,
                                    param_dtype=self.param_dtype, FeedBack_key=self.FeedBack_key),
                                    variable_broadcast=("params",'eligibility params'), name="ReadOut")
        
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

        Description:
        This method initializes the eligibility carries required for training the LSNN.
        The carries are initialized using a specified initialization function and saved in a dictionary for inputs, recurrent, and output layers.

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
        
