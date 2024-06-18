from jax import numpy as jnp
from jax import random
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API
from flax.linen import initializers
import spatial_embedings
# Just for typing the type of inputs of functions
from flax.linen.module import compact, nowrap

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
        key_x_pre,key_y_pre,key_x_post,key_y_post, M_key = random.split((key), 5)
        
        if local_connectivity:
            return spatial_embedings.twodMatrix(random.uniform(key_x_pre, shape=(n_rec,)), 
                                                random.uniform(key_y_pre, shape=(n_rec,)),
                                                random.uniform(key_x_post, shape=(n_rec,)),
                                                random.uniform(key_y_post, shape=(n_rec,)), 
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
  weights_init: Initializer = initializers.kaiming_normal()
  gain: Tuple[float, float] = (0.5,0.1)
  param_dtype: Dtype = jnp.float32 
  beta: Union[None, list] = None  #TODO: implement some test that guarantees that beta has the right shape and right values of 0 on the right positions
  gamma: float = 1
  local_connectivity: bool = True # create locally connection pattern for recurrent neurons
  sigma: float = 0.012 # controls probability of connection in the local connective mode according to distance between neurons
  local_connectivity_key: int = 0
  
 
  
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
    
    def calculate_betas(rho=rho.value, alpha=alpha.value, n_LIF=self.n_LIF, n_ALIF=self.n_ALIF): # since they are constants, should still work fine with jit if I dont pass attributes, but better to guarantee
      if self.beta is None:
        return 1.7 * (1 - rho) / (1 - alpha) * jnp.concatenate((jnp.zeros(n_LIF), jnp.ones(n_ALIF)))  # this was taken from paper            
              
      else:
          return self.beta
  
    # Initialize betas as a fixed variable
    betas = self.variable('eligibility params', 'betas', calculate_betas) # dim: (n_rec,)
    
    # Initialize connectivity mask. So far, only the 2D embeding provided is spatial_embedings.py can be used for this. In case local_connectivity is False, returns ones matrix (no mask)
    
    M = self.variable('connectivity mask', 'M',initialize_connectivity_mask(self.local_connectivity, key=random.key(self.local_connectivity_key), n_rec=self.n_LIF + self.n_ALIF, sigma = self.sigma, dtype = self.param_dtype))
    
        
    # Initialize weights (in @compact method, the init is done inside of __call__) 
    w_in = self.param(
      'input_weights',
      generalized_initializer(self.weights_init, gain=self.gain[0], avoid_self_recurrence=False),
      (jnp.shape(x)[-1], self.n_LIF + self.n_ALIF), # need to use self.n_LIF + self.n_ALIF instead of n_rec.value, because n_rec.value becomes traceable array, what does not work for jit 
      self.param_dtype,
    ) # dim: (n_in, n_rec)
    
    w_rec = self.param(
      'recurrent_weights',
      
      generalized_initializer(self.weights_init, gain=self.gain[1], avoid_self_recurrence=True, local_connectivity=M.value),
      (self.n_LIF + self.n_ALIF, self.n_LIF + self.n_ALIF),
      self.param_dtype,
    ) # dim: (n_rec, n_rec)    
    
    
    
    # Forward pass - Hidden state:  v: recurrent layer membrane potential a: threshold adaptation
    # Visible state: z: recurrent layer spike output, vo: output layer membrane potential (yo incl. activation function)
    
    v, a, A_thr, z = carry  # unpack the hidden stated of the ALIF cells: membrane potential v and adaptation variable a. although in the paper z is the observable state, for a better agreement with the structure of recurrent cells in FLAX, it was also defined here as a carry
    
    # For my computer with CPU, this version was running faster, might be different for GPU
    new_v = alpha.value * v + jnp.dot(z, w_rec) + jnp.dot(x, w_in) - z * self.thr #important, z and x should have dimension n_b, n_features and w n_features, output dimension
   
    # Maybe: for efficiency, combine the inputs from input layer and from recurrent layer, to have it a single dot multiplication
    #new_v = self.alpha * v + jnp.dot(jnp.concatenate([inputs,z], axis=-1), jnp.concatenate([w_in,w_rec], axis=-1)) - z * self.thr #important, z and x should have dimension n_b, n_features and w n_features, output dimension
    
    new_a = rho.value * a + z
    
   
    new_A_thr =   self.thr + new_a * betas.value # compute value of adapted threshold at t+1        
        
    new_z = jnp.float32(jnp.where(new_v > new_A_thr, jnp.array(1), jnp.array(0)))   
    new_carry =  (new_v, new_a, new_A_thr, new_z)   
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
    
    key1, key2, key3, key4 = random.split(rng, 4)
    hidden_shape = batch_dims + (self.n_ALIF + self.n_LIF,) # sum of tuples mean concatenation: (n_batches) + (n_rec) = (n_batches, n_rec)
    v = self.v_init(key1, hidden_shape, self.param_dtype)
    a = self.a_init(key2, hidden_shape, self.param_dtype)
    A_thr = self.A_thr_init(key3, hidden_shape, self.param_dtype)
    z = self.z_init(key4, hidden_shape, self.param_dtype)

    return (v, a, A_thr, z)
  
  
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
  
  def setup(self,): # ugly way to make it work as I wanted, probably better solution
    """
    Sets up the feedback weights initializer.
    """
    feedback_weights_initializer = generalized_initializer(self.feedback_init, gain=self.gain[1])
  @compact
  def __call__(self, carry, z):   
    
    # Initialize weights (in @compact method, the init is done inside of __call__)
    w_out = self.param(
      'readout_weights',
      generalized_initializer(self.weights_init, gain=self.gain[0], avoid_self_recurrence=False),
      (jnp.shape(z)[-1], self.n_out),
      self.param_dtype,
    ) # dim: (n_pre, n_post): in the paper architecture this is (n_rec, n_out)
    
     
    # initialized parameters for eligibility params (those are the params necessary to compute eligibility trace later)
    
    kappa = self.variable('eligibility params', 'kappa', lambda: jnp.exp(-self.dt/self.tau_out))
    B_out = self.variable('eligibility params', 'feedback_weights',
                          feedback_weights_initializer(self.feedback_init,random.key(self.FeedBack_key) , 
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
    beta: None | list = None  #TODO: implement some test that guarantees that beta has the right shape and right values of 0 on the right positions
    gamma: float = 1.0
    
    # Readout params
    tau_out: float = 0.8


    #carries init
    v_init: Initializer = initializers.zeros_init()
    a_init: Initializer = initializers.zeros_init()
    A_thr_init: Initializer = initializers.zeros_init() # TODO: deal with this later, should be initialized accordingly to a_init, z and betas
    z_init: Initializer = initializers.zeros_init()
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
    t_crop: int = 150
    classification: bool = True
    
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
                                    beta = self.beta, gamma = self.gamma, dt=self.dt, local_connectivity=self.local_connectivity, sigma = self.sigma,
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
        
        if self.classification:
            y = nn.softmax(y)

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
        