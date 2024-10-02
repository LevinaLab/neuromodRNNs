"""Train cue accumulation task"""

import logging
import numpy as np
import hydra
import jax
import optax
import orbax.checkpoint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from optax import losses
from flax.training import train_state, orbax_utils
from flax import struct
from flax.linen import softmax
from jax import random, numpy as jnp

import sys
import os
file_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir + "/../..")
from src.modRNN.models import LSSN
from src.modRNN import learning_rules,learning_utils, tasks, plots 

from typing import (
  Any,
  Callable,
  Dict,
  List,
  Optional,
  Sequence,
  Tuple,
  Iterable  
 )
from flax.typing import (PRNGKey, Dtype)

Array = jnp.ndarray

TrainState = train_state.TrainState


def model_from_config(cfg)-> LSSN:
  """Builds the LSSN model from a config.
  
    Note: not passing beta and b_out, because their functionality are not fully implemented
    and it will only work correclty with their default values. Also not passing any of the weight or
    carries init functions but they can be modified after initialization, before training starts. 
  """

  # generate different seed buy drawing random large ints
  key = random.PRNGKey(cfg.net_params.seed)
  subkey, key = random.split(key)
  feedback_seed,local_connectivity_seed, diff_kernel_seed, cell_loc_seed,input_sparsity_seed, readout_sparsity_seed = random.randint(subkey, (6,),10000, 10000000)
  # Not passing beta and b_out because are not fully implemented
  model = LSSN(n_ALIF=cfg.net_arch.n_ALIF,
              n_LIF=cfg.net_arch.n_LIF,
              n_out=cfg.net_arch.n_out,
              sigma = cfg.net_arch.sigma,
              gridshape=cfg.net_arch.gridshape,
              n_neuromodulators=cfg.net_arch.n_neuromodulators,
              sparse_connectivity=cfg.net_arch.sparse_connectivity,
              local_connectivity=cfg.net_arch.local_connectivity,
              thr=cfg.net_params.thr,
              tau_m=cfg.net_params.tau_m,
              tau_adaptation=cfg.net_params.tau_adaptation, 
              beta = cfg.net_params.beta,
              gamma=cfg.net_params.gamma,
              refractory_period= cfg.net_params.refractory_period,
              k=cfg.net_params.k,
              radius=cfg.net_params.radius,     
              input_sparsity= cfg.net_params.input_sparsity,         
              readout_sparsity= cfg.net_params.readout_sparsity,
              tau_out=cfg.net_params.tau_out,
              feedback=cfg.net_arch.feedback,
              input_sparsity_seed = input_sparsity_seed,
              readout_sparsity_seed = readout_sparsity_seed,                        
              FeedBack_seed=feedback_seed,     
              learning_rule=cfg.train_params.learning_rule,
              local_connectivity_seed= local_connectivity_seed,
              diff_kernel_seed=diff_kernel_seed,
              cell_loc_seed=cell_loc_seed,                                                
              gain=cfg.net_params.w_init_gain,
              dt=cfg.net_params.dt                
              )
  return model 

def get_initial_params(rng:PRNGKey, model:LSSN, input_shape:Tuple[int,...])->Tuple[Dict,]:
  """Returns randomly initialized parameters, eligibility parameters and connectivity mask."""
  dummy_x = jnp.ones(input_shape, dtype=jnp.float64)
  variables = model.init(rng, dummy_x)
  return variables['params'], variables['eligibility params'], variables['spatial params']
    

def get_init_eligibility_carries(rng:PRNGKey, model:LSSN, input_shape:Tuple[int,...])->Dict[str,Dict[str,Array]]:
  """Returns randomly initialized carries. In the default mode, they are all initialized as zeros arrays"""
  return model.initialize_eligibility_carry(rng, input_shape)

def get_init_error_grid(rng:PRNGKey, model:LSSN, input_shape:Tuple[int,...])->Dict[str,Dict[str,Array]]:
   """Return initial error grid initialized as zeros"""
   return model.initialize_grid(rng=rng, input_shape=input_shape)


# Create a custom TrainState to include both params and other variable collections
class TrainStateEProp(TrainState):
  """ Personalized TrainState for e-prop with local connectivity """
  eligibility_params: Dict[str, Array]
  spatial_params: Dict[str, Array]
  init_eligibility_carries: Dict[str, Array]
  init_error_grid: Array
  
  
def create_train_state(rng:PRNGKey, learning_rate:float, model:LSSN, input_shape:Tuple[int,...], batch_size:int, mini_batch_size:int)->train_state.TrainState:
  """Create initial training state."""
  key1, key2, key3 = random.split(rng, 3)
  params, eligibility_params, spatial_params = get_initial_params(key1, model, input_shape)
  init_eligibility_carries = get_init_eligibility_carries(key2, model, input_shape)
  init_error_grid = get_init_error_grid(key3, model, input_shape)

  tx = optax.adam(learning_rate=learning_rate)
  grad_accum_steps = int(batch_size/mini_batch_size)
  if grad_accum_steps > 1:
    tx = optax.MultiSteps(opt=tx, every_k_schedule=grad_accum_steps, use_grad_mean=False)
  state = TrainStateEProp.create(apply_fn=model.apply, params=params, tx=tx, 
                                  eligibility_params=eligibility_params,
                                  spatial_params = spatial_params,
                                  init_eligibility_carries=init_eligibility_carries,                                  
                                  init_error_grid=init_error_grid
                                  )
  return state



def optimization_loss(logits, labels, z, c_reg, f_target, trial_length):    
  """ Loss to be minimized by network, including task loss and any other, e.g. here also firing regularization
      Notes:
        1. logits is assumed to be non normalized logits
        2. labels are assumed to be one-hot encoded
  """
  # notice that optimization_loss is only called inside of learning_rules.compute_grads, and labels are already passed there as one-hot code and y is already softmax transformed
  task_loss = jnp.sum(losses.softmax_cross_entropy(logits=logits, labels=labels) ) # sum over batches and time --> it accumulates gradients, but in additively way (should not normalize batches)
  
  av_f_rate = learning_utils.compute_firing_rate(z=z, trial_length=trial_length)
  f_target = f_target / 1000 # f_target is given in Hz, bu av_f_rate is spikes/ms --> Bellec 2020 used the f_reg also in spikes/ms
  regularization_loss = 0.5 * c_reg * jnp.sum(jnp.mean(jnp.square(av_f_rate - f_target), 0)) # average over batches
  return task_loss + regularization_loss


def compute_cosine_similarity(grad_a, grad_b):
  """Given two grad matrices with shape (n_pre, n_post), compute the cosine similarity in degrees for the flattened grad vectors (n_pre * n_post)"""

  # Flatten the gradients for having it as a vector
  flattened_grad_a = grad_a.reshape(-1)  # (n_pre * n_post,)
  flattened_grad_b = grad_b.reshape(-1)  # (n_pre * n_post,)


  
  cosine_sim = losses.cosine_similarity(flattened_grad_a, flattened_grad_b)

  # Compute the angle in radians for each batch
  angle_in_radians = jnp.arccos(cosine_sim)  # Shape: (n_b,)

  # Compute angle in degrees
  angle_in_degrees = jnp.degrees(angle_in_radians)  # Shape: (n_b,)
  return angle_in_degrees

class Metrics(struct.PyTreeNode):
  """Computed metrics."""

  loss: float
  accuracy: Optional[float] = None
  count: Optional[int] = None

def compute_metrics(*, labels: Array, logits: Array) -> Metrics:
  """Computes the metrics, summed across the batch if a batch is provided.
     Normalization across the batches is done in a different function. 
     Notes:
     1. Important, expect labels to be one-hot encoded and logits unormalized
     2. For this task, we assume that the output is the accumulated evidence along the decision time
     (sum along decision time of the output readouts). The logits should already be this sum, so has dimension
     (n_batches, n_out=2)
  """
  
  # softmax_cross_entropy expects labels to be one-hot encoded 
  loss = losses.softmax_cross_entropy(labels=labels, logits=logits) # (n_b, n_t)
  loss = jnp.mean(loss, axis=-1) # average over time --> the average over batches is done in normalize_batch_metrics
  # Compute accuracy:
  # Inference: Although in my opnion somehow contradictory, inference is considered the cummulative
  # evidence during period where learning signal is available. Kept this way, but I'm passing as default
  # LS_avail as 1, so that anyways the decision is taken only looking at value of outputs at last time step.
  # In this case, sum is only getting rid of time dimension, which will have size 1. But code still
  # prepared to handle the scenario described in Bellec 2020
  inference = jnp.argmax(jnp.sum(logits, axis=1), axis=-1) #  jnp.argmax(jnp.sum(logits, axis=1), axis=-1) # sum the the output overtime, generate cummulative evidence. Select the one with higher evidence. (n_batches,)
  label = jnp.argmax(labels[:,0,:], axis=-1) # labels are same for every time step in the task
  binary_accuracy = jnp.equal(inference, label) 
  
  # metrics are summed over batches, counts are stored to normalize it later. This is important if paralellizing through multiple devices
  return Metrics(
      loss=jnp.sum(loss),
      accuracy= jnp.sum(binary_accuracy),
      count = logits.shape[0] # number of batches basically
       )

def normalize_batch_metrics(batch_metrics: Sequence[Metrics]) -> Metrics:
  """Consolidates and normalizes a list of per-batch metrics dicts."""
  # Here we sum the metrics that were already summed per batch.
  total_loss = np.sum([metrics.loss for metrics in batch_metrics])
  total_accuracy = np.sum([metrics.accuracy for metrics in batch_metrics])
  total = np.sum([metrics.count for metrics in batch_metrics])
  # Divide each metric by the total number of items in the data set.
  return Metrics(
      loss=total_loss.item() / total, accuracy=total_accuracy.item() / total
  )

def test_e_prop_grads( state: TrainState,
    train_batches: Dict[str, Array], 
    optimization_loss_fn: Callable,  
    LS_avail: int,     
    local_connectivity: bool,
    f_target: float,
    c_reg: float,
    task: str,
    shuffle: bool,
    shuffle_key: PRNGKey):

    accumulated_autodiff_grads = jax.tree_map(lambda x: jnp.zeros_like(x), state.params)
    accumulated_hardcoded_grads = jax.tree_map(lambda x: jnp.zeros_like(x), state.params)
    print("dubidu")
    for batch_idx, batch in enumerate(train_batches):
      print("bb")
      _, autodiff_grads = learning_rules.compute_grads(batch=batch, state=state, optimization_loss_fn=optimization_loss_fn, 
                                         LS_avail=LS_avail, local_connectivity=local_connectivity, f_target=f_target, 
                                         c_reg=c_reg, learning_rule="e_prop_autodiff",
                                         task=task, shuffle=shuffle, key=shuffle_key)
      mask = jnp.where(autodiff_grads['ALIFCell_0']['recurrent_weights']!=0.)
      
      accumulated_autodiff_grads = jax.tree_map(lambda a, g: a + g, accumulated_autodiff_grads, autodiff_grads)
    
      _, hardcoded_grads = learning_rules.compute_grads(batch=batch, state=state, optimization_loss_fn=optimization_loss_fn, 
                                        LS_avail=LS_avail, local_connectivity=local_connectivity, f_target=f_target, 
                                        c_reg=c_reg, learning_rule="e_prop_hardcoded",
                                        task=task, shuffle=shuffle, key=shuffle_key)
      accumulated_hardcoded_grads = jax.tree_map(lambda a, g: a + g, accumulated_hardcoded_grads, hardcoded_grads)

    # Autodiff grads
    autodiff_recurrent = accumulated_autodiff_grads['ALIFCell_0']['recurrent_weights']
    autodiff_inputs = accumulated_autodiff_grads['ALIFCell_0']['input_weights'] 
    jax.debug.print("max auto diff grad{}",jnp.max(jnp.abs(autodiff_recurrent)))
    # Hardcoded grads
    hardcoded_recurrent = accumulated_hardcoded_grads['ALIFCell_0']['recurrent_weights']
    hardcoded_inputs = accumulated_hardcoded_grads['ALIFCell_0']['input_weights'] 
    jax.debug.print("max hardcoded diff grad{}",jnp.max(jnp.abs(hardcoded_recurrent)))

    # cosine similarity
    recurrent_cos_sim = compute_cosine_similarity(autodiff_recurrent,hardcoded_recurrent)
    input_cos_sim = compute_cosine_similarity(autodiff_inputs, hardcoded_inputs)

    # max diff
    max_recurrent = jnp.max(jnp.abs(autodiff_recurrent-hardcoded_recurrent))
    max_input = jnp.max(jnp.abs(autodiff_inputs-hardcoded_inputs))

    return recurrent_cos_sim, input_cos_sim, max_recurrent, max_input


def train_step(
    state: TrainState,
    batch: Dict[str, Array], 
    optimization_loss_fn: Callable,  
    LS_avail: int,     
    local_connectivity: bool,
    f_target: float,
    c_reg: float,
    learning_rule: str,
    task: str,
    shuffle: bool,
    shuffle_key: PRNGKey
   
) -> Tuple[TrainState, Metrics]:
    
    """Train for a single step."""                                   
       
    #  Passing LS_avail will guarantee that it is only available during the last LS_avail    
    y, grads = learning_rules.compute_grads(batch=batch, state=state, optimization_loss_fn=optimization_loss_fn, 
                                         LS_avail=LS_avail, local_connectivity=local_connectivity, f_target=f_target, 
                                         c_reg=c_reg, learning_rule=learning_rule,
                                         task=task, shuffle=shuffle, key=shuffle_key)
    
    new_state = state.apply_gradients(grads=grads)

    # For computing loss, we use logits instead of already computed softmax
    metrics = compute_metrics(labels=batch['label'][:,-LS_avail:,:], logits=y[:,-LS_avail:,:])    
    
    return new_state, metrics, grads # return grads only for plotting reasons

def train_epoch(
    train_step_fn: Callable[..., Tuple[TrainState, Metrics]],
    state: TrainState,
    train_batches: Iterable,
    epoch: int,
    optimization_loss_fn: Callable,
    LS_avail: int,
    local_connectivity: bool,
    f_target: float,
    c_reg: float, 
    learning_rule: str,
    task: str,
    shuffle: bool,
    shuffle_key: PRNGKey
    ) -> Tuple[TrainState, Metrics]:

    """Train for a single epoch."""
    batch_metrics = []
    accumulated_grads = jax.tree_map(lambda x: jnp.zeros_like(x), state.params)
    for batch_idx, batch in enumerate(train_batches):
        state, metrics, grads = train_step_fn(state=state, batch=batch, optimization_loss_fn=optimization_loss_fn,
                                        LS_avail=LS_avail, local_connectivity=local_connectivity, f_target=f_target,
                                          c_reg=c_reg, learning_rule=learning_rule, task=task, 
                                          shuffle=shuffle, shuffle_key=shuffle_key)        
        batch_metrics.append(metrics)
        accumulated_grads = jax.tree_map(lambda a, g: a + g, accumulated_grads, grads) # this is only for plotting, the update is accumulated already using Optax.MultiSteps
    # Compute the metrics for this epoch.
    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics)

    logger = logging.getLogger(__name__)
    logger.info(
        'train epoch %03d loss %.4f accuracy %.2f',
        epoch,
        metrics.loss,
        metrics.accuracy * 100,
    )

    return state, metrics, accumulated_grads

def eval_step(
    state: TrainState, batch: Dict[str, Array], LS_avail:int) -> Metrics:   
    """Evaluate for single step"""
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'spatial params':state.spatial_params}
   
    _, logits = state.apply_fn(variables,  batch['input'] )
    metrics = compute_metrics(labels=batch['label'][:,-LS_avail:], logits=logits[:,-LS_avail:,:]) # metrics are computed only for decision making period
    
    return metrics

def evaluate_model(
    eval_step_fn: Callable[..., Any],
    state: TrainState,
    batches: int,
    epoch: int,
    LS_avail: int
) -> Metrics:
    """Evaluate a model on a dataset."""
    
    batch_metrics = []
    for batch_idx, batch in enumerate(batches):
        metrics = eval_step_fn(state, batch, LS_avail)
        batch_metrics.append(metrics)

    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics)

    logger = logging.getLogger(__name__)
    logger.info(
        'eval  epoch %03d loss %.4f accuracy %.2f',
        epoch,
        metrics.loss,
        metrics.accuracy * 100,
    )
    return metrics



def train_and_evaluate(cfg) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
    Returns:
    The final train state that includes the trained parameters.
    """

    if cfg.task.input_mode == "original":
       n_in = 4 * cfg.net_arch.n_neurons_channel
    elif cfg.task.input_mode == "modified":
       n_in = 3 * cfg.net_arch.n_neurons_channel
    else:
       raise ValueError
    

    # Create model and a state that contains the parameters.
    key = random.key(cfg.net_params.seed) # in model to config, consume the splits, not the key itself, so should be differetn
    shuffle_key, state_key, rng = random.split(key, 3)
    
    model = model_from_config(cfg)
    state = create_train_state(state_key, cfg.train_params.lr, model, input_shape=(cfg.train_params.train_mini_batch_size, n_in), batch_size=cfg.train_params.train_batch_size,
                                mini_batch_size=cfg.train_params.train_mini_batch_size )  

    # For plotting
    loss_training = []
    loss_eval = []
    accuracy_training = []
    accuracy_eval = []
    iterations = []
    if cfg.train_params.test_grads:
       recurrent_cos_sim_list = []
       input_cos_sim_list = []
       max_recurrent_list = []
       max_input_list = [] 
    # Compile step functions.
    train_step_fn = jax.jit(train_step, static_argnames=["LS_avail", "local_connectivity", 
                                                         "learning_rule", "task", "shuffle"])
    eval_step_fn = jax.jit(eval_step, static_argnames=["LS_avail"])
    
    # this is a trick to pass a Callable as argument of jitted function
    optimization_loss_fn = optimization_loss
    closure = jax.tree_util.Partial(optimization_loss_fn) 
    
  
    # Prepare Model Check pointer

    # ckpt = {'model': state}
    # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # save_args = orbax_utils.save_args_from_target(ckpt)
    # orbax_checkpointer.save('/tmp/flax_ckpt/orbax/single_save', ckpt, save_args=save_args)
    # options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager('/tmp/flax_ckpt/orbax/managed', orbax_checkpointer, options)
    # Loop Through Curriculum

    # output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
   
    logger = logging.getLogger(__name__)
    
    
    # Prepare datasets.
    # We want the test set to be always the same. So, instead of keeping generate the same data by fixing seed, generate data once and store it as a list of bathes
    eval_batch=  list(tasks.cue_accumulation_task(n_batches= cfg.train_params.test_batch_size, 
                                                             batch_size=cfg.train_params.test_mini_batch_size, 
                                                             seed = cfg.task.seed, n_cues=cfg.task.n_cues, min_delay=cfg.task.min_delay,
                                                             max_delay =cfg.task.max_delay, n_population= cfg.net_arch.n_neurons_channel, 
                                                             f_input =cfg.task.f_input, f_background=cfg.task.f_background,
                                                             t_cue = cfg.task.t_cue, t_cue_spacing = cfg.task.t_cue_spacing, 
                                                             p=cfg.task.p, input_mode=cfg.task.input_mode, dt=cfg.task.dt))
    
    # Main training loop.
    logger.info('Starting training...')
    # generate random seeds for using to generate reproducible training batches    
    train_task_seeds = random.randint(random.PRNGKey(cfg.task.seed), (cfg.train_params.iterations,), 10000, 10000000)
    
    for epoch, train_seed in zip(range(1, cfg.train_params.iterations+1), train_task_seeds): # change size of loop
        sub_shuffle_key, shuffle_key = random.split(shuffle_key) # splits key to get new one every epoch
        train_batch=  tasks.cue_accumulation_task(n_batches= cfg.train_params.train_batch_size, 
                                                             batch_size=cfg.train_params.test_mini_batch_size, 
                                                             seed = train_seed.item(), n_cues=cfg.task.n_cues, min_delay=cfg.task.min_delay,
                                                             max_delay =cfg.task.max_delay, n_population= cfg.net_arch.n_neurons_channel, 
                                                             f_input =cfg.task.f_input, f_background=cfg.task.f_background,
                                                             t_cue = cfg.task.t_cue, t_cue_spacing = cfg.task.t_cue_spacing, 
                                                             p=cfg.task.p, input_mode=cfg.task.input_mode, dt=cfg.task.dt)
        
        # Train for one epoch. 
        logger.info("\t Starting Epoch:{} ".format(epoch))     
        state, train_metrics, accumulated_grads = train_epoch(train_step_fn=train_step_fn, state=state, train_batches=train_batch, epoch=epoch, optimization_loss_fn=closure, LS_avail=cfg.task.LS_avail,
                                            local_connectivity=model.local_connectivity, f_target=cfg.train_params.f_target, c_reg=cfg.train_params.c_reg,
                                            learning_rule=cfg.train_params.learning_rule, task=cfg.task.task_type,
                                            shuffle=cfg.train_params.shuffle, shuffle_key=sub_shuffle_key)
       
        
 
        # Evaluate current model on the validation data.        
        if (epoch - 1) % 25 == 0:      
            eval_metrics = evaluate_model(eval_step_fn, state, eval_batch, epoch, LS_avail=cfg.task.LS_avail)  
            loss_training.append(train_metrics.loss)
            loss_eval.append(eval_metrics.loss)
            accuracy_training.append(train_metrics.accuracy)
            accuracy_eval.append(eval_metrics.accuracy)
            iterations.append(epoch - 1)  

            if cfg.train_params.test_grads:
              train_batch_test_grad=  tasks.cue_accumulation_task(n_batches= cfg.train_params.train_batch_size, 
                                                             batch_size=cfg.train_params.test_mini_batch_size, 
                                                             seed = train_seed.item(), n_cues=cfg.task.n_cues, min_delay=cfg.task.min_delay,
                                                             max_delay =cfg.task.max_delay, n_population= cfg.net_arch.n_neurons_channel, 
                                                             f_input =cfg.task.f_input, f_background=cfg.task.f_background,
                                                             t_cue = cfg.task.t_cue, t_cue_spacing = cfg.task.t_cue_spacing, 
                                                             p=cfg.task.p, input_mode=cfg.task.input_mode, dt=cfg.task.dt)
        

              recurrent_cos_sim, input_cos_sim, max_recurrent, max_input = test_e_prop_grads(state=state, train_batches=train_batch_test_grad, optimization_loss_fn=closure, LS_avail=cfg.task.LS_avail,
                                            local_connectivity=model.local_connectivity, f_target=cfg.train_params.f_target, c_reg=cfg.train_params.c_reg,
                                             task=cfg.task.task_type, shuffle=cfg.train_params.shuffle, shuffle_key=sub_shuffle_key)
            
              recurrent_cos_sim_list.append(recurrent_cos_sim)
              input_cos_sim_list.append(input_cos_sim)
              max_recurrent_list.append(max_recurrent)
              max_input_list.append(max_input)
                    
            # early stop
            if eval_metrics.accuracy > cfg.train_params.stop_criteria:
              accuracy_test = []
              for i in range(3):
                test_batch = tasks.cue_accumulation_task(n_batches= cfg.train_params.test_batch_size, 
                                                             batch_size=cfg.train_params.test_mini_batch_size, 
                                                             seed = cfg.task.seed, n_cues=cfg.task.n_cues, min_delay=cfg.task.min_delay,
                                                             max_delay =cfg.task.max_delay, n_population= cfg.net_arch.n_neurons_channel, 
                                                             f_input =cfg.task.f_input, f_background=cfg.task.f_background,
                                                             t_cue = cfg.task.t_cue, t_cue_spacing = cfg.task.t_cue_spacing, 
                                                             p=cfg.task.p, input_mode=cfg.task.input_mode, dt=cfg.task.dt)
                test_metrics = evaluate_model(eval_step_fn, state, test_batch, epoch, LS_avail=cfg.task.LS_avail)  
                if test_metrics.accuracy < cfg.train_params.stop_criteria:
                   break
                accuracy_test.append(test_metrics.accuracy)
              if len(accuracy_test) == 3:
                logger.info(f'Met early stopping criteria, breaking at epoch {epoch}')
                break



        # Plot hist of grads.        
        if (epoch - 1) % 100 == 0: 
          grad_fig_directory = os.path.join(output_dir, 'grad_figs')
          os.makedirs(grad_fig_directory, exist_ok=True)
          grad_save_path = os.path.join(grad_fig_directory, str(epoch))
          plots.plot_gradients(accumulated_grads, state.spatial_params, epoch, grad_save_path)
                  

    train_info_directory = os.path.join(output_dir, 'train_info')
    os.makedirs(train_info_directory, exist_ok=True)
    # Saving info for later plot
    def save_training_info(run_metrics:List[Array,], file_name):
      save_file= os.path.join(train_info_directory, f'{file_name}.pkl')
      with open(save_file, 'wb') as outfile:
        pickle.dump(run_metrics, outfile, pickle.HIGHEST_PROTOCOL)
       
    run_metrics = [loss_training, loss_eval, accuracy_training, accuracy_eval, iterations]   
    names = ["loss_training", "loss_eval", "accuracy_training", "accuracy_eval", "iterations"]   
    for run_metric, file_name in zip(run_metrics, names):
      save_training_info(run_metric, file_name)

    figures_directory = os.path.join(output_dir, 'figures')
    os.makedirs(figures_directory, exist_ok=True)
    
    layer_names = ["Input layer", "Recurrent layer", "Readout layer"]
    plots.plot_LSNN_weights(state,layer_names=layer_names,
                       save_path=os.path.join(figures_directory, "weights"))
  
    
    plots.plot_weights_spatially_indexed(state, cfg.net_arch.gridshape,os.path.join(figures_directory, "spatially_weights"))
    #
    fig_train, axs_train = plt.subplots(2, 2, figsize=(12, 10))

    # Plot training loss
    axs_train[0, 0].plot(iterations, loss_training, label='Training Loss', color='b')
    axs_train[0, 0].set_title('Training Loss')
    axs_train[0, 0].set_xlabel('Iterations')
    axs_train[0, 0].set_ylabel('Loss')
    axs_train[0, 0].legend()

    # Plot evaluation loss
    axs_train[0, 1].plot(iterations, loss_eval, label='Evaluation Loss', color='r')
    axs_train[0, 1].set_title('Evaluation Loss')
    axs_train[0, 1].set_xlabel('Iterations')
    axs_train[0, 1].set_ylabel('Loss')
    axs_train[0, 1].legend()

    # Plot training accuracy
    axs_train[1, 0].plot(iterations, accuracy_training, label='Training Accuracy', color='g')
    axs_train[1, 0].set_title('Training Accuracy')
    axs_train[1, 0].set_xlabel('Iterations')
    axs_train[1, 0].set_ylabel('Accuracy')
    axs_train[1, 0].legend()

    # Plot evaluation accuracy
    axs_train[1, 1].plot(iterations, accuracy_eval, label='Evaluation Accuracy', color='m')
    axs_train[1, 1].set_title('Evaluation Accuracy')
    axs_train[1, 1].set_xlabel('Iterations')
    axs_train[1, 1].set_ylabel('Accuracy')
    axs_train[1, 1].legend()

    # Adjust layout to prevent overlap
    fig_train.tight_layout()

    # Save the figure
    fig_train.savefig(os.path.join(figures_directory, "training"))
    plt.close(fig_train)


    visualization_batch = eval_batch[0]    
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'spatial params':state.spatial_params}
    recurrent_carries, output_logits = state.apply_fn(variables, visualization_batch['input']) 
    y = softmax(output_logits)
    v, a, A_thr, z, r = recurrent_carries
    firing_rates = 1000 * learning_utils.compute_firing_rate(z, visualization_batch["trial_duration"])
    logger.info('firing rate eval set  average %.1f max %.1f min %.1f',
                 jnp.mean(firing_rates),
                 jnp.max(firing_rates),
                 jnp.min(firing_rates))


  

    # Plot task and model

    # Create a GridSpec with 3 rows and 1 column
    input_example_1 = visualization_batch['input'][0,:,:]
    recurrent_example_1 = z[0,:,:]

    fig1 = plt.figure(figsize=(8, 10))
    gs1 = gridspec.GridSpec(4, 1, height_ratios=[2.5, 2.5, 2.5, 2.5])
    ax1_1 = fig1.add_subplot(gs1[0])
    ax1_2 = fig1.add_subplot(gs1[1])
    ax1_3 = fig1.add_subplot(gs1[2])
    ax1_4 = fig1.add_subplot(gs1[3])
    plots.plot_cue_accumulation_inputs(input_example_1, n_population = cfg.net_arch.n_neurons_channel, input_mode=cfg.task.input_mode, ax =ax1_1)
    plots.plot_recurrent(recurrent_example_1, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax1_2)
    plots.plot_recurrent(recurrent_example_1, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="ALIF", ax =ax1_3)
    plots.plot_softmax_output(y[0,:,0],ax= ax1_4)
    fig1.suptitle("Example 1: " + cfg.save_paths.condition)
    fig1.tight_layout()
    fig1.savefig(os.path.join(figures_directory, "example_1"))      
    plt.close(fig1)

    input_example_2 = visualization_batch['input'][1,:,:]
    recurrent_example_2 = z[1,:,:]

    fig2 = plt.figure(figsize=(8, 10))
    gs2 = gridspec.GridSpec(4, 1, height_ratios=[2.5, 2.5, 2.5, 2.5])
    ax2_1 = fig2.add_subplot(gs2[0])
    ax2_2 = fig2.add_subplot(gs2[1])
    ax2_3 = fig2.add_subplot(gs2[2])
    ax2_4 = fig2.add_subplot(gs2[3])
    plots.plot_cue_accumulation_inputs(input_example_2, n_population = cfg.net_arch.n_neurons_channel, input_mode=cfg.task.input_mode, ax =ax2_1)
    plots.plot_recurrent(recurrent_example_2, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax2_2)
    plots.plot_recurrent(recurrent_example_2, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="ALIF", ax =ax2_3)
    plots.plot_softmax_output(y[1,:,0],ax= ax2_4)
    fig2.suptitle("Example 2: " + cfg.save_paths.condition)
    fig2.tight_layout()
    fig2.savefig(os.path.join(figures_directory, "example_2"))      
    plt.close(fig2)


    input_example_3 = visualization_batch['input'][2,:,:]
    recurrent_example_3 = z[2,:,:]

    fig3 = plt.figure(figsize=(8, 10))
    gs3 = gridspec.GridSpec(4, 1, height_ratios=[2.5, 2.5, 2.5, 2.5])
    ax3_1 = fig3.add_subplot(gs3[0])
    ax3_2 = fig3.add_subplot(gs3[1])
    ax3_3 = fig3.add_subplot(gs3[2])
    ax3_4 = fig3.add_subplot(gs3[3])
    plots.plot_cue_accumulation_inputs(input_example_3, n_population = cfg.net_arch.n_neurons_channel, input_mode=cfg.task.input_mode, ax =ax3_1)
    plots.plot_recurrent(recurrent_example_3, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax3_2)
    plots.plot_recurrent(recurrent_example_3, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="ALIF", ax =ax3_3)
    plots.plot_softmax_output(y[2,:,0],ax= ax3_4)
    fig3.suptitle("Example 3: " + cfg.save_paths.condition)
    fig3.tight_layout()
    fig3.savefig(os.path.join(figures_directory, "example_3"))   
    plt.close(fig3)

    if cfg.train_params.test_grads:
      print("aaaa")
      grads_fig, axs = plt.subplots(1,2, figsize=(10, 6))
      print(recurrent_cos_sim_list)
      axs[0].plot(iterations, recurrent_cos_sim_list, label="Recurrent grads")
      axs[0].plot(iterations, input_cos_sim_list, label="input grads")
      axs[0].set_title('Cosine similarity between hardcoded and autodiff e-prop')
      axs[0].set_ylabel("Degrees")
      axs[0].set_xlabel('Iterations')
      axs[0].legend()


      print(max_recurrent_list)
      axs[1].plot(iterations, max_recurrent_list, label="Recurrent grads")
      axs[1].plot(iterations, max_input_list, label="input grads")
      axs[1].set_yscale("log")
      axs[1].set_title('Max difference between hardcoded and autodiff e-prop')
      axs[1].set_ylabel("Max abosulete difference")
      axs[1].set_xlabel('Iterations')
      axs[1].legend()
      grads_fig.tight_layout()
      grads_fig.savefig(os.path.join(figures_directory, "grads"))
      plt.close(grads_fig)