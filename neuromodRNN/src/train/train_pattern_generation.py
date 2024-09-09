"""Train pattern generation task"""

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
from jax.nn import one_hot
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
from flax.typing import (PRNGKey)
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
              dt=cfg.net_params.dt,         
              )
  return model 

def get_initial_params(rng:PRNGKey, model:LSSN, input_shape:Tuple[int,...])->Tuple[Dict[str,Array]]:
  """Returns randomly initialized parameters, eligibility parameters and connectivity mask."""
  dummy_x = jnp.ones(input_shape)
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
    
  if labels.ndim==2: # calling labels what normally people call targets in regression tasks
      labels = jnp.expand_dims(labels, axis=-1) # this is necessary because target labels might have only (n_batch, n_t) and predictions (n_batch, n_t, n_out=1)

  task_loss = jnp.sum(losses.squared_error(targets=labels, predictions=logits)) # sum over batches and time --> usually, take average, but biologically is unplausible that updates are averaged across batches, so sum
    
  av_f_rate = learning_utils.compute_firing_rate(z=z, trial_length=trial_length)
  f_target = f_target / 1000 # f_target is given in Hz, bu av_f_rate is spikes/ms --> Bellec 2020 used the f_reg also in spikes/ms
  regularization_loss = 0.5 * c_reg * jnp.sum(jnp.square(av_f_rate - f_target))
  return task_loss + regularization_loss
 



class Metrics(struct.PyTreeNode):
  """Computed metrics."""

  loss: float
  normalized_loss: Optional[float] = None
  count: Optional[int] = None

def compute_metrics(*, targets: Array, predictions: Array) -> Metrics:
  """Computes the metrics, summed across the batch if a batch is provided.
     Normalization across the batches is done in a different function.
     The logits and labels should already be croped for time where the are actually available  
  """

  if targets.ndim==2:
    targets = jnp.expand_dims(targets, axis=-1) # this is necessary because targets might have only (n_batch, n_t) and predictions (n_batch, n_t, n_out=1)
  
  loss = losses.squared_error(targets=targets, predictions=predictions) # n_batches, n_t
  

  #normalized loss (targets have zero mean by task construction)
  squared_sum_targets = jnp.sum(jnp.square(targets), axis=1) # n_batches
  normalized_loss = loss / squared_sum_targets[:,None,:]
  
  # get MSE
  loss = loss.mean(axis=1) # n_batches
  
  # metrics are summed over batches, counts are stored to normalize it later. This is important if paralellizing through multiple devices
  return Metrics(
      loss=jnp.sum(loss),
      normalized_loss= jnp.sum(normalized_loss),
      count = targets.shape[0] # number of batches basically
       )

def normalize_batch_metrics(batch_metrics: Sequence[Metrics]) -> Metrics:
  """Consolidates and normalizes a list of per-batch metrics dicts."""
  # Here we sum the metrics that were already summed per batch.
  total_loss = np.sum([metrics.loss for metrics in batch_metrics])
  total_normalized_loss= np.sum([metrics.normalized_loss for metrics in batch_metrics])
  total = np.sum([metrics.count for metrics in batch_metrics])
  # Divide each metric by the total number of items in the data set.
  return Metrics(
      loss=total_loss.item() / total, normalized_loss=total_normalized_loss.item() / total
  )
  


def train_step(
    state: TrainState,
    batch: Dict[str, Array], # change this, my batch will be different probably
    optimization_loss_fn: Callable,
    LS_avail: int,    
    local_connectivity: bool,
    f_target: float,
    c_reg: float,
    learning_rule: str,
    task: str
   
) -> Tuple[TrainState, Metrics]:
    
    """Train for a single step."""
                            
       
    #  Passing LS_avail will guarantee that it is only available during the last LS_avail    
    y, grads = learning_rules.compute_grads(batch=batch, state=state, optimization_loss_fn=optimization_loss_fn, 
                                         LS_avail=LS_avail, local_connectivity=local_connectivity, f_target=f_target, 
                                         c_reg=c_reg, learning_rule=learning_rule,
                                         task=task)
    
    new_state = state.apply_gradients(grads=grads)

    # For computing loss, we use logits instead of already computed softmax
    metrics = compute_metrics(targets=batch['label'][:,-LS_avail:], predictions=y[:,-LS_avail:,:])    
    return new_state, metrics

def train_epoch(
    train_step_fn: Callable[..., Tuple[TrainState, Metrics]],
    state: TrainState,
    train_batches: Iterable,
    epoch: int,
    optimization_loss_fn: Callable,
    LS_avail:int,
    local_connectivity:bool,
    f_target: float,
    c_reg: float,
    learning_rule:str,
    task:str
    ) -> Tuple[TrainState, Metrics]:

    """Train for a single epoch."""
    batch_metrics = []
    accumulated_grads = jax.tree_map(lambda x: jnp.zeros_like(x), state.params)
    for batch_idx, batch in enumerate(train_batches):
        state, metrics = train_step_fn(state=state, batch=batch, optimization_loss_fn=optimization_loss_fn,
                                       LS_avail=LS_avail, local_connectivity=local_connectivity,
                                       f_target=f_target, c_reg=c_reg, learning_rule=learning_rule, task=task)
        batch_metrics.append(metrics)
        accumulated_grads = jax.tree_map(lambda a, g: a + g, accumulated_grads, grads) # this is only for plotting, the update is accumulated already using Optax.MultiSteps
    # Compute the metrics for this epoch.
    batch_metrics = jax.device_get(batch_metrics)
    metrics = normalize_batch_metrics(batch_metrics)

    logger = logging.getLogger(__name__)
    logger.info(
        'train epoch %03d MSE %.4f NMSE %.3f',
        epoch,
        metrics.loss,
        metrics.normalized_loss,
    )

    return state, metrics

def eval_step(
    state: TrainState, batch: Dict[str, Array], LS_avail:int) -> Metrics:   
    """Evaluate for a single step"""
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'spatial params':state.spatial_params}
   
    _, y = state.apply_fn(variables,  batch['input'] )
    metrics = compute_metrics(targets=batch['label'][:,-LS_avail:], predictions=y[:,-LS_avail:, :]) # metrics are computed only for decision making period
    
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
        'eval  epoch %03d MSE %.4f nMSE %.2f',
        epoch,
        metrics.loss,
        metrics.normalized_loss
    )
    return metrics



def train_and_evaluate(
  cfg
) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
    Returns:
    The final train state that includes the trained parameters.
    """

 
    n_in = cfg.net_arch.n_neurons_channel

    
    # Create model and a state that contains the parameters.
    rng = random.key(cfg.net_params.seed) # in model to config, consume the splits, not the key itself, so should be differetn
    
    model = model_from_config(cfg)
    state = create_train_state(rng, cfg.train_params.lr, model, input_shape=(cfg.train_params.train_mini_batch_size, n_in), batch_size=cfg.train_params.train_batch_size, mini_batch_size=cfg.train_params.train_mini_batch_size)  
    
    # For plotting
    loss_training = []
    loss_eval = []
    normalized_loss_training = []
    normalized_loss_eval = []
    iterations = []
    # Compile step functions.
    train_step_fn = jax.jit(train_step, static_argnames=["LS_avail", "local_connectivity", "learning_rule", "task"])
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
  
    logger = logging.getLogger(__name__)
    
    
# Prepare datasets.

    # We want the test set to be always the same. So, instead of keeping generate the same data by fixing seed, generate data once and store it as a list of bathes
    eval_batch=  list(tasks.pattern_generation(n_batches= cfg.train_params.test_batch_size, 
                                                             batch_size=cfg.train_params.test_mini_batch_size, 
                                                             seed = cfg.task.seed, frequencies=cfg.task.frequencies,
                                                             n_population= cfg.net_arch.n_neurons_channel,
                                                             weights = cfg.task.weights,
                                                             f_input =cfg.task.f_input, trial_dur=cfg.task.trial_dur
                                                             ))
    
# Main training loop.
    logger.info('Starting training...')
    for epoch in range(1, cfg.train_params.iterations+1): # change size of loop
        train_batch=  tasks.pattern_generation(n_batches= cfg.train_params.train_batch_size, 
                                                             batch_size=cfg.train_params.test_mini_batch_size, 
                                                             seed = cfg.task.seed, frequencies=cfg.task.frequencies,
                                                             n_population= cfg.net_arch.n_neurons_channel,
                                                             weights = cfg.task.weights,
                                                             f_input =cfg.task.f_input, trial_dur=cfg.task.trial_dur)
        
        # Train for one epoch. 
        logger.info("\t Starting Epoch:{} ".format(epoch))     
        state, train_metrics = train_epoch(train_step_fn=train_step_fn, state=state, train_batches=train_batch, epoch=epoch, optimization_loss_fn=closure, LS_avail=cfg.task.LS_avail,
                                            local_connectivity=model.local_connectivity, f_target=cfg.train_params.f_target, c_reg=cfg.train_params.c_reg,
                                            learning_rule=cfg.train_params.learning_rule, task=cfg.task.task_type)
        
       
        
 
        # Evaluate current model on the validation data.        
        if (epoch - 1) % 25 == 0: 
               
            eval_metrics = evaluate_model(eval_step_fn, state, eval_batch, epoch, LS_avail=cfg.task.LS_avail)  
            loss_training.append(train_metrics.loss)
            loss_eval.append(eval_metrics.loss)
            normalized_loss_training.append(train_metrics.normalized_loss)
            normalized_loss_eval.append(eval_metrics.normalized_loss)            
            iterations.append(epoch - 1)  

            # early stop
            if eval_metrics.normalized_loss < cfg.train_params.stop_criteria:
              normalized_loss = []
              for i in range(3):
                test_batch = tasks.pattern_generation(n_batches= cfg.train_params.test_batch_size, 
                                                             batch_size=cfg.train_params.test_mini_batch_size, 
                                                             seed = cfg.task.seed, frequencies=cfg.task.frequencies,
                                                             n_population= cfg.net_arch.n_neurons_channel,
                                                             weights = cfg.task.weights,
                                                             f_input =cfg.task.f_input, trial_dur=cfg.task.trial_dur)
                
                test_metrics = evaluate_model(eval_step_fn, state, test_batch, epoch, LS_avail=cfg.task.LS_avail)  
                if test_metrics.normalized_loss > cfg.train_params.stop_criteria:
                   break
                normalized_loss.append(test_metrics.normalized_loss)
              if len(normalized_loss) == 3:
                logger.info(f'Met early stopping criteria, breaking at epoch {epoch}')
                break



    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    train_info_directory = os.path.join(output_dir, 'train_info')
    os.makedirs(train_info_directory, exist_ok=True)
    # Saving info for later plot
    def save_training_info(run_metrics:List[Array,], file_name):
      save_file= os.path.join(train_info_directory, f'{file_name}.pkl')
      with open(save_file, 'wb') as outfile:
        pickle.dump(run_metrics, outfile, pickle.HIGHEST_PROTOCOL)
       
    run_metrics = [loss_training, loss_eval, normalized_loss_training, normalized_loss_eval, iterations]   
    names = ["MSE_training", "MSE_eval", "nMSE_training", "nMSE_eval", "iterations"]   
    for run_metric, file_name in zip(run_metrics, names):
      save_training_info(run_metric, file_name)

    figures_directory = os.path.join(output_dir, 'figures')
    os.makedirs(figures_directory, exist_ok=True)
    
    layer_names = ["Input layer", "Recurrent layer", "Readout layer"]
    plots.plot_LSNN_weights(state,layer_names=layer_names,
                       save_path=os.path.join(figures_directory, "weights"))
  
    
    
    #
    fig_train, axs_train = plt.subplots(2, 2, figsize=(12, 10))

    # Plot training loss
    axs_train[0, 0].plot(iterations, loss_training, label='Training MSE', color='b')
    axs_train[0, 0].set_title('Training MSE')
    axs_train[0, 0].set_xlabel('Iterations')
    axs_train[0, 0].set_ylabel('MSE')
    axs_train[0, 0].set_ylim(0,1)
    axs_train[0, 0].legend()

    # Plot evaluation loss
    axs_train[0, 1].plot(iterations, loss_eval, label='Evaluation MSE', color='r')
    axs_train[0, 1].set_title('Evaluation MSE')
    axs_train[0, 1].set_xlabel('Iterations')
    axs_train[0, 1].set_ylabel('MSE')
    axs_train[0, 1].set_ylim(0,1)
    axs_train[0, 1].legend()

    # Plot training accuracy
    axs_train[1, 0].plot(iterations, normalized_loss_training, label='Training nMSE', color='g')
    axs_train[1, 0].set_title('Training nMSE')
    axs_train[1, 0].set_xlabel('Iterations')
    axs_train[1, 0].set_ylabel('nMSE')
    axs_train[1, 0].set_ylim(0,1)
    axs_train[1, 0].legend()

    # Plot evaluation accuracy
    axs_train[1, 1].plot(iterations, normalized_loss_eval, label='Evaluation nMSE', color='m')
    axs_train[1, 1].set_title('Evaluation nMSE')
    axs_train[1, 1].set_xlabel('Iterations')
    axs_train[1, 1].set_ylabel('nMSE')
    axs_train[1, 1].set_ylim(0,1)
    axs_train[1, 1].legend()

    # Adjust layout to prevent overlap
    fig_train.tight_layout()

    # Save the figure
    fig_train.savefig(os.path.join(figures_directory, "training"))
    plt.close(fig_train)

    # plot task and model
    visualization_batch = eval_batch[0]    
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'spatial params':state.spatial_params}
    recurrent_carries, y = state.apply_fn(variables, visualization_batch['input']) 
    
    v, a, A_thr, z, r = recurrent_carries
    firing_rates = 1000 * learning_utils.compute_firing_rate(z, visualization_batch["trial_duration"])
    logger.info('firing rate eval set  average %.1f max %.1f min %.1f',
                 jnp.mean(firing_rates),
                 jnp.max(firing_rates),
                 jnp.min(firing_rates))



    # Create a GridSpec with 3 rows and 1 column
    input_example_1 = visualization_batch['input'][0,:,:]
    recurrent_example_1 = z[0,:,:]

    fig1 = plt.figure(figsize=(8, 9))
    gs1 = gridspec.GridSpec(3, 1, height_ratios=[2.5, 2.5, 4])
    ax1_1 = fig1.add_subplot(gs1[0])
    ax1_2 = fig1.add_subplot(gs1[1])
    ax1_3 = fig1.add_subplot(gs1[2])
    
   
    plots.plot_pattern_generation_inputs(input_example_1, ax=ax1_1)
    plots.plot_recurrent(recurrent_example_1, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax1_2)
    plots.plot_pattern_generation_prediction(y[0,:,:], targets= visualization_batch["label"][0,:],  ax=ax1_3)

    
    fig1.suptitle("Example 1: " + cfg.save_paths.condition)
    fig1.tight_layout()
    fig1.savefig(os.path.join(figures_directory, "example_1"))      
    plt.close(fig1)

    input_example_2 = visualization_batch['input'][1,:,:]
    recurrent_example_2 = z[1,:,:]

    fig2 = plt.figure(figsize=(8, 9))
    gs2 = gridspec.GridSpec(3, 1, height_ratios=[2.5, 2.5, 4])
    ax2_1 = fig2.add_subplot(gs2[0])
    ax2_2 = fig2.add_subplot(gs2[1])
    ax2_3 = fig2.add_subplot(gs2[2])
   
    plots.plot_pattern_generation_inputs(input_example_2, ax=ax2_1)
    plots.plot_recurrent(recurrent_example_2, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax2_2)
    plots.plot_pattern_generation_prediction(y[1,:,:], targets= visualization_batch["label"][1,:],  ax=ax2_3)

  
    fig2.suptitle("Example 2: " + cfg.save_paths.condition)
    fig2.tight_layout()
    fig2.savefig(os.path.join(figures_directory, "example_2"))      
    plt.close(fig2)


    input_example_3 = visualization_batch['input'][2,:,:]
    recurrent_example_3 = z[2,:,:]

    fig3 = plt.figure(figsize=(8, 9))
    gs3 = gridspec.GridSpec(3, 1, height_ratios=[2.5, 2.5, 4])
    ax3_1 = fig3.add_subplot(gs3[0])
    ax3_2 = fig3.add_subplot(gs3[1])
    ax3_3 = fig3.add_subplot(gs3[2])
  

    plots.plot_pattern_generation_inputs(input_example_3, ax=ax3_1)
    plots.plot_recurrent(recurrent_example_3, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax3_2)
    plots.plot_pattern_generation_prediction(y[2,:,:], targets= visualization_batch["label"][2,:],  ax=ax3_3)

    fig3.suptitle("Example 3: " + cfg.save_paths.condition)
    fig3.tight_layout()
    fig3.savefig(os.path.join(figures_directory, "example_3"))   
    plt.close(fig3)

def test_func(cfg):
   print(cfg.task.task_name)