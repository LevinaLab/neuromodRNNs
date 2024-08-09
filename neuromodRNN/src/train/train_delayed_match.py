"""Train delayed match to sample"""

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
  feedback_seed,local_connectivity_seed, diff_kernel_seed, cell_loc_seed = random.randint(subkey, (4,),10000, 10000000)
  # Not passing beta and b_out because are not fully implemented
  model = LSSN(n_ALIF=cfg.net_arch.n_ALIF,
              n_LIF=cfg.net_arch.n_LIF,
              n_out=cfg.net_arch.n_out,
              feedback=cfg.net_arch.feedback,                        
              FeedBack_seed=feedback_seed,
              local_connectivity=cfg.net_arch.local_connectivity,
              gridshape=cfg.net_arch.gridshape,
              n_neuromodulators=cfg.net_arch.n_neuromodulators,
              sigma = cfg.net_arch.sigma,
              beta = cfg.net_params.beta,
              local_connectivity_seed= local_connectivity_seed,
              diff_kernel_seed=diff_kernel_seed,
              cell_loc_seed=cell_loc_seed,
              thr=cfg.net_params.thr,
              tau_m=cfg.net_params.tau_m,
              tau_adaptation=cfg.net_params.tau_adaptation,                      
              gamma=cfg.net_params.gamma,
              refractory_period= cfg.net_params.refractory_period,
              tau_out=cfg.net_params.tau_out,                                    
              gain=cfg.net_params.w_init_gain,
              dt=cfg.net_params.dt,
              k=cfg.net_params.k,
              radius=cfg.net_params.radius,
                            
              )
  return model 

def get_initial_params(rng:PRNGKey, model:LSSN, input_shape:Tuple[int,...])->Tuple[Dict[str:Array]]:
  """Returns randomly initialized parameters, eligibility parameters and connectivity mask."""
  dummy_x = jnp.ones(input_shape)
  variables = model.init(rng, dummy_x)
  return variables['params'], variables['eligibility params'], variables['spatial params']
    

def get_init_eligibility_carries(rng:PRNGKey, model:LSSN, input_shape:Tuple[int,...])->Dict[Dict[str:Array]]:
  """Returns randomly initialized carries. In the default mode, they are all initialized as zeros arrays"""
  return model.initialize_eligibility_carry(rng, input_shape)

def get_init_error_grid(rng:PRNGKey, model:LSSN, input_shape:Tuple[int,...])->Dict[Dict[str:Array]]:
   """Return initial error grid initialized as zeros"""
   return model.initialize_grid(rng=rng, input_shape=input_shape)


# Create a custom TrainState to include both params and other variable collections
class TrainStateEProp(TrainState):
  """ Personalized TrainState for e-prop with local connectivity """
  eligibility_params: Dict[str, Array]
  spatial_params: Dict[str, Array]
  init_eligibility_carries: Dict[str, Array]
  init_error_grid: Array
  
def create_train_state(rng:PRNGKey, learning_rate:float, model:LSSN, input_shape:Tuple[int,...])->train_state.TrainState:
  """Create initial training state."""
  key1, key2, key3 = random.split(rng, 3)
  params, eligibility_params, spatial_params = get_initial_params(key1, model, input_shape)
  init_eligibility_carries = get_init_eligibility_carries(key2, model, input_shape)
  init_error_grid = get_init_error_grid(key3, model, input_shape)

  tx = optax.adam(learning_rate=learning_rate)

  state = TrainStateEProp.create(apply_fn=model.apply, params=params, tx=tx, 
                                  eligibility_params=eligibility_params,
                                  spatial_params = spatial_params,
                                  init_eligibility_carries=init_eligibility_carries,                                  
                                  init_error_grid=init_error_grid
                                  )
  return state

# TODO: change this accordingly
class Metrics(struct.PyTreeNode):
  """Computed metrics."""

  loss: float
  accuracy: Optional[float] = None
  count: Optional[int] = None

def compute_metrics(*, labels: Array, logits: Array) -> Metrics:
  """Computes the metrics, summed across the batch if a batch is provided.
     Normalization across the batches is done in a different function.
     The logits and labels should already be croped for time where the are actually available 
  """


  # softmax_cross_entropy expects labels to be one-hot encoded
  loss = losses.softmax_cross_entropy(labels=one_hot(labels,2), logits=logits) 
  
  inference = jnp.argmax(jnp.sum(logits, axis=1), axis=-1) #  jnp.argmax(jnp.sum(logits, axis=1), axis=-1) # sum the the output overtime, generate cummulative evidence. Select the one with higher evidence. (n_batches,)
    
  binary_accuracy = jnp.equal(inference, labels[:,0]) 
  
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
  


def train_step(
    state: TrainState,
    batch: Dict[str, Array], # change this, my batch will be different probably
    LS_avail: int,    
    local_connectivity: bool,
    f_target: float,
    c_reg: float,
    learning_rule: str 
   
) -> Tuple[TrainState, Metrics]:
    
    """Train for a single step."""

    # Since not using grads, don't need to keep usual structure of defining loss_fn with params as argument
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'spatial params':state.spatial_params}
   
    # the network returns logits
    recurrent_carries, output_logits = state.apply_fn(variables, batch['input'])                                  
    
    
    # Compute e-prop Updates
    # the eprop update function expects y to be already the assigned probability
    # for classification task.     
    y = softmax(output_logits) # (n_batches, n_t, n_)
    
    # eligiblity_inputs: y_batch, true_y_batch, v,a, A_thr, z, x (recurrent_carries: v,a, A_thr, z,)  
    eligibility_inputs = (y, one_hot(batch['label'],2),batch['trial_duration'], recurrent_carries, batch['input']) # for gradients, labels must be one-hot encoded

    #  Passing LS_avail will guarantee that it is only available during the last LS_avail    
    grads = learning_rules.compute_grads(eligibility_inputs = eligibility_inputs, state=state, LS_avail=LS_avail, local_connectivity=local_connectivity, f_target=f_target, c_reg=c_reg, learning_rule=learning_rule)
    
    new_state = state.apply_gradients(grads=grads)

    # For computing loss, we use logits instead of already computed softmax
    metrics = compute_metrics(labels=batch['label'][:,-LS_avail:], logits=output_logits[:,-LS_avail:,:])    
    
    return new_state, metrics

def train_epoch(
    train_step_fn: Callable[..., Tuple[TrainState, Metrics]],
    state: TrainState,
    train_batches: Iterable,
    epoch: int,
    LS_avail:int,
    local_connectivity:bool,
    f_target: float,
    c_reg: float ,
    learning_rule:str
    ) -> Tuple[TrainState, Metrics]:

    """Train for a single epoch."""
    batch_metrics = []
    for batch_idx, batch in enumerate(train_batches):
        state, metrics = train_step_fn(state=state, batch=batch, LS_avail=LS_avail, local_connectivity=local_connectivity, f_target=f_target, c_reg=c_reg, learning_rule=learning_rule)
        batch_metrics.append(metrics)

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

    return state, metrics

def eval_step(
    state: TrainState, batch: Dict[str, Array], LS_avail:int) -> Metrics:   
    "Evaluate for a single batch"

    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'spatial params':state.spatial_params}
   
    _, y = state.apply_fn(variables,  batch['input'] )
    metrics = compute_metrics(labels=batch['label'][:,-LS_avail:], logits=y[:,-LS_avail:,:]) # metrics are computed only for decision making period
    
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

    n_in = 3 * cfg.net_arch.n_neurons_channel

    
    # Create model and a state that contains the parameters.
    rng = random.key(cfg.net_params.seed) # in model to config, consume the splits, not the key itself, so should be differetn
    
    model = model_from_config(cfg)
    state = create_train_state(rng, cfg.train_params.lr, model, input_shape=(cfg.train_params.train_sub_batch_size, n_in))  

    # For plotting
    loss_training = []
    loss_eval = []
    accuracy_training = []
    accuracy_eval = []
    iterations = []
    # Compile step functions.
    train_step_fn = jax.jit(train_step, static_argnames=["LS_avail", "local_connectivity", "learning_rule"])
    eval_step_fn = jax.jit(eval_step, static_argnames=["LS_avail"])
    
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
    eval_batch=  list(tasks.delayed_match_task(n_batches= cfg.train_params.test_batch_size, 
                                                             batch_size=cfg.train_params.test_sub_batch_size, 
                                                             n_population=cfg.net_arch.n_neurons_channel,
                                                             f_background=cfg.task.f_background, f_input=cfg.task.f_input,
                                                             p = cfg.task.p, fixation_time=cfg.task.fixation_time, 
                                                             cue_time=cfg.task.cue_time,  delay_time=cfg.task.delay_time,
                                                             seed = cfg.task.seed ))
    
# Main training loop.
    logger.info('Starting training...')
    for epoch in range(1, cfg.train_params.iterations+1): # change size of loop
        train_batch=  tasks.delayed_match_task(n_batches= cfg.train_params.train_batch_size, 
                                                             batch_size=cfg.train_params.train_sub_batch_size, 
                                                             n_population=cfg.net_arch.n_neurons_channel,
                                                             f_background=cfg.task.f_background, f_input=cfg.task.f_input,
                                                             p = cfg.task.p, fixation_time=cfg.task.fixation_time, 
                                                             cue_time=cfg.task.cue_time,  delay_time=cfg.task.delay_time,
                                                             seed = cfg.task.seed)
        
        # Train for one epoch. 
        logger.info("\t Starting Epoch:{} ".format(epoch))     
        state, train_metrics = train_epoch(train_step_fn=train_step_fn, state=state, train_batches=train_batch, epoch=epoch, LS_avail=cfg.task.LS_avail,
                                            local_connectivity=model.local_connectivity, f_target=cfg.train_params.f_target, c_reg=cfg.train_params.c_reg,
                                            learning_rule=cfg.train_params.learning_rule)
       
       
        
 
        # Evaluate current model on the validation data.        
        if (epoch - 1) % 25 == 0:      
            eval_metrics = evaluate_model(eval_step_fn, state, eval_batch, epoch, LS_avail=cfg.task.LS_avail)  
            loss_training.append(train_metrics.loss)
            loss_eval.append(eval_metrics.loss)
            accuracy_training.append(train_metrics.accuracy)
            accuracy_eval.append(eval_metrics.accuracy)
            iterations.append(epoch - 1)  

            # early stop
            if eval_metrics.accuracy > cfg.train_params.stop_criteria:
              accuracy_test = []
              for i in range(3):
                test_batch = tasks.delayed_match_task(n_batches= cfg.train_params.test_batch_size, 
                                                             batch_size=cfg.train_params.test_sub_batch_size, 
                                                             n_population=cfg.net_arch.n_neurons_channel,
                                                             background_f=cfg.task.f_background, f_input=cfg.task.f_input,
                                                             p = cfg.task.p, fixation_time=cfg.task.fixation_time, 
                                                             cue_time=cfg.task.cue_time,  delay_time=cfg.task.delay_time,
                                                             seed = cfg.task.seed )
                test_metrics = evaluate_model(eval_step_fn, state, test_batch, epoch, LS_avail=cfg.task.LS_avail)  
                if test_metrics.accuracy < cfg.train_params.stop_criteria:
                   break
                accuracy_test.append(test_metrics.accuracy)
              if len(accuracy_test) == 3:
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
       
    run_metrics = [loss_training, loss_eval, accuracy_training, accuracy_eval, iterations]   
    names = ["loss_training", "loss_eval", "accuracy_training", "accuracy_eval", "iterations"]   
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

    # Plot task and model
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
    
   
    # Create a GridSpec with 3 rows and 1 column
    input_example_1 = visualization_batch['input'][0,:,:]
    recurrent_example_1 = z[0,:,:]

    fig1 = plt.figure(figsize=(8, 10))
    gs1 = gridspec.GridSpec(4, 1, height_ratios=[2.5, 2.5, 2.5, 2.5])
    ax1_1 = fig1.add_subplot(gs1[0])
    ax1_2 = fig1.add_subplot(gs1[1])
    ax1_3 = fig1.add_subplot(gs1[2])
    ax1_4 = fig1.add_subplot(gs1[3])
    plots.plot_delayed_match_inputs(input_example_1, n_population = cfg.net_arch.n_neurons_channel, ax =ax1_1)
    plots.plot_recurrent(recurrent_example_1, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax1_2)
    plots.plot_recurrent(recurrent_example_1, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="ALIF", ax =ax1_3)
    plots.plot_softmax_output(y[0,:,1],ax= ax1_4, label="Probability of 1", title="Softmaxt Output: neuron coding 1")
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
    plots.plot_delayed_match_inputs(input_example_2, n_population = cfg.net_arch.n_neurons_channel, ax =ax2_1)
    plots.plot_recurrent(recurrent_example_2, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax2_2)
    plots.plot_recurrent(recurrent_example_2, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="ALIF", ax =ax2_3)
    plots.plot_softmax_output(y[1,:,1],ax= ax2_4, label="Probability of 1", title="Softmaxt Output: neuron coding 1")
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
    plots.plot_delayed_match_inputs(input_example_3, n_population = cfg.net_arch.n_neurons_channel, ax =ax3_1)
    plots.plot_recurrent(recurrent_example_3, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="LIF", ax =ax3_2)
    plots.plot_recurrent(recurrent_example_3, n_LIF=cfg.net_arch.n_LIF, n_ALIF=cfg.net_arch.n_ALIF, neuron_type="ALIF", ax =ax3_3)
    plots.plot_softmax_output(y[2,:,1],ax= ax3_4, label="Probability of 1", title="Softmaxt Output: neuron coding 1")
    fig3.suptitle("Example 3: " + cfg.save_paths.condition)
    fig3.tight_layout()
    fig3.savefig(os.path.join(figures_directory, "example_3"))   
    plt.close(fig3)
