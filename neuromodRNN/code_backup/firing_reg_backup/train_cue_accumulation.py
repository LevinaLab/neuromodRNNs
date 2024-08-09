

import logging
import numpy as np

import hydra
import jax

from jax import random, numpy as jnp
import matplotlib.gridspec as gridspec
import optax
from optax import losses
import pickle
from flax.training import train_state, orbax_utils
from flax.training.early_stopping import EarlyStopping
from flax import struct
from flax.linen import softmax
import orbax.checkpoint
from jax.nn import one_hot
import matplotlib.pyplot as plt

import sys
import os
file_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir + "/../..")
from src.e_prop.models import LSSN
from src.e_prop import plots, e_prop_updates_regularization, tasks
import src.e_prop.tasks as tasks
import  src.e_prop.plots as plots
import src.e_prop.utils as utils
#from absl import logging
#import e_prop_updates_regularization
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

Array = jnp.ndarray

TrainState = train_state.TrainState



def model_from_config(cfg)-> LSSN:
  """Builds the LSSN model from a config.
  
    Note: not passing beta and b_out, because their functionality are not fully implemented
    and it will only work correclty with their default values. Also not passing any of the weight or
    carries init functions but they can be modified after initialization, before training starts. 
  """
  # Not passing beta and b_out because are not fully implemented
  model = LSSN(n_ALIF=cfg.net_arch.n_ALIF,
              n_LIF=cfg.net_arch.n_LIF,
              n_out=cfg.net_arch.n_out,
              feedback=cfg.net_arch.feedback,                        
              FeedBack_key=cfg.net_arch.FeedBack_key,
              local_connectivity=cfg.net_arch.local_connectivity,
              sigma = cfg.net_arch.sigma,
              local_connectivity_key= cfg.net_arch.local_connectivity_key,
              thr=cfg.net_params.thr,
              tau_m=cfg.net_params.tau_m,
              tau_adaptation=cfg.net_params.tau_adaptation,                      
              gamma=cfg.net_params.gamma,
              refractory_period= cfg.net_params.refractory_period,
              tau_out=cfg.net_params.tau_out,                                    
              gain=cfg.net_params.w_init_gain,
              dt=cfg.net_params.dt,                        
              )
  return model 

def get_initial_params(rng, model, input_shape):
  """Returns randomly initialized parameters, eligibility parameters and connectivity mask."""
  dummy_x = jnp.ones(input_shape)
  variables = model.init(rng, dummy_x)
  return variables['params'], variables['eligibility params'], variables['connectivity mask']
    

def get_init_eligibility_carries(rng, model, input_shape):
  """Returns randomly initialized carries. In the default mode, they are all initialized as zeros arrays"""
  return model.initialize_eligibility_carry(rng, input_shape)


# Create a custom TrainState to include both params and other variable collections
class TrainStateEProp(TrainState):
  """ Personalized TrainState for e-prop with local connectivity """
  eligibility_params: Dict[str, Array]
  init_eligibility_carries: Dict[str, Array]
  connectivity_mask: Array
  
def create_train_state(rng, learning_rate, model, input_shape):
  """Create initial training state."""
  key1, key2 = random.split(rng)
  params, eligibility_params, connectivity_mask = get_initial_params(key1, model, input_shape)
  init_eligibility_carries = get_init_eligibility_carries(key2, model, input_shape)

  tx = optax.adam(learning_rate=learning_rate)

  state = TrainStateEProp.create(apply_fn=model.apply, params=params, tx=tx, 
                                  eligibility_params=eligibility_params,
                                  init_eligibility_carries=init_eligibility_carries,
                                  connectivity_mask = connectivity_mask
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
     Notes:
     For this task, we assume that the output is the accumulated evidence along the decision time
     (sum along decision time of the output readouts). The logits should already be this sum, so has dimension
     (n_batches, n_out=2)
  """


  # softmax_cross_entropy expects labels to be one-hot encoded
  loss = losses.softmax_cross_entropy(labels=one_hot(labels,2), logits=logits) 
  
  # Compute accuracy:
  # Inference: Although in my opnion somehow contradictory, inference is considered the cummulative
  # evidence during period where learning signal is available. Kept this way, but I'm passing as default
  # LS_avail as 1, so that anyways the decision is taken only looking at value of outputs at last time step.
  # In this case, sum is only getting rid of time dimension, which will have size 1. But code still
  # prepared to handle the scenario described in Bellec 2020
  inference = jnp.argmax(jnp.sum(logits, axis=1), axis=-1) #  jnp.argmax(jnp.sum(logits, axis=1), axis=-1) # sum the the output overtime, generate cummulative evidence. Select the one with higher evidence. (n_batches,)
  
  # TODO: change this, in my loader each batch will have a single label, not through time since it is redudant for hte task
  # using the same dataloader as torch implementation, where labels are either 0 or 1 tensors, with shape (n_batch, n_t), but n_t is redundant, since all entries are equal over this dimension, therefore select only one element   
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
    c_reg: float 
   
) -> Tuple[TrainState, Metrics]:
    
    """Train for a single step."""

    # Since not using grads, don't need to keep usual structure of defining loss_fn with params as argument
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'connectivity mask':state.connectivity_mask}
   
    # the network returns logits
    recurrent_carries, output_logits = state.apply_fn(variables, batch['input'])  
                                   
    
    
    # Compute e-prop Updates
    # the eprop update function expects y to be already the assigned probability
    # for classification task.     
    y = softmax(output_logits) # (n_batches, n_t, n_)
    
    # eligiblity_inputs: y_batch, true_y_batch, v,a, A_thr, z, x (recurrent_carries: v,a, A_thr, z,)  
    eligibility_inputs = (y, one_hot(batch['label'],2),batch['trial_duration'], recurrent_carries, batch['input']) # for gradients, labels must be one-hot encoded
    
    # e-prop grads. Passing LS_avail will guarantee that it is only available during the last LS_avail
    # time steps
    grads = e_prop_updates_regularization.e_prop_grads(eligibility_inputs, state, LS_avail, local_connectivity, f_target, c_reg)
    
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
    c_reg: float 
    ) -> Tuple[TrainState, Metrics]:

    """Train for a single epoch."""
    batch_metrics = []
    for batch_idx, batch in enumerate(train_batches):
        state, metrics = train_step_fn(state, batch, LS_avail, local_connectivity, f_target, c_reg)
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

    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'connectivity mask':state.connectivity_mask}
   
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

    if cfg.task.input_mode == "original":
       n_in = 4 * cfg.net_arch.n_neurons_channel
    elif cfg.task.input_mode == "modified":
       n_in = 3 * cfg.net_arch.n_neurons_channel
    else:
       raise ValueError
    
    # Create model and a state that contains the parameters.
    rng = random.key(cfg.net_params.state_key)
    model = model_from_config(cfg)
    state = create_train_state(rng, cfg.train_params.lr, model, input_shape=(cfg.train_params.train_sub_batch_size, n_in))  

    # For plotting
    loss_training = []
    loss_eval = []
    accuracy_training = []
    accuracy_eval = []
    iterations = []
    # Compile step functions.
    train_step_fn = jax.jit(train_step, static_argnames=["LS_avail", "local_connectivity"])
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
    eval_batch=  list(tasks.cue_accumulation_task(n_batches= cfg.train_params.test_batch_size, 
                                                             batch_size=cfg.train_params.test_sub_batch_size, 
                                                             seed = cfg.task.seed, n_cues=cfg.task.n_cues, min_delay=cfg.task.min_delay,
                                                             max_delay =cfg.task.max_delay, n_population= cfg.net_arch.n_neurons_channel, 
                                                             f_input =cfg.task.f_input, f_background=cfg.task.f_background,
                                                             t_cue = cfg.task.t_cue, t_cue_spacing = cfg.task.t_cue_spacing, 
                                                             p=cfg.task.p, input_mode=cfg.task.input_mode, dt=cfg.task.dt))
    
# Main training loop.
    logger.info('Starting training...')
    for epoch in range(1, cfg.train_params.iterations+1): # change size of loop
        train_batch=  tasks.cue_accumulation_task(n_batches= cfg.train_params.train_batch_size, 
                                                             batch_size=cfg.train_params.test_sub_batch_size, 
                                                             seed = cfg.task.seed, n_cues=cfg.task.n_cues, min_delay=cfg.task.min_delay,
                                                             max_delay =cfg.task.max_delay, n_population= cfg.net_arch.n_neurons_channel, 
                                                             f_input =cfg.task.f_input, f_background=cfg.task.f_background,
                                                             t_cue = cfg.task.t_cue, t_cue_spacing = cfg.task.t_cue_spacing, 
                                                             p=cfg.task.p, input_mode=cfg.task.input_mode, dt=cfg.task.dt)
        
        # Train for one epoch. 
        logger.info("\t Starting Epoch:{} ".format(epoch))     
        state, train_metrics = train_epoch(train_step_fn, state, train_batch, epoch, LS_avail=cfg.task.LS_avail,
                                            local_connectivity=model.local_connectivity, f_target=cfg.train_params.f_target, c_reg=cfg.train_params.c_reg)
        
 
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
                test_batch = tasks.cue_accumulation_task(n_batches= cfg.train_params.test_batch_size, 
                                                             batch_size=cfg.train_params.test_sub_batch_size, 
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
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'connectivity mask':state.connectivity_mask}
    recurrent_carries, output_logits = state.apply_fn(variables, visualization_batch['input']) 
    y = softmax(output_logits)
    v, a, A_thr, z, r = recurrent_carries
    firing_rates = 1000 * utils.compute_firing_rate(z, visualization_batch["trial_duration"])
    logger.info('firing rate eval set  average %.1f max %.1f min %.1f',
                 jnp.mean(firing_rates),
                 jnp.max(firing_rates),
                 jnp.min(firing_rates))



    

    
    # TODO: decide where to save figs, and how to save figures from different experiments
    # TODO: modify titles

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
