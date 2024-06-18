import numpy as np
import jax_models
import e_prop_updates
import jax

from jax import random, numpy as jnp
import flax
import optax
from optax import losses
from flax.training import train_state, orbax_utils
from flax import struct
import orbax.checkpoint
from jax.nn import one_hot
import matplotlib.pyplot as plt
import experiment_setup as experiment_setup
#from absl import logging
#import e_prop_updates_regularization
from typing import (
  Any,
  Callable,
  Dict,
  Optional,
  Sequence,
  Tuple,
  Iterable
 
  
 )

from flax.typing import (
  Array
)

Array = jnp.ndarray

TrainState = train_state.TrainState




def model_from_args(args):
    """Builds a text classification model from a config."""
    model = jax_models.LSSN(n_ALIF=args.n_ALIF,
                        n_LIF=args.n_LIF,
                        n_out=args.n_out,
                        thr=args.thr,
                        tau_m=args.tau_m,
                        tau_adaptation=args.tau_adaptation,
                        beta=args.beta,
                        gamma=args.gamma,
                        tau_out=args.tau_out,              
                        b_out=args.bias_out,            
                        gain=args.w_init_gain,
                        dt=args.dt,
                        t_crop=args.t_crop,
                        classification=args.classification,
                        feedback=args.feedback,                        
                        FeedBack_key=args.FeedBack_key,
                        local_connectivity=args.local_connectivity,
                        sigma = args.sigma,
                        local_connectivity_key= args.local_connectivity_key
                        
                    )
    return model      


def get_initial_params(rng, model, input_shape):
    """Returns randomly initialized parameters."""
    dummy_x = jnp.ones(input_shape)
    variables = model.init(rng, dummy_x)
    return variables['params'], variables['eligibility params'], variables['connectivity mask']
    

def get_init_eligibility_carries(rng, model, input_shape):
    """Returns randomly initialized carries. In the default mode, they are all initialized as zeroes arrays"""
    return model.initialize_eligibility_carry(rng, input_shape)


# Create a custom TrainState to include both params and other variable collections
class TrainStateEProp(TrainState):
    """ Personalized TrainState for e-prop with local connectivity """
    eligibility_params: flax.core.FrozenDict
    init_eligibility_carries: Dict[str, Array]
    connectivity_mask: Array

    #param_dtype: Dtype


# 

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
  accuracy: float
  count: Optional[int] = None


 
  
def compute_metrics(*, labels: Array, logits: Array) -> Metrics:
  """Computes the metrics, summed across the batch if a batch is provided.
     Normalization across the batches is done in a different function.    
  """
  # check if necessary, probably not, believe that the function already deals with it
  if labels.ndim == 1:  # Prevent the labels from broadcasting over the logits.
    labels = jnp.expand_dims(labels, axis=1)


  loss = losses.softmax_cross_entropy(labels=one_hot(labels,2), logits=logits) # mean over individual losses (batches loss and maybe maybe time steps loss, if my model outputs it)
  
  # Compute accuracy:
  # Inference: cummulative evidence --> sum the the output overtime (n_batches, n_out). Argmax to select the decision (either 0 or 1, (n_batches))
  inference = jnp.argmax(jnp.sum(logits, axis=1), axis=-1) # sum the the output overtime, generate cummulative evidence. Select the one with higher evidence. (n_batches,)
  # using the same dataloader as torch implementation, where logits are either 0 or 1 tensors, with shape (n_batch, n_t), but n_t is redundant, since all entries are equal over this dimension, therefore select only one element   
  binary_accuracy = jnp.equal(inference, labels[:,0]) 
  
  # metrics are summed over batches, counts are stored to normalize it later. This is important if paralellizing through multiple devices
  return Metrics(
      loss=jnp.sum(loss),
      accuracy= jnp.sum(binary_accuracy),
      count = logits.shape[0]
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

def batch_to_numpy(batch: Dict[str, Array]) -> Dict[str, Array]: # TODO: change the type
  """Converts a batch with Torch tensors to a batch of NumPy arrays."""
  # _numpy() reuses memory, does not make a copy.
  # pylint: disable=protected-access
  return jax.tree_util.tree_map(lambda x: x.numpy(), batch)


def train_step(
    state: TrainState,
    batch: Dict[str, Array], # change this, my batch will be different probably
    t_crop: int,
    local_connectivity: bool   
   
) -> Tuple[TrainState, Metrics]:
    
    """Train for a single step."""

    # Since not using grads, don't need to keep usual structure of defining loss_fn with params as argument
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'connectivity mask':state.connectivity_mask}
   
   
    recurrent_carries, y = state.apply_fn(variables,
                                          batch['input'] # TODO: check how batch will be organized     
                                         )
    
    # Compute e-prop Updates
    
    # eligiblity_inputs: y_batch, true_y_batch, v,a, A_thr, z, x (recurrent_carries: v,a, A_thr, z,)  
    eligibility_inputs = (y, one_hot(batch['label'],2), recurrent_carries, batch['input']) # for gradients, labels must be one-hot encoded
    # e-prop grads
    grads = e_prop_updates.e_prop_grads(eligibility_inputs, state, t_crop, local_connectivity)
    
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(labels=batch['label'][:,-t_crop:], logits=y[:,-t_crop:,:])
    
    
    return new_state, metrics

def train_epoch(
    train_step_fn: Callable[..., Tuple[TrainState, Metrics]],
    state: TrainState,
    train_batches: Iterable,
    epoch: int,
    t_crop:int,
    local_connectivity:bool
) -> Tuple[TrainState, Metrics]:
  
  """Train for a single epoch."""
  batch_metrics = []
  for batch_idx, batch in enumerate(train_batches):
    
    batch = batch_to_numpy(batch)
    # Check if the current batch index is a multiple of 50
    if (batch_idx + 1) % 50 == 0:
      print(f"\t\t Processed {batch_idx + 1} batches")
    state, metrics = train_step_fn(state, batch, t_crop, local_connectivity)
    batch_metrics.append(metrics)

  # Compute the metrics for this epoch.
  batch_metrics = jax.device_get(batch_metrics)
  metrics = normalize_batch_metrics(batch_metrics)

  # logging.info(
  #     'train epoch %03d loss %.4f accuracy %.2f',
  #     epoch,
  #     metrics.loss,
  #     metrics.accuracy * 100,
  # )

  return state, metrics


def eval_step(
    state: TrainState, batch: Dict[str, Array], t_crop:int) -> Metrics:
    """Evaluate for a single step. Model should be in deterministic mode (not relevant)."""
      # Since not using grads, don't need to keep usual structure of defining loss_fn with params as argument
    variables = {'params': state.params, 'eligibility params':state.eligibility_params, 'connectivity mask':state.connectivity_mask}
   
    recurrent_carries, y = state.apply_fn(variables,  batch['input'] )
    metrics = compute_metrics(labels=batch['label'][:,-t_crop:], logits=y[:,-t_crop:,:]) # metrics are computed only for decision making period
    
    return metrics


def evaluate_model(
    eval_step_fn: Callable[..., Any],
    state: TrainState,
    batches: int,
    epoch: int,
    t_crop: int
) -> Metrics:
  """Evaluate a model on a dataset."""
  batch_metrics = []
  for batch_idx, batch in enumerate(batches):
    
    
    batch = batch_to_numpy(batch)
    metrics = eval_step_fn(state, batch, t_crop)
    batch_metrics.append(metrics)

  batch_metrics = jax.device_get(batch_metrics)
  metrics = normalize_batch_metrics(batch_metrics)
  # logging.info(
  #     'eval  epoch %03d loss %.4f accuracy %.2f',
  #     epoch,
  #     metrics.loss,
  #     metrics.accuracy * 100,
  # )
  return metrics

def train_and_evaluate(
  args
) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The final train state that includes the trained parameters.
  """
  
  # Create model and a state that contains the parameters.
  rng = random.key(args.state_key)
  model = model_from_args(args)
  state = create_train_state(rng, args.learning_rate, model, input_shape=(args.batch_size, args.n_in)) # TODO: get input shape from args, get key from args, get learning rate from 
  jax.debug.print("initial rec_weights: {}", state.params['ALIFCell_0']['recurrent_weights'])
  # For plotting
  loss_training = []
  loss_eval = []
  accuracy_training = []
  accuracy_eval = []
  iterations = []
  # Compile step functions.
  train_step_fn = jax.jit(train_step, static_argnames=["t_crop", "local_connectivity"])
  eval_step_fn = jax.jit(eval_step, static_argnames=["t_crop"])
  
  # ckpt = {'model': state}
  # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  # save_args = orbax_utils.save_args_from_target(ckpt)
  # orbax_checkpointer.save('/tmp/flax_ckpt/orbax/single_save', ckpt, save_args=save_args)
  # options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
  # checkpoint_manager = orbax.checkpoint.CheckpointManager('/tmp/flax_ckpt/orbax/managed', orbax_checkpointer, options)
  # Loop Through Curriculum
  for trial,n_cues in enumerate(args.curriculum):    
    # Prepare datasets.
    _, _, eval_loader = experiment_setup.load_dataset_cue_accumulation(args, n_cues=n_cues)

    print("Starting Trial:{}".format(trial+1))
 


    # Main training loop.
    # logging.info('Starting training...')
    for epoch in range(1, args.epochs+1): # change size of loop
      train_loader, _ ,_ = experiment_setup.load_dataset_cue_accumulation(args, n_cues=n_cues)
      # Train for one epoch. 
      print("\t Starting Epoch:{} ".format(epoch))     
      state, train_metrics = train_epoch(train_step_fn, state, train_loader, epoch, t_crop=model.t_crop, local_connectivity=model.local_connectivity)
        
      print("\t Score on training set:{} ".format(train_metrics.accuracy))
      print("\t Loss on training set:{} ".format(train_metrics.loss))
      # Evaluate current model on the validation data.
        
      if (epoch - 1) % 25 == 0:      
        eval_metrics = evaluate_model(eval_step_fn, state, eval_loader, epoch, t_crop=model.t_crop)
        print("\t Score on test set:{} ".format(eval_metrics.accuracy))
        print("\t Loss on test set:{} ".format(eval_metrics.loss))
        loss_training.append(train_metrics.loss)
        loss_eval.append(eval_metrics.loss)
        accuracy_training.append(train_metrics.accuracy)
        accuracy_eval.append(train_metrics.accuracy)
        iterations.append(epoch - 1)      
        
        # checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})
 
 
 
  # Plot
  jax.debug.print("final rec_weights: {}", state.params['ALIFCell_0']['recurrent_weights'])
  fig, axs = plt.subplots(2, 2, figsize=(12, 10))

  # Plot training loss
  axs[0, 0].plot(iterations, loss_training, label='Training Loss', color='b')
  axs[0, 0].set_title('Training Loss')
  axs[0, 0].set_xlabel('Iterations')
  axs[0, 0].set_ylabel('Loss')
  axs[0, 0].legend()

  # Plot evaluation loss
  axs[0, 1].plot(iterations, loss_eval, label='Evaluation Loss', color='r')
  axs[0, 1].set_title('Evaluation Loss')
  axs[0, 1].set_xlabel('Iterations')
  axs[0, 1].set_ylabel('Loss')
  axs[0, 1].legend()

  # Plot training accuracy
  axs[1, 0].plot(iterations, accuracy_training, label='Training Accuracy', color='g')
  axs[1, 0].set_title('Training Accuracy')
  axs[1, 0].set_xlabel('Iterations')
  axs[1, 0].set_ylabel('Accuracy')
  axs[1, 0].legend()

  # Plot evaluation accuracy
  axs[1, 1].plot(iterations, accuracy_eval, label='Evaluation Accuracy', color='m')
  axs[1, 1].set_title('Evaluation Accuracy')
  axs[1, 1].set_xlabel('Iterations')
  axs[1, 1].set_ylabel('Accuracy')
  axs[1, 1].legend()

  # Adjust layout to prevent overlap
  plt.tight_layout()

  # Save the figure
  plt.savefig("training_metrics_7_cues_600delay_updtade64_lr0001_firinreg.png")
  plt.close()
  return state




            
