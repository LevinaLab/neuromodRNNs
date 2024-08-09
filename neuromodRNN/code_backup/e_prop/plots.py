import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import jax.numpy as jnp

import os
#TODOÇ raster plot for ALIF and LIF, the y ticks are considering at least 10 neurons

def plot_cue_accumulation_inputs(neuron_spikes, n_population = 10, input_mode="original", ax =None):
    """plot spike rasters for a single neuron sorted by condition

 
    """
    if ax is None:
        fig, ax = plt.subplots()

    offset_constant = 0.6
    
    ax.set_title("Inputs rasterplot")
    if input_mode == "original":
        input_channels = 4
        y_ticks = ["noise", "cue", "left", "right"]
        left_anchor = 2 * offset_constant + 0.3
        right_anchor = 3 * offset_constant + 0.3
    elif input_mode == "modified":
        input_channels = 3
        y_ticks = ["cue", "left", "right"]
        left_anchor = 1 * offset_constant + 0.3
        right_anchor = 2 * offset_constant + 0.3
    # TODO: handle this correctly
    else:
        raise ValueError
    
    neuron_spikes = neuron_spikes.T
    for channel in range(input_channels):
        channel_spikes = neuron_spikes[channel *n_population:channel *n_population+n_population]
        offset = offset_constant * channel # this is only to get how much should offset in y axis
        for i, neuron in enumerate(channel_spikes):
            spike_times = np.where(neuron == 1)[0]
            ax.vlines(spike_times, 0.3 + i*0.03 + offset, 0.3+ i*0.03 + offset+ 0.015, color='black')
        
    ax.set_ylabel('Input Channel')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Input Example')
    ax.tick_params(which="both", left=False, pad=5)
    
    ax.set_yticks([0.3 + 0.03 * n_population/2 + offset_constant*j for j in range(input_channels)], y_ticks)

    ax.set_xlim(0, neuron_spikes.shape[1]+10)
    
    ax.add_patch(patches.Rectangle((0, left_anchor),
                                   neuron_spikes.shape[1], n_population * 0.03,
                                   facecolor="red", alpha=0.2))
    
        
    ax.add_patch(patches.Rectangle((0, right_anchor),
                                   neuron_spikes.shape[1], n_population * 0.03,
                                   facecolor="blue", alpha=0.2))

def plot_delayed_match_inputs(neuron_spikes, n_population = 10, ax =None):
    """plot spike rasters for a single neuron sorted by condition

 
    """
    if ax is None:
        fig, ax = plt.subplots()

    offset_constant = 0.6
    
    ax.set_title("Inputs rasterplot")
 
    input_channels = 3
    y_ticks = ["noise", "1st cue", "2nd cue"]
    first_anchor =  offset_constant + 0.3
    second_anchor = 2 * offset_constant + 0.3


    
    neuron_spikes = neuron_spikes.T
    for channel in range(input_channels):
        channel_spikes = neuron_spikes[channel *n_population:channel *n_population+n_population]
        offset = offset_constant * channel # this is only to get how much should offset in y axis
        for i, neuron in enumerate(channel_spikes):
            spike_times = np.where(neuron == 1)[0]
            ax.vlines(spike_times, 0.3 + i*0.03 + offset, 0.3+ i*0.03 + offset+ 0.015, color='black')
        
    ax.set_ylabel('Input Channel')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Input Example')
    ax.tick_params(which="both", left=False, pad=5)
    
    ax.set_yticks([0.3 + 0.03 * n_population/2 + offset_constant*j for j in range(input_channels)], y_ticks)

    ax.set_xlim(0, neuron_spikes.shape[1]+10)
    
    ax.add_patch(patches.Rectangle((0, first_anchor),
                                   neuron_spikes.shape[1], n_population * 0.03,
                                   facecolor="red", alpha=0.2))
    
        
    ax.add_patch(patches.Rectangle((0, second_anchor),
                                   neuron_spikes.shape[1], n_population * 0.03,
                                   facecolor="blue", alpha=0.2))


def plot_recurrent(z, n_LIF, n_ALIF, neuron_type="LIF", ax =None):
    """plot spike rasters for a single neuron sorted by condition

 
    """

    

    if ax is None:
        fig, ax = plt.subplots()
    z = np.array(z)

    offset_constant = 0.6
    if neuron_type == "LIF":
        z = z[:,:n_LIF]
        if n_LIF > 100:
            z = z[:, :100]

    elif neuron_type == "ALIF":
        z = z[:, n_LIF:]
        if n_ALIF > 100:
            z = z[:, :100]

    else:
        raise ValueError
    z = z.T
    
    

    # TODO: handle this correctly

    
    
    for i, neuron in enumerate(z):
            spike_times = np.where(neuron == 1)[0]
            ax.vlines(spike_times, 0.3 + i*0.06 , 0.3+ i*0.06 + 0.02, color='black',linewidth=0.8)
        
    ax.set_ylabel('Neuron Index')
    ax.set_xlabel('Time (ms)')
    ax.set_title("example {} rasterplot".format(neuron_type))
    ax.tick_params(which="both", left=False, pad=5)
    
    #ax.set_yticks([0.3 + 0.03 * n_population/2 + offset_constant*j for j in range(input_channels)], y_ticks)
    # This considers at least 10 neurons
    y_ticks = list(range(0,z.shape[0], 10))
   
    ax.set_xlim(0, z.shape[1]+10)
    ax.set_yticks([0.3 + j*0.06 + 0.01 for j in range(0,z.shape[0], 10)], y_ticks)


def plot_softmax_output(y, ax=None, label='Probability of Left', title="Softmaxt Output: left"):
    y = y.T
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_ylabel(label)
    ax.set_xlabel('Time (ms)')
    ax.set_title(title)
    ax.plot(y, color='red')

def plot_pattern_generation_inputs(input_spikes, ax=None):
    input_spikes = input_spikes.T

    if ax is None:
        fig, ax = plt.subplots()

    for i, neuron in enumerate(input_spikes):
        spike_times = np.where(neuron == 1)[0]
        ax.vlines(spike_times, 0.3 + i*0.06 , 0.3+ i*0.06 + 0.02, color='black',linewidth=0.8)
        
    ax.set_ylabel('Neuron Index')
    ax.set_xlabel('Time (ms)')
    ax.set_title("example inputs rasterplot")
    ax.tick_params(which="both", left=False, pad=5)

    y_ticks = list(range(0,input_spikes.shape[0], 10))
   
    ax.set_xlim(0, input_spikes.shape[1]+10)
    ax.set_yticks([0.3 + j*0.06 + 0.01 for j in range(0,input_spikes.shape[0], 10)], y_ticks)


def plot_pattern_generation_prediction(y, targets, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(y, label="prediction", color='red', linestyle='-')
    ax.plot(targets, label="target", color='black')
    ax.legend()
    ax.set_title("Predictions")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Output")

def plot_LSNN_weights(state, layer_names, save_path):
    """
    Plots multiple sets of weights in the same figure.
    
    Args:
    - weights_list (list of jnp.array): A list of weight arrays to be plotted.
    - layer_names (list of str): A list of layer names corresponding to the weight arrays.
    """
  
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    weights_list = [state.params['ALIFCell_0']['input_weights'],state.params['ALIFCell_0']['recurrent_weights'],
                    state.params['ReadOut_0']['readout_weights']]
    
    for i, (weights, ax) in enumerate(zip(weights_list, axs)):
        # Normalize weights for better visualization
        #weights_min = jnp.min(weights)
        #weights_max = jnp.max(weights)
        #weights_normalized = (weights - weights_min) / (weights_max - weights_min + 0.00001)
        
        # Convert to numpy array for plotting
       # weights_normalized = jnp.array(weights_normalized)
        
        # Plot the weights
        cax = ax.imshow(weights, cmap='viridis',interpolation='nearest', aspect='auto')
     
        
        # Add a color bar
        fig.colorbar(cax, ax=ax)
        
        # Set titles and labels
        ax.set_title(f'Weights of {layer_names[i]}')
        ax.set_xlabel('Post Neuron index')
        ax.set_ylabel('Pre Neuron index')
    axs[2].set_xlim(0, 1)
    axs[2].set_xticks([0.25, 0.75], ["left", "rigth"])
    # Adjust layout to prevent overlap
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()
    

"""
def cue_accumulation(output_dir,iterations, loss_training, loss_eval, accuracy_training, accuracy_eval, eval_batch, state,):
    
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
    from flax.linen import softmax



    

    
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
    plot_cue_accumulation_inputs(input_example_1, n_population = cfg.net_arch.n_neurons_channel, input_mode=cfg.task.input_mode, ax =ax1_1)
    plot_recurrent(recurrent_example_1, n_LIF=50, n_ALIF=50, neuron_type="LIF", ax =ax1_2)
    plot_recurrent(recurrent_example_1, n_LIF=50, n_ALIF=50, neuron_type="ALIF", ax =ax1_3)
    plot_softmax_output(y[0,:,0],ax= ax1_4)
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
    plots.plot_recurrent(recurrent_example_2, n_LIF=50, n_ALIF=50, neuron_type="LIF", ax =ax2_2)
    plots.plot_recurrent(recurrent_example_2, n_LIF=50, n_ALIF=50, neuron_type="ALIF", ax =ax2_3)
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
    plots.plot_recurrent(recurrent_example_3, n_LIF=50, n_ALIF=50, neuron_type="LIF", ax =ax3_2)
    plots.plot_recurrent(recurrent_example_3, n_LIF=50, n_ALIF=50, neuron_type="ALIF", ax =ax3_3)
    plots.plot_softmax_output(y[2,:,0],ax= ax3_4)
    fig3.suptitle("Example 3: " + cfg.save_paths.condition)
    fig3.tight_layout()
    fig3.savefig(os.path.join(figures_directory, "example_3"))   
    plt.close(fig3)
"""