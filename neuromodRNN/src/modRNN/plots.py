"""plots funcitons"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Just for typing the type of inputs of functions
from typing import (
 List  
 )

from flax.typing import (
  Array,
)




def plot_cue_accumulation_inputs(neuron_spikes: Array, n_population:int = 10, input_mode:str="original", ax =None):
    """
    plot spike rasters of input neurons of cue_accumulation task for single trial
    """
    if ax is None:
        fig, ax = plt.subplots()

    offset_constant = 0.6
    
    ax.set_title("Inputs rasterplot")
    # original mode 4 populations: left cue, right cue, decision time, background
    if input_mode == "original":
        input_channels = 4
        y_ticks = ["noise", "cue", "left", "right"]
        left_anchor = 2 * offset_constant + 0.3 # anchor to set patch for left input
        right_anchor = 3 * offset_constant + 0.3 # anchor to set patch for tight input
    
    # modified mode 3 populations: left cue, right cue, decision time: the three also contain noise
    elif input_mode == "modified":
        input_channels = 3
        y_ticks = ["cue", "left", "right"]
        left_anchor = 1 * offset_constant + 0.3 # anchor to set patch for left input
        right_anchor = 2 * offset_constant + 0.3 # anchor to set patch for tight input
    
    else:
        raise NotImplementedError("The requested mode {} has not being implemented. Try either 'original' or 'modified'".format(input_mode))
    
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

def plot_delayed_match_inputs(neuron_spikes:Array, n_population:int = 10, ax =None):
    """plot spike rasters for input neurons of delayed match task for single trial"""
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


def plot_recurrent(z:Array, n_LIF:int, n_ALIF:int, neuron_type:str="LIF", ax =None):
    """
    Plot spike rasters for recurrent layer, either LIF or ALIF neurons. The maximum value of neurons is truncated to be 100. 
    """

    

    if ax is None:
        fig, ax = plt.subplots()
    z = np.array(z)

    offset_constant = 0.6
    if neuron_type == "LIF":
        z = z[:,:n_LIF] # LIF are by construction of model the first n_LIF neurons
        if n_LIF > 100:
            z = z[:, :100]

    elif neuron_type == "ALIF":
        z = z[:, n_LIF:] # ALIF are by construction of model the neurons after the firt n_LIF neurons
        if n_ALIF > 100:
            z = z[:, :100]

    else:
        raise NotImplementedError("The requested neuron model {} has not being implemented. Try either 'LIF' or 'ALIF'".format(neuron_type))
    
    z = z.T # in a trial, time is the leading axis. For plotting want to have neuron as leading axis
            
    for i, neuron in enumerate(z):
            spike_times = np.where(neuron == 1)[0]
            ax.vlines(spike_times, 0.3 + i*0.06 , 0.3+ i*0.06 + 0.02, color='black',linewidth=0.8)
        
    ax.set_ylabel('Neuron Index')
    ax.set_xlabel('Time (ms)')
    ax.set_title("example {} rasterplot".format(neuron_type))
    ax.tick_params(which="both", left=False, pad=5)    
    
    # This considers at least 10 neurons
    y_ticks = list(range(0,z.shape[0], 10))
   
    ax.set_xlim(0, z.shape[1]+10)
    ax.set_yticks([0.3 + j*0.06 + 0.01 for j in range(0,z.shape[0], 10)], y_ticks)


def plot_softmax_output(y:Array, ax=None, label:str='Probability of Left', title:str="Softmaxt Output: left"):
    """
    Plot the softmax output for a chosen output neuron
    """    
    y = y.T
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_ylabel(label)
    ax.set_xlabel('Time (ms)')
    ax.set_title(title)
    ax.plot(y, color='red')

def plot_pattern_generation_inputs(input_spikes:Array, ax=None):
    """  Plot the softmax output for a chosen output neuron """    
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


def plot_pattern_generation_prediction(y:Array, targets:Array, ax=None):

    """ plot spike rasters for input neurons of delayed match task for single trial """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(y, label="prediction", color='red', linestyle='-')
    ax.plot(targets, label="target", color='black')
    ax.legend()
    ax.set_title("Predictions")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Output")

def plot_LSNN_weights(state, layer_names:List, save_path):
    """ Plot weights of the different layers.  """
  
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    weights_list = [state.params['ALIFCell_0']['input_weights'],state.params['ALIFCell_0']['recurrent_weights'],
                    state.params['ReadOut_0']['readout_weights']]
    
    for i, (weights, ax) in enumerate(zip(weights_list, axs)):
        # Normalization commented out
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
    
