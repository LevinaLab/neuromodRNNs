"""plots funcitons"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import jax.numpy as jnp
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
 
    input_channels = 4
    y_ticks = ["noise", "decision cue", "1st cue", "2nd cue"]
    first_anchor =  2 * offset_constant + 0.3
    second_anchor = 3 * offset_constant + 0.3


    
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
    
def plot_weights_spatially_indexed(state, gridshape, save_path):
    # Get weights
    input_weights = state.params['ALIFCell_0']['input_weights']
    recurrent_weights = state.params['ALIFCell_0']['recurrent_weights']
    output_weights = state.params['ReadOut_0']['readout_weights']

    # Get location of recurrent neurons in cell
    rec_cell_loc = state.spatial_params["ALIFCell_0"]["cells_loc"]

    # convert to 1D indexing location code 
    rec_cell_loc_ind = jnp.lexsort((rec_cell_loc[:, 1], rec_cell_loc[:, 0])) # sort by row, then column 

    # sort weights
    sorted_input_weights = input_weights[:, rec_cell_loc_ind] # (n_in, n_rec)
    sorted_recurrent_weights =  recurrent_weights[jnp.ix_(rec_cell_loc_ind, rec_cell_loc_ind)]# (n_rec, n_rec) sorts both rows and columns 
    
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot input weights
    cax1 = axs[0,0].imshow(sorted_input_weights, cmap='viridis',interpolation='nearest', aspect='auto') 
    
    # Add a color bar
    fig.colorbar(cax1, ax=axs[0,0])
    
    # Set titles and labels
    axs[0,0].set_title(f'Input weights')
    axs[0,0].set_xlabel('Pre Neuron index')
    axs[0,0].set_ylabel('Post Neuron index')


    # Plot recurrent weights
    cax2 = axs[0,1].imshow(sorted_recurrent_weights, cmap='viridis',interpolation='nearest', aspect='auto')     
    
    # Add a color bar
    fig.colorbar(cax2, ax=axs[0,1])
    
    # Set titles and labels
    axs[0,1].set_title(f'Recurrent weights')
    axs[0,1].set_xlabel('Pre Neuron index')
    axs[0,1].set_ylabel('Post Neuron index')


    # plot output weights    
    # Create a 2D grid 
    grid = jnp.full(gridshape, jnp.nan)    
    
    # Plot first readout neuron weight
    # Populate the grid with weights using the neuron positions
    grid = grid.at[rec_cell_loc[:, 0], rec_cell_loc[:, 1]].set(output_weights[:,0])
    cax3 = axs[1,0].imshow(grid, cmap='viridis',interpolation='nearest', aspect='auto')    
    fig.colorbar(cax3, ax=axs[1,0])
    axs[1,0].set_title(f'Readout 1 weights(Left/0)')
    axs[1,0].set_xlabel('x position grid')
    axs[1,0].set_ylabel('y position grid')
    
    # if exist, plot second readout
    if jnp.shape(output_weights)[1] > 1:
        grid = jnp.full(gridshape, jnp.nan)
    
        # Plot first readout neuron weight
        # Populate the grid with weights using the neuron positions
        grid = grid.at[rec_cell_loc[:, 0], rec_cell_loc[:, 1]].set(output_weights[:,1])
        cax4 = axs[1,1].imshow(grid, cmap='viridis',interpolation='nearest', aspect='auto')    
        fig.colorbar(cax4, ax=axs[1,1])
        axs[1,1].set_title(f'Readout 2 weights(Right/1)')
        axs[1,1].set_xlabel('x position grid')
        axs[1,1].set_ylabel('y position grid')
    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()

def plot_gradients(grads, spatial_params, epoch, save_path):
    """
    Plot histogram of grads for the different layers, already excluding gradients masked to 0 due to sparse or local connectivity
    figured inspired on https://hassaanbinaslam.github.io/myblog/posts/2022-10-23-pytorch-vanishing-gradients-deep-neural-networks.html
    """
    fig, axs= plt.subplots(1, 3, figsize=(10.5,3))
    
    #input layer grads
    input_grads_ind = jnp.where(spatial_params['ALIFCell_0']['sparse_input'] == 1)
    sns.histplot(data=grads['ALIFCell_0']['input_weights'][input_grads_ind], bins=30, ax=axs[0], kde=True)
    axs[0].set_title('Input Weights')
    axs[0].set_xlabel("Grad magnitude", fontsize=11)
    # Count the number of NaNs in input grads
    input_nan_count = jnp.sum(jnp.isnan(grads['ALIFCell_0']['input_weights'][input_grads_ind]))
    # Count the number of infs in input grads
    input_inf_count = jnp.sum(jnp.isinf(grads['ALIFCell_0']['input_weights'][input_grads_ind]))
    axs[0].text(
    0.95,  # x-position (relative to plot, 0 is left, 1 is right)
    0.95,  # y-position (relative to plot, 0 is bottom, 1 is top)
    f'NaN count: {input_nan_count}\nInf count: {input_inf_count}',
    horizontalalignment='right',
    verticalalignment='top',
    transform=axs[0].transAxes,  # Position relative to the specific axis (axs[0])
    bbox=dict(facecolor='white', alpha=0.5)  # Optional: adds a semi-transparent background box
    )
    
    #recurrent layer grads
    recurrent_grads_ind = jnp.where(spatial_params['ALIFCell_0']['M'] == 1)
    sns.histplot(data=grads['ALIFCell_0']['recurrent_weights'][recurrent_grads_ind], bins=30, ax=axs[1], kde=True)
    axs[1].set_title('Recurrent Weights')
    axs[1].set_xlabel("Grad magnitude", fontsize=11)
    # Count the number of NaNs in recurrent grads
    recurrent_nan_count = jnp.sum(jnp.isnan(grads['ALIFCell_0']['recurrent_weights'][recurrent_grads_ind]))
    # Count the number of infs in recurrent grads
    recurrent_inf_count = jnp.sum(jnp.isinf(grads['ALIFCell_0']['recurrent_weights'][recurrent_grads_ind]))
    axs[1].text(
    0.95,  # x-position (
    0.95,  # y-position 
    f'NaN count: {recurrent_nan_count}\nInf count: {recurrent_inf_count}',
    horizontalalignment='right',
    verticalalignment='top',
    transform=axs[1].transAxes,  # Position relative to the specific axis (axs[0])
    bbox=dict(facecolor='white', alpha=0.5)  
    )

        
    # Output layer grads
    output_grads_ind = jnp.where(spatial_params['ReadOut_0']['sparse_readout'] == 1)
    sns.histplot(data=grads['ReadOut_0']['readout_weights'][output_grads_ind] , bins=10, ax=axs[2], kde=True)
    axs[2].set_title('Output Weights')
    axs[2].set_xlabel("Grad magnitude", fontsize=11)
        # Count the number of NaNs in output grads
    output_nan_count = jnp.sum(jnp.isnan(grads['ReadOut_0']['readout_weights'][output_grads_ind] ))
    # Count the number of infs in input grads
    output_inf_count = jnp.sum(jnp.isinf(grads['ReadOut_0']['readout_weights'][output_grads_ind] ))
    axs[2].text(
    0.95,  # x-position (
    0.95,  # y-position 
    f'NaN count: {output_nan_count}\nInf count: {output_inf_count}',
    horizontalalignment='right',
    verticalalignment='top',
    transform=axs[2].transAxes,  # Position relative to the specific axis (axs[0])
    bbox=dict(facecolor='white', alpha=0.5)  
    )


    fig.suptitle(f"Epoch: {epoch}", fontsize=16)
    # Adjust layout to prevent overlap
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()