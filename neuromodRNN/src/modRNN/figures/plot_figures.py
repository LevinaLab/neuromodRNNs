import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_multiruns(experiment_folder, label, metric_file, ax):

    loss_all_runs = []
    
    # Traverse through each run folder inside the experiment folder
    for run_folder in os.listdir(experiment_folder):
        run_path = os.path.join(experiment_folder, run_folder)
        if os.path.isdir(run_path):
            train_info_folder = os.path.join(run_path, 'train_info')
                # Look for .pkl files inside the train_info folder
            if os.path.exists(train_info_folder):
                file = metric_file
                file_path = os.path.join(train_info_folder, file) 
                # Load the .pkl file and extract the loss
                with open(file_path, 'rb') as f:
                    loss_all_runs.append(pickle.load(f))
                # Convert list of loss into a numpy array (padding shorter runs with NaN)
        
    max_epochs = max(len(run_loss) for run_loss in loss_all_runs)
    losses_padded = np.array([np.pad(run_loss, (0, max_epochs - len(run_loss)), 'constant', constant_values=np.nan)
                                for run_loss in loss_all_runs])

    # Compute mean and variance, ignoring NaNs
    mean_loss = np.nanmean(losses_padded, axis=0)
    
    variance_loss = np.nanvar(losses_padded, axis=0)

    # Plot the average loss with variance as a shaded area
    epochs = np.arange(0, max_epochs * 25, 25)

    ax.plot(epochs, mean_loss, label=label)
    ax.fill_between(epochs, mean_loss - np.sqrt(variance_loss), mean_loss + np.sqrt(variance_loss), alpha=0.2)
    ax.legend()



