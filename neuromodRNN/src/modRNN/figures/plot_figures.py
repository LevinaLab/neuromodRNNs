import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_multiruns(experiment_folder, label, metric_file, ax, color):

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

    lower_percentile = np.nanpercentile(losses_padded, 5, axis=0)
    upper_percentile = np.nanpercentile(losses_padded, 95, axis=0)
    

    
    

    # Plot the average loss with variance as a shaded area
    epochs = np.arange(0, max_epochs * 25, 25)

    ax.plot(epochs, mean_loss, label=label, c=color)
    ax.fill_between(epochs, lower_percentile, upper_percentile, color=color, alpha=0.4)
    ax.legend()



def boxplot_multiruns(experiment_folder, label, metric_file, ax, position, color):
    """
    Plot a boxplot of the final metric values across multiple runs of an experiment.

    Parameters:
    - experiment_folder (str): Path to the main folder containing all runs for the experiment.
    - label (str): Label for the box plot (experiment name).
    - metric_file (str): Name of the file with the metric data (e.g., 'loss.pkl').
    - ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
    - position (int): Position of the boxplot on the x-axis (for distinguishing different experiments).
    - color (str or tuple): Color for the boxplot.

    """
    final_metrics = []

    # Traverse through each run folder inside the experiment folder
    for run_folder in os.listdir(experiment_folder):
        run_path = os.path.join(experiment_folder, run_folder)
        if os.path.isdir(run_path):
            train_info_folder = os.path.join(run_path, 'train_info')
            # Look for .pkl files inside the train_info folder
            if os.path.exists(train_info_folder):
                file = metric_file
                file_path = os.path.join(train_info_folder, file) 
                
                # Load the .pkl file and extract the last metric value
                with open(file_path, 'rb') as f:
                    run_metric = pickle.load(f)
                    # Append only the last metric value of the current run
                    final_metrics.append(run_metric[-1] if len(run_metric) > 0 else np.nan)

    # Filter out any NaN values (if a run had no data or a shorter series)
    final_metrics = [metric for metric in final_metrics if not np.isnan(metric)]

    # Plot boxplot at the given position with the specified color
    ax.boxplot(final_metrics, positions=[position], patch_artist=True,
               boxprops=dict(facecolor=color, color=color), 
               medianprops=dict(color="black"))
    

    
