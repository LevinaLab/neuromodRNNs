# TODO: check if need traintestloader for early stop or something else, ow remove it


import numpy as np






def delayed_match_task(n_batches,batch_size,n_population=10, f_background = 10, f_input = 40, p=0.5, seed=None, 
                       fixation_time=50, cue_time=150, delay_time=750  ):
    """
    Generates batches of data for a delayed match-to-sample task.
    
    Important: both background_f and inputs_f should be given in Hz.
    
    Task description:
    
    Task starts with a brief period of fixation (no cue) followed by two sequential cues, each last cue_time and separated by delay_time. Each one of the cues
    has probability p of being equal to 1 and 1-p of being equal to 0. At the end of the second cue, the goal of the network is to output 1 if both cues had the same value
    (either 0 and 0 or 1 and 1) and output 0 if the cues didn't match (0 and 1 or 1 and 0).
    
    The inputs are defined by 3 populations, each one of size n_neurons_populaiton. The first population (resp second) fires with rate "inputs_f" during the periond where the first cue (resp second) is
    1, and is quiscient otherwise. The third population generates background noise with frequency "backgroud_f" during the whole trial.

    Parameters:
    n_batches (int): The total number of batches to generate.
    batch_size (int): The size of each batch.
    n_neurons_population (int, optional): The number of neurons in each population. Default is 10.
    background_f (int, optional): The background firing rate in Hz. Default is 10 Hz.
    inputs_f (int, optional): The input firing rate for cues in Hz. Default is 40 Hz.

    Yields:
    dict: A dictionary containing:
        - "input" (numpy.ndarray): The input data of shape (batch_size, trial_length, 3 * n_neurons_population), 
                                   where trial_length is the total duration of a trial in ms.
        - "label" (numpy.ndarray): The labels for each batch, indicating if the cues match (1) or do not match (0).

    Description:
    This function simulates a delayed match-to-sample task. Each trial consists of:
    - A fixation period (50 ms)
    - A first cue period (150 ms)
    - A delay period (300 ms)
    - A second cue period (150 ms)
.

    The cues for each batch are randomly generated with a 50% probability for each cue to be 1.
    The labels indicate whether the first and second cues match.


    Variables:
    - fixation_time (int): Duration of the fixation period (50 ms).
    - cue_time (int): Duration of each cue period (150 ms).
    - delay_time (int): Duration of the delay period (300 ms).
    - trial_length (int): Total duration of a trial (fixation_time + 2 * cue_time + delay_time).
    - background_f (float): Background firing rate in spikes/ms.
    - inputs_f (float): Input firing rate for cues in spikes/ms.
    - p (float): Probability of any cue being 1.
    - cues_labels (numpy.ndarray): Randomly generated cues for each batch.
    - input_shape (tuple): Shape of the input data (n_batches, trial_length, n_neurons_population).
    - back_ground_channel (numpy.ndarray): Background activity channel.
    - input_1_channel (numpy.ndarray): Activity channel for the first cue.
    - input_2_channel (numpy.ndarray): Activity channel for the second cue.
    - inputs (numpy.ndarray): Concatenated input data from background and cue channels.
    - labels (numpy.ndarray): Labels indicating if the cues match.
    - match_idx (numpy.ndarray): Indices where the first and second cues match.
    """

    
  
    trial_dur = fixation_time + 2*cue_time + delay_time
    f_background = f_background / 1000 # assumes that firing rate was given in Hz
    f_input = f_input / 1000 # assumes that firing rate was given in Hz
    p = 0.5 # probability of any cue being equal to 1
    
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    
    # generate cues
    cues_labels = rng.binomial(n=1, p=p, size=(n_batches, 2)) # independently draw 1 or 0 from bernoulli distribution for each cue and batch
    
    # initialize input popuplations
    input_shape = (n_batches,trial_dur, n_population)
    
    back_ground_channel = (rng.uniform(size=input_shape) < (f_background)) # backgroud
    
    input_1_channel = np.zeros(input_shape)
   
    idx_1 = np.where(cues_labels[:, 0]== 1)[0]  # get batches where first cue is 1
    n_trials_first_on = np.sum(cues_labels[:,0])    
    input_1_channel[idx_1,fixation_time:fixation_time+cue_time, :] = rng.uniform(size=(n_trials_first_on,cue_time, n_population)) < (f_input)

    input_2_channel = np.zeros(input_shape)
    idx_2 = np.where(cues_labels[:, 1]== 1)[0]  # get batches where second cue is 1  
    n_trials_second_on = np.sum(cues_labels[:,1]) 
    input_2_channel[idx_2,-cue_time:, :] = rng.uniform(size=(n_trials_second_on,cue_time, n_population)) < (f_input)
    
    inputs = np.concatenate((back_ground_channel, input_1_channel, input_2_channel), axis=2).astype(float)
    
    
    # get labels
    labels = np.zeros(n_batches)
    match_idx = np.where(cues_labels[:,0] == cues_labels[:,1])[0]
    labels[match_idx] = 1
    labels = np.repeat(labels[:, np.newaxis], trial_dur, axis=1)

    for start_idx in range(0, n_batches, batch_size):
        end_idx = min(start_idx + batch_size, n_batches)
        batch_inputs = inputs[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        batch_trial_duration = np.full(batch_size, trial_dur)
        yield {"input": batch_inputs, "label": batch_labels, 'trial_duration':batch_trial_duration}


def cue_accumulation_task(n_batches, batch_size, seed=None, n_cues=[1,3,5,7], min_delay=500, max_delay=1500, 
                                     n_population=10, f_input=40, f_background=10, t_cue=100, t_cue_spacing=150, 
                                     p=0.5, input_mode='original', dt=1000):
    
    if type(n_cues) is int:
        n_cues = [n_cues]
    f_background /= dt
    f_input /= dt
    max_trial_duration = max(n_cues) * t_cue_spacing + max_delay
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    
    # Generate batch of trials
    trials = np.full((n_batches, max_trial_duration), -1)
    delays = rng.integers(low=min_delay, high=max_delay, size=n_batches)
    trial_n_cues = rng.choice(n_cues, size=n_batches)

    trial_duration = t_cue_spacing * trial_n_cues + delays
   
    
    # Set initial trial states
    for i in range(n_batches):
        trials[i, :delays[i]] = 1
    
    trials[:, :t_cue_spacing] = 2
    
    # Generate cue assignments for all trials
    prob_choices = np.array([p, 1 - p], dtype=np.float32)
    idx = rng.choice([0, 1], size=n_batches)
    p_for_binomial = prob_choices[idx]
    
    max_n_cues = np.max(trial_n_cues)
    cues_assignment = np.zeros((n_batches, max_n_cues), dtype=int)
    
    for i in range(n_batches):
        cues_assignment[i, :trial_n_cues[i]] = rng.binomial(n=1, p=p_for_binomial[i], size=trial_n_cues[i])
   
    # Assign cues to trials
    for i in range(n_batches):
        for j in range(trial_n_cues[i]):
            start_cue = delays[i] + j * t_cue_spacing
            trials[i, start_cue:start_cue + t_cue] = cues_assignment[i, j] + 3
            trials[i, start_cue + t_cue:start_cue + t_cue_spacing] = 0
    
    trials = trials[:,::-1]
    
    # Initialize input populations
    input_shape = (n_batches, max_trial_duration, n_population)
    
    if input_mode == "original":
        background_channel = rng.uniform(size=input_shape) < f_background
        left_channel = np.zeros(input_shape)
        right_channel = np.zeros(input_shape)
        decision_channel = np.zeros(input_shape)
        
        for i in range(n_batches):
            no_trial_mask = trials[i] == -1
            background_channel[i, no_trial_mask] = 0
            
            left_cues_idx = np.where(trials[i] == 3)[0]
            left_channel[i, left_cues_idx] = rng.uniform(size=(left_cues_idx.shape[0], n_population)) < f_input
            
            right_cues_idx = np.where(trials[i] == 4)[0]
            right_channel[i, right_cues_idx] = rng.uniform(size=(right_cues_idx.shape[0], n_population)) < f_input
            
            decision_time_idx = np.where(trials[i] == 2)[0]
            decision_channel[i, decision_time_idx] = rng.uniform(size=(decision_time_idx.shape[0], n_population)) < f_input
        
        inputs = np.concatenate((background_channel, decision_channel, left_channel, right_channel), axis=2).astype(float)
        
        # get labels
        labels = np.zeros(n_batches)
        right_mask = np.sum(trials == 4, axis=1) > np.sum(trials == 3, axis=1)
        labels[right_mask] = 1
        labels =  np.repeat(labels[:, np.newaxis], max_trial_duration, axis=1)

        for start_idx in range(0, n_batches, batch_size):
            end_idx = min(start_idx + batch_size, n_batches)
            batch_inputs = inputs[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch_trial_duration = trial_duration[start_idx:end_idx]
            yield {"input": batch_inputs, "label": batch_labels, 'trial_duration':batch_trial_duration}
        
    

    if input_mode == "modified":                
        left_channel = rng.uniform(size=input_shape) < (f_background / 3)
        right_channel = rng.uniform(size=input_shape) < (f_background / 3)
        decision_channel = rng.uniform(size=input_shape) < (f_background / 3)

        for i in range(n_batches):
            no_trial_mask = trials[i] == -1

            left_channel[i, no_trial_mask] = 0
            left_cues_idx = np.where(trials[i] == 3)[0]
            left_channel[i, left_cues_idx] = rng.uniform(size=(left_cues_idx.shape[0], n_population)) < f_input

            right_channel[i, no_trial_mask] = 0
            right_cues_idx = np.where(trials[i] == 4)[0]
            right_channel[i, right_cues_idx] = rng.uniform(size=(right_cues_idx.shape[0], n_population)) < f_input

            decision_channel[i, no_trial_mask] = 0
            decision_time_idx = np.where(trials[i] == 2)[0]
            decision_channel[i, decision_time_idx] = rng.uniform(size=(decision_time_idx.shape[0], n_population)) < f_input

            inputs = np.concatenate((decision_channel, left_channel, right_channel), axis=2).astype(float)

        # get labels
        labels = np.zeros(n_batches)
        right_mask = np.sum(trials == 4, axis=1) > np.sum(trials == 3, axis=1)
        labels[right_mask] = 1
        labels =  np.repeat(labels[:, np.newaxis], max_trial_duration, axis=1)
        
        for start_idx in range(0, n_batches, batch_size):
            end_idx = min(start_idx + batch_size, n_batches)
            batch_inputs = inputs[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch_trial_duration = trial_duration[start_idx:end_idx]
            yield {"input": batch_inputs, "label": batch_labels, 'trial_duration':batch_trial_duration}
        
    

def pattern_generation(n_batches, batch_size, seed, frequencies, 
                       weights,n_population, f_input, trial_dur):

    # adjust frequencies to be compatible with ms
    f_input /= 1000
    frequencies = [f/1000 for f in frequencies]

    # create time vector
    t = np.arange(trial_dur)

    # input spikes
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    
    input_shape = (n_batches, trial_dur, n_population )    
    inputs = (rng.uniform(size=input_shape) < f_input).astype(float)

    # generate signal (label
    signal = np.zeros(trial_dur)
    for f, w in zip(frequencies, weights):
        signal = signal + w * np.sin(2* np.pi * f * t)
    
    # center signal so that has mean 0
    centered_signal = signal - np.mean(signal)

    # # min-max normalization
    # signal_min = np.min(signal)
    # signal_max = np.max(signal)
    # signal_normalized = (signal - signal_min) / (signal_max - signal_min)

    # replicate through batches
    labels = np.repeat(centered_signal[np.newaxis, :], n_batches, axis=0)

    for start_idx in range(0, n_batches, batch_size):
        end_idx = min(start_idx + batch_size, n_batches)
        batch_inputs = inputs[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        batch_trial_duration = np.full(batch_size, trial_dur)
        yield {"input": batch_inputs, "label": batch_labels, 'trial_duration':batch_trial_duration}
    return inputs
   