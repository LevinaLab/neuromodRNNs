# TODO: check if need traintestloader for early stop or something else, ow remove it


import numpy as np
from jax.nn import one_hot
from typing import (
 List, Union 
 )





def delayed_match_task(n_batches:int,batch_size:int,n_population:int=10, f_background:float = 10., f_input:float = 40., p:float=0.5, seed:Union[None,int]=None, 
                       fixation_time:int=50, cue_time:int=150, delay_time:int=350, decision_time:int=150  ):
    """
    Generates batches of data for a delayed match-to-sample task.
    
    Important: both background_f and inputs_f should be given in Hz.
    
    Task description:
    
    Task starts with a brief period of fixation (no cue) followed by two sequential cues, each last cue_time and separated by delay_time. Each one of the cues
    has probability p of being equal to 1 and 1-p of being equal to 0. At the end of the second cue, the goal of the network is to output 1 if both cues had the same value
    (either 0 and 0 or 1 and 1) and output 0 if the cues didn't match (0 and 1 or 1 and 0).
    
    The inputs are defined by 3 populations, each one of size n_neurons_populaiton. The first population (resp second) fires with rate "inputs_f" during the period where the first cue (resp second) is
    1, and is quiscient otherwise. The third population generates background noise with frequency "backgroud_f" during the whole trial.

    Parameters:
    n_batches: int 
        The total number of batches to generate.
    batch_size: int 
        The size of each batch.
    n_neurons_population: int
        The number of neurons in each population. Default is 10.
    f_background: int
        The background firing rate in Hz. Default is 10 Hz.
    f_input: int
        The input firing rate for cues in Hz. Default is 40 Hz.
    p:  float
        Probability of any cue being 1.
    fixation_time: int
        Duration of the fixation period in ms.  Default is 50 ms
    seed: int or None
        Seed for RNG. If a seed is passed, function will always yield the same group of batches when called, but batches will still be different within each other.
    cue_time: int
        Duration of each cue period in ms. Default is 150 ms.
    delay_time: int
        Duration of the delay period in ms. Default is 350ms.

    Yields
    dict: A dictionary containing:
        "input": Array (batch_size, trial_length, 3 * n_neurons_population)
            The spike train of input channels.        
        "label":  Array (batch_size, n_t, 1)
            The labels for each batch, indicating if the cues match (1) or do not match (0).
        "trial_duration": Array (batch_size,)
             Duration of each trial in the batch (equal for every trial). 
    """

    
  
    trial_dur = fixation_time + 2*cue_time + 2 * delay_time + decision_time
    f_background = f_background / 1000 # assumes that firing rate was given in Hz
    f_input = f_input / 1000 # assumes that firing rate was given in Hz
    
    
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    
    # generate cues
    cues_labels = rng.binomial(n=1, p=p, size=(n_batches, 2)) # independently draw 1 or 0 from bernoulli distribution for each cue and batch
    
    # initialize input popuplations
    input_shape = (n_batches,trial_dur, n_population)
    
    back_ground_channel = (rng.uniform(size=input_shape) < (f_background)) # backgroud
    
    # channel coding for first cue: 40Hz if cue is 1, 0Hz otherwise
    input_1_channel = np.zeros(input_shape)
   
    idx_1 = np.where(cues_labels[:, 0]== 1)[0]  # get batches where first cue is 1
    n_trials_first_on = np.sum(cues_labels[:,0])    
    input_1_channel[idx_1,fixation_time:fixation_time+cue_time, :] = rng.uniform(size=(n_trials_first_on,cue_time, n_population)) < (f_input)

    # channel coding for second cue: 40Hz if cue is 1, 0Hz otherwise
    input_2_channel = np.zeros(input_shape)
    idx_2 = np.where(cues_labels[:, 1]== 1)[0]  # get batches where second cue is 1  
    n_trials_second_on = np.sum(cues_labels[:,1]) 
    input_2_channel[idx_2, fixation_time+cue_time+delay_time:fixation_time+2*cue_time+delay_time, :] = rng.uniform(size=(n_trials_second_on,cue_time, n_population)) < (f_input)
    
    
    # channel coding for decision time
    decision_time_channel = np.zeros(input_shape)
    decision_time_channel[:, -decision_time:] = (rng.uniform(size=(n_batches,decision_time,n_population )) < (f_input))
    
    
    inputs = np.concatenate((back_ground_channel, decision_time_channel, input_1_channel, input_2_channel), axis=2).astype(float)
    
    
    # get labels
    labels = np.zeros(n_batches)
    match_idx = np.where(cues_labels[:,0] == cues_labels[:,1])[0]
    labels[match_idx] = 1
    labels = np.repeat(labels[:, np.newaxis], trial_dur, axis=1)
    # one-hot labels
    labels = one_hot(labels, 2)

    for start_idx in range(0, n_batches, batch_size):
        end_idx = min(start_idx + batch_size, n_batches)
        batch_inputs = inputs[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        batch_trial_duration = np.full(batch_size, trial_dur)
        yield {"input": batch_inputs, "label": batch_labels, 'trial_duration':batch_trial_duration}


def cue_accumulation_task(n_batches:int, batch_size:int, seed:Union[None,int]=None, n_cues:List[int]=[1,3,5,7], min_delay:int=500, max_delay:int=1500, 
                                     n_population:int=10, f_input:float=40., f_background:float=10., t_cue:int=100, t_cue_spacing:int=150, 
                                     p:float=0.5, input_mode:str='original', dt:int=1000):
    
    """
    Generates batches of data for a due accumulation task similar to the one described in 
    Bellec et al. 2020: A solution to the learning dilemma for recurrent networks of spiking neurons.

    Task description: 
    Task description: a trial consists of n_cues, where each individual cue can be presented either on the left side or right. 
    After the last cue, there is delay time when any cue is presented. After this delay, decision time arrives
    and at the end of this period, the network has to indicate if the majority of cues were presented either on the left
    or on the right side, independent of the order.The number of cues in a trial is draw randomly from the provided
    n_cues list (if it contains a single value, all the trials have same number of cues).
    The delay time is draw randomly with uniform distribution between min_delay (included) and max_delay (excluded). 
    If max_delay - min_delay=1, all trials will have the same delay time: min_delay.Each cue is presented for t_cue ms, 
    and the time between two consecutives cues is given by t_cue_spacing - t_cue. After decided the number of cues for the trial,
    each cue is assigned to either left or right independently, with probability p or (1-p) 
    (which one is for each side, is also decided randomly for each trial).
    The input is encoded as follows:    
    “original mode”: input consists of 4 populations, each one with n_population neurons on it. A first population responds
    with a poisson noise of input_f Hz while a cue on the left side is present, 0Hz otherwise. 
    A second population behaves similar, but for cues on the right side. The third population encodes decision time, 
    consisting of the last 150ms of the trial, responding also with input_f Hz in this period and 0Hz during the rest of trial. 
    The last population correspond to background noise, firing with a frequency of background_f during whole trial.     
    “modified mode”: input consists of 3 populations: left cue, right cue and decision time. They behave similar as in the
    original mode, with the difference that when they are not responding to their “encoding stimulus”, they fire with a 
    frequency of background_f, composing so the backgorund noise.

Important!!!
If n_cues contain more than one value, or/and max_delay-min_delay>1, the trials might have different lengths.
This would cause much trouble, so shorter trials are padded with 0 at the beginning so that the array size matches with the 
maximal duration possible given the conditions. The 0s dont affect the behavior of the network or the learning rules.
The length of each trial can then be accessed using the trial_duration output.

If n_cues contain more than one value, or/and max_delay-min_delay>1, the trials might have different lengths. This would cause much trouble, so shorter trials are padded with 0 at the beginning so that the array size matches with the maximal duration possible given the conditions. The 0s dont affect the behavior of the network or the learning rules. The length of each trial can then be accessed using the trial_duration output

Parameters:
-----------
    n_batches: int 
        The total number of batches to generate.
    batch_size: int 
        The size of each batch.
    seed: int or None
        Seed for RNG. If a seed is passed, function will always yield the same group of batches when called, but batches will still be different within each other.
    n_cues: List of ints
        List contain the number of possible cues in a trial. If a single value is passed, then all trials will have the same number of cues, equal to the passed value.
        If the list contains more than one value, then for each trial the number of cues in randomly picked from the options with uniform distribution
    min_delay: int
        The minimum delay between cues and decision time. For each trial, the delay time is picked randomly between min_delay and max_delay, with unifor distribution
    max_delay: int
        The maximum delay between cues and decision time. For each trial, the delay time is picked randomly between min_delay and max_delay, with unifor distribution    
    n_population: int
        The number of neurons in each population. Default is 10.
    f_input: int
        The input firing rate for cues in Hz. Default is 40 Hz.
    f_background: int
        The background firing rate in Hz. Default is 10 Hz.
    t_cue: int
        time duration of a cue in ms. Default is 100ms
    t_cue_spacing: int
        time interval between the start of a cue and the start of the next in ms. Default is 150ms. The space between two cues is given by t_cue_spacing - t_cue. Defa
    p:  float
        Probability of any cue being on left side. Default is 0.5
    fixation_time: int
        Duration of the fixation period in ms.  Default is 50 ms
    input_mode: str
        Decides between two modes. In original, inputs have 4 different populations. Left, right, decision time and background noise. The populations coding for cues and decision time have no noise built in
        and respond only to when a respective cue is presented (or decision time for the decision time channel). In the modified task, inputs have 3 populations: left, right and decision time. The background
        noise is built into this 3 populations, instead of a dedicated background noise population 
    delay_time: int
        Duration of the delay period in ms. Default is 750ms.

    dt: int
        bad name for the variable. It is supposed to convert frequencies in Hz to spike/ms, since time units are in ms

    Yields
    dict: A dictionary containing:
        "input": Array (batch_size, trial_length, 4 * n_neurons_population) for "original" mode (batch_size, trial_length, 3 * n_neurons_population) for "modified" mode
            The spike train of input channels.
        "label": Array  (batch_size, n_t, 1)
            The labels for each batch, indicating if the most cues were on left (0) or right (1).
        "trial_duration": Array (batch_size,)
            Duration of each trial in the batch (might be different for trial depending on values of n_cues, min_delay and max_delay). 
    """



    if type(n_cues) is int:
        n_cues = [n_cues]
        
    f_background /= 1000
    f_input /= 1000
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
        labels = one_hot(labels, 2)
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
        labels = one_hot(labels, 2)
        for start_idx in range(0, n_batches, batch_size):
            end_idx = min(start_idx + batch_size, n_batches)
            batch_inputs = inputs[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch_trial_duration = trial_duration[start_idx:end_idx]
            yield {"input": batch_inputs, "label": batch_labels, 'trial_duration':batch_trial_duration}
        
    

def pattern_generation(n_batches:int, batch_size:int, seed:int, frequencies:List[float], 
                       n_population:int, f_input:float, trial_dur:int):
    """
    Generate batches of data for a pattern generation task.
    
    Task description:
    The network receives input from n_population neurons firing at f_input Hz during whole trial. The goal of the network
    is to generate an output that matches a fixed signal, consisting of a weight sum of sinusoids of different frequencies.
    

    Parameters:
    -----------
    n_batches: int 
        The total number of batches to generate.
    batch_size: int 
        The size of each batch.
    seed: int or None
        Seed for RNG. If a seed is passed, function will always yield the same group of batches when called, 
        but batches will still be different within each other.
    frequencies: List of floats
        Frequencies of sinusoids used to create output goal
    n_population: int
        The number of neurons in input population
    f_input: int
        The input firing rate for cues in Hz.
    trial_dur: int
        Duration of trial in ms
    

    Yields
    ------
    dict: A dictionary containing:
        "input": Array (batch_size, trial_dur, n_neurons_population) 
            The spike train of input channels.
        "label": Array  (batch_size, n_t)
            The weighted sum of sinusoids with trial_duration length. Every trial has the same sequence
        "trial_duration": Array (batch_size,)
            Duration of each trial in the batch (same for all trials=trial_dur)
    """

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

    # generate random weights
    weights = rng.uniform(size=5)
    # normalize weights to sum up to 1
    weights /= np.sum(weights)
    
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
   