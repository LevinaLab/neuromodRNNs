# TODO: check if need traintestloader for early stop or something else, ow remove it

import torch
import numpy as np
import numpy.random as rd
import sys


class CueAccumulationDataset(torch.utils.data.Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation"""

    def __init__(self, args, type, n_cues=3):
      
        self.n_cues     = n_cues
        f0         = 40
        t_cue      = 100
        t_wait     = 0
        n_symbols  = 4
        p_group    = 0.5
        
        self.dt         = 1e-3
        self.t_interval = 150 #each cue is presented for 100s and they are separated by 50 ms, so 150ms per cue
        self.seq_len    = n_cues*self.t_interval + t_wait
        self.n_in       = args.n_in  
        self.n_out      = 2    # This is a binary classification task, so using two output units with a softmax activation redundant
        n_channel       = self.n_in // n_symbols
        prob0           = f0 * self.dt
        t_silent        = self.t_interval - t_cue
        
        if (type == 'train'):
            length = args.train_len
        else:
            length = args.test_len
            
    
        # Randomly assign group A and B
        prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
        idx = rd.choice([0, 1], length)
        probs = np.zeros((length, 2), dtype=np.float32)
        # Assign input spike probabilities
        probs[:, 0] = prob_choices[idx]
        probs[:, 1] = prob_choices[1 - idx]
    
        cue_assignments = np.zeros((length, self.n_cues), dtype=int)
        # For each example in batch, draw which cues are going to be active (left or right)
        for b in range(length):
            cue_assignments[b, :] = rd.choice([0, 1], self.n_cues, p=probs[b])
    
        # Generate input spikes
        input_spike_prob = np.zeros((length, self.seq_len, self.n_in))
        
        t_silent = self.t_interval - t_cue
        for b in range(length):
            for k in range(self.n_cues):
                # Input channels only fire when they are selected (left or right)
                c = cue_assignments[b, k]
                input_spike_prob[b, t_silent+k*self.t_interval:t_silent+k*self.t_interval+t_cue, c*n_channel:(c+1)*n_channel] = prob0
    
        # Recall cue and background noise
        input_spike_prob[:, -self.t_interval:, 2*n_channel:3*n_channel] = prob0
        input_spike_prob[:, :, 3*n_channel:] = prob0/4.
        input_spikes = generate_poisson_noise_np(input_spike_prob)
        self.x = torch.tensor(input_spikes).float()
        
        # Generate targets
        target_nums = np.zeros((length, self.seq_len), dtype=int)
        target_nums[:, :] = np.transpose(np.tile(np.sum(cue_assignments, axis=1) > int(self.n_cues/2), (self.seq_len, 1)))
        self.y = torch.tensor(target_nums).long() # Label encoded as 0 or 1 (for e-prop update, need to convert to one-hot encoded --see training). (n_batches, n_t), event though n_t dimension is unnecessary, since it is simply a copy
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {"input": self.x[index], "label": self.y[index]}
    
    
    
def load_dataset_cue_accumulation(args, n_cues=5, **kwargs):
    
    if args.dataset == "cue_accumulation":
        print("=== Loading cue evidence accumulation dataset...")
    
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)

    trainset = CueAccumulationDataset(args,"train", n_cues)
    testset  = CueAccumulationDataset(args,"test", n_cues)

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=args.shuffle, **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False       , **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False       , **kwargs)
    
    args.n_classes      = trainset.n_out
    args.n_steps        = trainset.seq_len
    args.n_inputs       = trainset.n_in
    args.dt             = trainset.dt
    args.classif        = True
    args.full_train_len = len(trainset)
    args.full_test_len  = len(testset)
    args.delay_targets  = trainset.t_interval # also defines the time of decision make, where the learning signal is available
    args.skip_test      = False
            
    print("Training set length: "+str(args.full_train_len))
    print("Test set length: "+str(args.full_test_len))
    return (train_loader, traintest_loader, test_loader)


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes




def delayed_match_task(n_batches,batch_size,n_neurons_population=10, background_f = 10, inputs_f = 40):
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

    fixation_time = 50 # ms
    cue_time = 150 # ms
    delay_time = 300 # ms
    trial_length = fixation_time + 2*cue_time + delay_time
    background_f = background_f / 1000 # assumes that firing rate was given in Hz
    inputs_f = inputs_f / 1000 # assumes that firing rate was given in Hz
    p = 0.5 # probability of any cue being equal to 1
    
    
    # generate cues
    cues_labels = np.random.binomial(n=1, p=p, size=(n_batches, 2)) # independently draw 1 or 0 from bernoulli distribution for each cue and batch
    
    # initialize input popuplations
    input_shape = (n_batches,trial_length, n_neurons_population)
    
    back_ground_channel = np.random.rand(n_batches,trial_length, n_neurons_population) < (background_f) # backgroud
    
    input_1_channel = np.zeros(input_shape)
   
    idx_1 = np.where(cues_labels[:, 0]== 1)[0]  # get batches where first cue is 1
    n_trials_first_on = np.sum(cues_labels[:,0])
    input_1_channel[idx_1,fixation_time:fixation_time+cue_time, :] = np.random.rand(n_trials_first_on,cue_time, n_neurons_population) < (inputs_f)

    input_2_channel = np.zeros(input_shape)
    idx_2 = np.where(cues_labels[:, 1]== 1)[0]  # get batches where first cue is 2   
    n_trials_second_on = np.sum(cues_labels[:,1]) 
    input_2_channel[idx_2,-cue_time:, :] = np.random.rand(n_trials_second_on,cue_time, n_neurons_population) < (inputs_f)
    
    inputs = np.concatenate((back_ground_channel, input_1_channel, input_2_channel), axis=2)
    
    
    # get labels
    labels = np.zeros(n_batches)
    match_idx = np.where(cues_labels[:,0] == cues_labels[:,1])[0]
    labels[match_idx] = 1

    for start_idx in range(0, n_batches, batch_size):
        end_idx = min(start_idx + batch_size, n_batches)
        batch_inputs = inputs[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        yield {"input": batch_inputs, "label": batch_labels}
        