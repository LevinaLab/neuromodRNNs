import argparse
import train_cue_accumulation
import train_delayed_match



def main():
    
    # Training Configuration
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--dataset', type=str, choices = ['cue_accumulation'], default='cue_accumulation', help='Choice of the dataset')
    parser.add_argument('--shuffle', action='store_true', default=True, help='Enables shuffling sample order in datasets after each epoch')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--train_len', type=int, default=64, help='Number of sample in training data per epoch')
    parser.add_argument('--test_len', type=int, default=512, help='Number of sample in testing data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--curriculum', nargs='+', type=int, default=[7], help='Curriculum for training')
    
    # Network Architecture
    parser.add_argument('--n_LIF', type=int, default=50, help='Number of recurrent LIF units')
    parser.add_argument('--n_ALIF', type=int, default=50, help='Number of recurrent ALIF units') 
    parser.add_argument('--n_in', type=int, default=40, help='Number of input units') 
    parser.add_argument('--n_out', type=int, default=2, help='Number of readout units') 

    # Network Hyperparams
    parser.add_argument('--thr', type=float, default=0.6, help='Firing threshold in the recurrent layer')
    parser.add_argument('--tau_m', type=float, default=20, help='Membrane potential leakage time constant in the recurrent layer (in ms)') # in this code it was originally 2000ms, but with ALIF can use the one from original paper: 20ms
    parser.add_argument('--tau-out', type=float, default=20, help='Membrane potential leakage time constant in the output layer (in ms)')
    parser.add_argument('--tau-adaptation', type=float, default=2000, help='Firing Threshold adaptation time constant in the recurrent (in ms)')
    parser.add_argument('--beta', type=float, default=None, help='Multiplicative factor for threshold adaptation ') # So far should be passed as None, but need to change to also accept valid lists or customized values
    parser.add_argument('--bias_out', type=float, default=0.0, help='Bias of the output layer') # so far, update of bias is not coded, should not pass any value different from 0
    parser.add_argument('--gamma', type=float, default=0.3, help='Surrogate derivative magnitude parameter')
    parser.add_argument('--w_init_gain', type=float, nargs='+', default=(0.5,0.1,0.5, 0.5), help='Gain parameter for the He Normal initialization of the input, recurrent and output layer weights and feedback weight')    
    parser.add_argument('--dt', type=float, default=1, help='time step of simulation in ms')                      
    parser.add_argument('--t_crop', type=int, default=1, help='time where learning signal is available counting from the end (value of 150ms means that only at last 150ms the learning signal is avaialble). Use 0 for learning signal avaialble through the whole duration')                 
    parser.add_argument('--feedback', type=str, choices = ['Symmetric', 'Random'], default='Symmetric', help='Choice of feedback type. Symmetric feedback weights are same feedforward, while Random they are drawn randomly.') # The way it is coded, if you pass anything that it is not 'Symmetric', it will work as random
    parser.add_argument('--classification', type=str, choices = ['classification'], default='classification', help='Nature of the task: classification or regression') # So far only classification coded
    parser.add_argument('--FeedBack_key', type=int, default=1, help='Random seed for drawing feedback weight in random mode') 
    parser.add_argument('--state_key', type=int, default=42, help='Random seed for initializing state')     
    parser.add_argument('--local_connectivity', action='store_true', default=True, help='If model should have local connectivity') 
    parser.add_argument('--local_connectivity_key', type=int, default=0, help='Seed for key used in initialization of local connectivity mask') 
    parser.add_argument('--sigma', type=float, default=0.012, help='Parameter that controls probability of connection as funciton of distance in local connectivity mode') 
    parser.add_argument('--save_checkpoint', type=str, default=r"\Users\j1559\Documents\Tuebingen\SS_24\MasterThesis\eprop_jax\checkpoint", help='Where to save checkpoints') 
    #parser.add_argument('--c_reg', type=float, default=0, help='Random seed for initializing state')                
       
    #parser.add_argument('--f_target', type=float, default=10, help='Random seed for initializing state')
    args = parser.parse_args()
                       


    train_cue_accumulation.train_and_evaluate(args)
  
if __name__ == '__main__':
    main()
    
