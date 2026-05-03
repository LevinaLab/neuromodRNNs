"""
Schema for the non-task portion of the config: network architecture,
network parameters, training parameters, save paths.
 
The defaults declared here are the "schema fallbacks" — what you get if
no YAML file or CLI override applies. For the actual values used in real
experiments, see `conf/task/<task>.yaml` and `conf/experiment/<exp>.yaml`.
Notice that different tasks/experiments have different default configs,
so expect many parameters do be override

"""
 
from dataclasses import dataclass, field
from typing import Tuple
 
from omegaconf import MISSING
 
 
@dataclass
class NetworkArchitecture:
    """Recurrent network shape and connectivity scheme."""
    n_neurons_channel: int = 10
    n_ALIF: int = 50
    n_LIF: int = 50
    n_out: int = 2
    connectivity_rec_layer: str = "local"  # "local" | "sparse" | "full"
    sigma: float = 0.012  # local-connectivity decay scale
    feedback: str = "Symmetric" # "Symmetric" | "Random" | "Random_sparse"
    gridshape: Tuple[int, int] = (10, 10)
    n_neuromodulators: int = 1
    sparse_input: bool = True
    sparse_readout: bool = True
 
 
@dataclass
class NetworkParams:
    """ALIF cell parameters and weight-init / diffusion settings."""
    thr: float = 0.03
    tau_m: float = 20
    tau_out: float = 20
    tau_adaptation: float = 2000
    beta: float = 1.8
    bias_out: float = 0.0
    gamma: float = 0.3
    refractory_period: int = 5
    w_init_gain: Tuple[float, ...] = field(default_factory=lambda: (1., 1., 1., 1.))
    dt: float = 1  # simulation time step (ms)
    seed: int = 42
    k: float = 0  # diffusion decay
    radius: int = 1  # diffusion kernel radius
    input_sparsity: float = 0.1
    readout_sparsity: float = 0.1
    recurrent_sparsity: float = 0.1
 
 
@dataclass
class TrainParams:
    """Optimizer, batching, and training-loop control."""
    lr: float = 0.0025
    train_batch_size: int = 64
    # Sub-batch (mini-batch) size: number of trials processed per step;
    # gradient accumulation is used if train_batch_size > train_mini_batch_size.
    train_mini_batch_size: int = 8
    test_batch_size: int = 512
    test_mini_batch_size: int = 8
    iterations: int = 2000
    stop_criteria: float = 1.0
    f_target: float = 10.
    c_reg: float = 1.0
    learning_rule: str = "e_prop_hardcoded"  # "BPTT" | "e_prop_hardcoded" | "e_prop_autodiff" | "diffusion"
    diffusion_mode: str = "aligned" # one of: "aligned", "shuffled_per_step", "shuffled_fixed"
    test_grads: bool = False  # debug: compare autodiff vs hardcoded grads
 
 
@dataclass
class SaveFiles:
    """Per-job naming for output directory."""
    experiment_name: str = MISSING
    condition: str = MISSING