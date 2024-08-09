from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class NetworkArchitecture:
    n_neurons_channel: int = 10 
    n_ALIF: int = 50
    n_LIF: int = 60
    n_out: int = 2
    local_connectivity: bool = True
    sigma: float = 0.012
    feedback: str = "Symmetric"
    
    
@dataclass
class NetworkParams:
    thr: float = 0.6
    tau_m: float = 20
    tau_out: float = 20
    tau_adaptation: float = 2000
    beta: float = 1.7
    bias_out: float = 0.0
    gamma: float = 0.3
    refractory_period: int = 5
    w_init_gain: Tuple[float,...] = field(default_factory=lambda: (0.5, 0.1, 0.5, 0.5))
    dt: float = 1 # time step of simulation in ms  
    seed: int = 42
    
@dataclass
class TrainParams:
    lr: float = 0.001 # learning rate
    train_batch_size: int = 64 # how many samples before weight update
    train_sub_batch_size: int = 8 # Due to memory limitations, load subsets of the batch at each time
    test_batch_size: int = 512
    test_sub_batch_size: int = 8 # Due to memory limitations, load subsets of the batch at each time 
    iterations: int = 2000  # how many time steps
    stop_criteria: float = 0.95
    f_target:float = 10.
    c_reg: float = 1.
    
@dataclass
class SaveFiles:
    experiment_name: str = MISSING
    condition: str = MISSING


