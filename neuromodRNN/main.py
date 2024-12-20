import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path, get_object
from omegaconf import DictConfig, OmegaConf, MISSING
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional
import os
import sys
import logging
from jax import config, numpy as jnp
#config.update("jax_enable_x64", True)
#config.update("jax_default_matmul_precision", "float32")

file_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
from src.train import train_cue_accumulation, train_delayed_match, train_pattern_generation

from src.config.config_dataclass import NetworkArchitecture, NetworkParams, TrainParams, SaveFiles
from src.config.tasks_dataclass import Task, register_configs
from hydra.types import TaskFunction
from typing import (
  Any,
  Callable,
  Dict,
  List,
  Optional,
  Sequence,
  Tuple,
  Iterable  
 )
from flax.typing import Dtype
#from conf.task.tasks_setup import Task
#from conf.task.tasks_setup import register_configs



@dataclass
class NetworkArchitecture:
    n_neurons_channel: int = 10 
    n_ALIF: int = 50
    n_LIF: int = 50
    n_out: int = 2
    connectivity_rec_layer: str = "local"
    sigma: float = 0.012
    feedback: str = "Symmetric"
    gridshape: Tuple[int, int] = (10, 10)
    n_neuromodulators: int =1
    sparse_input: bool = True
    sparse_readout: bool = True


@dataclass
class NetworkParams:
    thr: float = 0.03
    tau_m: float = 20
    tau_out: float = 20
    tau_adaptation: float = 2000
    beta: float = 1.8
    bias_out: float = 0.0
    gamma: float = 0.3
    refractory_period: int = 5
    w_init_gain: Tuple[float,...] = field(default_factory=lambda: (1.,1.,1.,1.))
    dt: float = 1 # time step of simulation in ms  
    seed: int = 42
    k: float = 0 # decay of diffusion
    radius:int = 1 # radius of difussion kernel,should probably be kept as one
    input_sparsity: float = 0.1
    readout_sparsity: float = 0.1
    recurrent_sparsity: float = 0.1
    
@dataclass
class TrainParams:
    lr: float = 0.0025 # learning rate
    train_batch_size: int = 64
    # how many samples before weight update
    train_mini_batch_size: int = 8 # Due to memory limitations, load subsets of the batch at each time
    test_batch_size: int = 512
    test_mini_batch_size: int = 8 # Due to memory limitations, load subsets of the batch at each time 
    iterations: int = 2000  # how many time steps
    stop_criteria: float = 1.0
    f_target:float = 10.
    c_reg: float = 1.0
    learning_rule:str = 'e_prop_hardcoded'
    shuffle:bool = False # shuffle or not the diffusion grid. Only relevant for diffusion
    test_grads:bool = False # only used to compare autodiff with hardcoded during training (P.S. it only works appropriately if learning rule is e_prop_autodiff, otherwise BPTT will be performed)
    
    
@dataclass
class SaveFiles:
    experiment_name: str = MISSING
    condition: str = MISSING



@dataclass
class ConfigTrain:
    net_arch: NetworkArchitecture = field(default_factory=NetworkArchitecture)
    net_params: NetworkParams  = field(default_factory=NetworkParams)
    train_params: TrainParams  = field(default_factory=TrainParams)
    task: Task  = MISSING
    save_paths: SaveFiles  = field(default_factory=SaveFiles)

cs = ConfigStore.instance()
cs.store(name='base_config', node=ConfigTrain)
register_configs()




@hydra.main(version_base=None, config_path="conf",config_name="config")
def main(cfg: ConfigTrain) -> None:
    
    if cfg.task.task_name == "cue_accumulation":
        
        train_cue_accumulation.train_and_evaluate(cfg)
    elif cfg.task.task_name == "delayed_match":
        train_delayed_match.train_and_evaluate(cfg)
    
    elif cfg.task.task_name == "pattern_generation":
        train_pattern_generation.train_and_evaluate(cfg)
        

    else:
        logger = logging.getLogger(__name__)
        logger.error("The requested task hasnt been implemented")
if __name__ == "__main__":    
    main()
