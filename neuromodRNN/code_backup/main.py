import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path, get_object
from omegaconf import DictConfig, OmegaConf, MISSING
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional
import os
import sys
from hydra.plugins.sweeper import Sweeper
from hydra.core.plugins import Plugins

file_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
from src.train import train_cue_accumulation
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
#from conf.task.tasks_setup import Task
#from conf.task.tasks_setup import register_configs



@dataclass
class NetworkArchitecture:
    n_neurons_channel: int = 10 
    n_ALIF: int = 50
    n_LIF: int = 50
    n_out: int = 2
    local_connectivity: bool = True
    sigma: float = 0.012
    local_connectivity_key: int = 42
    feedback: str = "Symmetric"
    FeedBack_key: int = 2 
    
@dataclass
class NetworkParams:
    thr: float = 0.6
    tau_m: float = 20
    tau_out: float = 20
    tau_adaptation: float = 2000
    beta: Optional[list] = None
    bias_out: float = 0.0
    gamma: float = 0.3
    w_init_gain: Tuple[float,...] = field(default_factory=lambda: (0.5, 0.1, 0.5, 0.5))
    dt: float = 1 # time step of simulation in ms  
    state_key: int = 0
    
@dataclass
class TrainParams:
    lr: float = 0.001 # learning rate
    train_batch_size: int = 64 # how many samples before weight update
    train_sub_batch_size: int = 8 # Due to memory limitations, load subsets of the batch at each time
    test_batch_size: int = 512
    test_sub_batch_size: int = 8 # Due to memory limitations, load subsets of the batch at each time 
    iterations: int = 2000  # how many time steps
    stop_criteria: float = 0.9

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
      
    train_cue_accumulation.train_and_evaluate(cfg)
    

if __name__ == "__main__":    
    main()