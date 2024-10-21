from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import sys
import os



@dataclass
class Task:
    task_name: str = MISSING
    task_type: str = MISSING    
    LS_avail: int = MISSING



@dataclass
class CueAccumulation(Task):
    task_name: str = "cue_accumulation"
    task_type: str = "classification"
    LS_avail: int = 150
    n_cues: List[int] = field(default_factory=lambda: [7])
    seed: Optional[int] = 42
    min_delay: int = 1500
    max_delay:int = 1501
    f_input: float = 40.
    f_background: float = 10.
    t_cue: int = 100
    t_cue_spacing: int = 150
    p: float = 0.5
    input_mode: str = "original"
    dt: int = 1000
   
                       
@dataclass
class DelayedMatch(Task):
    task_name: str = "delayed_match"
    task_type: str = "classification"
    LS_avail: int = 150    
    seed: Optional[int] = None
    f_input: float = 40.
    f_background: float = 10.
    fixation_time: int = 50
    cue_time: int = 150
    delay_time: int=350 
    decision_time: int=150
    p: float = 0.5
    

@dataclass
class PatternGeneration(Task):
    task_name: str = "pattern_generation"
    task_type: str = "regression"
    LS_avail: int = 0    
    seed: Optional[int] = None
    frequencies: List[float] = field(default_factory=lambda: [0.5, 1., 2., 3., 4.])
    f_input: float = 10.
    trial_dur: int = 2000


                                                         

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="task_class/task",
        name="cue_accumulation",
        node=CueAccumulation,
    )
    cs.store(
    group="task_class/task",
    name="delayed_match",
    node=DelayedMatch,
    )

    cs.store(
    group="task_class/task",
    name="pattern_generation",
    node=PatternGeneration,
    )
