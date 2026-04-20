"""
Shared training pipeline for LSSN tasks.
 
Typical use from a task-specific train script:
 
    from modRNN.training import TaskSpec, train_and_evaluate
 
    spec = TaskSpec(...)
 
    def main(cfg):
        train_and_evaluate(cfg, spec)
"""
 
from src.modRNN.training.task_spec import TaskSpec
from src.modRNN.training.common import (
    TrainStateEProp,
    model_from_config,
    create_train_state,
    train_and_evaluate,
)

__all__ = [
    "TaskSpec",
    "TrainStateEProp",
    "model_from_config",
    "create_train_state",
    "train_and_evaluate",
]

