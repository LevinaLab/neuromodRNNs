"""
src.modRNN.training — primitives + experiment-type orchestration.
 
Public API:
 
  TaskSpec                — bundle of task-specific hooks.
  TrainStateEProp         — Flax TrainState + e-prop fields.
  model_from_config       — build LSSN from a Hydra config.
  create_train_state      — initialize TrainState.
  epoch_seed_sequence     — deterministic per-epoch seed derivation.
  train_and_evaluate      — standard training experiment entry point.
 
Internal modules:
 
  primitives              — shared building blocks (model, state,
                            train_step, seed helpers).
  training_loop           — standard training experiment orchestration
                            (periodic eval, early stop, plotting).
  task_spec               — TaskSpec dataclass.
  plotting                — shared end-of-training plotting helpers.
 
Future experiment types live alongside training_loop as their own
modules (e.g. alignment.py), reusing primitives directly.
"""
 
from src.modRNN.training.primitives import (
    TrainStateEProp,
    create_train_state,
    epoch_seed_sequence,
    model_from_config,
)
from src.modRNN.training.task_spec import TaskSpec
from src.modRNN.training.training_loop import train_and_evaluate
 
__all__ = [
    "TaskSpec",
    "TrainStateEProp",
    "create_train_state",
    "epoch_seed_sequence",
    "model_from_config",
    "train_and_evaluate",
]