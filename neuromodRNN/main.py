"""Entry point. Composes a Hydra config and dispatches to the right entry point."""
 
import logging
import os
import sys
from dataclasses import dataclass, field
 
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
 
# Make the project root importable so that `from src...` works regardless
# of where the script is invoked from. This is a workaround for the project
# not yet being pip-installable; revisit if a pyproject.toml is added.
file_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
 
from src.train import (
    train_cue_accumulation,
    train_delayed_match,
    train_pattern_generation,
    align_cue_accumulation,
    align_delayed_match,
    align_pattern_generation,
)
from src.config.config_dataclass import (
    NetworkArchitecture,
    NetworkParams,
    TrainParams,
    SaveFiles,
)
from src.config.tasks_dataclass import Task, register_configs
from src.config.alignment_dataclass import AlignmentConfig
 
 
@dataclass
class ConfigTrain:
    """Top-level config schema. Combined with task and experiment YAMLs at runtime."""
    net_arch: NetworkArchitecture = field(default_factory=NetworkArchitecture)
    net_params: NetworkParams = field(default_factory=NetworkParams)
    train_params: TrainParams = field(default_factory=TrainParams)
    task: Task = MISSING
    save_paths: SaveFiles = field(default_factory=SaveFiles)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    # "training" — standard learning experiment (train + eval).
    # "alignment" — train with BPTT, periodically measure rule alignment.
    experiment_type: str = "training"
 
 
# Register ConfigTrain as the schema for "base_config", which is the
# starting point referenced by conf/config.yaml's defaults list.
cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigTrain)
# Register the three task subclasses for the "task" config group.
register_configs()
 
 
# Map from (experiment_type, task_name) to the right entry point. Adding
# a new experiment type or task means adding entries here; the dispatcher
# below looks up the pair.
_ENTRYPOINTS = {
    ("training", "cue_accumulation"):    train_cue_accumulation.train_and_evaluate_entry,
    ("training", "delayed_match"):       train_delayed_match.train_and_evaluate_entry,
    ("training", "pattern_generation"):  train_pattern_generation.train_and_evaluate_entry,
    ("alignment", "cue_accumulation"):   align_cue_accumulation.train_and_evaluate_entry,
    ("alignment", "delayed_match"):      align_delayed_match.train_and_evaluate_entry,
    ("alignment", "pattern_generation"): align_pattern_generation.train_and_evaluate_entry,
}
 
 
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: ConfigTrain) -> None:
    logger = logging.getLogger(__name__)
    key = (cfg.experiment_type, cfg.task.task_name)
    entrypoint = _ENTRYPOINTS.get(key)
    if entrypoint is None:
        logger.error(
            "Unknown (experiment_type, task) pair %r. Known pairs: %s",
            key,
            sorted(_ENTRYPOINTS.keys()),
        )
        return
    entrypoint(cfg)
 
 
if __name__ == "__main__":
    main()