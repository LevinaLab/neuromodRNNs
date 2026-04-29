"""Entry point. Composes a Hydra config and dispatches to the right train script."""
 
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
)
from src.config.config_dataclass import (
    NetworkArchitecture,
    NetworkParams,
    TrainParams,
    SaveFiles,
)
from src.config.tasks_dataclass import Task, register_configs
 
 
@dataclass
class ConfigTrain:
    """Top-level config schema. Combined with task and experiment YAMLs at runtime."""
    net_arch: NetworkArchitecture = field(default_factory=NetworkArchitecture)
    net_params: NetworkParams = field(default_factory=NetworkParams)
    train_params: TrainParams = field(default_factory=TrainParams)
    task: Task = MISSING
    save_paths: SaveFiles = field(default_factory=SaveFiles)
 
 
# Register ConfigTrain as the schema for "base_config", which is the
# starting point referenced by conf/config.yaml's defaults list.
cs = ConfigStore.instance()
cs.store(name="base_config", node=ConfigTrain)
# Register the three task subclasses for the "task" config group.
register_configs()
 
 
# Map from the task_name field to the train script's entry point. Adding
# a new task means: register its dataclass in tasks_dataclass.py, add a
# YAML in conf/task/, and add an entry here.
_TRAIN_ENTRYPOINTS = {
    "cue_accumulation": train_cue_accumulation.train_and_evaluate_entry,
    "delayed_match": train_delayed_match.train_and_evaluate_entry,
    "pattern_generation": train_pattern_generation.train_and_evaluate_entry,
}
 
 
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: ConfigTrain) -> None:
    logger = logging.getLogger(__name__)
    entrypoint = _TRAIN_ENTRYPOINTS.get(cfg.task.task_name)
    if entrypoint is None:
        logger.error(
            "Unknown task %r. Known tasks: %s",
            cfg.task.task_name,
            sorted(_TRAIN_ENTRYPOINTS.keys()),
        )
        return
    entrypoint(cfg)
 
 
if __name__ == "__main__":
    main()