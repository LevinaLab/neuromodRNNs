"""
The three implemented tasks (cue_accumulation, delayed_match, pattern_generation)
only differ in the fields of this object; the training loop itself is identical.

To add a new task:
 
  1. Write a batch generator in `src/modRNN/tasks.py`.
  2. Create a `TaskSpec` in your task's `train/train_<task>.py`.
  3. Hand that spec to `common.train_and_evaluate(cfg, spec)`.
"""
 
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable
 
from flax.typing import Array
 
 
# --- Type aliases for readability ------------------------------------------
BatchGenerator = Callable[..., Iterable[Dict[str, Array]]]
LossFn = Callable[..., Array]
MetricsFn = Callable[..., Any]
ExamplePlotFn = Callable[..., None]
 
 
@dataclass(frozen=True)
class TaskSpec:
    """
    Task-specific hooks used by the shared training pipeline.
 
    Attributes
    ----------
    name
        Short identifier, used for logging.
    task_type
        "classification" or "regression". Controls which code path
        `learning_rules.compute_grads` takes.
    input_dim_from_cfg
        Returns the number of input channels `n_in` given the config.
    make_train_batch
        Callable(cfg, *, epoch) -> iterable of batches.
    make_eval_batch
        Callable(cfg) -> iterable of batches. The result is list()-ified
        by the shared pipeline so it can be reused each evaluation.
    make_test_batch
        Callable(cfg, *, offset) -> iterable of batches, used for the
        early-stopping confirmation. `offset` is an integer 0, 1, 2 that
        lets the task decide whether to vary the test set (classification
        tasks do; pattern_generation ignores it).
    optimization_loss
        Loss used by autodiff (BPTT, e_prop_autodiff).
    metrics_class
        The PyTreeNode subclass used for this task's metrics.
    compute_metrics
        Reduces (labels, predictions) -> a `metrics_class` instance.
    metric_names
        Names of the scalar metrics this task reports, in display order.
        The first must be "loss".
    early_stop_metric
        Which field to use for early stopping.
    early_stop_better
        "higher" or "lower" — direction in which the metric is improving.
    log_format
        printf-style format for per-epoch logging.
    log_scale
        How to scale each metric for display (e.g. accuracy * 100).
    plot_examples
        Draws and saves the per-task example figures at end of training.
    plot_example_count
        How many examples to plot.
    """
    name: str
    task_type: str  # "classification" | "regression"
    input_dim_from_cfg: Callable[[Any], int]
    make_train_batch: Callable[..., Iterable[Dict[str, Array]]]
    make_eval_batch: Callable[[Any], Iterable[Dict[str, Array]]]
    make_test_batch: Callable[..., Iterable[Dict[str, Array]]]
    optimization_loss: LossFn
    metrics_class: type
    compute_metrics: MetricsFn
    metric_names: tuple
    early_stop_metric: str
    early_stop_better: str  # "higher" | "lower"
    log_format: str
    log_scale: tuple
    plot_examples: ExamplePlotFn
    plot_example_count: int = 3
 
    def __post_init__(self):
        # Lightweight sanity checks — caught at import time, not mid-training.
        if self.task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got {self.task_type!r}"
            )
        if self.early_stop_better not in ("higher", "lower"):
            raise ValueError(
                f"early_stop_better must be 'higher' or 'lower', "
                f"got {self.early_stop_better!r}"
            )
        if self.early_stop_metric not in self.metric_names:
            raise ValueError(
                f"early_stop_metric {self.early_stop_metric!r} not in "
                f"metric_names {self.metric_names}"
            )
        if self.metric_names[0] != "loss":
            raise ValueError(
                f"First metric must be 'loss' by convention, "
                f"got metric_names={self.metric_names}"
            )
        if len(self.log_scale) != len(self.metric_names):
            raise ValueError(
                f"log_scale (len {len(self.log_scale)}) must match "
                f"metric_names (len {len(self.metric_names)})"
            )
 
    def early_stop_satisfied(self, value: float, criterion: float) -> bool:
        """True if the metric `value` meets the early-stopping `criterion`."""
        if self.early_stop_better == "higher":
            return value > criterion
        return value < criterion