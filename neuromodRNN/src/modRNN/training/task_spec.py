"""
The three implemented tasks (cue_accumulation, delayed_match, pattern_generation)
only differ in the fields of this object; the training loop itself is identical.

To add a new task:
 
  1. Write a batch generator in `src/modRNN/tasks.py`.
  2. Create a `TaskSpec` in your task's `train/train_<task>.py`.
  3. Hand that spec to `common.train_and_evaluate(cfg, spec)`.
"""
 
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional
 
from flax.typing import Array
 
 
# A batch generator yields dicts with keys "input", "label", "trial_duration".
BatchGenerator = Callable[..., Iterable[Dict[str, Array]]]
 
# A loss function takes (logits, labels, z, c_reg, f_target, trial_length) -> scalar.
LossFn = Callable[..., Array]
 
# A metrics function takes (labels, predictions) -> Metrics pytree.
MetricsFn = Callable[..., Any]
 
# An example-plot function takes (cfg, state, eval_batch, figures_dir) -> None.
ExamplePlotFn = Callable[..., None]
 
 
@dataclass(frozen=True)
class TaskSpec:
    """
    Everything a task needs to supply to the shared training pipeline.
 
    Attributes
    ----------
    name
        Short identifier, used only for logging ("cue_accumulation", etc.).
    task_type
        Either "classification" or "regression". Controls which code path
        `learning_rules.compute_grads` takes (softmax vs raw logits).
    input_dim_from_cfg
        Returns the number of input channels `n_in` given the config.
        This is task-dependent because each task has its own input encoding
    make_batch
        Callable that, given (cfg, n_batches, batch_size, seed), returns an
        iterable of batches. 
    optimization_loss
        Loss used for autodiff gradients (cross-entropy for classification,
        MSE for regression, both plus firing-rate regularization).
    metrics_class
        The PyTreeNode subclass used for this task's metrics.
    compute_metrics
        Reduces (labels, predictions) to a single `metrics_class` instance.
    metric_names
        Names of the scalar metrics this task reports, in the order they are
        displayed/saved. E.g. ("loss", "accuracy") for classification,
        ("loss", "normalized_loss") for regression. The first must be "loss".
    early_stop_metric
        Which metric field to use for early stopping.
    early_stop_better
        "higher" if a higher value is better (accuracy), "lower" if lower is
        better (normalized_loss). Used to decide the direction of the
        stop_criteria comparison.
    log_format
        printf-style format for per-epoch logging, takes the metric values.
        E.g. "epoch %03d loss %.4f accuracy %.2f%%".
    log_scale
        How to scale each metric for display (e.g. accuracy * 100 for percent).
        Same length as metric_names.
    plot_examples
        Draws and saves the per-task example figures at end of training.
    plot_example_count
        How many examples to plot at end of training (usually 3).
    """
    name: str
    task_type: str  # "classification" | "regression"
    input_dim_from_cfg: Callable[[Any], int]
    make_batch: Callable[..., Iterable[Dict[str, Array]]]
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