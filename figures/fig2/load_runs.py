"""
Output structure expected by load_runs :

  Alignment experiments:
    <root>/<task>/<experiment_name>/<seed_dir>/align_info/
        iterations.pkl
        <tag>_angle_per_layer.pkl     (when metric="angle")
        <tag>_gradnorm_rule.pkl       (when metric="norm")
"""
from __future__ import annotations
 
import logging
import pickle
from pathlib import Path
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

# Default layer to plot 
DEFAULT_LAYER = "recurrent_weights"
 
# Map short layer names (CLI-friendly) to the dict keys used in
# <tag>_angle_per_layer.pkl files.
_LAYER_KEYS = {
    "input_weights":     "ALIFCell_0.input_weights",
    "recurrent_weights": "ALIFCell_0.recurrent_weights",
    "readout_weights":   "ReadOut_0.readout_weights",
}
 
 
# Per-metric configuration: which file suffix to look for, what unit
# label to put on the y-axis, and how to convert raw stored values to
# display values. Angles are stored in radians and shown in degrees;
# norms are stored and shown as-is.
_METRICS = {
    "angle": {
        "file_suffix": "_angle_per_layer.pkl",
        "convert":     lambda x: np.asarray(x) * 180.0 / np.pi,
        "unit_label":  "alignment angle (°)",
        "diff_label":  "Δ angle vs. reference (°)",
    },
    "norm": {
        "file_suffix": "_gradnorm_rule.pkl",
        "convert":     lambda x: np.asarray(x, dtype=float),
        "unit_label":  "gradient norm",
        "diff_label":  "Δ gradient norm vs. reference",
    },
}
 
 
def _metric_config(metric: str) -> dict[str, Any]:
    if metric not in _METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Choose from {sorted(_METRICS)}."
        )
    return _METRICS[metric]
 
 
# =============================================================================
# Data loading
# =============================================================================

def _read_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
def _list_seed_dirs(experiment_dir: Path) -> list[Path]:
    """Return all subdirectories of `experiment_dir` that contain align_info/."""
    return sorted(
        d for d in experiment_dir.iterdir()
        if d.is_dir() and (d / "align_info").is_dir()
    )
 
 
def _discover_tags(seed_dir: Path, suffix: str) -> list[str]:
    """
    Return the list of comparison tags by examining the file names in
    <seed_dir>/align_info/. Each tag has a corresponding
    <tag><suffix> file (e.g. <tag>_angle_per_layer.pkl).
    """
    align_dir = seed_dir / "align_info"
    tags = []
    for f in align_dir.iterdir():
        if f.name.endswith(suffix):
            tags.append(f.name[: -len(suffix)])
    return sorted(tags)
 
 
def load_runs(
    root: Path,
    experiment_name: str,
    tasks: list[str],
    layer: str = DEFAULT_LAYER,
    metric: str = "angle",
) -> dict[str, dict]:
    """
    Load alignment angles or gradient norms for the given tasks and
    the given layer.
 
    Parameters
    ----------
    metric
        Either "angle" (loads <tag>_angle_per_layer.pkl, values in
        radians) or "norm" (loads <tag>_gradnorm_rule.pkl, values
        as-is).
 
    Returns
    -------
    Dict structured as:
        {
            task_name: {
                "iterations": np.ndarray of shape (n_iterations,),
                "tags": list of comparison tags,
                "metric": str, the metric that was loaded,
                "values": {
                    tag: np.ndarray of shape (n_seeds, n_iterations),
                    ...
                },
                "n_seeds": int,
                "seeds": list of seed-dir names,
            },
            ...
        }
 
    Values are stored in their raw on-disk units (radians for angles,
    raw norm units for norms). Conversion to display units happens at
    plot time.
    """
    if layer not in _LAYER_KEYS:
        raise ValueError(
            f"Unknown layer {layer!r}. Choose from {sorted(_LAYER_KEYS)}."
        )
    layer_key = _LAYER_KEYS[layer]
 
    cfg = _metric_config(metric)
    file_suffix: str = cfg["file_suffix"]
 
    out: dict[str, dict] = {}
 
    for task in tasks:
        experiment_dir = root / task / experiment_name
        if not experiment_dir.is_dir():
            logger.warning(
                "Task %s: experiment dir %s does not exist; skipping.",
                task, experiment_dir,
            )
            continue
 
        seed_dirs = _list_seed_dirs(experiment_dir)
        if not seed_dirs:
            logger.warning("Task %s: no seed dirs found; skipping.", task)
            continue
 
        # Discover tags from the first seed dir; assume the rest are
        # consistent (warn if they aren't).
        tags = _discover_tags(seed_dirs[0], file_suffix)
        if not tags:
            logger.warning(
                "Task %s, seed %s: no <tag>%s files; skipping.",
                task, seed_dirs[0].name, file_suffix,
            )
            continue
 
        # Load the iteration list from the first seed; assume all seeds
        # share the same iteration grid (alignment experiment doesn't
        # early-stop, so this should hold).
        iterations_path = seed_dirs[0] / "align_info" / "iterations.pkl"
        if not iterations_path.is_file():
            logger.warning(
                "Task %s, seed %s: no iterations.pkl; skipping task.",
                task, seed_dirs[0].name,
            )
            continue
        iterations = np.asarray(_read_pickle(iterations_path))
        n_iterations = len(iterations)
 
        # For each tag, collect a (n_seeds, n_iterations) array of the
        # metric at the chosen layer, in raw on-disk units.
        values_by_tag: dict[str, list[list[float]]] = {tag: [] for tag in tags}
        used_seeds: list[str] = []
 
        for seed_dir in seed_dirs:
            seed_tags = _discover_tags(seed_dir, file_suffix)
            if seed_tags != tags:
                logger.warning(
                    "Task %s, seed %s: tag set %s differs from first-seed "
                    "tag set %s; skipping this seed.",
                    task, seed_dir.name, seed_tags, tags,
                )
                continue
 
            seed_iterations_path = seed_dir / "align_info" / "iterations.pkl"
            seed_iterations = np.asarray(_read_pickle(seed_iterations_path))
            if not np.array_equal(seed_iterations, iterations):
                logger.warning(
                    "Task %s, seed %s: iterations differ from first seed; "
                    "skipping this seed.",
                    task, seed_dir.name,
                )
                continue
 
            ok = True
            seed_values: dict[str, list[float]] = {}
            for tag in tags:
                pkl_path = seed_dir / "align_info" / f"{tag}{file_suffix}"
                per_layer = _read_pickle(pkl_path)
                if layer_key not in per_layer:
                    logger.warning(
                        "Task %s, seed %s, tag %s: no key %r in %s pickle; "
                        "skipping seed.",
                        task, seed_dir.name, tag, layer_key, metric,
                    )
                    ok = False
                    break
                values = np.asarray(per_layer[layer_key], dtype=float)
                if len(values) != n_iterations:
                    logger.warning(
                        "Task %s, seed %s, tag %s: %d %s values but %d "
                        "iterations; skipping seed.",
                        task, seed_dir.name, tag, len(values), metric,
                        n_iterations,
                    )
                    ok = False
                    break
                seed_values[tag] = values
 
            if not ok:
                continue
 
            for tag in tags:
                values_by_tag[tag].append(seed_values[tag])
            used_seeds.append(seed_dir.name)
 
        if not used_seeds:
            logger.warning("Task %s: no usable seeds; skipping.", task)
            continue
 
        out[task] = {
            "iterations": iterations,
            "tags": tags,
            "metric": metric,
            "values": {
                tag: np.asarray(per_seed_lists)
                for tag, per_seed_lists in values_by_tag.items()
            },
            "n_seeds": len(used_seeds),
            "seeds": used_seeds,
        }
        logger.info(
            "Task %s (%s): loaded %d seed(s) x %d iteration(s) x %d tag(s).",
            task, metric, len(used_seeds), n_iterations, len(tags),
        )
 
    return out