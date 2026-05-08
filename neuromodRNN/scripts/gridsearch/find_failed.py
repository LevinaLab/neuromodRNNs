#!/usr/bin/env python3
"""
Detect failed gridsearch runs and emit a retry combinations file.
 
For one (task, experiment) pair, scans every (lr, c_reg, seed) combination
in the original combinations file and checks whether the corresponding run
directory contains a complete training output. Writes the failed combinations
to a new file in the same format as the original — directly submittable
via launcher_grid.sh.
 
A run is "complete" if:
    <root>/<task>/gridsearch/<experiment>/lr_<lr>__creg_<c_reg>/seed_<seed>/train_info/loss_training.pkl
exists and is non-empty.
 
Anything else (whole directory missing, train_info missing, pickle missing,
zero-byte pickle) is treated as a failure and emitted.
 
Usage:
    python find_failed.py \\
        --combinations combinations/combinations_cue.txt \\
        --root outputs \\
        --task cue_accumulation \\
        --experiment e_prop_hardcoded \\
        --output combinations/combinations_cue_retry.txt
 
Then resubmit the retry file via the existing launcher:
    bash launcher_grid.sh combinations/combinations_cue_retry.txt \\
                          cue_accumulation e_prop_hardcoded
 
If the retry file is empty, no jobs need to be resubmitted.
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
from typing import List, Tuple
 
 
# Marker file used to determine "complete run." Loss is saved by every
# task, so this works regardless of cue_accumulation/delayed_match/
# pattern_generation. If you want a stricter check, add more markers
# (e.g., final figures) — but loss_training.pkl is sufficient for "did
# the run reach the end of training."
COMPLETION_MARKER = "loss_training.pkl"
 
 
def parse_combinations_file(path: Path) -> List[Tuple[str, str, str]]:
    """
    Read a combinations file and return a list of (lr, c_reg, seed) tuples.
 
    Skips blank/comment lines (consistent with launcher_grid.sh).
    """
    combinations = []

    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                print(
                    f"WARNING: line {lineno} of {path} does not have "
                    f"exactly 3 tab-separated fields; skipping: {line!r}",
                    file=sys.stderr,
                )
                continue
            lr, c_reg, seed = fields
            combinations.append((lr, c_reg, seed))
    return combinations
 
 
def is_run_complete(seed_dir: Path) -> bool:
    """Return True if the seed directory contains a complete training run."""
    marker = seed_dir / "train_info" / COMPLETION_MARKER
    if not marker.is_file():
        return False
    try:
        return marker.stat().st_size > 0
    except OSError:
        return False
 
 
def diagnose(seed_dir: Path) -> str:
    """One-line description of why a run is incomplete."""
    if not seed_dir.exists():
        return "seed dir missing"
    if not seed_dir.is_dir():
        return "seed path is not a directory"
    train_info = seed_dir / "train_info"
    if not train_info.is_dir():
        return "train_info missing"
    marker = train_info / COMPLETION_MARKER
    if not marker.is_file():
        return f"{COMPLETION_MARKER} missing"
    try:
        if marker.stat().st_size == 0:
            return f"{COMPLETION_MARKER} is empty"
    except OSError as exc:
        return f"stat error: {exc}"
    return "complete"
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combinations",
        type=Path,
        required=True,
        help="Original combinations file used for the gridsearch.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Output root (typically 'outputs').",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., cue_accumulation).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g., e_prop_hardcoded).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the retry combinations file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per failed combination with diagnosis.",
    )
    args = parser.parse_args()
 
    if not args.combinations.is_file():
        print(f"ERROR: combinations file not found: {args.combinations}",
              file=sys.stderr)
        sys.exit(1)
 
    base_dir = args.root / args.task / "gridsearch" / args.experiment
    if not base_dir.exists():
        # All combinations are failures — no runs even started.
        print(f"WARNING: no gridsearch directory exists at {base_dir}.")
        print("         Treating all combinations as failed.")
 
    combinations = parse_combinations_file(args.combinations)
    if not combinations:
        print(f"ERROR: combinations file is empty: {args.combinations}",
              file=sys.stderr)
        sys.exit(1)
 
    failed: List[Tuple[str, str, str]] = []
    completed_count = 0
    for lr, c_reg, seed in combinations:
        seed_dir = base_dir / f"lr_{lr}__creg_{c_reg}" / f"seed_{seed}"
        if is_run_complete(seed_dir):
            completed_count += 1
        else:
            failed.append((lr, c_reg, seed))
            if args.verbose:
                print(f"  FAIL  lr={lr}\tc_reg={c_reg}\tseed={seed}\t({diagnose(seed_dir)})")
 
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for lr, c_reg, seed in failed:
            f.write(f"{lr}\t{c_reg}\t{seed}\n")
 
    n_total = len(combinations)
    n_failed = len(failed)
    print()
    print(f"Total combinations:     {n_total}")
    print(f"Completed:              {completed_count}")
    print(f"Failed (need retry):    {n_failed}")
    print()
    print(f"Wrote {n_failed} failed combinations to {args.output}")
    if n_failed > 0:
        print()
        print("To resubmit the failures:")
        print(f"  bash launcher_grid.sh {args.output} {args.task} {args.experiment}")
 
 
if __name__ == "__main__":
    main()