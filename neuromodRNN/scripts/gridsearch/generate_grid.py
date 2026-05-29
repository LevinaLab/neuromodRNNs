"""
Generate a tab-separated file listing (lr, c_reg, seed) combinations.
Usage:
    python generate_grid.py \
        --output combinations/combinations_cue_accumulation.txt \
        --lrs 1e-4 5e-3 1e-3 1e-2 5e-2 \
        --cregs 0.0 1e-5 1e-4 5e-3 5e-2 \
        --seeds 32132 43244 41235 9934 23431
    
    python generate_grid.py \
        --output combinations/combinations_delayed_match.txt \
        --lrs 5e-4 5e-3 1e-2 \
        --cregs 1e-4 5e-3 1e-2 1e-1 \
        --seeds 32132 43244 41235 9934 23431

"""
 
from __future__ import annotations
 
import argparse
import itertools
import sys
from pathlib import Path
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the combinations file.",
    )
    parser.add_argument(
        "--lrs", nargs="+", required=True,
        help="Learning rate values (passed verbatim to Hydra; use '1e-3' etc.).",
    )
    parser.add_argument(
        "--cregs", nargs="+", required=True,
        help="c_reg values (passed verbatim to Hydra; use '1.0' etc.).",
    )
    parser.add_argument(
        "--seeds", nargs="+", required=True,
        help="Seed values (integers).",
    )
    args = parser.parse_args()
 
    # Validate seeds are integers.
    for s in args.seeds:
        try:
            int(s)
        except ValueError:
            print(f"ERROR: seed '{s}' is not an integer.", file=sys.stderr)
            sys.exit(1)
 
    args.output.parent.mkdir(parents=True, exist_ok=True)
 
    n_lines = 0
    with open(args.output, "w") as f:
        for lr, c_reg, seed in itertools.product(args.lrs, args.cregs, args.seeds):
            f.write(f"{lr}\t{c_reg}\t{seed}\n")
            n_lines += 1
 
    print(f"Wrote {n_lines} combinations to {args.output}")
    print(f"  {len(args.lrs)} learning rates: {args.lrs}")
    print(f"  {len(args.cregs)} c_reg values: {args.cregs}")
    print(f"  {len(args.seeds)} seeds: {args.seeds}")
 
 
if __name__ == "__main__":
    main()

