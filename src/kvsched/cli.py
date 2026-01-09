from __future__ import annotations
import argparse
from .experiments.run_one import run

def main():
    p = argparse.ArgumentParser(prog="kvsched", description="KV-aware scheduling research scaffold")
    p.add_argument("--scheduler", default="kv_heuristic", help="random|rr|load_only|kv_only|kv_heuristic")
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/raw/demo")
    args = p.parse_args()
    summary = run(scheduler_name=args.scheduler, ticks=args.ticks, seed=args.seed, out_dir=args.out)
    print(summary)

if __name__ == "__main__":
    main()
