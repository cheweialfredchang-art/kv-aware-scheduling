from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import csv

from kvsched.config.loader import load_yaml, deep_merge
from kvsched.cli import run_once  # reuse core runner


def _dictify(x):
    return x if isinstance(x, dict) else {}


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    return deep_merge(cfg, overrides)


def run_suite(
    suite_yaml: str | Path,
    *,
    base_cfg: str | Path = "configs/base.yaml",
    out_root: str | Path = "results/raw",
    index_path: str | Path = "results/aggregated/suite_index.csv",
    fail_fast: bool = False,
) -> Path:
    suite = load_yaml(suite_yaml)
    common = _dictify(suite.get("common", {}))
    seeds = common.get("seeds", [0, 1, 2])
    ticks = int(common.get("ticks", 1000))
    schedulers = common.get("schedulers", ["kv_heuristic", "load_only", "kv_only", "random"])

    rows: List[Dict[str, Any]] = []

    for exp in suite.get("experiments", []):
        exp_id = exp.get("id", "EXP")
        scenario = exp.get("scenario")
        if scenario is None:
            raise ValueError(f"Experiment {exp_id} missing 'scenario'")

        sweeps = exp.get("sweeps", [{}]) or [{}]
        for si, override in enumerate(sweeps):
            label = override.get("label", f"{exp_id}_sweep{si}")
            # scheduler hyper-params for kv_heuristic can be provided under override.scheduler_params
            sched_params = _dictify(override.get("scheduler_params", {}))

            for sch in schedulers:
                for seed in seeds:
                    try:
                        row = run_once(
                            base=str(base_cfg),
                            scenario=str(scenario),
                            scheduler=str(sch),
                            ticks=int(override.get("ticks", ticks)),
                            seed=int(seed),
                            out_root=str(out_root),
                            prefetch_on=bool(override.get("prefetch", False)),
                            overlap_on=bool(override.get("overlap", False)),
                            strict_network=bool(override.get("strict_network", False)),
                        )
                        row["exp_id"] = exp_id
                        row["sweep_label"] = label

                        # attach scheduler hyper-params only for kv_heuristic (for analysis)
                        if sch in ("kv_heuristic", "kv_aware", "kaware"):
                            row["sched_weights"] = str(sched_params.get("weights", ""))
                            row["sched_policy"] = str(sched_params.get("policy", ""))

                        rows.append(row)
                        print(f"[OK] {exp_id} {label} {sch} seed={seed}")
                    except Exception as e:
                        print(f"[ERROR] {exp_id} {label} {sch} seed={seed}: {e}")
                        if fail_fast:
                            raise

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return index_path
