from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from kvsched.config.loader import load_yaml, deep_merge
from kvsched.models.node import NodeState, GPUDevice, NodeLoad
from kvsched.models.network import (
    NetworkModel, NetworkTopology, NetworkLink, TrafficProfile
)
from kvsched.models.notation import gib
from kvsched.scheduler.registry import get_scheduler
from kvsched.simulator.workload import make_synthetic_requests
from kvsched.simulator.engine import run_simulation
from kvsched.simulator.reporters import write_summary


# -------------------------------------------------
# Builders
# -------------------------------------------------

def build_nodes(node_cfg: dict) -> dict[str, NodeState]:
    nodes = {}
    for nid, cfg in node_cfg.items():
        nodes[nid] = NodeState(
            node_id=nid,
            gpu=GPUDevice(
                name=cfg["gpu"]["name"],
                vram_total_bytes=gib(cfg["gpu"]["vram_total_gb"]),
                vram_free_bytes=gib(cfg["gpu"]["vram_free_gb"]),
            ),
            load=NodeLoad(
                running_prefill=cfg.get("running_prefill", 0),
                queued_prefill=cfg.get("queued_prefill", 0),
                running_decode=cfg.get("running_decode", 0),
                queued_decode=cfg.get("queued_decode", 0),
            ),
            tags=cfg.get("tags", {}),
        )
    return nodes


def build_network(net_cfg: dict) -> NetworkModel:
    topo = NetworkTopology()
    for l in net_cfg.get("links", []):
        topo.add_link(NetworkLink(
            src=l["src"],
            dst=l["dst"],
            bandwidth_Gbps=l["bandwidth_Gbps"],
            rtt_ms=l["rtt_ms"],
        ))

    topo.traffic = TrafficProfile(
        default_util=net_cfg.get("default_util", 0.0),
        util_by_pair={
            tuple(k.split("->")): v
            for k, v in net_cfg.get("util_by_pair", {}).items()
        }
    )
    return NetworkModel(topo=topo)


# -------------------------------------------------
# Single run
# -------------------------------------------------

def run_one(
    base_cfg: dict,
    scenario_path: Path,
    scheduler_name: str,
    seed: int,
    ticks: int,
    out_root: Path
) -> dict:

    scenario_cfg = load_yaml(scenario_path)
    cfg = deep_merge(base_cfg, scenario_cfg)
    scenario_id = cfg["scenario_id"]

    nodes = build_nodes(cfg["nodes"])
    network = build_network(cfg["network"])
    scheduler = get_scheduler(scheduler_name, net=network, seed=seed)

    requests = make_synthetic_requests(
        n=cfg["workload"].get("num_requests", 50),
        owner_nodes=list(nodes.keys())
    )

    metrics = run_simulation(
        requests=requests,
        nodes=nodes,
        scheduler=scheduler,
        ticks=ticks,
        seed=seed,
    )

    out_dir = out_root / scenario_id / scheduler_name / f"seed={seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_summary(out_dir / "summary.json", metrics)

    row = metrics.summary()
    row.update({
        "scenario": scenario_id,
        "scheduler": scheduler_name,
        "seed": seed,
        "ticks": ticks,
        "path": str(out_dir),
    })
    return row


# -------------------------------------------------
# Batch entry
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser("kvsched-batch")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--scenarios", required=True,
                    help="Directory or comma-separated yaml list")
    ap.add_argument("--schedulers", default="kv_heuristic,load_only,kv_only")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--index", default="results/aggregated/index.csv")
    args = ap.parse_args()

    base_cfg = load_yaml(args.base)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if Path(args.scenarios).is_dir():
        scenario_files = sorted(Path(args.scenarios).glob("*.yaml"))
    else:
        scenario_files = [Path(x.strip()) for x in args.scenarios.split(",")]

    schedulers = [s.strip() for s in args.schedulers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    rows: List[dict] = []

    for sc in scenario_files:
        for sch in schedulers:
            for seed in seeds:
                print(f"[RUN] {sc.name} | {sch} | seed={seed}")
                rows.append(
                    run_one(
                        base_cfg, sc, sch, seed,
                        ticks=args.ticks,
                        out_root=out_root
                    )
                )

    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario", "scheduler", "seed", "ticks",
        "p50_ms", "p95_ms", "p99_ms",
        "kv_migrations", "kv_migration_bytes", "path"
    ]

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[DONE] index written to {index_path}")


if __name__ == "__main__":
    main()
