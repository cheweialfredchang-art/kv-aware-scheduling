from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

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


# ---------------- Builders ----------------

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


def validate_cfg(cfg: dict):
    for k in ("scenario_id", "nodes", "network", "workload"):
        if k not in cfg:
            raise KeyError(f"Missing '{k}' in config")


# ---------------- Run once ----------------

def run_once(base, scenario, scheduler, ticks, seed, out_root):
    base_cfg = load_yaml(base)
    sc_cfg = load_yaml(scenario)
    cfg = deep_merge(base_cfg, sc_cfg)
    validate_cfg(cfg)

    nodes = build_nodes(cfg["nodes"])
    network = build_network(cfg["network"])
    sched = get_scheduler(scheduler, net=network, seed=seed)

    reqs = make_synthetic_requests(
        n=int(cfg["workload"].get("num_requests", 50)),
        owner_nodes=list(nodes.keys()),
    )

    metrics = run_simulation(
        requests=reqs,
        nodes=nodes,
        scheduler=sched,
        ticks=ticks,
        seed=seed,
    )

    out_dir = Path(out_root) / cfg["scenario_id"] / scheduler / f"seed={seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.json"
    write_summary(summary, metrics, include_samples=True)

    row = metrics.summary()
    row.update({
        "scenario": cfg["scenario_id"],
        "scheduler": scheduler,
        "seed": seed,
        "ticks": ticks,
        "summary_path": str(summary),
    })
    return row


# ---------------- Commands ----------------

def cmd_run(args):
    row = run_once(
        args.base, args.scenario, args.scheduler,
        args.ticks, args.seed, args.out
    )
    print(row)
    return 0


def cmd_batch(args):
    scenarios = []
    p = Path(args.scenarios)
    if p.exists() and p.is_dir():
        scenarios = sorted(p.glob("*.yaml"))
    else:
        scenarios = [Path(x) for x in args.scenarios.split(",")]

    schedulers = [s.strip() for s in args.schedulers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    rows = []
    for sc in scenarios:
        for sch in schedulers:
            for seed in seeds:
                try:
                    row = run_once(
                        args.base, str(sc), sch,
                        args.ticks, seed, args.out
                    )
                    rows.append(row)
                    print(f"[OK] {sc.name} {sch} seed={seed}")
                except Exception as e:
                    print(f"[ERROR] {sc.name} {sch} seed={seed}: {e}")
                    if args.fail_fast:
                        raise

    index = Path(args.index)
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Optional plots
    if not args.no_plots:
        try:
            from kvsched.evaluation.plots import generate_all_plots
            generate_all_plots(index, args.figdir)
        except ImportError:
            print("⚠ pandas/matplotlib not installed; skip plotting")

    return 0


# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser("kvsched")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--base", default="configs/base.yaml")
    pr.add_argument("--scenario", required=True)
    pr.add_argument("--scheduler", default="kv_heuristic")
    pr.add_argument("--ticks", type=int, default=1000)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default="results/raw")
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("batch")
    pb.add_argument("--base", default="configs/base.yaml")
    pb.add_argument("--scenarios", required=True)
    pb.add_argument("--schedulers", default="kv_heuristic,load_only,kv_only,random")
    pb.add_argument("--seeds", default="0,1,2")
    pb.add_argument("--ticks", type=int, default=1000)
    pb.add_argument("--out", default="results/raw")
    pb.add_argument("--index", default="results/aggregated/index.csv")
    pb.add_argument("--figdir", default="results/figures")
    pb.add_argument("--no-plots", action="store_true")
    pb.add_argument("--fail-fast", action="store_true")
    pb.set_defaults(func=cmd_batch)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
