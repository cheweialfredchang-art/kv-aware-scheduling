from __future__ import annotations

import argparse
from pathlib import Path
import csv
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


# -------------------------------------------------
# Builders shared by run & batch
# -------------------------------------------------

def build_nodes(node_cfg: dict) -> dict[str, NodeState]:
    nodes: dict[str, NodeState] = {}
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
# Subcommand: run
# -------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    base_cfg = load_yaml(args.base)
    scenario_cfg = load_yaml(args.scenario)
    cfg = deep_merge(base_cfg, scenario_cfg)
    scenario_id = cfg["scenario_id"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = build_nodes(cfg["nodes"])
    network = build_network(cfg["network"])
    scheduler = get_scheduler(args.scheduler, net=network, seed=args.seed)

    requests = make_synthetic_requests(
        n=int(cfg.get("workload", {}).get("num_requests", 50)),
        owner_nodes=list(nodes.keys()),
    )

    metrics = run_simulation(
        requests=requests,
        nodes=nodes,
        scheduler=scheduler,
        ticks=args.ticks,
        seed=args.seed,
    )

    summary = metrics.summary()
    summary_path = out_dir / f"{scenario_id}_{args.scheduler}_seed{args.seed}.json"
    write_summary(summary_path, metrics)

    print("=" * 60)
    print(f"Scenario : {scenario_id}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Ticks    : {args.ticks}")
    print(f"Seed     : {args.seed}")
    print("-" * 60)
    print(f"P50 latency : {summary['p50_ms']:.2f} ms")
    print(f"P95 latency : {summary['p95_ms']:.2f} ms")
    print(f"P99 latency : {summary['p99_ms']:.2f} ms")
    print(f"KV migrations       : {int(summary['kv_migrations'])}")
    print(f"KV migration bytes  : {summary['kv_migration_bytes'] / (1024**3):.2f} GiB")
    print(f"Output -> {summary_path}")
    print("=" * 60)

    return 0


# -------------------------------------------------
# Subcommand: batch
# -------------------------------------------------

def _resolve_scenarios(s: str) -> List[Path]:
    p = Path(s)
    if p.exists() and p.is_dir():
        return sorted(p.glob("*.yaml"))
    return [Path(x.strip()) for x in s.split(",") if x.strip()]


def _run_one(base_cfg: dict, scenario_path: Path, scheduler_name: str, seed: int, ticks: int, out_root: Path) -> dict:
    scenario_cfg = load_yaml(scenario_path)
    cfg = deep_merge(base_cfg, scenario_cfg)
    scenario_id = cfg["scenario_id"]

    nodes = build_nodes(cfg["nodes"])
    network = build_network(cfg["network"])
    scheduler = get_scheduler(scheduler_name, net=network, seed=seed)

    requests = make_synthetic_requests(
        n=int(cfg.get("workload", {}).get("num_requests", 50)),
        owner_nodes=list(nodes.keys()),
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
        "summary_path": str(out_dir / "summary.json"),
    })
    return row


def cmd_batch(args: argparse.Namespace) -> int:
    base_cfg = load_yaml(args.base)
    scenario_paths = _resolve_scenarios(args.scenarios)
    schedulers = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    total = len(scenario_paths) * len(schedulers) * len(seeds)
    done = 0

    for sp in scenario_paths:
        for sch in schedulers:
            for seed in seeds:
                done += 1
                label = f"[{done}/{total}] scenario={sp.name} scheduler={sch} seed={seed}"
                try:
                    rows.append(_run_one(base_cfg, sp, sch, seed, args.ticks, out_root))
                    print(f"{label} -> OK")
                except Exception as e:
                    print(f"{label} -> ERROR: {e}")
                    if args.fail_fast:
                        raise

    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario", "scheduler", "seed", "ticks",
        "p50_ms", "p95_ms", "p99_ms",
        "kv_migrations", "kv_migration_bytes",
        "summary_path",
    ]

    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Index written to: {index_path}")
    print(f"Completed runs: {len(rows)} / {total}")
    return 0


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="kvsched",
        description="KV-aware scheduling research scaffold CLI (run/batch)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="Run a single scenario once")
    pr.add_argument("--base", default="configs/base.yaml", help="Base config YAML")
    pr.add_argument("--scenario", required=True, help="Scenario YAML path")
    pr.add_argument("--scheduler", default="kv_heuristic",
                    help="random|rr|load_only|kv_only|kv_heuristic")
    pr.add_argument("--ticks", type=int, default=1000)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default="results/raw", help="Output directory")
    pr.set_defaults(func=cmd_run)

    # batch
    pb = sub.add_parser("batch", help="Batch run scenarios × schedulers × seeds")
    pb.add_argument("--base", default="configs/base.yaml", help="Base config YAML")
    pb.add_argument("--scenarios", required=True,
                    help="Scenario directory OR comma-separated list of scenario YAMLs")
    pb.add_argument("--schedulers", default="kv_heuristic,load_only,kv_only,random",
                    help="Comma-separated scheduler names")
    pb.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    pb.add_argument("--ticks", type=int, default=1000)
    pb.add_argument("--out", default="results/raw", help="Output root directory")
    pb.add_argument("--index", default="results/aggregated/index.csv", help="CSV index output path")
    pb.add_argument("--fail-fast", action="store_true", help="Stop immediately on first error")
    pb.set_defaults(func=cmd_batch)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
