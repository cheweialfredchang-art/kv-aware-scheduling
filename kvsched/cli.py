from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def build_network(net_cfg: dict, *, strict_network: bool = False) -> NetworkModel:
    topo = NetworkTopology(strict_network=strict_network)
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


# ---------------- Core runner ----------------

def run_once(base, scenario, scheduler, ticks, seed, out_root, *, prefetch_on=False, overlap_on=False, strict_network=False):
    base_cfg = load_yaml(base)
    sc_cfg = load_yaml(scenario)
    cfg = deep_merge(base_cfg, sc_cfg)
    validate_cfg(cfg)

    nodes = build_nodes(cfg["nodes"])
    network = build_network(cfg["network"], strict_network=strict_network)
    sched = get_scheduler(
        scheduler,
        net=network,
        seed=seed,
        prefetch_on=prefetch_on,
        overlap_on=overlap_on,
    )

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
        "prefetch": int(prefetch_on),
        "overlap": int(overlap_on),
        "strict_network": int(strict_network),
        "label": f"{scheduler}|pf{int(prefetch_on)}|ov{int(overlap_on)}",
        "summary_path": str(summary),
    })
    return row


# ---------------- Commands ----------------

def cmd_run(args):
    row = run_once(
        args.base, args.scenario, args.scheduler,
        args.ticks, args.seed, args.out,
        prefetch_on=args.prefetch,
        overlap_on=args.overlap,
        strict_network=args.strict_network,
    )
    print(row)
    return 0


def cmd_batch(args):
    p = Path(args.scenarios)
    scenarios = sorted(p.glob("*.yaml")) if p.is_dir() else [p]

    schedulers = [s.strip() for s in args.schedulers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    rows = []
    for sc in scenarios:
        for sch in schedulers:
            for seed in seeds:
                try:
                    row = run_once(
                        args.base, sc, sch,
                        args.ticks, seed, args.out,
                        prefetch_on=args.prefetch,
                        overlap_on=args.overlap,
                        strict_network=args.strict_network,
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

    if not args.no_plots:
        try:
            from kvsched.evaluation.plots import generate_all_plots
            generate_all_plots(index, args.figdir)
        except ImportError:
            print("⚠ plotting skipped (pandas/matplotlib not installed)")
    return 0


def cmd_lint_network(args):
    from kvsched.tools.lint_network import lint_scenario_network, lint_scenarios_dir
    p = Path(args.scenarios)
    report = (
        lint_scenarios_dir(args.base, p,
                           require_bidirectional=args.require_bidirectional,
                           require_complete=args.require_complete)
        if p.is_dir() else
        lint_scenario_network(args.base, p,
                              require_bidirectional=args.require_bidirectional,
                              require_complete=args.require_complete)
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_compare_modes(args):
    try:
        from kvsched.evaluation.compare_modes import generate_mode_comparison_plots
        figs = generate_mode_comparison_plots(args.relaxed, args.strict, args.out)
        for k, v in figs.items():
            print(f"{k}: {v}")
    except ImportError:
        print("⚠ compare-modes requires pandas + matplotlib + numpy")
        return 2
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
    pr.add_argument("--prefetch", action="store_true")
    pr.add_argument("--overlap", action="store_true")
    pr.add_argument("--strict-network", action="store_true")
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
    pb.add_argument("--prefetch", action="store_true")
    pb.add_argument("--overlap", action="store_true")
    pb.add_argument("--strict-network", action="store_true")
    pb.add_argument("--no-plots", action="store_true")
    pb.add_argument("--fail-fast", action="store_true")
    pb.set_defaults(func=cmd_batch)

    pl = sub.add_parser("lint-network")
    pl.add_argument("--base", default="configs/base.yaml")
    pl.add_argument("--scenarios", required=True)
    pl.add_argument("--require-bidirectional", action="store_true")
    pl.add_argument("--require-complete", action="store_true")
    pl.set_defaults(func=cmd_lint_network)

    pc = sub.add_parser("compare-modes")
    pc.add_argument("--relaxed", required=True)
    pc.add_argument("--strict", required=True)
    pc.add_argument("--out", default="results/figures")
    pc.set_defaults(func=cmd_compare_modes)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
