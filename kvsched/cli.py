from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def _resolve_index_path(p: str) -> str:
    from pathlib import Path
    pp = Path(p)
    if pp.is_dir():
        cand = pp / "index.csv"
        if cand.exists():
            return str(cand)
        cand2 = pp / "aggregated" / "index.csv"
        if cand2.exists():
            return str(cand2)
    return str(pp)


def _parse_json_or_none(s: str | None):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return json.loads(s)


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

def run_once(base, scenario, scheduler, ticks, seed, out_root, *, prefetch_on=False, overlap_on=False, strict_network=False, weights=None, policy=None):
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
        weights=_parse_json_or_none(getattr(args, "weights_json", None)) if "args" in locals() else None,
        policy=_parse_json_or_none(getattr(args, "policy_json", None)) if "args" in locals() else None,
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
        base, args.scenario, args.scheduler,
        args.ticks, args.seed, args.out,
        prefetch_on=args.prefetch,
        overlap_on=args.overlap,
        strict_network=args.strict_network,
        weights=_parse_json_or_none(args.weights_json),
        policy=_parse_json_or_none(args.policy_json),
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
                        base, sc, sch,
                        args.ticks, seed, args.out,
                        prefetch_on=args.prefetch,
                        overlap_on=args.overlap,
                        strict_network=args.strict_network,
                        weights=_parse_json_or_none(args.weights_json),
                        policy=_parse_json_or_none(args.policy_json),
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
            figdir = getattr(args, "figdir", "") or str(Path(args.out) / "figures")
            generate_all_plots(index, figdir)
        except ImportError:
            print("⚠ plotting skipped (pandas/matplotlib not installed)")
        except Exception as e:
            print(f"[WARN] plot generation failed: {e}")
    return 0


def cmd_suite(args):
    from kvsched.tools.experiment_suite import run_suite
    idx = run_suite(
        args.suite,
        base_cfg=args.base,
        out_root=args.out,
        index_path=args.index,
        fail_fast=args.fail_fast,
    )
    print(f"Suite index -> {idx}")
    return 0



def cmd_lint_network(args):
    base = getattr(args, 'base', 'configs/base.yaml')
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


def cmd_topology_table(args):
    base = getattr(args, 'base', 'configs/base.yaml')
    from kvsched.tools.topology_table import write_topology_tables
    p = Path(args.scenarios)
    scenarios = sorted(p.glob("*.yaml")) if p.is_dir() else [p]
    out = write_topology_tables(
        args.base,
        scenarios,
        args.out,
        fmt=args.format,
        require_bidirectional=args.require_bidirectional,
    )
    print(f"Topology tables -> {out}")
    return 0



def cmd_compare_modes(args):
    from pathlib import Path
    from kvsched.evaluation.compare_modes import generate_mode_comparison_plots

    relaxed = _resolve_index_path(args.relaxed)
    strict = _resolve_index_path(args.strict)

    if not Path(relaxed).exists():
        print(f"[ERROR] relaxed index not found: {relaxed}")
        print("Hint: generate it with batch, e.g.:")
        print("  python -m kvsched.cli batch --scenarios configs/scenarios/S1_kv_vs_load.yaml --schedulers kv_heuristic --seeds 0,1,2 --out results/relaxed --index results/relaxed/index.csv")
        return 2

    if not Path(strict).exists():
        print(f"[ERROR] strict index not found: {strict}")
        print("Hint: generate it with strict-network batch, e.g.:")
        print("  python -m kvsched.cli batch --scenarios configs/scenarios/S1_kv_vs_load.yaml --schedulers kv_heuristic --seeds 0,1,2 --out results/strict --index results/strict/index.csv --strict-network")
        return 2

    figs = generate_mode_comparison_plots(relaxed, strict, args.out)
    for k, v in figs.items():
        print(f"{k}: {v}")
    return 0


# ---------------- Main ----------------

def main():
    import argparse

    p = argparse.ArgumentParser(prog="kvsched")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run")
    pr.add_argument("--base", default="configs/base.yaml")
    pr.add_argument("--scenario", required=True)
    pr.add_argument("--scheduler", required=True)
    pr.add_argument("--ticks", type=int, default=1000)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default="results/raw")
    pr.add_argument("--prefetch", action="store_true")
    pr.add_argument("--overlap", action="store_true")
    pr.add_argument("--strict-network", action="store_true")
    pr.add_argument("--weights-json", default="")
    pr.add_argument("--policy-json", default="")
    pr.set_defaults(func=cmd_run)

    # batch
    pb = sub.add_parser("batch")
    pb.add_argument("--base", default="configs/base.yaml")
    pb.add_argument("--scenarios", required=True)
    pb.add_argument("--schedulers", required=True)
    pb.add_argument("--seeds", required=True)
    pb.add_argument("--ticks", type=int, default=1000)
    pb.add_argument("--out", default="results/raw")
    pb.add_argument("--index", default="results/aggregated/index.csv")
    pb.add_argument("--prefetch", action="store_true")
    pb.add_argument("--overlap", action="store_true")
    pb.add_argument("--strict-network", action="store_true")
    pb.add_argument("--weights-json", default="")
    pb.add_argument("--policy-json", default="")
    pb.add_argument("--no-plots", action="store_true")
    pb.add_argument("--figdir", default="")
    pb.set_defaults(func=cmd_batch)

    # compare-modes
    pc = sub.add_parser("compare-modes")
    pc.add_argument("--relaxed", required=True, help="CSV path OR directory containing index.csv")
    pc.add_argument("--strict", required=True, help="CSV path OR directory containing index.csv")
    pc.add_argument("--out", default="results/figures")
    pc.set_defaults(func=cmd_compare_modes)

    # lint-network
    pl = sub.add_parser("lint-network")
    pl.add_argument("--base", default="configs/base.yaml")
    pl.add_argument("--scenarios", required=True)
    pl.add_argument("--require-bidirectional", action="store_true")
    pl.set_defaults(func=cmd_lint_network)

    # topology-table
    pt = sub.add_parser("topology-table")
    pt.add_argument("--base", default="configs/base.yaml")
    pt.add_argument("--scenarios", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--format", choices=["md", "latex"], default="md")
    pt.add_argument("--require-bidirectional", action="store_true")
    pt.set_defaults(func=cmd_topology_table)

    # suite
    ps = sub.add_parser("suite")
    ps.add_argument("--suite", default="configs/experiments/suite.yaml")
    ps.add_argument("--base", default="configs/base.yaml")
    ps.add_argument("--out", default="results/raw")
    ps.add_argument("--index", default="results/aggregated/suite_index.csv")
    ps.add_argument("--fail-fast", action="store_true")
    ps.set_defaults(func=cmd_suite)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
