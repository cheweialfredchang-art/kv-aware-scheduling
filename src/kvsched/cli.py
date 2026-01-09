from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

from kvsched.config.loader import load_yaml, deep_merge
from kvsched.models.node import NodeState, GPUDevice, NodeLoad
from kvsched.models.network import (
    NetworkModel, NetworkTopology, NetworkLink, TrafficProfile
)
from kvsched.models.notation import gib
from kvsched.scheduler.registry import get_scheduler
from kvsched.scheduler.base import Scheduler, SchedulingDecision
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

    # allow optional overlap settings in YAML; default in model is OK
    net = NetworkModel(topo=topo)
    net.topo = net.topo.model_copy(update={"strict_network": strict_network})
    if "overlap" in net_cfg:
        net = net.model_copy(update={"overlap": net.overlap.model_copy(update=net_cfg["overlap"])})
    return net


def validate_cfg(cfg: dict):
    for k in ("scenario_id", "nodes", "network", "workload"):
        if k not in cfg:
            raise KeyError(f"Missing '{k}' in config")


# ---------------- Scheduler wrappers for ablation ----------------

class PrefetchMaskScheduler(Scheduler):
    """Force prefetch to a constant value for ablation."""
    def __init__(self, inner: Scheduler, prefetch_on: bool):
        self.inner = inner
        self.prefetch_on = prefetch_on

    def decide(self, tick: int, req, nodes) -> SchedulingDecision:
        d = self.inner.decide(tick, req, nodes)
        return d.model_copy(update={"prefetch": bool(self.prefetch_on) and bool(d.will_migrate)})


# ---------------- Run once ----------------

def run_once(base, scenario, scheduler_name, ticks, seed, out_root, *, prefetch_on: bool, overlap_on: bool, strict_network: bool):
    base_cfg = load_yaml(base)
    sc_cfg = load_yaml(scenario)
    cfg = deep_merge(base_cfg, sc_cfg)
    validate_cfg(cfg)

    nodes = build_nodes(cfg["nodes"])
    net = build_network(cfg["network"], strict_network=strict_network)

    sched = get_scheduler(scheduler_name, net=net, seed=seed)
    # Ablation: force prefetch on/off regardless of heuristic policy
    sched = PrefetchMaskScheduler(sched, prefetch_on=prefetch_on)

    reqs = make_synthetic_requests(
        n=int(cfg["workload"].get("num_requests", 50)),
        owner_nodes=list(nodes.keys()),
    )

    metrics = run_simulation(
        requests=reqs,
        nodes=nodes,
        scheduler=sched,
        net=net,
        ticks=ticks,
        seed=seed,
        overlap_on=overlap_on,
    )

    out_dir = Path(out_root) / cfg["scenario_id"] / scheduler_name / f"seed={seed}" / f"pf={int(prefetch_on)}_ov={int(overlap_on)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.json"
    write_summary(summary, metrics, include_samples=True)

    row = metrics.summary()
    label = f"{scheduler_name}|pf{int(prefetch_on)}|ov{int(overlap_on)}"
    row.update({
        "scenario": cfg["scenario_id"],
        "scheduler": scheduler_name,
        "label": label,
        "prefetch": int(prefetch_on),
        "overlap": int(overlap_on),
        "seed": seed,
        "ticks": ticks,
        "summary_path": str(summary),
    })
    return row


# ---------------- Commands ----------------

def cmd_run(args):
    row = run_once(
        args.base, args.scenario, args.scheduler,
        args.ticks, args.seed, args.out,
        prefetch_on=bool(args.prefetch),
        overlap_on=bool(args.overlap),
        strict_network=bool(args.strict_network),
    )
    print(row)
    return 0


def _iter_ablations(mode: str, prefetch: bool, overlap: bool) -> List[Tuple[bool, bool]]:
    mode = (mode or "none").lower()
    if mode in ("none", "off"):
        return [(prefetch, overlap)]
    # default: full on/off ablation over two switches
    return [(False, False), (True, False), (False, True), (True, True)]


def cmd_batch(args):
    # scenarios
    p = Path(args.scenarios)
    if p.exists() and p.is_dir():
        scenarios = sorted(p.glob("*.yaml"))
    else:
        scenarios = [Path(x.strip()) for x in args.scenarios.split(",") if x.strip()]

    schedulers = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    abls = _iter_ablations(args.ablations, bool(args.prefetch), bool(args.overlap))

    rows = []
    for sc in scenarios:
        for sch in schedulers:
            for seed in seeds:
                for pf, ov in abls:
                    try:
                        row = run_once(
                            args.base, str(sc), sch,
                            args.ticks, seed, args.out,
                            prefetch_on=pf,
                            overlap_on=ov,
                        )
                        rows.append(row)
                        print(f"[OK] {sc.name} {sch} seed={seed} pf={int(pf)} ov={int(ov)}")
                    except Exception as e:
                        print(f"[ERROR] {sc.name} {sch} seed={seed} pf={int(pf)} ov={int(ov)}: {e}")
                        if args.fail_fast:
                            raise

    index = Path(args.index)
    index.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No successful runs; index.csv not written")

    # stable field order
    fieldnames = [
        "scenario", "scheduler", "label", "prefetch", "overlap", "seed", "ticks",
        "p50_ms", "p95_ms", "p99_ms",
        "p50_prefill_ms", "p95_prefill_ms", "p99_prefill_ms",
        "p50_decode_ms", "p95_decode_ms", "p99_decode_ms",
        "kv_migrations", "kv_migration_bytes",
        "summary_path",
    ]

    with index.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Optional plots
    if not args.no_plots:
        try:
            from kvsched.evaluation.plots import generate_all_plots
            figdir = Path(args.figdir)
            generate_all_plots(index, figdir)
            print(f"Plots -> {figdir}")
        except ImportError:
            print("⚠ Plotting skipped (requires pandas + matplotlib + numpy). Install via: pip install pandas matplotlib numpy")

    return 0

def cmd_lint_network(args):
    from kvsched.tools.lint_network import lint_scenario_network, lint_scenarios_dir
    p = Path(args.scenarios)
    if p.exists() and p.is_dir():
        report = lint_scenarios_dir(
            args.base, p,
            require_bidirectional=bool(args.require_bidirectional),
            require_complete=bool(args.require_complete),
        )
    else:
        report = lint_scenario_network(
            args.base, p,
            require_bidirectional=bool(args.require_bidirectional),
            require_complete=bool(args.require_complete),
        )

    # Pretty print summary
    if isinstance(report, dict) and "results" in report:
        ok = report.get("ok", False)
        print(f"OK={ok}  scenarios={len(report['results'])}")
        for r in report["results"]:
            if not r["ok"]:
                print(f"- {r['scenario']}: issues={len(r['issues'])}")
    else:
        print(f"OK={report.get('ok', False)}  scenario={report.get('scenario')} issues={len(report.get('issues', []))}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report -> {out}")
    return 0


def cmd_compare_modes(args):
    try:
        from kvsched.evaluation.compare_modes import generate_mode_comparison_plots
        figs = generate_mode_comparison_plots(args.relaxed, args.strict, args.out)
        for k, v in figs.items():
            print(f"{k}: {v}")
    except ImportError:
        print("⚠ compare-modes requires pandas + matplotlib + numpy. Install via: pip install pandas matplotlib numpy")
        return 2
    return 0



# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser("kvsched")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run a single scenario")
    pr.add_argument("--base", default="configs/base.yaml")
    pr.add_argument("--scenario", required=True)
    pr.add_argument("--scheduler", default="kv_heuristic")
    pr.add_argument("--ticks", type=int, default=1000)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default="results/raw")
    pr.add_argument("--prefetch", action="store_true", help="Force prefetch ON for ablation")
    pr.add_argument("--overlap", action="store_true", help="Enable overlap model (default OFF for run unless set)")
    pr.add_argument("--strict-network", action="store_true", help="Require explicit network links; error if missing")
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("batch", help="Batch run scenarios × schedulers × seeds × ablations")
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

    pb.add_argument("--prefetch", action="store_true", help="Default prefetch value if --ablations none")
    pb.add_argument("--overlap", action="store_true", help="Default overlap value if --ablations none")
    pb.add_argument("--strict-network", action="store_true", help="Require explicit network links; error if missing")
    pb.add_argument("--ablations", default="prefetch_overlap",
                    help="none|prefetch_overlap. If prefetch_overlap, runs 4 combos: (pf,ov) in {0,1}^2")

    pb.set_defaults(func=cmd_batch)

# ---- lint-network ----
pl = sub.add_parser("lint-network", help="Lint scenario network completeness")
pl.add_argument("--base", default="configs/base.yaml")
pl.add_argument("--scenarios", required=True, help="Scenario YAML file or directory containing *.yaml")
pl.add_argument("--require-bidirectional", action="store_true", help="Require dst->src for every src->dst")
pl.add_argument("--require-complete", action="store_true", help="Require explicit link for every directed node pair")
pl.add_argument("--json", dest="json_out", default="", help="Write lint report to JSON file")
pl.set_defaults(func=cmd_lint_network)

# ---- compare-modes ----
pc = sub.add_parser("compare-modes", help="Plot strict vs relaxed deltas from two index.csv files")
pc.add_argument("--relaxed", required=True, help="Path to relaxed index.csv")
pc.add_argument("--strict", required=True, help="Path to strict index.csv")
pc.add_argument("--out", default="results/figures", help="Directory for output figures")
pc.set_defaults(func=cmd_compare_modes)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
