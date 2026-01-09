from __future__ import annotations
from pathlib import Path
from typing import Dict

from ..models.node import NodeState, GPUDevice, NodeLoad
from ..models.network import NetworkModel, NetworkTopology, NetworkLink, TrafficProfile
from ..scheduler.registry import get_scheduler
from ..simulator.workload import make_synthetic_requests
from ..simulator.engine import run_simulation
from ..simulator.reporters import write_summary
from ..models.notation import gib

def build_demo_nodes() -> Dict[str, NodeState]:
    nodeA = NodeState(
        node_id="nodeA",
        gpu=GPUDevice(name="A100_80GB", vram_total_bytes=gib(80), vram_free_bytes=gib(60)),
        load=NodeLoad(running_decode=5, queued_decode=2),
        tags={"tier": "cloud"},
    )
    nodeB = NodeState(
        node_id="nodeB",
        gpu=GPUDevice(name="L40S_48GB", vram_total_bytes=gib(48), vram_free_bytes=gib(30)),
        load=NodeLoad(running_decode=2, queued_decode=1),
        tags={"tier": "edge"},
    )
    return {"nodeA": nodeA, "nodeB": nodeB}

def build_demo_network() -> NetworkModel:
    topo = NetworkTopology()
    topo.add_link(NetworkLink(src="nodeA", dst="nodeB", bandwidth_Gbps=25.0, rtt_ms=0.3))
    topo.add_link(NetworkLink(src="nodeB", dst="nodeA", bandwidth_Gbps=25.0, rtt_ms=0.3))
    topo.traffic = TrafficProfile(default_util=0.1, util_by_pair={("nodeA", "nodeB"): 0.4})
    return NetworkModel(topo=topo)

def run(scheduler_name: str = "kv_heuristic", ticks: int = 200, seed: int = 0, out_dir: str = "results/raw/demo"):
    nodes = build_demo_nodes()
    net = build_demo_network()
    sched = get_scheduler(scheduler_name, net=net, seed=seed)
    reqs = make_synthetic_requests(20, owner_nodes=list(nodes.keys()))
    metrics = run_simulation(reqs, nodes, sched, ticks=ticks, seed=seed)
    write_summary(Path(out_dir) / f"{scheduler_name}_seed{seed}.json", metrics)
    return metrics.summary()
