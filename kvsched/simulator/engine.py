from __future__ import annotations
from typing import Dict, List
import random
from ..models.request import InferenceRequest
from ..models.node import NodeState
from ..scheduler.base import Scheduler
from .metrics import RunMetrics

def run_simulation(requests: List[InferenceRequest], nodes: Dict[str, NodeState], scheduler: Scheduler, ticks: int = 100, seed: int = 0) -> RunMetrics:
    rng = random.Random(seed)
    m = RunMetrics()
    if not requests:
        return m

    for t in range(ticks):
        req = requests[t % len(requests)]
        dec = scheduler.decide(t, req, nodes)

        if dec.will_migrate:
            m.kv_migrations += 1
            m.kv_migration_bytes += req.kv.kv_bytes
            req.kv = req.kv.model_copy(update={"owner_node_id": dec.dst_node_id, "version": req.kv.version + 1})

        dst = nodes[dec.dst_node_id]
        q = dst.load.queued_decode + dst.load.running_decode
        latency_ms = 10.0 + 1.0 * q + (5.0 if dec.will_migrate else 0.0)
        m.latencies_ms.append(latency_ms)

        dst.load = dst.load.model_copy(update={"running_decode": max(0, dst.load.running_decode + rng.choice([-1, 0, 1]))})

    return m
