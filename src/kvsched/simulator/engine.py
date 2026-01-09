from __future__ import annotations

from typing import Dict, List
import random

from ..models.request import InferenceRequest, Stage
from ..models.node import NodeState
from ..scheduler.base import Scheduler
from ..models.network import NetworkModel
from .metrics import RunMetrics


def _pick_prefill_node(nodes: Dict[str, NodeState]) -> str:
    # Prefer nodes with tag role=prefill; fallback to all nodes
    prefill_candidates = []
    for nid, n in nodes.items():
        role = getattr(n, "tags", {}).get("role")
        if role == "prefill":
            prefill_candidates.append(nid)
    cand = prefill_candidates or list(nodes.keys())
    # choose by smallest (queued+running prefill)
    best = None
    best_q = None
    for nid in cand:
        n = nodes[nid]
        q = n.load.queued_prefill + n.load.running_prefill
        if best_q is None or q < best_q:
            best_q = q
            best = nid
    assert best is not None
    return best


def run_simulation(
    requests: List[InferenceRequest],
    nodes: Dict[str, NodeState],
    scheduler: Scheduler,
    net: NetworkModel | None = None,
    ticks: int = 100,
    seed: int = 0,
    overlap_on: bool = True,
) -> RunMetrics:
    rng = random.Random(seed)
    m = RunMetrics()
    if not requests:
        return m

    # If overlap is disabled, we will treat effective transfer as raw transfer (overlap ratio = 0)
    if net is not None and not overlap_on:
        net = net.model_copy(update={"overlap": net.overlap.model_copy(update={"overlap_ratio": 0.0})})

    for t in range(ticks):
        tmpl = requests[t % len(requests)]
        # Each tick = independent request instance for stable CDFs
        req = tmpl.model_copy(deep=True)

        # ----- PREFILL -----
        prefill_node = _pick_prefill_node(nodes)
        pn = nodes[prefill_node]
        q_prefill = pn.load.queued_prefill + pn.load.running_prefill

        # Simple prefill latency model (throughput-ish): base + queue + prompt factor
        prefill_latency_ms = 20.0 + 0.06 * req.profile.prompt_tokens + 2.0 * q_prefill
        m.prefill_latencies_ms.append(prefill_latency_ms)

        # Prefill produces KV and sets its owner at prefill node
        req.runtime = req.runtime.model_copy(update={"stage": Stage.DECODE, "seq_len": req.profile.prompt_tokens})
        req.kv = req.kv.model_copy(update={
            "owner_node_id": prefill_node,
            "num_cached_tokens": req.profile.prompt_tokens,
        })

        # Small random load fluctuation on prefill node
        pn.load = pn.load.model_copy(update={
            "running_prefill": max(0, pn.load.running_prefill + rng.choice([-1, 0, 1])),
            "queued_prefill": max(0, pn.load.queued_prefill + rng.choice([-1, 0, 1])),
        })

        # ----- DECODE (routing/scheduling) -----
        dec = scheduler.decide(t, req, nodes)

        # compute baseline decode latency as queueing + constant
        dst = nodes[dec.dst_node_id]
        q_dec = dst.load.queued_decode + dst.load.running_decode
        decode_compute_ms = 10.0 + 1.0 * q_dec

        mig_ms = 0.0
        if dec.will_migrate:
            m.kv_migrations += 1
            m.kv_migration_bytes += int(req.kv.kv_bytes)

            if net is not None:
                mig_ms = float(net.est_effective_transfer_ms(
                    dec.src_node_id, dec.dst_node_id, int(req.kv.kv_bytes),
                    compute_ms=decode_compute_ms,
                    prefetch=bool(dec.prefetch),
                ))
            else:
                # fallback: constant migration penalty if no network model
                mig_ms = 8.0

        decode_latency_ms = decode_compute_ms + mig_ms
        m.decode_latencies_ms.append(decode_latency_ms)
        m.total_latencies_ms.append(prefill_latency_ms + decode_latency_ms)

        # Update decode node load (toy dynamics)
        dst.load = dst.load.model_copy(update={
            "running_decode": max(0, dst.load.running_decode + rng.choice([-1, 0, 1])),
            "queued_decode": max(0, dst.load.queued_decode + rng.choice([-1, 0, 1])),
        })

    return m
