from __future__ import annotations
from typing import Dict, Optional
from ..base import Scheduler, SchedulingDecision
from ...models.request import InferenceRequest
from ...models.network import NetworkModel
from .scoring import ScoreWeights, mem_pressure
from .policies import KVSwitchPolicy

class KVAwareHeuristic(Scheduler):
    def __init__(self, net: Optional[NetworkModel] = None, weights: ScoreWeights | None = None, policy: KVSwitchPolicy | None = None):
        self.net = net
        self.weights = weights or ScoreWeights()
        self.policy = policy or KVSwitchPolicy()

    def decide(self, tick: int, req: InferenceRequest, nodes: Dict[str, object]) -> SchedulingDecision:
        src = req.kv.owner_node_id
        best_k, best_score, best_prefetch = None, None, False

        for k, n in nodes.items():
            ld = getattr(n, "load", None)
            q = (getattr(ld, "queued_decode", 0) + getattr(ld, "running_decode", 0)) if ld is not None else 0

            if self.net is not None and src != k:
                mig_ms = self.net.est_effective_transfer_ms(src, k, req.kv.kv_bytes, compute_ms=0.0, prefetch=self.policy.enable_prefetch)
            else:
                mig_ms = 0.0 if src == k else 1e6

            gpu = getattr(n, "gpu", None)
            mp = mem_pressure(getattr(gpu, "vram_free_bytes", 0), getattr(gpu, "vram_total_bytes", 1)) if gpu is not None else 0.0

            score = self.weights.w_queue * q + self.weights.w_migration_ms * mig_ms + self.weights.w_mem_pressure * mp

            if best_score is None or score < best_score:
                best_score = score
                best_k = k
                best_prefetch = (src != k and self.policy.enable_prefetch)

        assert best_k is not None
        return SchedulingDecision(tick=tick, request_id=req.request_id, src_node_id=src, dst_node_id=best_k,
                                  will_migrate=(src != best_k), prefetch=best_prefetch, reason="kv_aware_heuristic")
