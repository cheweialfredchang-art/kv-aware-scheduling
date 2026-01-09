from __future__ import annotations
from typing import Dict, Optional
from ..base import Scheduler, SchedulingDecision
from ...models.request import InferenceRequest
from ...models.network import NetworkModel
from .scoring import ScoreWeights, mem_pressure
from .policies import KVSwitchPolicy

def _coerce_weights(w):
    from .scoring import ScoreWeights
    if w is None:
        return ScoreWeights()
    if isinstance(w, ScoreWeights):
        return w
    if isinstance(w, dict):
        return ScoreWeights(**w)
    return ScoreWeights()

def _coerce_policy(p):
    from .policies import KVSwitchPolicy
    if p is None:
        return KVSwitchPolicy()
    if isinstance(p, KVSwitchPolicy):
        return p
    if isinstance(p, dict):
        return KVSwitchPolicy(**p)
    return KVSwitchPolicy()

class KVAwareHeuristic(Scheduler):

    def __init__(self, net: Optional[NetworkModel] = None, weights: ScoreWeights | dict | None = None, policy: KVSwitchPolicy | dict | None = None):
        self.net = net
        self.weights = _coerce_weights(weights)
        self.policy = _coerce_policy(policy)

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
                h = getattr(self.policy, "prefetch_horizon_tokens", 0) or 0
                best_prefetch = (src != k and self.policy.enable_prefetch and req.profile.max_new_tokens >= h)

        assert best_k is not None
        return SchedulingDecision(tick=tick, request_id=req.request_id, src_node_id=src, dst_node_id=best_k,
                                  will_migrate=(src != best_k), prefetch=best_prefetch, reason="kv_aware_heuristic")
