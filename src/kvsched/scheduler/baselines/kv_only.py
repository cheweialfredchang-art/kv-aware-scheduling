from __future__ import annotations
from typing import Dict
from ..base import Scheduler, SchedulingDecision
from ...models.request import InferenceRequest

class KVOnlyScheduler(Scheduler):
    def decide(self, tick: int, req: InferenceRequest, nodes: Dict[str, object]) -> SchedulingDecision:
        src = req.kv.owner_node_id
        return SchedulingDecision(tick=tick, request_id=req.request_id, src_node_id=src, dst_node_id=src,
                                  will_migrate=False, reason="kv_affinity")
