from __future__ import annotations
from typing import Dict
from ..base import Scheduler, SchedulingDecision
from ...models.request import InferenceRequest

class LoadOnlyScheduler(Scheduler):
    def decide(self, tick: int, req: InferenceRequest, nodes: Dict[str, object]) -> SchedulingDecision:
        def load(n):
            ld = getattr(n, "load", None)
            if ld is None:
                return 0
            return getattr(ld, "queued_decode", 0) + getattr(ld, "running_decode", 0)

        dst = min(nodes.keys(), key=lambda k: load(nodes[k]))
        src = req.kv.owner_node_id
        return SchedulingDecision(tick=tick, request_id=req.request_id, src_node_id=src, dst_node_id=dst,
                                  will_migrate=(src != dst), reason="load_only")
