from __future__ import annotations
import random
from typing import Dict
from ..base import Scheduler, SchedulingDecision
from ...models.request import InferenceRequest

class RandomScheduler(Scheduler):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def decide(self, tick: int, req: InferenceRequest, nodes: Dict[str, object]) -> SchedulingDecision:
        dst = self.rng.choice(list(nodes.keys()))
        src = req.kv.owner_node_id
        return SchedulingDecision(tick=tick, request_id=req.request_id, src_node_id=src, dst_node_id=dst,
                                  will_migrate=(src != dst), reason="random")

class RoundRobinScheduler(Scheduler):
    def __init__(self):
        self.i = 0

    def decide(self, tick: int, req: InferenceRequest, nodes: Dict[str, object]) -> SchedulingDecision:
        keys = list(nodes.keys())
        dst = keys[self.i % len(keys)]
        self.i += 1
        src = req.kv.owner_node_id
        return SchedulingDecision(tick=tick, request_id=req.request_id, src_node_id=src, dst_node_id=dst,
                                  will_migrate=(src != dst), reason="round_robin")
