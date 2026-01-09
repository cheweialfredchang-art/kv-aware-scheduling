from __future__ import annotations
from .baselines.random_rr import RandomScheduler, RoundRobinScheduler
from .baselines.load_only import LoadOnlyScheduler
from .baselines.kv_only import KVOnlyScheduler
from .kaware.heuristic import KVAwareHeuristic

def get_scheduler(name: str, **kwargs):
    name = name.lower()
    if name in ("random", "rand"):
        return RandomScheduler(seed=int(kwargs.get("seed", 0)))
    if name in ("rr", "round_robin"):
        return RoundRobinScheduler()
    if name in ("load", "load_only"):
        return LoadOnlyScheduler()
    if name in ("kv", "kv_only", "kv_affinity"):
        return KVOnlyScheduler()
    if name in ("kv_heuristic", "kv_aware", "kaware"):
        return KVAwareHeuristic(
        net=kwargs.get("net"),
        weights=kwargs.get("weights"),
        policy=kwargs.get("policy"),
    )
    raise ValueError(f"Unknown scheduler: {name}")
