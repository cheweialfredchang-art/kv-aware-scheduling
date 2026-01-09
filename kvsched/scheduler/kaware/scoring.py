from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ScoreWeights:
    w_queue: float = 1.0
    w_migration_ms: float = 1.0
    w_mem_pressure: float = 0.2

def mem_pressure(vram_free: int, vram_total: int) -> float:
    if vram_total <= 0:
        return 1.0
    used = max(0, vram_total - vram_free)
    return used / vram_total
