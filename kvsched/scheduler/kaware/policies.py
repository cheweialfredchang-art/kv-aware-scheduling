from __future__ import annotations
from dataclasses import dataclass

@dataclass
class KVSwitchPolicy:
    hysteresis_ms: float = 5.0
    enable_prefetch: bool = True
    enable_overlap: bool = True
    prefetch_horizon_tokens: int = 0  # enable prefetch only if remaining tokens >= horizon
