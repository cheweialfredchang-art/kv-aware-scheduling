from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np

@dataclass
class RunMetrics:
    latencies_ms: List[float] = field(default_factory=list)
    kv_migrations: int = 0
    kv_migration_bytes: int = 0

    def p(self, q: float) -> float:
        if not self.latencies_ms:
            return float("nan")
        return float(np.quantile(np.array(self.latencies_ms), q))

    def summary(self) -> Dict[str, float]:
        return {
            "p50_ms": self.p(0.5),
            "p95_ms": self.p(0.95),
            "p99_ms": self.p(0.99),
            "kv_migrations": float(self.kv_migrations),
            "kv_migration_bytes": float(self.kv_migration_bytes),
        }

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(self.summary())
        if include_samples:
            d["latencies_ms"] = list(map(float, self.latencies_ms))
        return d
