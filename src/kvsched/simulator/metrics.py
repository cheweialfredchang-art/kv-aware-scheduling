from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass
class RunMetrics:
    # Stage latencies
    prefill_latencies_ms: List[float] = field(default_factory=list)
    decode_latencies_ms: List[float] = field(default_factory=list)
    total_latencies_ms: List[float] = field(default_factory=list)

    # Migration stats
    kv_migrations: int = 0
    kv_migration_bytes: int = 0

    def _q(self, arr: List[float], q: float) -> float:
        if not arr:
            return float("nan")
        a = np.asarray(arr, dtype=float)
        return float(np.quantile(a, q))

    def summary(self) -> Dict[str, float]:
        # Keep legacy keys (p50/p95/p99) as TOTAL latency for compatibility
        return {
            "p50_ms": self._q(self.total_latencies_ms, 0.5),
            "p95_ms": self._q(self.total_latencies_ms, 0.95),
            "p99_ms": self._q(self.total_latencies_ms, 0.99),

            "p50_prefill_ms": self._q(self.prefill_latencies_ms, 0.5),
            "p95_prefill_ms": self._q(self.prefill_latencies_ms, 0.95),
            "p99_prefill_ms": self._q(self.prefill_latencies_ms, 0.99),

            "p50_decode_ms": self._q(self.decode_latencies_ms, 0.5),
            "p95_decode_ms": self._q(self.decode_latencies_ms, 0.95),
            "p99_decode_ms": self._q(self.decode_latencies_ms, 0.99),

            "kv_migrations": float(self.kv_migrations),
            "kv_migration_bytes": float(self.kv_migration_bytes),
        }

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(self.summary())
        if include_samples:
            d["prefill_latencies_ms"] = list(map(float, self.prefill_latencies_ms))
            d["decode_latencies_ms"] = list(map(float, self.decode_latencies_ms))
            d["total_latencies_ms"] = list(map(float, self.total_latencies_ms))
        return d
