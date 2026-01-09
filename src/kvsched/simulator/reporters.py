from __future__ import annotations
import json
from pathlib import Path
from .metrics import RunMetrics

def write_summary(path: str | Path, metrics: RunMetrics, *, include_samples: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = metrics.to_dict(include_samples=include_samples)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
