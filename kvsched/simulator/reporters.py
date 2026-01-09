from __future__ import annotations
import json
from pathlib import Path
from .metrics import RunMetrics

def write_summary(path: str | Path, metrics: RunMetrics) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics.summary(), indent=2), encoding="utf-8")
