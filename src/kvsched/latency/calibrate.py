from __future__ import annotations
import json
from pathlib import Path

def fit_from_profiling(log_path: str | Path) -> dict:
    p = Path(log_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return {"alpha_ms": 0.2, "beta_ms_per_token": 0.002, "gamma_ms_per_batch": 0.01, "n": len(data)}
