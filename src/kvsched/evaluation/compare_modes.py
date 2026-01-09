from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def _read_index(path: str | Path):
    if pd is None:
        raise ImportError("pandas is required")
    df = pd.read_csv(path)
    # standardize numeric
    for c in df.columns:
        if c.endswith("_ms") or c.endswith("_bytes") or c in ("seed", "ticks", "prefetch", "overlap"):
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
    return df


def _agg(df, value_col: str):
    # mean across seeds for each (scenario,label)
    g = df.groupby(["scenario", "label"], as_index=False)[value_col].mean()
    return g


def compare_strict_relaxed(
    relaxed_index: str | Path,
    strict_index: str | Path,
    out_dir: str | Path,
    *,
    value_col: str = "p99_ms",
    title: str = "Strict vs Relaxed",
    out_name: str = "strict_vs_relaxed_p99.png",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r = _agg(_read_index(relaxed_index), value_col)
    s = _agg(_read_index(strict_index), value_col)

    # Merge on scenario+label; compute delta (strict - relaxed)
    m = r.merge(s, on=["scenario", "label"], suffixes=("_relaxed", "_strict"))
    m["delta"] = m[f"{value_col}_strict"] - m[f"{value_col}_relaxed"]

    scenarios = list(sorted(m["scenario"].unique()))
    labels = list(sorted(m["label"].unique()))
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(labels))

    plt.figure()
    for i, lab in enumerate(labels):
        vals = []
        for sc in scenarios:
            row = m[(m["scenario"] == sc) & (m["label"] == lab)]
            vals.append(float(row["delta"].iloc[0]) if len(row) else np.nan)
        plt.bar(x + i * width, vals, width=width, label=lab)

    plt.xticks(x + (len(labels) - 1) * width / 2, scenarios, rotation=0)
    plt.ylabel(f"Δ {value_col} (strict - relaxed)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = out_dir / out_name
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def generate_mode_comparison_plots(relaxed_index: str | Path, strict_index: str | Path, out_dir: str | Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs: Dict[str, Path] = {}
    figs["delta_p99_total"] = compare_strict_relaxed(
        relaxed_index, strict_index, out_dir,
        value_col="p99_ms",
        title="Δ P99 total latency (strict - relaxed)",
        out_name="delta_p99_total.png",
    )
    if "p99_decode_ms" in _read_index(relaxed_index).columns:
        figs["delta_p99_decode"] = compare_strict_relaxed(
            relaxed_index, strict_index, out_dir,
            value_col="p99_decode_ms",
            title="Δ P99 decode latency (strict - relaxed)",
            out_name="delta_p99_decode.png",
        )
    figs["delta_migration_gib"] = compare_strict_relaxed(
        relaxed_index, strict_index, out_dir,
        value_col="kv_migration_bytes",
        title="Δ KV migration bytes (strict - relaxed)",
        out_name="delta_migration_bytes.png",
    )
    return figs
