from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_index(index_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    # ensure numeric
    for c in ("p50_ms", "p95_ms", "p99_ms", "kv_migrations", "kv_migration_bytes"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_latencies(summary_path: str | Path) -> List[float]:
    p = Path(summary_path)
    if not p.exists():
        return []
    obj = json.loads(p.read_text(encoding="utf-8"))
    return list(map(float, obj.get("latencies_ms", [])))


def plot_cdf_for_scenario(df: pd.DataFrame, scenario: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["scenario"] == scenario].copy()

    plt.figure()
    for sch in sorted(sub["scheduler"].unique()):
        paths = sub[sub["scheduler"] == sch]["summary_path"].tolist()
        lat: List[float] = []
        for sp in paths:
            lat.extend(_load_latencies(sp))
        lat = [x for x in lat if np.isfinite(x)]
        if not lat:
            continue
        x = np.sort(np.array(lat))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=sch)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Empirical CDF")
    plt.title(f"CDF: {scenario}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = out_dir / f"cdf_{scenario}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def _group_bar(df: pd.DataFrame, value_col: str, title: str, out_path: Path) -> None:
    # mean across seeds
    g = df.groupby(["scenario", "scheduler"], as_index=False)[value_col].mean()
    scenarios = list(sorted(g["scenario"].unique()))
    schedulers = list(sorted(g["scheduler"].unique()))

    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(schedulers))

    plt.figure()
    for i, sch in enumerate(schedulers):
        vals = []
        for sc in scenarios:
            row = g[(g["scenario"] == sc) & (g["scheduler"] == sch)]
            vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
        plt.bar(x + i * width, vals, width=width, label=sch)

    plt.xticks(x + (len(schedulers) - 1) * width / 2, scenarios, rotation=0)
    plt.ylabel(value_col)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_all_plots(index_csv: str | Path, out_dir: str | Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_index(index_csv)

    figs: Dict[str, Path] = {}

    # CDF per scenario (requires samples saved in summary.json)
    for sc in sorted(df["scenario"].unique()):
        figs[f"cdf_{sc}"] = plot_cdf_for_scenario(df, sc, out_dir)

    # P99 bar
    p99_path = out_dir / "p99_bar.png"
    _group_bar(df, "p99_ms", "P99 latency (mean over seeds)", p99_path)
    figs["p99_bar"] = p99_path

    # migration bytes bar (GiB)
    df2 = df.copy()
    df2["kv_migration_gib"] = df2["kv_migration_bytes"] / (1024 ** 3)
    mig_path = out_dir / "migration_bytes_bar.png"
    _group_bar(df2, "kv_migration_gib", "KV migration bytes (GiB, mean over seeds)", mig_path)
    figs["migration_bytes_bar"] = mig_path

    return figs
