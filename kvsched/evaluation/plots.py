from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def _read_index(index_csv: str | Path):
    if pd is None:
        raise ImportError("pandas is required for plotting")
    df = pd.read_csv(index_csv)
    for c in df.columns:
        if c.endswith("_ms") or c.endswith("_bytes") or c in ("seed", "ticks", "prefetch", "overlap"):
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    return df


def _load_list(summary_path: str | Path, key: str) -> List[float]:
    p = Path(summary_path)
    if not p.exists():
        return []
    obj = json.loads(p.read_text(encoding="utf-8"))
    return list(map(float, obj.get(key, [])))


def _plot_cdf(df, scenario: str, out_dir: Path, key: str, title_suffix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["scenario"] == scenario].copy()

    plt.figure()
    for label in sorted(sub["label"].unique()):
        paths = sub[sub["label"] == label]["summary_path"].tolist()
        lat: List[float] = []
        for sp in paths:
            lat.extend(_load_list(sp, key))
        lat = [x for x in lat if np.isfinite(x)]
        if not lat:
            continue
        x = np.sort(np.array(lat))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=label)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Empirical CDF")
    plt.title(f"CDF {title_suffix}: {scenario}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = out_dir / f"cdf_{title_suffix.lower()}_{scenario}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def _group_bar(df, value_col: str, title: str, out_path: Path) -> None:
    g = df.groupby(["scenario", "label"], as_index=False)[value_col].mean()
    scenarios = list(sorted(g["scenario"].unique()))
    labels = list(sorted(g["label"].unique()))

    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(labels))

    plt.figure()
    for i, lab in enumerate(labels):
        vals = []
        for sc in scenarios:
            row = g[(g["scenario"] == sc) & (g["label"] == lab)]
            vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
        plt.bar(x + i * width, vals, width=width, label=lab)

    plt.xticks(x + (len(labels) - 1) * width / 2, scenarios, rotation=0)
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

    # CDF per scenario, split by stage
    for sc in sorted(df["scenario"].unique()):
        figs[f"cdf_prefill_{sc}"] = _plot_cdf(df, sc, out_dir, "prefill_latencies_ms", "Prefill")
        figs[f"cdf_decode_{sc}"] = _plot_cdf(df, sc, out_dir, "decode_latencies_ms", "Decode")

    # P99 bar (TOTAL latency; legacy p99_ms)
    p99_path = out_dir / "p99_total_bar.png"
    _group_bar(df, "p99_ms", "P99 total latency (mean over seeds)", p99_path)
    figs["p99_total_bar"] = p99_path

    # Migration bytes bar (GiB)
    df2 = df.copy()
    df2["kv_migration_gib"] = df2["kv_migration_bytes"] / (1024 ** 3)
    mig_path = out_dir / "migration_bytes_bar.png"
    _group_bar(df2, "kv_migration_gib", "KV migration bytes (GiB, mean over seeds)", mig_path)
    figs["migration_bytes_bar"] = mig_path

    return figs
