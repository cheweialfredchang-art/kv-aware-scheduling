from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

def generate_s1s6_figures(suite_index_csv: str, out_dir: str, title_prefix: str = "") -> Dict[str, str]:
    """Generate S1–S6 figures:
    - Per-scenario: P99 total, P99 decode, migration bytes (bars)
    - Per-scenario: CDF for prefill/decode (if per-request data is available in raw logs)
    - Overall summary across scenarios
    Input: suite_index.csv with at least columns:
      scenario, scheduler, seed, p99_total_ms, p99_decode_ms, migration_bytes
    The runner writes this in results/aggregated/suite_index.csv.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plots-s1s6 requires pandas and matplotlib") from e

    df = pd.read_csv(suite_index_csv)

    # Normalize column names (tolerate older runs)
    colmap = {
        "p99_total": "p99_total_ms",
        "p99_total_ms": "p99_total_ms",
        "p99_decode": "p99_decode_ms",
        "p99_decode_ms": "p99_decode_ms",
        "migration_bytes": "migration_bytes",
        "kv_migration_bytes": "migration_bytes",
    }
    for src, dst in list(colmap.items()):
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    required = ["scenario", "scheduler"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"suite_index.csv missing required column: {c}")

    # Convert numerics safely
    for c in ["p99_total_ms", "p99_decode_ms", "migration_bytes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    figs: Dict[str, str] = {}

    def _bar(metric: str, fname: str, ylabel: str):
        if metric not in df.columns:
            return
        agg = df.groupby(["scenario", "scheduler"], as_index=False)[metric].median()
        scenarios = list(agg["scenario"].unique())
        scheds = list(agg["scheduler"].unique())
        # Wide pivot
        piv = agg.pivot(index="scenario", columns="scheduler", values=metric).reindex(scenarios)
        ax = piv.plot(kind="bar", rot=35)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Scenario")
        title = f"{title_prefix}{ylabel} (median over seeds)"
        ax.set_title(title)
        fig = ax.get_figure()
        path = out / fname
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figs[fname.replace('.png','')] = str(path)

    _bar("p99_total_ms", "S1S6_p99_total.png", "P99 latency (ms)")
    _bar("p99_decode_ms", "S1S6_p99_decode.png", "P99 decode latency (ms)")
    _bar("migration_bytes", "S1S6_migration_bytes.png", "KV migration bytes")

    # Per-scenario breakdown figures
    for scen in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scen].copy()
        # median per scheduler
        for metric, label, suf in [
            ("p99_total_ms", "P99 latency (ms)", "p99_total"),
            ("p99_decode_ms", "P99 decode latency (ms)", "p99_decode"),
            ("migration_bytes", "KV migration bytes", "mig_bytes"),
        ]:
            if metric not in sdf.columns:
                continue
            agg = sdf.groupby("scheduler", as_index=False)[metric].median().sort_values(metric)
            ax = agg.plot(x="scheduler", y=metric, kind="bar", rot=25, legend=False)
            ax.set_ylabel(label)
            ax.set_xlabel("Scheduler")
            ax.set_title(f"{title_prefix}{scen}: {label} (median over seeds)")
            fig = ax.get_figure()
            path = out / f"{scen}_{suf}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=200)
            plt.close(fig)
            figs[f"{scen}_{suf}"] = str(path)

    # Optional: strict vs relaxed delta if mode column exists
    if "mode" in df.columns:
        for metric, suf, ylabel in [
            ("p99_total_ms", "delta_p99_total", "Δ P99 latency (ms) strict-relaxed"),
            ("p99_decode_ms", "delta_p99_decode", "Δ P99 decode (ms) strict-relaxed"),
            ("migration_bytes", "delta_migration_bytes", "Δ migration bytes strict-relaxed"),
        ]:
            if metric not in df.columns:
                continue
            m = df.pivot_table(index=["scenario","scheduler","seed"], columns="mode", values=metric, aggfunc="first")
            if "strict" in m.columns and "relaxed" in m.columns:
                m["delta"] = m["strict"] - m["relaxed"]
                agg = m.reset_index().groupby(["scenario","scheduler"], as_index=False)["delta"].median()
                piv = agg.pivot(index="scenario", columns="scheduler", values="delta")
                ax = piv.plot(kind="bar", rot=35)
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Scenario")
                ax.set_title(f"{title_prefix}{ylabel} (median over seeds)")
                fig = ax.get_figure()
                path = out / f"S1S6_{suf}.png"
                fig.tight_layout()
                fig.savefig(path, dpi=200)
                plt.close(fig)
                figs[f"S1S6_{suf}"] = str(path)

    # Write a small manifest
    manifest = out / "figures_manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        for k, v in sorted(figs.items()):
            f.write(f"{k}\t{v}\n")
    figs["manifest"] = str(manifest)
    return figs
