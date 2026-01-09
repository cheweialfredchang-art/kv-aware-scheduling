from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import json

from kvsched.config.loader import load_yaml, deep_merge
from kvsched.tools.lint_network import lint_scenario_network


def _role(tags: dict) -> str:
    return str(tags.get("role", "")) if isinstance(tags, dict) else ""


def _tier(tags: dict) -> str:
    return str(tags.get("tier", "")) if isinstance(tags, dict) else ""


def _nodes_table(cfg: dict) -> List[Dict[str, str]]:
    nodes = cfg.get("nodes", {}) or {}
    rows: List[Dict[str, str]] = []
    for nid, c in nodes.items():
        tags = c.get("tags", {}) or {}
        gpu = c.get("gpu", {}) or {}
        rows.append({
            "node": str(nid),
            "tier": _tier(tags),
            "role": _role(tags),
            "gpu": str(gpu.get("name", "")),
            "vram_total_gb": str(gpu.get("vram_total_gb", "")),
            "vram_free_gb": str(gpu.get("vram_free_gb", "")),
        })
    return rows


def _links_table(cfg: dict) -> List[Dict[str, str]]:
    net = cfg.get("network", {}) or {}
    links = net.get("links", []) or []
    rows: List[Dict[str, str]] = []
    for l in links:
        rows.append({
            "src": str(l.get("src", "")),
            "dst": str(l.get("dst", "")),
            "bandwidth_Gbps": str(l.get("bandwidth_Gbps", "")),
            "rtt_ms": str(l.get("rtt_ms", "")),
        })
    return rows


def _load_merged_cfg(base_cfg_path: str | Path, scenario_path: str | Path) -> dict:
    base = load_yaml(base_cfg_path)
    sc = load_yaml(scenario_path)
    return deep_merge(base, sc)


def topology_report(
    base_cfg_path: str | Path,
    scenario_path: str | Path,
    *,
    require_bidirectional: bool = True,
) -> Dict[str, object]:
    cfg = _load_merged_cfg(base_cfg_path, scenario_path)
    lint = lint_scenario_network(
        base_cfg_path,
        scenario_path,
        require_bidirectional=require_bidirectional,
        require_complete=False,
    )
    return {
        "scenario": cfg.get("scenario_id", Path(scenario_path).stem),
        "nodes": _nodes_table(cfg),
        "links": _links_table(cfg),
        "lint_ok": bool(lint.get("ok", False)),
        "lint_issues": lint.get("issues", []),
        "node_count": len(_nodes_table(cfg)),
        "link_count": len(_links_table(cfg)),
    }


def _md_table(headers: List[str], rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_(none)_\n"
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out) + "\n"


def _latex_escape(s: str) -> str:
    return (s.replace("&", "\\&")
             .replace("%", "\\%")
             .replace("#", "\\#")
             .replace("_", "\\_"))


def _latex_table(caption: str, label: str, headers: List[str], rows: List[Dict[str, str]]) -> str:
    cols = "l" * len(headers)
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{_latex_escape(caption)}}}")
    out.append(f"\\label{{{label}}}")
    out.append(f"\\begin{{tabular}}{{{cols}}}")
    out.append("\\toprule")
    out.append(" & ".join(_latex_escape(h) for h in headers) + " \\")
    out.append("\\midrule")
    for r in rows:
        out.append(" & ".join(_latex_escape(str(r.get(h, ""))) for h in headers) + " \\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def write_topology_tables(
    base_cfg_path: str | Path,
    scenarios: List[str | Path],
    out_path: str | Path,
    *,
    fmt: str = "md",
    require_bidirectional: bool = True,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: List[str] = []
    all_reports = []

    for sp in scenarios:
        rep = topology_report(base_cfg_path, sp, require_bidirectional=require_bidirectional)
        all_reports.append(rep)

        sc = rep["scenario"]
        nodes = rep["nodes"]
        links = rep["links"]
        ok = rep["lint_ok"]
        issues = rep["lint_issues"]

        if fmt == "json":
            continue

        if fmt == "md":
            blocks.append(f"## {sc}\n")
            blocks.append(f"- Lint OK: **{ok}**  | Nodes: {rep['node_count']} | Links: {rep['link_count']}\n")
            if not ok and issues:
                blocks.append("**Issues:**\n")
                for it in issues[:20]:
                    blocks.append(f"- {it.get('kind')}: {it.get('src')} -> {it.get('dst')} ({it.get('detail')})")
                blocks.append("")
            blocks.append("### Nodes\n")
            blocks.append(_md_table(["node", "tier", "role", "gpu", "vram_total_gb", "vram_free_gb"], nodes))
            blocks.append("### Links\n")
            blocks.append(_md_table(["src", "dst", "bandwidth_Gbps", "rtt_ms"], links))

        elif fmt == "latex":
            blocks.append(_latex_table(
                caption=f"Scenario {sc}: node configuration (tier/role/GPU).",
                label=f"tab:topo-nodes-{str(sc).lower()}",
                headers=["node", "tier", "role", "gpu", "vram_total_gb", "vram_free_gb"],
                rows=nodes,
            ))
            blocks.append(_latex_table(
                caption=f"Scenario {sc}: directed network links.",
                label=f"tab:topo-links-{str(sc).lower()}",
                headers=["src", "dst", "bandwidth_Gbps", "rtt_ms"],
                rows=links,
            ))
            if (not ok) and issues:
                blocks.append("% Lint issues (not for camera-ready unless needed)")
                for it in issues[:30]:
                    blocks.append(f"% {it.get('kind')}: {it.get('src')} -> {it.get('dst')} ({it.get('detail')})")

        elif fmt == "csv":
            # write two CSVs: nodes and links
            base = out_path.with_suffix("")
            nodes_csv = base.parent / (base.name + "_nodes.csv")
            links_csv = base.parent / (base.name + "_links.csv")
            with nodes_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["scenario","node","tier","role","gpu","vram_total_gb","vram_free_gb"])
                w.writeheader()
                for r in nodes:
                    w.writerow({"scenario": sc, **r})
            with links_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["scenario","src","dst","bandwidth_Gbps","rtt_ms"])
                w.writeheader()
                for r in links:
                    w.writerow({"scenario": sc, **r})
            return out_path

        else:
            raise ValueError(f"Unknown fmt: {fmt}")

    if fmt == "json":
        out_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    else:
        out_path.write_text("\n".join(blocks), encoding="utf-8")
    return out_path
