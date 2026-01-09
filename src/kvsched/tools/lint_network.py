from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from kvsched.config.loader import load_yaml, deep_merge


@dataclass
class LintIssue:
    kind: str
    src: str
    dst: str
    detail: str


def _node_ids(cfg: dict) -> List[str]:
    nodes = cfg.get("nodes", {})
    if isinstance(nodes, dict):
        return list(nodes.keys())
    return []


def _links(cfg: dict) -> Set[Tuple[str, str]]:
    net = cfg.get("network", {}) or {}
    links = net.get("links", []) or []
    out: Set[Tuple[str, str]] = set()
    for l in links:
        try:
            out.add((str(l["src"]), str(l["dst"])))
        except Exception:
            continue
    return out


def _directed_pairs(nodes: List[str], *, include_self: bool = False) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for a in nodes:
        for b in nodes:
            if (not include_self) and a == b:
                continue
            pairs.add((a, b))
    return pairs


def lint_scenario_network(
    base_cfg_path: str | Path,
    scenario_path: str | Path,
    *,
    require_bidirectional: bool = True,
    require_complete: bool = False,
) -> Dict[str, object]:
    """Lint a scenario network definition.

    Checks:
      - link endpoints exist in nodes
      - (optional) every directed pair has a link (complete graph requirement)
      - (optional) every link is bidirectional (src->dst implies dst->src)
    """
    base_cfg = load_yaml(base_cfg_path)
    sc_cfg = load_yaml(scenario_path)
    cfg = deep_merge(base_cfg, sc_cfg)

    nodes = _node_ids(cfg)
    link_set = _links(cfg)

    issues: List[LintIssue] = []

    # endpoint existence
    node_set = set(nodes)
    for (s, d) in sorted(link_set):
        if s not in node_set:
            issues.append(LintIssue("unknown_endpoint", s, d, "src not in nodes"))
        if d not in node_set:
            issues.append(LintIssue("unknown_endpoint", s, d, "dst not in nodes"))

    # bidirectional requirement
    if require_bidirectional:
        for (s, d) in sorted(link_set):
            if (d, s) not in link_set:
                issues.append(LintIssue("missing_reverse_link", s, d, "dst->src missing"))

    # completeness requirement (very strict)
    if require_complete:
        need = _directed_pairs(nodes, include_self=False)
        missing = sorted(list(need - link_set))
        for (s, d) in missing:
            issues.append(LintIssue("missing_link", s, d, "no explicit link for directed pair"))

    ok = len(issues) == 0
    return {
        "ok": ok,
        "scenario": str(Path(scenario_path).name),
        "node_count": len(nodes),
        "link_count": len(link_set),
        "issues": [issue.__dict__ for issue in issues],
    }


def lint_scenarios_dir(
    base_cfg_path: str | Path,
    scenarios_dir: str | Path,
    *,
    require_bidirectional: bool = True,
    require_complete: bool = False,
) -> Dict[str, object]:
    p = Path(scenarios_dir)
    yamls = sorted(list(p.glob("*.yaml")))
    results = []
    ok_all = True
    for y in yamls:
        r = lint_scenario_network(
            base_cfg_path, y,
            require_bidirectional=require_bidirectional,
            require_complete=require_complete,
        )
        results.append(r)
        ok_all = ok_all and bool(r["ok"])
    return {"ok": ok_all, "results": results}
