from __future__ import annotations

from typing import List, Dict, Any
import random

from ..models.request import (
    InferenceRequest, RequestProfile, RequestQoS, RequestRuntime,
    KVCacheState, Stage, Priority
)


def _choose_from_dist(rng: random.Random, dist: Dict[int, float], default: int) -> int:
    if not dist:
        return default
    items = list(dist.items())
    vals, probs = zip(*items)
    total = float(sum(probs)) if probs else 0.0
    if total <= 0:
        return default
    r = rng.random() * total
    c = 0.0
    for v, p in items:
        c += float(p)
        if r <= c:
            return int(v)
    return int(vals[-1])


def make_synthetic_requests(
    n: int,
    owner_nodes: list[str],
    workload_cfg: Dict[str, Any] | None = None,
    *,
    seed: int = 0,
) -> List[InferenceRequest]:
    """Create request templates (Prefill stage).

    The simulator deep-copies a template per tick, so each tick is an independent request.
    workload_cfg supports:
      - prompt_tokens_dist: {tokens: weight, ...}
      - max_new_tokens_dist: {tokens: weight, ...}
      - kv_bytes_per_token: int (bytes/token)
      - kv_base_bytes: int (bytes)
      - model_id: str
      - decode_micro_batch: int
    """
    cfg = workload_cfg or {}
    rng = random.Random(seed)

    prompt_dist = cfg.get("prompt_tokens_dist", {}) or {}
    new_dist = cfg.get("max_new_tokens_dist", {}) or {}

    default_prompt = int(cfg.get("default_prompt_tokens", 512))
    default_new = int(cfg.get("default_max_new_tokens", 256))

    kv_bytes_per_token = int(cfg.get("kv_bytes_per_token", 4096))  # ~4KB/token (placeholder)
    kv_base_bytes = int(cfg.get("kv_base_bytes", 0))

    model_id = str(cfg.get("model_id", "llama-8b"))
    decode_micro_batch = int(cfg.get("decode_micro_batch", 1))

    reqs: List[InferenceRequest] = []
    for i in range(n):
        owner = owner_nodes[i % len(owner_nodes)] if owner_nodes else "node0"
        prompt_tokens = _choose_from_dist(rng, prompt_dist, default_prompt) if prompt_dist else default_prompt
        max_new = _choose_from_dist(rng, new_dist, default_new) if new_dist else default_new

        # approximate KV size; grows with context length
        kv_bytes = kv_base_bytes + kv_bytes_per_token * int(prompt_tokens)

        reqs.append(InferenceRequest(
            request_id=f"r-{i:05d}",
            profile=RequestProfile(
                model_id=model_id,
                prompt_tokens=int(prompt_tokens),
                max_new_tokens=int(max_new),
                decode_micro_batch=decode_micro_batch,
            ),
            qos=RequestQoS(priority=Priority.NORMAL),
            runtime=RequestRuntime(stage=Stage.PREFILL, seq_len=int(prompt_tokens), generated_tokens=0),
            kv=KVCacheState(owner_node_id=owner, kv_bytes=int(kv_bytes), num_cached_tokens=0),
            tags={"synthetic": "1"},
        ))
    return reqs
