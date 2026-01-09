from __future__ import annotations
from typing import List
from ..models.request import InferenceRequest, RequestProfile, RequestQoS, RequestRuntime, KVCacheState, Stage, Priority

def make_synthetic_requests(n: int, owner_nodes: list[str]) -> List[InferenceRequest]:
    """Create request templates.

    We create templates in PREFILL stage. The simulator will deep-copy a template
    per tick so that each tick represents an independent request instance.
    """
    reqs: List[InferenceRequest] = []
    for i in range(n):
        # "owner" here is just an initial hint; PREFILL will assign actual KV owner
        owner = owner_nodes[i % len(owner_nodes)] if owner_nodes else "node0"
        prompt_tokens = 256 if (i % 3) else 512
        max_new = 128 if (i % 4) else 256

        reqs.append(InferenceRequest(
            request_id=f"r-{i:05d}",
            profile=RequestProfile(model_id="llama-8b", prompt_tokens=prompt_tokens, max_new_tokens=max_new, decode_micro_batch=1),
            qos=RequestQoS(priority=Priority.NORMAL),
            runtime=RequestRuntime(stage=Stage.PREFILL, seq_len=prompt_tokens, generated_tokens=0),
            kv=KVCacheState(owner_node_id=owner, kv_bytes=2 * 1024**3, num_cached_tokens=0),
            tags={"synthetic": "1"},
        ))
    return reqs
