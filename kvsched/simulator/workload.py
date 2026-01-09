from __future__ import annotations
from typing import List
from ..models.request import InferenceRequest, RequestProfile, RequestQoS, RequestRuntime, KVCacheState, Stage, Priority

def make_synthetic_requests(n: int, owner_nodes: list[str]) -> List[InferenceRequest]:
    reqs: List[InferenceRequest] = []
    for i in range(n):
        owner = owner_nodes[i % len(owner_nodes)]
        reqs.append(InferenceRequest(
            request_id=f"r-{i:04d}",
            profile=RequestProfile(model_id="llama-8b", prompt_tokens=256, max_new_tokens=128, decode_micro_batch=1),
            qos=RequestQoS(priority=Priority.NORMAL),
            runtime=RequestRuntime(stage=Stage.DECODE, seq_len=256, generated_tokens=0),
            kv=KVCacheState(owner_node_id=owner, kv_bytes=2 * 1024**3, num_cached_tokens=256),
            tags={"synthetic": "1"},
        ))
    return reqs
