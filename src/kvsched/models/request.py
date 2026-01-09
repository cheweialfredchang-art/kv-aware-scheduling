from __future__ import annotations
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator
import time

class Stage(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    DONE = "done"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class KVCacheState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_node_id: str = Field(..., description="Current KV cache owner node id")
    kv_bytes: int = Field(..., ge=0, description="KV cache size in bytes")
    num_cached_tokens: int = Field(..., ge=0, description="Number of cached tokens")
    version: int = Field(0, ge=0, description="KV state version (increment on migration/update)")

class RequestQoS(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priority: Priority = Priority.NORMAL
    deadline_ms: Optional[int] = Field(None, ge=1)
    target_p99_ms: Optional[int] = Field(None, ge=1)
    max_queue_ms: Optional[int] = Field(None, ge=1)

class RequestProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(..., min_length=1)
    prompt_tokens: int = Field(..., ge=0)
    max_new_tokens: int = Field(..., ge=1)
    decode_micro_batch: int = Field(1, ge=1)
    avg_tpot_ms_hint: Optional[float] = Field(None, gt=0)

class RequestRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Stage = Stage.PREFILL
    seq_len: int = Field(0, ge=0)
    generated_tokens: int = Field(0, ge=0)
    arrival_ts: float = Field(default_factory=lambda: time.time())
    queue_enter_ts: float = Field(default_factory=lambda: time.time())
    last_scheduled_ts: Optional[float] = None

class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(..., min_length=1)
    profile: RequestProfile
    qos: RequestQoS = Field(default_factory=RequestQoS)
    runtime: RequestRuntime = Field(default_factory=RequestRuntime)
    kv: KVCacheState
    tags: Dict[str, str] = Field(default_factory=dict)
    user_group: Optional[str] = None

    @model_validator(mode="after")
    def _sanity_check(self) -> "InferenceRequest":
        if self.runtime.seq_len < self.profile.prompt_tokens and self.runtime.stage != Stage.PREFILL:
            raise ValueError("runtime.seq_len cannot be smaller than prompt_tokens after prefill.")
        if self.runtime.generated_tokens > self.profile.max_new_tokens:
            raise ValueError("generated_tokens exceeds max_new_tokens.")
        if self.kv.num_cached_tokens > self.runtime.seq_len and self.runtime.seq_len > 0:
            raise ValueError("kv.num_cached_tokens cannot exceed runtime.seq_len.")
        return self
