from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator

class GPUArch(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    OTHER = "other"

class NodeRole(str, Enum):
    PREFILL = "prefill_node"
    DECODE = "decode_node"
    HYBRID = "hybrid_node"

class GPUDevice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arch: GPUArch = GPUArch.NVIDIA
    vram_total_bytes: int = Field(..., ge=0)
    vram_free_bytes: int = Field(..., ge=0)
    peak_tflops_fp16: Optional[float] = Field(None, gt=0)
    max_batch_prefill: Optional[int] = Field(None, ge=1)
    max_batch_decode: Optional[int] = Field(None, ge=1)

    @model_validator(mode="after")
    def _vram_check(self) -> "GPUDevice":
        if self.vram_free_bytes > self.vram_total_bytes:
            raise ValueError("vram_free_bytes cannot exceed vram_total_bytes.")
        return self

class NodeLoad(BaseModel):
    model_config = ConfigDict(extra="forbid")
    running_prefill: int = Field(0, ge=0)
    running_decode: int = Field(0, ge=0)
    queued_prefill: int = Field(0, ge=0)
    queued_decode: int = Field(0, ge=0)
    sm_utilization: Optional[float] = Field(None, ge=0, le=1)
    mem_utilization: Optional[float] = Field(None, ge=0, le=1)

class KVPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kv_used_bytes: int = Field(0, ge=0)
    kv_request_ids: List[str] = Field(default_factory=list)

class NodeCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: NodeRole = NodeRole.HYBRID
    supports_prefetch: bool = True
    supports_overlap: bool = True

class NodeState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str = Field(..., min_length=1)
    gpu: GPUDevice
    caps: NodeCapabilities = Field(default_factory=NodeCapabilities)
    load: NodeLoad = Field(default_factory=NodeLoad)
    kv: KVPlacement = Field(default_factory=KVPlacement)
    tags: Dict[str, str] = Field(default_factory=dict)

    def can_host_kv(self, kv_bytes: int) -> bool:
        return self.gpu.vram_free_bytes >= kv_bytes
