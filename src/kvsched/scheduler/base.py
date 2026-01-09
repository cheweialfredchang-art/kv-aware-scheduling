from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional
from ..models.request import InferenceRequest

class SchedulingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tick: int = Field(..., ge=0)
    request_id: str
    src_node_id: str
    dst_node_id: str
    will_migrate: bool
    prefetch: bool = False
    reason: Optional[str] = None

class Scheduler(ABC):
    @abstractmethod
    def decide(self, tick: int, req: InferenceRequest, nodes: Dict[str, object]) -> SchedulingDecision:
        raise NotImplementedError
