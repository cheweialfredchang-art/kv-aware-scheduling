from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(..., min_length=1)
    scheduler: str = Field(..., min_length=1)
    seed: int = Field(0, ge=0)
    ticks: int = Field(1000, ge=1)

    workload: Dict[str, Any] = Field(default_factory=dict)
    nodes: Dict[str, Any] = Field(default_factory=dict)
    network: Dict[str, Any] = Field(default_factory=dict)
    scheduler_params: Dict[str, Any] = Field(default_factory=dict)

class BatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenarios: list[str]
    schedulers: list[str]
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    ticks: int = 1000
