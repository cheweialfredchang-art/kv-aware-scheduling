from __future__ import annotations
from enum import Enum
from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict

class LinkType(str, Enum):
    ETHERNET = "ethernet"
    INFINIBAND = "infiniband"
    PCIE = "pcie"
    NVLINK = "nvlink"
    OTHER = "other"

class CongestionModel(str, Enum):
    NONE = "none"
    UTIL_LINEAR = "util_linear"
    MM1 = "mm1"

class NetworkLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    src: str
    dst: str
    link_type: LinkType = LinkType.ETHERNET
    bandwidth_Gbps: float = Field(..., gt=0)
    rtt_ms: float = Field(..., ge=0)
    loss_rate: float = Field(0.0, ge=0, le=1)
    jitter_ms: float = Field(0.0, ge=0)
    utilization: float = Field(0.0, ge=0, le=0.99)
    congestion_model: CongestionModel = CongestionModel.UTIL_LINEAR
    setup_ms: float = Field(0.0, ge=0)

    def capacity_bytes_per_sec(self) -> float:
        return self.bandwidth_Gbps * 1e9 / 8.0

    def est_queue_delay_ms(self, flow_bytes_per_sec: float) -> float:
        if self.congestion_model == CongestionModel.NONE:
            return 0.0
        cap = self.capacity_bytes_per_sec()
        offered = min(0.999, self.utilization + (flow_bytes_per_sec / cap))
        eps = 1e-6
        if self.congestion_model == CongestionModel.UTIL_LINEAR:
            return (self.rtt_ms) * (offered / max(eps, (1.0 - offered)))
        if self.congestion_model == CongestionModel.MM1:
            return (offered / max(eps, (1.0 - offered))) * self.rtt_ms
        return 0.0

    def est_transfer_ms(self, bytes_to_send: int, flow_bytes_per_sec: float = 0.0, include_rtt: bool = True) -> float:
        if bytes_to_send <= 0:
            return 0.0
        cap = self.capacity_bytes_per_sec()
        eff_cap = max(1.0, cap * (1.0 - self.utilization))
        serialization_ms = (bytes_to_send / eff_cap) * 1000.0
        queue_ms = self.est_queue_delay_ms(flow_bytes_per_sec=flow_bytes_per_sec)
        base = self.setup_ms + serialization_ms + queue_ms
        if include_rtt:
            base += self.rtt_ms
        if self.loss_rate > 0:
            base *= (1.0 / max(1e-6, (1.0 - self.loss_rate)))
        return base

class TrafficProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_util: float = Field(0.0, ge=0, le=0.99)
    util_by_pair: Dict[Tuple[str, str], float] = Field(default_factory=dict)
    def get_util(self, src: str, dst: str) -> float:
        return float(self.util_by_pair.get((src, dst), self.default_util))

class NetworkTopology(BaseModel):
    strict_network: bool = False

    model_config = ConfigDict(extra="forbid")
    links: Dict[Tuple[str, str], NetworkLink] = Field(default_factory=dict)
    traffic: TrafficProfile = Field(default_factory=TrafficProfile)
    def add_link(self, link: NetworkLink) -> None:
        self.links[(link.src, link.dst)] = link
    def get_link(self, src: str, dst: str) -> NetworkLink:
        """Return a link from src->dst.

        If strict_network=True, require explicit link.
        Otherwise:
          1) try reverse link (symmetric)
          2) fall back to conservative default
        """
        link = self.links.get((src, dst))
        if link is not None:
            return link
        if self.strict_network:
            raise ValueError(f"No link found for {src} -> {dst} (strict mode)")
        rev = self.links.get((dst, src))
        if rev is not None:
            return rev.model_copy(update={"src": src, "dst": dst})
        return NetworkLink(src=src, dst=dst, bandwidth_Gbps=1.0, rtt_ms=20.0)
    def est_path_transfer_ms(self, src: str, dst: str, bytes_to_send: int, flow_bytes_per_sec: float = 0.0) -> float:
        link = self.get_link(src, dst)
        util = self.traffic.get_util(src, dst)
        link = link.model_copy(update={"utilization": util})
        return link.est_transfer_ms(bytes_to_send, flow_bytes_per_sec=flow_bytes_per_sec, include_rtt=True)

class OverlapModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overlap_ratio: float = Field(0.0, ge=0, le=1)
    max_hide_ms: Optional[float] = Field(None, ge=0)
    def effective_transfer_ms(self, transfer_ms: float, compute_ms: float) -> float:
        hidden = min(transfer_ms, compute_ms) * self.overlap_ratio
        if self.max_hide_ms is not None:
            hidden = min(hidden, self.max_hide_ms)
        return max(0.0, transfer_ms - hidden)

class NetworkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topo: NetworkTopology
    overlap: OverlapModel = Field(default_factory=OverlapModel)
    def est_transfer_ms(self, src: str, dst: str, bytes_to_send: int, flow_bytes_per_sec: float = 0.0) -> float:
        return self.topo.est_path_transfer_ms(src, dst, bytes_to_send, flow_bytes_per_sec)
    def est_effective_transfer_ms(self, src: str, dst: str, bytes_to_send: int, compute_ms: float, prefetch: bool = False) -> float:
        raw = self.est_transfer_ms(src, dst, bytes_to_send)
        ratio = self.overlap.overlap_ratio + (0.3 if prefetch else 0.0)
        tmp = self.overlap.model_copy(update={"overlap_ratio": min(1.0, ratio)})
        return tmp.effective_transfer_ms(raw, compute_ms)
