from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GPUParams:
    name: str
    alpha_ms: float
    beta_ms_per_token: float
    gamma_ms_per_batch: float

def est_compute_delay_ms(seq_len: int, batch_size: int, gpu: GPUParams) -> float:
    return gpu.alpha_ms + gpu.beta_ms_per_token * seq_len + gpu.gamma_ms_per_batch * batch_size

def est_decode_step_ms(context_len: int, micro_batch: int, gpu: GPUParams) -> float:
    return est_compute_delay_ms(seq_len=context_len, batch_size=micro_batch, gpu=gpu)
