from __future__ import annotations
from .analytic import GPUParams

GPU_TABLE = {
    "A100_80GB": GPUParams(name="A100_80GB", alpha_ms=0.2, beta_ms_per_token=0.002, gamma_ms_per_batch=0.01),
    "L40S_48GB": GPUParams(name="L40S_48GB", alpha_ms=0.25, beta_ms_per_token=0.0025, gamma_ms_per_batch=0.012),
}
