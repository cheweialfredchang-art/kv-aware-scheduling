from __future__ import annotations
from .analytic import GPUParams

GPU_TABLE = {
    "A100_80GB": GPUParams(name="A100_80GB", alpha_ms=0.2, beta_ms_per_token=0.002, gamma_ms_per_batch=0.01),
    "L40S_48GB": GPUParams(name="L40S_48GB", alpha_ms=0.25, beta_ms_per_token=0.0025, gamma_ms_per_batch=0.012),
"RTX6000_ADA_48GB": GPUParams(name="RTX6000_ADA_48GB", alpha_ms=0.22, beta_ms_per_token=0.0022, gamma_ms_per_batch=0.011),
"RTX4000_ADA_20GB": GPUParams(name="RTX4000_ADA_20GB", alpha_ms=0.28, beta_ms_per_token=0.0030, gamma_ms_per_batch=0.014),
"RTX2000_ADA_16GB": GPUParams(name="RTX2000_ADA_16GB", alpha_ms=0.32, beta_ms_per_token=0.0035, gamma_ms_per_batch=0.016),
}
