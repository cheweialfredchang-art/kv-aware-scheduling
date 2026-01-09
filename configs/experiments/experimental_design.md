# Experimental Design Overview (S1–S6)

This project includes an experiment suite that decomposes KV-aware scheduling into six controlled scenarios.

## Baselines
- **load_only**: routes by decode queue/load only (ignores KV locality)
- **kv_only**: routes by KV owner only (ignores load)
- **random / rr**: ignores both
- **kv_heuristic**: proposed KV-aware heuristic (locality + load + migration cost), optionally with **prefetch/overlap**

## Common Settings
- Disaggregated inference: Prefill / Decode separated by node tags (`role=prefill` / `role=decode`).
- Online scheduling, repeated across multiple seeds.
- Metrics: end-to-end latency (P50/P95/P99), stage latencies (prefill/decode), KV migration bytes, migration rate, and optional network stall time.

## Scenarios
### S1: KV-locality vs Load
Stress the trade-off between load balancing and KV locality.

### S2: Prompt Length Distribution (KV size impact)
Vary prompt-token distributions to change KV size and migration cost.

### S3: Prefill/Decode Ratio (Phase-aware necessity)
Vary max_new_tokens distributions to create prefill-heavy vs decode-heavy workloads.

### S4: Network Scan (Prefetch/Overlap effective regime)
Scan bandwidth/RTT to find where prefetch/overlap is beneficial.

### S5: Hysteresis & Prefetch Horizon (stability vs waste)
Sweep scheduler policy params (hysteresis, prefetch horizon) to study stability and wasted prefetch.

### S6: Heterogeneous GPUs (latency model necessity)
Mix GPUs with different latency profiles; evaluate scheduler sensitivity to heterogeneity.

See `configs/experiments/suite.yaml` for the runnable sweep plan.
