# kv-aware-scheduling

Research/prototype project for **KV-aware scheduling** in **distributed LLM inference** (Prefill/Decode disaggregation).

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m kvsched.cli --help

### Examples
```bash
python -m kvsched.cli run --scenario configs/scenarios/S2_edge_cloud.yaml --scheduler kv_heuristic --ticks 1000 --seed 0 --out results/raw/S2_edge_cloud
python -m kvsched.cli batch --scenarios configs/scenarios/ --schedulers kv_heuristic,load_only,kv_only --seeds 0,1,2 --ticks 1000 --out results/raw --index results/aggregated/index.csv
```
```

## Repo layout (high level)
- `src/kvsched/models/`: Request / Node / Network models (Pydantic v2)
- `src/kvsched/scheduler/`: baselines + KV-aware heuristic
- `src/kvsched/simulator/`: simple discrete-time simulator + metrics
- `configs/`: experiment & scheduler YAML configs
- `results/`: outputs (raw / aggregated / figures)

This repository is a **research scaffold**: extend the simulator and schedulers to match your paper/proposal.
