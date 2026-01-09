# kv-aware-scheduling

Research/prototype project for **KV-aware scheduling** in **distributed LLM inference** (Prefill/Decode disaggregation).

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m kvsched.cli --help
```

## Repo layout (high level)
- `src/kvsched/models/`: Request / Node / Network models (Pydantic v2)
- `src/kvsched/scheduler/`: baselines + KV-aware heuristic
- `src/kvsched/simulator/`: simple discrete-time simulator + metrics
- `configs/`: experiment & scheduler YAML configs
- `results/`: outputs (raw / aggregated / figures)

This repository is a **research scaffold**: extend the simulator and schedulers to match your paper/proposal.
