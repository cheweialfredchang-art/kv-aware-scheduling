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


## Batch + Plots

```bash
python -m kvsched.cli batch --scenarios configs/scenarios/ --schedulers kv_heuristic,load_only,kv_only,random --seeds 0,1,2 --ticks 1000 --out results/raw --index results/aggregated/index.csv
# Figures will be saved to results/figures (cdf_*.png, p99_bar.png, migration_bytes_bar.png)
```


## Prefill/Decode CDF + Prefetch/Overlap Ablation

Batch will run 4 ablation combos by default (pf,ov) in {0,1}^2 and generate figures if plotting deps are installed.

```bash
python -m kvsched.cli batch --scenarios configs/scenarios/ --schedulers kv_heuristic,load_only,kv_only,random --seeds 0,1,2 --ticks 1000
```

Figures:
- results/figures/cdf_prefill_<scenario>.png
- results/figures/cdf_decode_<scenario>.png
- results/figures/p99_total_bar.png
- results/figures/migration_bytes_bar.png


## Network lint

```bash
python -m kvsched.cli lint-network --scenarios configs/scenarios/ --require-bidirectional --json results/lint_report.json
```

## Strict vs relaxed comparison plots

Run twice (relaxed and strict), then compare:
```bash
python -m kvsched.cli batch --scenarios configs/scenarios/ --schedulers kv_heuristic --seeds 0,1,2 --index results/relaxed/index.csv
python -m kvsched.cli batch --scenarios configs/scenarios/ --schedulers kv_heuristic --seeds 0,1,2 --strict-network --index results/strict/index.csv
python -m kvsched.cli compare-modes --relaxed results/relaxed/index.csv --strict results/strict/index.csv --out results/figures
```


## Changelog

- v9: Remove pandas FutureWarning by avoiding to_numeric(errors='ignore') in plotting utilities.


## Topology table generation

```bash
python -m kvsched.cli topology-table --scenarios configs/scenarios/ --out results/topology.md --format md --require-bidirectional
```
LaTeX output:
```bash
python -m kvsched.cli topology-table --scenarios configs/scenarios/ --out results/topology_tables.tex --format latex --require-bidirectional
```


## Experiment suite (S1–S6)

Run the full experiment suite:
```bash
python -m kvsched.cli suite --suite configs/experiments/suite.yaml
```
Generate topology tables (Markdown/LaTeX):
```bash
python -m kvsched.cli topology-table --scenarios configs/scenarios/ --out results/topology_tables.tex --format latex --require-bidirectional
```
Scheduler hyper-parameters for kv_heuristic (JSON):
```bash
python -m kvsched.cli run --scenario configs/scenarios/S5_hysteresis_horizon.yaml --scheduler kv_heuristic \
  --policy-json '{"hysteresis_ms":10, "enable_prefetch":true, "enable_overlap":true, "prefetch_horizon_tokens":256}'
```


## Windows (run without install)

This repo includes a top-level `kvsched/` package so you can run:
```powershell
python -m kvsched.cli ...
```

Compare modes accepts either CSV paths or directories containing `index.csv`:
```powershell
python -m kvsched.cli compare-modes --relaxed results/relaxed --strict results/strict --out results/figures
```


### Batch plots output
`batch` writes figures into `--figdir` (default: `<out>/figures`).
