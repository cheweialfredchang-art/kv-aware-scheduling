#!/usr/bin/env bash
set -e
python -m kvsched.cli --scheduler kv_heuristic --ticks 200 --seed 0 --out results/raw/demo
