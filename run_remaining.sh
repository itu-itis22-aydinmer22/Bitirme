#!/usr/bin/env bash
# Runs the remaining experiments sequentially. Invoked via run_in_background.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p results/logs

echo "[run_remaining] starting multi_seed at $(date)"
python -u experiments/multi_seed.py > results/logs/multi_seed.log 2>&1
echo "[run_remaining] multi_seed done"

echo "[run_remaining] starting benchmark at $(date)"
python -u experiments/benchmark_models.py > results/logs/benchmark.log 2>&1
echo "[run_remaining] benchmark done"

echo "[run_remaining] starting ablation at $(date)"
python -u experiments/ablation.py > results/logs/ablation.log 2>&1
echo "[run_remaining] ablation done"

echo "[run_remaining] ALL EXPERIMENTS DONE at $(date)"
