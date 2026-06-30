#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source .venv_vllm/bin/activate
for k in 1 2 3 5 0; do
  python3 run_drift_stats.py facebook/cwm --app togglelist --n_roll 24 --steps 8 --reground_k $k \
    --tag base_k$k --out results/knee_base_k$k.json 2>&1 | grep -aE "mean_step_acc"
done
echo "=== KNEE DONE ==="
