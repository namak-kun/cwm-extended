#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source .venv_vllm/bin/activate
python3 run_drift_stats.py facebook/cwm --app togglelist --n_roll 16 --steps 8 --reground_k 0 --tag base_k0 --out results/driftstats_base_k0.json 2>&1 | grep -aE "\[|per-step"
python3 run_drift_stats.py facebook/cwm --app togglelist --n_roll 16 --steps 8 --reground_k 3 --tag base_k3 --out results/driftstats_base_k3.json 2>&1 | grep -aE "\[|per-step"
python3 run_drift_stats.py facebook/cwm --app togglelist --n_roll 16 --steps 8 --reground_k 0 --lora adapters/cwm_cascade --tag sft_k0 --out results/driftstats_sft_k0.json 2>&1 | grep -aE "\[|per-step"
python3 run_drift_stats.py facebook/cwm --app togglelist --n_roll 16 --steps 8 --reground_k 3 --lora adapters/cwm_cascade --tag sft_k3 --out results/driftstats_sft_k3.json 2>&1 | grep -aE "\[|per-step"
echo "=== DRIFT STUDY DONE ==="
