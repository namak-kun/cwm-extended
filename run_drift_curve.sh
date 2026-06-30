#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source .venv_vllm/bin/activate
echo "=== base free-roll (k=0) ==="
python3 render_rollout.py facebook/cwm --tp 4 --app tabs --steps 12 --seed 42 --tag base_k0 --out_dir results/drift_curve 2>&1 | grep -aE "per-step|GIF"
echo "=== base re-grounded (k=3) ==="
python3 render_rollout.py facebook/cwm --tp 4 --app tabs --steps 12 --seed 42 --reground_k 3 --tag base_k3 --out_dir results/drift_curve 2>&1 | grep -aE "per-step|GIF"
echo "=== SFT free-roll (k=0) ==="
python3 render_rollout.py facebook/cwm --tp 4 --app tabs --steps 12 --seed 42 --lora adapters/cwm_cascade --tag sft_k0 --out_dir results/drift_curve 2>&1 | grep -aE "per-step|GIF"
echo "=== DONE ==="
