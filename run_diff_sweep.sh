#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source .venv_vllm/bin/activate
for arm in fdm0:adapters/cwm_gametick_stepover idmr2:adapters/cwm_fdm_idm_r2; do
  nm=${arm%%:*}; ad=${arm##*:}
  echo "===== HARD K6-8 T4-6 $nm ====="
  python3 run_gametick_abstract.py facebook/cwm --tp 4 --n 30 --seed 999 --lora "$ad" \
    --kmin 6 --kmax 8 --tmin 4 --tmax 6 --out "results/diff_hard_${nm}.json" 2>&1 \
    | grep -aE "per_tick_state_acc|all_ticks|compression|saved|Error"
done
python3 -c "import json;[print(n,json.load(open(f'results/diff_hard_{n}.json'))['per_tick_state_acc'],json.load(open(f'results/diff_hard_{n}.json'))['all_ticks_correct_rate']) for n in ['fdm0','idmr2']]"
