#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source .venv_vllm/bin/activate
for name in idm_r2 oracle_r2; do
  ad="adapters/cwm_fdm_${name}"
  echo "===== MULTITICK $name ====="
  python3 run_gametick_abstract.py facebook/cwm --tp 4 --n 40 --seed 999 --lora "$ad" \
    --out "results/flywheel_eval_fdm_${name}_n40_s999.json" 2>&1 | grep -aE "per_tick_state_acc|all_ticks|saved|Error"
  echo "===== ACTSENS $name ====="
  python3 run_action_sensitivity.py facebook/cwm --tp 4 --n 120 --seed 999 --lora "$ad" \
    --out "results/action_sens_fdm_${name}_n120_s999.json" 2>&1 | grep -aE "OVERALL|SEPARABLE|\[contact|\[stomp|\[death|saved|Error"
done
echo "=== R2 SUMMARY ==="
python3 - <<'PY'
import json,glob
for n in ["fdm_idm","fdm_idm_r2","fdm_oracle_r2"]:
    e=sorted(glob.glob(f"results/flywheel_eval_{n}_n40_s999.json")); s=sorted(glob.glob(f"results/action_sens_{n}_n120_s999.json"))
    pt=json.load(open(e[-1]))["per_tick_state_acc"] if e else "?"; sd=json.load(open(s[-1]))["action_separable"]["true_acc"] if s else "?"
    print(f"{n:14s} multitick_per_tick={pt}  single_true_acc={sd}")
PY
