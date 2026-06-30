#!/usr/bin/env bash
# Decisive 3-way flywheel eval (REPORT §32).
# ONE vLLM process per adapter (vllm 0.23 LoRA-switch bug: must isolate adapters).
# All arms: held-out seed 999 (disjoint from unlabeled-train seed 4321), n programs identical per seed.
set -u
cd "$(dirname "$0")"
source .venv_vllm/bin/activate
N=${1:-40}
SEED=${2:-999}
declare -A ARMS=(
  [fdm0]="adapters/cwm_gametick_stepover"
  [fdm_idm]="adapters/cwm_fdm_idm_r1"
  [fdm_oracle]="adapters/cwm_fdm_oracle_r1"
)
for name in fdm0 fdm_idm fdm_oracle; do
  adapter=${ARMS[$name]}
  out="results/flywheel_eval_${name}_n${N}_s${SEED}.json"
  echo "=================== EVAL $name ($adapter) -> $out ==================="
  python3 run_gametick_abstract.py facebook/cwm --tp 4 --n "$N" --seed "$SEED" \
     --lora "$adapter" --out "$out" 2>&1 | grep -aE "loaded|per_tick_state_acc|compression|all_ticks|saved|Error|Traceback"
  echo "[done $name]"
done
echo "=== SUMMARY ==="
python3 - <<'PY'
import json
for name in ["fdm0","fdm_idm","fdm_oracle"]:
    import glob
    fs=sorted(glob.glob(f"results/flywheel_eval_{name}_*.json"))
    if not fs: 
        print(f"{name}: (no result)"); continue
    d=json.load(open(fs[-1]))
    print(f"{name:11s} per_tick={d['per_tick_state_acc']}  all_ticks={d['all_ticks_correct_rate']}  n={d['n']} seed={d['seed']}  ({fs[-1]})")
PY
