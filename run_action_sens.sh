#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source .venv_vllm/bin/activate
N=${1:-120}; SEED=${2:-999}
for name in fdm0 fdm_idm fdm_oracle; do
  case $name in
    fdm0) ad="adapters/cwm_gametick_stepover";;
    fdm_idm) ad="adapters/cwm_fdm_idm_r1";;
    fdm_oracle) ad="adapters/cwm_fdm_oracle_r1";;
  esac
  echo "=========== ACTION-SENS $name ($ad) ==========="
  python3 run_action_sensitivity.py facebook/cwm --tp 4 --n "$N" --seed "$SEED" \
    --lora "$ad" --out "results/action_sens_${name}_n${N}_s${SEED}.json" 2>&1 \
    | grep -aE "CWM |OVERALL|SEPARABLE|INSENS|\[|saved|Error|Traceback"
done
echo "=== SENS SUMMARY ==="
python3 - <<'PY'
import json,glob
for name in ["fdm0","fdm_idm","fdm_oracle"]:
    fs=sorted(glob.glob(f"results/action_sens_{name}_*.json"))
    if not fs: print(name,"(none)"); continue
    d=json.load(open(fs[-1])); s=d["action_separable"]; o=d["overall"]
    print(f"{name:11s} SEP: true={s['true_acc']} swap={s['swap_acc']} (Δ={s['true_minus_swap']}) pred_div={s['pred_diversity']}/4 | overall track={o['track_acc']} gt_div={o['gt_diversity']}")
PY
