#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source .venv_vllm/bin/activate
for cfg in "togglelist 0 base_tl_k0" "togglelist 3 base_tl_k3"; do
  set -- $cfg
  echo "=== $1 k=$2 ($3) ==="
  python3 render_rollout.py facebook/cwm --tp 4 --app $1 --steps 8 --seed 7 --reground_k $2 --tag $3 --out_dir results/drift2 2>&1 | grep -aE "per-step|GIF"
done
echo "=== DONE ==="
