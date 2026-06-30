#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source .venv_vllm/bin/activate
python3 run_uitrans_probe.py facebook/cwm --tp 4 --lora adapters/cwm_heldapp --data "data/uidom_togglelist_heldout.jsonl" --out results/heldapp_sft.json 2>&1 | grep -aE "\[uidom|OVERALL"
python3 run_uitrans_probe.py facebook/cwm --tp 4 --lora adapters/cwm_cascade --data "data/uidom_togglelist_heldout.jsonl" --out results/heldapp_cascade.json 2>&1 | grep -aE "\[uidom|OVERALL"
echo DONE
