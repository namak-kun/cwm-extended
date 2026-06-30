"""Upload the 15 trained LoRA adapters to a HF model repo, PRIORITY-ORDERED so the most
important ones (game-tick SFT win + flywheel) land first if the run is interrupted."""
import os, sys, time
from huggingface_hub import HfApi, create_repo

REPO = "nmk-kun/cwm-extended-adapters"
ROOT = "adapters"
# priority: headline results first, controls/studies last
PRIORITY = [
    "cwm_gametick_stepover",   # SFT win 0.017->0.692
    "cwm_fdm_idm_r1",          # flywheel round 1 (headline)
    "cwm_fdm_idm_r2",          # flywheel round 2 (stable)
    "cwm_cascade",             # UI cascade SFT
    "cwm_fdm_oracle_r1",       # oracle control
    "cwm_fdm_oracle_r2",
    "cwm_fdm_hardoracle",      # hard-K oracle
    "cwm_heldapp",             # abstraction held-out
    "cwm_mixed_expanded",      # forgetting fix
    "cwm_oop_expanded",        # phi-expansion
    "cwm_arith_gold", "cwm_arith_drift", "cwm_arith_wholetrace",
    "cwm_dagger_gold", "cwm_dagger_drift",
]

api = HfApi()
create_repo(REPO, repo_type="model", exist_ok=True, private=False)
print(f"[hf] repo ready: https://huggingface.co/{REPO}", flush=True)

# upload the README at root first (orientation)
readme = f"""---
license: other
license_name: fair-noncommercial-research
base_model: facebook/cwm
tags: [lora, peft, code-world-model, world-model]
---

# CWM-Extended LoRA adapters

Trained LoRA adapters (r=16, attn+MLP) for the **CWM interactive/visual world-model** project
(code + full REPORT: https://github.com/namak-kun/cwm-extended). Base model: `facebook/cwm`.

Each subfolder is a loadable PEFT adapter. Headline ones:

| adapter | what it teaches | result |
|---|---|---|
| `cwm_gametick_stepover` | one-shot game-tick transition | 0.017 -> 0.692 |
| `cwm_fdm_idm_r1` / `_r2` | FDM<->IDM self-labeling flywheel | 0.525 -> 0.683 (~= oracle), stable 2 rounds |
| `cwm_cascade` | UI DOM-cascade step-over | in-dist win (uidom 0.80->1.0); NB negative transfer to real JS |
| `cwm_fdm_oracle_r1/_r2`, `cwm_fdm_hardoracle` | oracle-label controls | flywheel validation |
| `cwm_heldapp` | UI minus togglelist (abstraction test) | cross-app 0.44 -> 0.56 |
| `cwm_oop_expanded`, `cwm_mixed_expanded` | phi-expansion + forgetting fix | 0.02 -> 0.93 |
| `cwm_arith_*`, `cwm_dagger_*` | arithmetic / DAgger studies | mostly neutral (capability hole) |

Load (PEFT):
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("facebook/cwm")
model = PeftModel.from_pretrained(base, "{REPO}", subfolder="cwm_gametick_stepover")
```
"""
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=REPO, repo_type="model")
print("[hf] root README uploaded", flush=True)

done = []
for name in PRIORITY:
    src = os.path.join(ROOT, name)
    if not os.path.isdir(src):
        print(f"[hf] SKIP missing {name}", flush=True); continue
    t0 = time.time()
    print(f"[hf] uploading {name} ...", flush=True)
    for attempt in range(3):
        try:
            api.upload_folder(folder_path=src, path_in_repo=name, repo_id=REPO, repo_type="model",
                              ignore_patterns=["README.md"],
                              commit_message=f"add adapter {name}")
            done.append(name)
            print(f"[hf] DONE {name} ({len(done)}/{len(PRIORITY)}) in {time.time()-t0:.0f}s", flush=True)
            break
        except Exception as e:
            print(f"[hf] retry {name} attempt {attempt+1}: {e}", flush=True); time.sleep(10)
    else:
        print(f"[hf] FAILED {name} after 3 attempts", flush=True)
print(f"[hf] ALL DONE: {len(done)}/{len(PRIORITY)} -> https://huggingface.co/{REPO}", flush=True)
