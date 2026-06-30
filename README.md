# CWM-WM — extending the Code World Model into an interactive / visual world model

Research code that extends Meta's **[Code World Model (CWM)](https://github.com/facebookresearch/cwm)** — a
32B LLM trained to predict code *execution-trace state* — from a passive trace predictor into an
**execution-free, interactive, pixel-level world model** for programs, GUI apps, and games.

> **Thesis:** given a program's source as context, CWM can predict how its *symbolic state* evolves under
> a stream of user **inputs/actions**, with **no interpreter/engine in the loop** — and that predicted
> state can be **rendered to pixels** (a real browser/UI frame). So: *given app code + an action sequence,
> generate a video of the UI responding* — without running the app.

This repo is the experiment harness, the trained adapters' recipes, the pixel-rendering pipeline, and a
detailed empirical record. **The master record is [`results/REPORT.md`](results/REPORT.md)** (§0–§39) — read
its TL;DR first.

---

## Headline results

| Axis | Result |
|---|---|
| **Real-app world model** | Base CWM (no SFT) free-rolls the **real TodoMVC reducer** under a user session and renders each predicted state to a UI frame → a faithful **UI video**: **8/8** clean, **16/16** stress (6 todos; add/toggle/**edit**/**delete**/filter), zero drift. (§39) |
| **Real JS execution** | Base CWM predicts a real vanilla-JS form-validator's transitions at **0.75** (full-trace). Earlier "real apps fail" numbers were **harness/token-budget artifacts**, not capability. (§34) |
| **Game-tick dynamics (SFT win)** | Base CWM *cannot* one-shot a game tick (**0.017**); step-over LoRA SFT teaches it → **0.692**. (§30) |
| **FDM↔IDM flywheel** | CWM **self-labels** unlabeled game trajectories (forward-search inverse dynamics) and bootstraps action-conditioned dynamics: per-tick **0.525 → 0.683 ≈ true-action oracle**; stable over 2 rounds. (§32) |
| **DOM render-FDM** | State = canonical DOM-JSON; base **0.75 exact / 0.986 field-F1**; abstraction validated 3 ways (fresh-instance, element-extrapolation, cross-app held-out). (§35) |
| **Pixel pipeline** | DOM-JSON → HTML → headless Chromium → PNG/GIF. Frame-as-generation + free-roll video both work. (§37) |
| **Honest open problem** | Narrow Python-cascade SFT shows **negative transfer to real JS** (0.75 → 0.35). Fix (JS-aware/diverse replay) is unfinished. (§36) |

**Key finding:** base CWM is already a strong execution-free FDM for *in-distribution* code (real web apps);
SFT is what unlocks *compressed/OOD* regimes (game-tick step-over, the flywheel). The recurring lesson across
the project: most "dramatic CWM failures" turned out to be **harness artifacts** (token-cap truncation,
metric confounds), not model limits.

---

## Repo layout

```
wm_probe/
├── results/REPORT.md         # MASTER RECORD (§0–§39). Start here.
├── models/cwm_trace.py       # self-contained CWM trace client over in-process vLLM (verified token IDs)
├── ui_tick.py  ui_dom.py     # UI/DOM world-model apps (state = canonical DOM tree)
├── game_tick.py              # game-tick world (player + K enemies + within-tick side-effects)
├── build_*.py                # SFT/eval data builders (contract-format)
├── run_*.py                  # probes/evals (uitrans, gametick, drift, flywheel, …)
├── dom_render.py             # DOM-JSON -> HTML -> headless-Chromium PNG
├── render_rollout.py         # free-roll DOM rollout -> frames + GIF (--reground_k)
├── todomvc_video.py          # REAL-app capstone: free-roll real TodoMVC -> UI video
├── run_drift_stats.py        # powered drift study (batched rollouts + bootstrap CIs)
├── uidata/CONTRACT.md        # data contract; harvested real-app transitions
├── data/                     # generated + harvested transition/SFT data (.jsonl)
└── jsdeps/                   # JS deps (jsdom) for parsing real JS reducers; `npm install`
```

> Note: `models/cwm_trace.py` **re-implements** CWM's trace prompt/parse format over vLLM (it does *not*
> import the upstream `cwm` package). The original `cwm/` checkout was used only as reference
> (`PROMPTING_GUIDE`, `demos/cwmdbg.py`) and was **never modified** — hence this is a standalone repo, not a
> fork.

---

## Setup

Two virtualenvs are used (vLLM and the trainer pin slightly different torch builds):

```bash
# Inference / eval / rendering (needs a CUDA box; CWM 32B runs tp=4 on 4×~46GB)
python -m venv .venv_vllm && . .venv_vllm/bin/activate
pip install -r requirements.txt
python -m playwright install chromium      # for the pixel renderer
deactivate

# Training (LoRA SFT)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt
deactivate

# JS reducer parsing (TodoMVC etc.)
cd jsdeps && npm install && cd ..
```

The base model is **`facebook/cwm`** (downloaded from the HF Hub on first use).

---

## Reproduce the headline demo (real-app UI video)

```bash
. .venv_vllm/bin/activate
# clean 8-action session
python todomvc_video.py facebook/cwm --tp 4 --out_dir results/todomvc_video
# harder 16-action stress session (6 todos, edit/delete/filters)
python todomvc_video.py facebook/cwm --tp 4 --stress --out_dir results/todomvc_video_stress
# -> results/todomvc_video*/todomvc.gif  + step*_OK.png frames
```

See `results/REPORT.md` §38 for the full reproducibility appendix (every probe/eval command).

---

## Trained adapters

15 LoRA adapters (r=16, attn+MLP) were trained — the actual weight changes (game-tick step-over, the
flywheel rounds, UI cascade, oop/φ-expansion, arithmetic/DAgger studies). Each `adapter_model.safetensors`
is ~477MB, so they are **not committed here** (exceed GitHub's 100MB/file limit). They are hosted on the
Hugging Face Hub:

**🤗 [`nmk-kun/cwm-extended-adapters`](https://huggingface.co/nmk-kun/cwm-extended-adapters)**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("facebook/cwm")
model = PeftModel.from_pretrained(base, "nmk-kun/cwm-extended-adapters",
                                  subfolder="cwm_gametick_stepover")
```

The **training recipes and data builders are in this repo** and can reproduce them on a CUDA box
(`upload_adapters_hf.py` re-uploads them).

---

## Attribution

Builds on Meta FAIR's **Code World Model**: <https://github.com/facebookresearch/cwm>. The trace
prompt/parse logic in `models/cwm_trace.py` is a vLLM re-implementation informed by the upstream
`demos/cwmdbg.py` and `PROMPTING_GUIDE`. Upstream CWM is distributed under the FAIR Noncommercial Research
License; this research code is intended for noncommercial research use consistent with that.
