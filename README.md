# Extending the Code World Model — from a passive trace predictor to an interactive, self‑improving world model over code

Research code built on Meta's **[Code World Model (CWM)](https://github.com/facebookresearch/cwm)** — a 32B LLM
that predicts a program's *execution state* (locals → JSON), frame‑by‑frame, **without running the code**.

CWM out of the box is a **passive tracer of autonomous execution**: feed it source, it imagines the trace.
This project asks what happens when you treat that as the *dynamics model of an interactive system* and build
on top of it. The goal is to use a code‑trained LLM to **predict, edit, and test program behavior without
paying to build or run the program** — including a learned, execution‑free form of "what does this input do,
where does it break, what does the screen become."

> **The master record is [`results/REPORT.md`](results/REPORT.md)** (§0–§39, the experiment log incl. dead ends
> and corrections). A paper‑style synthesis is in [`ACADEMIC_WRITEUP.md`](ACADEMIC_WRITEUP.md). All results are
> LoRA adapters over a **frozen** `facebook/cwm`; the upstream model is unmodified.

---

## What we add over CWM

CWM gives us `source → autonomous execution trace`. On top of that frozen capability we add six things:

| # | Addition (ours) | What CWM does alone | Evidence |
|---|---|---|---|
| **1. Abstraction (tick‑level)** | **Finetune CWM to predict a whole *tick/event* in one shot** (collapse the line‑by‑line interior), ~10× compressed. This is what makes long/expensive programs tractable. | Predicts line‑by‑line; **cannot one‑shot a tick** (measured **0.017**). | §30: 0.017 → **0.692** |
| **2. Self‑improvement (flywheel + inverse dynamics)** | Run CWM **backwards** (forward‑search the FDM to recover the action), use that to **label its own unlabeled trajectories**, and retrain — **zero new labels**. | No inverse/abductive mode; static, no self‑training over its own predictions. | §31–§32: **0.525 → 0.683 = oracle (0.679)** |
| **3. Cross‑language transfer of finetuning** | A skill taught in **Python** (object‑state observability) **transfers to C++/other languages** — fixes are not per‑language. | Traces Rust/Java/JS/C **zero‑shot**, but has no notion of *carrying a targeted fix* across languages. | §11, §28 (Python SFT fixed a C++ class) |
| **4. Object‑state observability** | SFT teaches CWM to **render encapsulated object/container state** each frame — closing the OOP/STL gap. | Keeps objects **opaque** behind method calls and *confabulates* their fields. | §22, §24: oop **0.02 → 0.93** |
| **5. Interactivity** | Treat a program as a dynamical system: **input/action as a first‑class driver** of the trace, `(state, action) → next_state`. | Traces *autonomous* execution; no notion of "user does X, then what." | §32–§39 |
| **6. Render‑grounding (early)** | A `state → pixels` read‑out (DOM‑JSON → real browser → frame/video), so predicted state is viewable. | Stops at text/JSON state. | §35–§39 (nascent; see *Status*) |

And a seventh, non‑code contribution: **a characterization of CWM as a world model** — where it holds, where it
breaks, and why (the wall is *compounding rollout drift*, not per‑step capability; the difficulty envelope of
self‑labeling; the native‑code/concurrency walls; an honest negative‑transfer result; and a set of
measurement‑artifact lessons).

### The salient features: abstraction, self‑teaching, and language‑agnostic fixes

The additions that carry the project are all **our finetuning**, not anything in the base model:

- **Step‑over SFT — the abstraction angle (addition #1).** Base CWM is a step‑by‑step computer — it can't
  compute a full game tick (multi‑entity chase + within‑tick side‑effects) in one shot (0.017). A short LoRA on
  the *step‑over abstraction* (predict `s_{t+1}` from `s_t`, skipping the interior) installs the one‑shot tick
  at **0.692**, ~10× compressed, loss converging cleanly. **Why it matters:** you can advance an expensive
  program one *tick* per query instead of one *line*, and re‑ground every k ticks to bound drift.
- **The FDM↔IDM flywheel (addition #2).** Because the forward model can be searched to act as an inverse model,
  CWM can **manufacture its own action labels** for raw, unlabeled trajectories and retrain on them. This lifts
  held‑out tick accuracy **0.525 → 0.683, matching a true‑action oracle (0.679)**, is genuinely
  action‑conditioned (wrong‑action swap accuracy **0.0**; a distinct prediction per action), and is **stable
  over two rounds** (0.696). **Why it matters:** new dynamics can be learned from logged states alone — no
  engine, wrapper, or human action labels.
- **Language‑agnostic finetuning (additions #3–#4).** CWM is already a strong multi‑language base — Rust, Java,
  JavaScript (loop & map/reduce), and C trace **zero‑shot**, and it tracks evolving **floats exactly** (gradient
  descent). Crucially, **our targeted fixes transfer across languages**: a Python‑only object‑state SFT did not
  narrow the model and actually **repaired a C++ class** (§28). **Why it matters:** you improve and test
  behavior across a polyglot codebase with **one model**, finetuned **once**, not a harness per language.

---

## Why this is useful

The point of an execution‑free model of program behavior is **leverage where building/running is expensive or
awkward**:

1. **Test & edit code without the expensive build/run loop — across languages.** CWM free‑rolls entire
   deterministic programs **perfectly to ~200 frames** and tracks teacher‑forced state to **depth 247** (§8,
   §16), and it does this for **Rust, Java, JavaScript, and C zero‑shot** with exact float tracking (§11). So
   you can predict "what state does this code reach on this input?" — to probe a function, check an edit, or
   generate inputs that drive toward an interesting state — *without standing up the system*, and with **one
   model instead of a per‑language harness**. Games and long‑running simulations are the clearest win: the
   step‑over unit advances them a tick at a time (§30).

2. **A learned, execution‑free form of abstract interpretation for bug‑finding.** CWM's native trace format
   emits **exceptions as a first‑class event** (`exception_sep`), and our harness already measures **invariant
   violations** (out‑of‑bounds, schema‑break) along a predicted rollout (§17). The mechanism to say *"this
   input raises / goes out of range / corrupts the schema at line L"* — without running — therefore already
   exists. Used this way the model is a cheap **proposer**: flag likely‑buggy paths and synthesize adversarial
   test inputs, which a single real run then confirms. **Honest caveat:** this is *not* sound abstract
   interpretation — it's a learned predictor that can be **confidently wrong** (our arithmetic study, §27, is
   exactly this failure), so it generates *candidates*, not proofs. Evaluating CWM as a bug oracle is a
   proposed next step, not a finished result (see *Status*).

3. **Tick‑level prediction of hard, long‑running programs.** For programs too expensive to run many ticks
   (game engines, simulations, agent loops), the recipe is **step‑over SFT (raise the per‑tick ceiling) +
   periodic re‑grounding (kill residual drift)**; the two **compose** (§37.4: base 0.44 → 0.75, non‑overlapping
   CIs; re‑grounding knee at k≈2). The honest boundary: *numerically* continuous, chaotic value computation
   (heavy floating‑point arithmetic) is where per‑step drift dominates (teacher‑forced 0.73 vs free‑roll 0.14)
   and needs frequent re‑grounding or tool‑use, not more imitation (§26–§27).

---

## Status: what's proven vs. what's early

**Proven (held‑out, with controls / CIs):**
- Step‑over SFT teaches the one‑shot tick: **0.017 → 0.692** (§30).
- The self‑labeling flywheel reaches **oracle parity** and is action‑conditioned + 2‑round stable (§32).
- **Multi‑language reach:** Rust/Java/JS/C trace zero‑shot with exact floats, and a Python‑only object‑state SFT
  **transfers to C++** (§11, §28).
- Base CWM is a strong execution‑free state predictor: deterministic free‑roll perfect to ~200 frames (§16);
  DOM/UI next‑state **exact 0.823** on audited metrics that beat the copy baseline (§35); real **TodoMVC** small
  states **1.0**, real vanilla‑JS validator **0.75** full‑trace (§34).

**Early / aspirational (a direction, not a destination):**
- **The visual/pixel axis is nascent.** The `state → pixels` renderer works and the **TodoMVC free‑roll video**
  is faithful (8/8, 16/16; §39) — but that is a *small, in‑distribution app*, not a general visual world model.
  Treat the "video of any app responding to input" framing as the *target*, not a shipped capability.
- **Bug‑finding / abstract interpretation** has the *mechanism* (exception + invariant prediction) but has **not
  yet been evaluated** as a bug oracle on a real bug benchmark.
- **JavaScript is unresolved:** narrow Python‑cascade SFT **negatively transfers** to real JS (full‑trace
  **0.75 → 0.35**, §36); base CWM is the better JS predictor today. The fix (mixed‑corpus/JS replay) is known
  but blocked by the Python‑only tracer for SFT‑target generation.

The throughline: most "dramatic CWM failures" we hit were **measurement artifacts** (token‑cap truncation,
copy‑inflated metrics, wrong extraction mode), not model limits — which is why the proven results above use
audited metrics, budget scaling, and powered drift studies (§7 of the writeup).

---

## Trained adapters (the finetuning)

15 LoRA adapters (r=16, α=32, attn+MLP) over frozen `facebook/cwm` — the actual weight changes. Headlines:
`cwm_gametick_stepover` (tick FDM, addition #2), `cwm_fdm_idm_r1`/`_r2` (the flywheel, additions #3–#4),
`cwm_cascade`/`cwm_heldapp` (UI/DOM render‑FDM + abstraction tests), `cwm_oop_expanded`/`cwm_mixed_expanded`
(object‑state observability + a forgetting fix that **transfers across languages**, §28). Full catalogue with
results and REPORT links: **[`ADAPTERS.md`](ADAPTERS.md)**.

Hosted on the Hub (each `adapter_model.safetensors` ~477MB, > GitHub's 100MB limit):
**🤗 [`nmk-kun/cwm-extended-adapters`](https://huggingface.co/nmk-kun/cwm-extended-adapters)**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("facebook/cwm")
model = PeftModel.from_pretrained(base, "nmk-kun/cwm-extended-adapters", subfolder="cwm_gametick_stepover")
```

---

## Repo layout

```
wm_probe/
├── results/REPORT.md         # MASTER LOG (§0–§39). Start here.
├── ACADEMIC_WRITEUP.md       # paper-style synthesis;  ADAPTERS.md = adapter catalogue
├── models/cwm_trace.py       # self-contained CWM trace client over in-process vLLM (verified token IDs)
├── game_tick.py              # game-tick world (player + K enemies + within-tick side-effects)
├── ui_tick.py  ui_dom.py     # UI/DOM world-model apps (state = canonical DOM tree)
├── build_*.py  run_*.py      # SFT/eval data builders + probes (gametick, flywheel, uitrans, drift)
├── dom_render.py             # DOM-JSON -> HTML -> headless-Chromium PNG  (the pixel read-out)
├── render_rollout.py         # free-roll DOM rollout -> frames + GIF (--reground_k)
├── todomvc_video.py          # real-app demo: free-roll the real TodoMVC reducer -> UI video
├── run_drift_stats.py        # powered drift study (batched rollouts + bootstrap CIs)
└── data/  uidata/            # generated + harvested (303 real) transitions; data contract in uidata/CONTRACT.md
```

> `models/cwm_trace.py` **re‑implements** CWM's trace prompt/parse format over vLLM (it does **not** import the
> upstream `cwm` package). The original checkout was reference only and never modified — hence this is a
> **standalone repo, not a fork**.

---

## Setup

```bash
# Inference / eval / rendering (CUDA box; CWM 32B runs tp=4 on 4×~46GB)
python -m venv .venv_vllm && . .venv_vllm/bin/activate
pip install -r requirements.txt
python -m playwright install chromium      # for the pixel read-out
deactivate
# Training (LoRA SFT)
python -m venv .venv && . .venv/bin/activate && pip install -r requirements-train.txt && deactivate
# JS reducer parsing (TodoMVC etc.)
cd jsdeps && npm install && cd ..
```

Reproduce the real‑app demo: `python todomvc_video.py facebook/cwm --tp 4 [--stress]`. Every probe/eval command
is in `results/REPORT.md` §38.

---

## Attribution

Builds on Meta FAIR's **Code World Model** (<https://github.com/facebookresearch/cwm>). The trace prompt/parse
logic in `models/cwm_trace.py` is a vLLM re‑implementation informed by the upstream `demos/cwmdbg.py` and
`PROMPTING_GUIDE`. Intended for **noncommercial research** consistent with the FAIR Noncommercial Research
License.
