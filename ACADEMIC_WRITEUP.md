# From Execution Traces to Interactive Pixels: Extending the Code World Model into a Self‑Improving, Render‑Grounded World Model for Programs, GUIs, and Games

**Author:** @namak-kun
**Artifacts:** Code — <https://github.com/namak-kun/cwm-extended> · Adapters — <https://huggingface.co/nmk-kun/cwm-extended-adapters> · Full experimental log — `results/REPORT.md` (§0–§39)
**Base model:** `facebook/cwm` (32B Code World Model, Meta FAIR)
**Status:** research prototype; all results are LoRA adapters over a frozen base; no modification to the upstream CWM.

---

## Abstract

Meta's **Code World Model (CWM)** is a 32B language model trained to predict the *execution state* of a program frame‑by‑frame — effectively a symbolic world model over code that does not run the code. We ask whether this capability can be lifted from passive trace prediction into an **interactive, visual world model**: given a program's source and a stream of user **actions**, predict how the program's *state* evolves, and **render that state to pixels** — without an interpreter or engine in the loop. We pursue this along two axes.

**(1) A self‑improving game world model.** We teach CWM to predict an entire game **tick** in one shot — a forward dynamics model (FDM) at ≈10× temporal compression (per‑tick state accuracy **0.017 → 0.692** after a short LoRA SFT). We then derive an **inverse dynamics model (IDM) for free** by forward‑searching the FDM, and close a **flywheel**: the IDM self‑labels *unlabeled* trajectories with actions, and the FDM is retrained on them. With **zero new action labels**, the flywheel lifts held‑out per‑tick accuracy **0.525 → 0.683**, *matching a true‑action oracle (0.679)*, and single‑step action‑conditioned accuracy **0.483 → 0.808**. We show the gain is genuinely action‑conditioned (wrong‑action swap accuracy **0.0**; four distinct predictions for four actions), statistically credible (bootstrap CI excludes 0), and stable across two self‑labeling rounds (0.683 → 0.696, no collapse). We also map the method's **operating envelope**: on hard arenas (6–8 simultaneous entities) self‑labeling collapses to chance, but oracle labels still break the ceiling (0.284 → 0.369).

**(2) A render‑grounded GUI world model.** We represent UI state as a canonical **DOM tree** and show base CWM is already a strong execution‑free render‑state predictor (exact **0.823**, field‑F1 **0.988**). A real headless browser rasterizes predicted DOM‑state to pixels, so CWM never emits pixels yet drives a faithful image/video. As a capstone we **free‑roll the real TodoMVC reducer** under a scripted user session and render each predicted state into a UI video: **8/8** exact on a short session and **16/16** on a harder 6‑item session with edits, deletes and filters, with zero drift.

A recurring methodological finding cuts across both axes: most apparent "dramatic CWM failures" were **measurement artifacts** — token‑budget truncation and copy‑inflated metrics — not capability limits. We report one genuine negative result honestly: narrow Python‑cascade SFT **negatively transfers** to real JavaScript execution (full‑trace exact 0.75 → 0.35), which scopes where finetuning helps versus hurts. **Net:** a frozen, code‑trained LLM can serve as an execution‑free, action‑conditioned, render‑grounded world model for real programs, and can bootstrap game dynamics from raw state sequences without a per‑game engine, wrapper, or labels.

---

## 1. Introduction

### 1.1 Motivation

Running code to observe its behavior is the default, but it is not always cheap or even possible: game engines take many ticks to evolve, integration environments are heavyweight and game‑specific (each needs a bespoke wrapper, observation parser, and reward shim), and during testing one often wants to probe a *single* function or a partial state without standing up the whole system. A model that can **predict program behavior from source** would provide leverage exactly where execution is expensive, and would do so in a uniform interface across programs rather than one engine at a time.

Meta's **Code World Model (CWM)** is a promising substrate for this. It is trained on execution traces and predicts, given source as context, the sequence of local‑variable states a program passes through — *without executing it*. This is a symbolic world model over code. Our question is whether it can be extended from **"trace a program"** to **"simulate an interactive program and show me what it looks like."**

### 1.2 The extension thesis

We treat a program under interaction as a discrete dynamical system. Let `s_t` be the program's *logic state* (the variables that matter for behavior), `a_t` a user action/input, and `f_t` the rendered frame (pixels). The system factorizes as:

```
perception:   f_t  ──►  s_t          (pixels → state)
dynamics:    (s_t, a_t) ──► s_{t+1}   (CWM, execution‑free)
rendering:    s_t  ──►  f_t           (state → pixels)
```

CWM is a natural fit for the **dynamics** term. The **rendering** term, for a large class of GUI programs, is a deterministic function of the logic state (the DOM is a sufficient statistic for the screen), so a real renderer — not a learned decoder — suffices and provides an exact pixel oracle. The **perception** term is, in this work, deliberately scoped out of the critical path (Section 8): because the logic state is the interpretable latent, we ground in state and let a real renderer produce pixels, rather than training a heavy pixel encoder/decoder pair that would itself need to be code‑aware.

Two properties distinguish this from CWM's original setting: **(i) action as a first‑class token** in the trace (CWM was trained on autonomous execution, not interactive input), and **(ii) long‑running interactive loops**, which stress *compounding error* in free rollout — the central failure mode we characterize and mitigate.

### 1.3 Contributions

1. **A one‑shot game‑tick FDM via step‑over abstraction.** We show base CWM *cannot* predict a full game tick in one shot (0.017), and that a short LoRA SFT on the step‑over abstraction teaches it (0.692) at ≈10× compression — a scalable game‑world‑model unit (Section 6.1).
2. **CWM as an inverse dynamics model, derived for free.** Forward‑searching the FDM over a small action set recovers the action that produced an observed transition at ≈0.72 on a representative set (Section 6.2).
3. **The FDM↔IDM flywheel: self‑improvement without action labels.** Self‑labeling unlabeled trajectories and retraining lifts per‑tick 0.525→0.683 (= oracle), with rigorous controls establishing genuine action‑conditioning and a stable two‑round plateau (Section 6.3).
4. **The operating envelope of self‑labeling.** We locate the difficulty boundary where self‑labeling collapses and show oracle labels break it, yielding a concrete scaling recipe (Section 6.4).
5. **A render‑grounded GUI world model.** Canonical DOM‑state prediction (base exact 0.823) plus a real‑browser renderer composes into frame‑as‑generation and free‑roll **video**; a capstone free‑rolls the **real TodoMVC reducer** at 8/8 and 16/16 exact (Sections 6.5–6.7).
6. **A disciplined account of measurement.** We document repeated cases where harness choices (token caps, extraction modes, copy‑inflated metrics) masqueraded as model failures, and we provide audited metrics and powered (CI‑backed) drift studies. We also report a genuine negative‑transfer result (Sections 7, 6.6).

---

## 2. Background and Related Work

**Code World Model (CWM).** CWM (Meta FAIR, 2025) is a decoder‑only LLM mid‑trained on Python execution traces in a structured format: a source context followed by frames, each frame emitting the *changed* local variables (diff‑based, JSON‑valued) and the executed source line, with explicit separator tokens for call/line/return/exception events. Our client (`models/cwm_trace.py`) re‑implements this prompt/parse protocol over vLLM with verified token IDs; it does not import or modify upstream CWM.

**World models.** Learning a model of environment dynamics to plan or imagine rollouts is long‑standing (Ha & Schmidhuber, *World Models*, 2018; Hafner et al., *Dreamer*, 2019–2023). Those models operate on pixels/latents of physical or game environments. Our setting is distinct: the "environment" is *defined by source code*, the model is a *code‑trained LLM*, and the latent is the program's *symbolic state*, which is human‑readable and exactly renderable.

**Inverse dynamics and self‑supervision.** Inverse dynamics models — predict the action from `(s_t, s_{t+1})` — are used for representation learning and exploration (Pathak et al., ICM, 2017). We do not train a separate IDM; we *derive* one from the FDM by forward search, then use it to manufacture action labels, related in spirit to self‑training/ReST (Gulcehre et al., 2023) and to model‑based data augmentation.

**On‑policy imitation.** DAgger (Ross, Gordon & Bagnell, 2011) addresses compounding error by training on the learner's own (drifted) state distribution. We run gold‑prefix vs drift‑prefix ablations and find on‑policy data helps only when the residual is *drift‑induced* rather than *structural* or a *capability hole* (Sections 6.8, 7).

**GUI/web agents and state.** Web‑interaction benchmarks (Shi et al., *World of Bits*, 2017; MiniWoB) and the TodoMVC reference application motivate our real‑app sources. Unlike GUI *agents* (which act to maximize task reward), we model the *environment's* response — the transition and its rendering — which is the world‑model dual of an agent.

---

## 3. Problem Formulation

### 3.1 Execution‑free forward dynamics

Given source `P` and an action set `A`, we want `F_θ(s, a; P) ≈ s'` where `s'` is the next logic state produced by `P` on input `a` from state `s`, computed **without executing `P`**. CWM supplies `F_θ` by predicting the trace of the relevant handler/step and reading off the resulting state. We evaluate `F_θ` against a real engine's `s'` (the oracle), but the oracle is used **only for evaluation/label studies**, never inside the rollout.

### 3.2 State representations

- **Game tick.** `s` = a structured dict: player `{x,y,hp,score}` plus `K` enemies, with within‑tick side‑effects (stomp, contact damage, death). A *tick* aggregates many interpreter lines; predicting `s_{t+1}` from `s_t` in one shot is the **step‑over** abstraction.
- **UI/DOM.** `s` = a canonical **DOM tree** (ordered nodes with tag/attrs/text/children). This is an order‑sensitive structural object; we measure exact match and field‑level metrics, and it is directly renderable to pixels.
- **Real app (TodoMVC).** `s = {filter, todos:[{id,title,completed}]}`, evolved by the *actual* TodoMVC reducer (`cloneTodos`/`nextId`/`dispatch`).

### 3.3 The rendering decomposition

For GUI programs we adopt `render(s)` as a deterministic browser rasterization of the DOM. This makes pixels an *exact read‑out* of the predicted symbolic state and supplies a pixel oracle for free. The empirical question "must the renderer be learned?" is answered negatively for this class (Section 6.5): a fixed renderer reproduces the screen whenever the logic state is a sufficient statistic, which holds for the apps studied.

### 3.4 Metrics and baselines

We report, depending on the unit: **exact‑match** (whole state correct), **field‑F1** (token/field overlap), **changed‑field accuracy** (only fields that actually change — defeats the copy baseline), **delta‑exact** (the set of changes is exactly right), **per‑tick state accuracy** and **all‑ticks‑correct rate** (game), **action‑recovery accuracy** (IDM), and **true‑vs‑swap accuracy** (action‑conditioning). Baselines include **oracle** (true‑action labels), **copy/no‑change**, and the **base (un‑finetuned) CWM**. The copy baseline is essential because, in DOM/UI states, most fields are unchanged per step (≈8% change on `ui_dom`), so field‑F1 alone is **copy‑inflated** (Section 7.1).

---

## 4. Methods

### 4.1 Step‑over abstraction (teaching the one‑shot tick)

We build SFT targets by tracing a program with `trace_program(src, entry, stepover_depth=1)` and serializing the *abstracted* trace, in which an entire tick's interior collapses to a single `s_t → s_{t+1}` step. (Harvested sources guard their entry under `if __name__=="__main__"`, which is false under `exec`, so we append a bare `main()` before tracing.) A short LoRA (r=16, α=32, attention+MLP) is trained for tens of steps; the loss converges cleanly, indicating the abstraction is *fittable* — unlike drift‑heavy arithmetic (Section 6.8).

### 4.2 FDM‑as‑IDM by forward search

To label which action `a∈A` produced an observed transition `s_t→s_{t+1}^{obs}`, we run the FDM forward for each candidate `a` and select `argmin_a dist(F_θ(s_t,a), s_{t+1}^{obs})`. No separate inverse model is trained. On a representative 490‑transition set this recovers the true action at **≈0.72** (not the 1.00 seen on an easy n=24 slice — an important self‑correction; Section 6.2). Errors concentrate on near‑equivalent (zero‑margin) action ties, which is why the residual noise is benign downstream.

### 4.3 The flywheel

(1) Train FDM₀ via step‑over SFT on a small labeled seed. (2) Use FDM₀‑as‑IDM to self‑label a pool of *unlabeled* trajectories with actions. (3) Continue‑train FDM₀ on the self‑labeled data → FDM₁. (4) Optionally repeat. A 3‑way comparison — **FDM₀** vs **flywheel (self‑labeled)** vs **oracle (true‑action)** — isolates the contribution of self‑labeling from the contribution of simply training more. We add **margin filtering** (drop zero‑margin ties) as the multi‑round anti‑collapse guardrail, which raises effective label recovery to 99%.

### 4.4 DOM render‑FDM and the pixel pipeline

`ui_dom.py` defines apps whose state is a canonical DOM tree and whose dispatch is a single loop (one‑shot‑able). `dom_render.py` converts predicted DOM‑JSON → HTML → a **headless‑Chromium** PNG (via Playwright); `render_rollout.py` free‑rolls the DOM state and assembles frames into a GIF, with an optional **re‑grounding** every `k` steps (replace the predicted state with the true state to bound drift). `todomvc_video.py` does the same over the *real* TodoMVC reducer, predicting each step by full‑trace.

### 4.5 Honest measurement protocol

Because extraction mode and token budget repeatedly produced metric artifacts, we standardized: (i) a robust parser (`robust_parse`: Python‑literal / JSON / JS‑literal via node) so parse failures are not silently scored wrong; (ii) **budget scaling** — `max_tokens` grows with state size, since longer states yield longer traces; (iii) audited metrics (exact + changed‑field + delta + copy baseline); and (iv) powered drift studies using the *rollout* as the unit with bootstrap confidence intervals. A `gpt‑5.5` model served as an adversarial "rubber‑duck" reviewer that repeatedly drove these corrections.

---

## 5. Experimental Setup

**Model.** `facebook/cwm` (32B), tensor‑parallel across 4× RTX A6000 (46 GB, no NVLink) under vLLM 0.23; throughput ≈150 tok/s (communication‑bound). All capabilities are added as **LoRA adapters** over the frozen base; one adapter per vLLM process (the engine pins a single adapter per session). A separate training venv (PEFT 0.19, transformers 5.12, torch 2.12) performs SFT; the inference/render venv adds Playwright/Chromium.

**Phase 0 (stand‑in) vs the real model.** Early scaffolding used Qwen2.5‑Coder {7B,14B,32B} as a CWM stand‑in to validate the harness; **all headline numbers in this paper are on the real `facebook/cwm`** in its native trace format. As a calibration, on correct history CWM predicts the next execution frame to depth **247** (deepest tested) and free‑rolls entire programs to **≈106 frames** before drift, versus the stand‑in drifting by step 10–20 — confirming the bottleneck is *compounding error*, not per‑step capability.

**Data.** Game‑tick trajectories are generated from a self‑contained arena (no public game code, to defeat memorization). UI data comes from `ui_dom`/`ui_tick` generators and from **303 verified real transitions** harvested by research subagents (TodoMVC 83, vanilla‑JS 80, Streamlit 56, MiniWoB 84), plus a small‑state TodoMVC slice and the real TodoMVC reducer. The anti‑memorization stance (bespoke/rule‑mutated worlds) is a deliberate design choice so that correct predictions require *reading the provided code*, not recalling a known game.

---

## 6. Results

### 6.1 Step‑over: CWM cannot one‑shot a game tick, but SFT teaches it

| Game‑tick FDM (held‑out) | per‑tick state acc | all‑ticks‑correct | compression |
|---|---:|---:|---:|
| Base CWM (one‑shot tick) | **0.017** | 0.00 | 9.6× |
| + step‑over LoRA SFT (`cwm_gametick_stepover`) | **0.692** | 0.55 | 9.6× |

Base CWM nails salient player `x/y/hp` but cannot compute the multi‑enemy chase plus within‑tick stomp/contact side‑effects in a single shot — it behaves as a step‑by‑step computer, not a one‑shot transition predictor. SFT on the step‑over abstraction installs the one‑shot transition at preserved ≈10× compression, with loss converging cleanly. This is the scalable game‑world‑model unit: a compressed `s_{t+1}|s_t` predictor whose residual is slow tick‑level drift, addressable by periodic re‑grounding. (REPORT §30.)

### 6.2 CWM as an inverse dynamics model (derived free)

Forward‑searching the SFT'd FDM over the 4‑action set recovers the true action at **≈0.72** on a representative 490‑transition set (by‑event: move‑only 0.76, stomp 0.76, death 0.76, contact 0.66; 34% of transitions are zero‑margin ties). The earlier 1.00 on a 24‑transition slice was an easy‑slice artifact — recorded as a self‑correction. The state‑distance metric is dominated by player position, so the hard cases are buried within‑tick `hp`‑contact side‑effects. Usable but noisy (~28% wrong labels) → margin/event‑aware filtering is needed. (REPORT §31.)

### 6.3 The FDM↔IDM flywheel: self‑improvement without action labels

| Held‑out game prediction (n=40, seed 999) | per‑tick state acc | all‑ticks‑correct |
|---|---:|---:|
| FDM₀ (`cwm_gametick_stepover`) | 0.525 | 0.375 |
| **Flywheel R1 — self‑labeled** (`cwm_fdm_idm_r1`) | **0.683** | 0.525 |
| Oracle R1 — true‑action (`cwm_fdm_oracle_r1`) | 0.679 | 0.525 |
| **Flywheel R2 — self‑labeled** (`cwm_fdm_idm_r2`) | **0.696** | 0.625 |
| Oracle R2 (`cwm_fdm_oracle_r2`) | 0.692 | 0.575 |

The self‑labeled flywheel **matches the true‑action oracle** (0.683 vs 0.679) using **zero new action labels**, and a second round is a **stable plateau** (0.696), not a collapse — margin filtering (zero‑margin ties dropped, recovery 72%→99%) is the key guardrail. Why 28% IDM label noise is benign: the training target is the *observed next state*, and label errors concentrate on near‑equivalent actions whose next states nearly coincide, so the supervision signal is preserved. (REPORT §32.)

**Action‑conditioning controls (the win is not a state prior).**

| Single‑step (n=120, seed 999) | true‑action acc | swap‑action acc | pred. diversity |
|---|---:|---:|---:|
| FDM₀ | 0.483 | **0.000** | 4/4 |
| Flywheel R1 | **0.808** | **0.000** | 4/4 |

Swapping in a wrong action **never** reproduces the observed next state (swap accuracy 0.0), and the FDM emits a **distinct** prediction for each of the 4 actions (pred. diversity 4/4) — the model is genuinely action‑conditioned, not action‑ignoring. The flywheel's gain lands on the hardest buried side‑effects: contact 0.279→0.754, stomp 0.312→0.812, death 0.312→0.812 (move‑only already 0.776→0.837). A program‑level bootstrap CI for IDM−FDM₀ excludes 0; IDM−oracle includes 0 (i.e., indistinguishable from oracle). (REPORT §32.6–32.7.)

### 6.4 Operating envelope: where self‑labeling collapses, and how to break it

On hard arenas (K=6–8 simultaneous entities), the FDM is too weak to forward‑discriminate actions: the IDM labeler hits chance (action‑recovery 0.258, 100% zero‑margin ties, 0 labels kept after filtering) versus 0.72 at K=3–5. So the flywheel only bootstraps regimes the FDM can already *partly* model. However, **oracle/engine labels still break the ceiling**: hard‑oracle SFT lifts per‑tick **0.284 → 0.369** (FDM₀ 0.117) where self‑labeling stalls at chance. The scaling recipe is therefore explicit: **flywheel for easy–mid difficulty; engine/oracle labels + a K‑curriculum for hard.** (REPORT §32.9–32.10.)

### 6.5 DOM render‑FDM: base CWM is already a strong UI state predictor

| UI/DOM transition (n=96, audited) | exact | field‑F1 | changed‑field | delta‑exact | copy‑F1 (baseline) |
|---|---:|---:|---:|---:|---:|
| Base CWM — overall | 0.823 | 0.988 | — | — | — |
|  · `ui_dom` | 0.812 | 0.991 | 0.875 | 0.812 | 0.907 |
|  · `ui_tick` | 0.833 | 0.984 | 1.000 | 0.875 | 0.700 |

Out of the box, base CWM predicts the next DOM/UI state well. Crucially, it **beats the copy baseline on the honest metrics** (e.g. `ui_tick` changed‑field 1.0 vs copy‑F1 0.70), so the competence is real and not an artifact of mostly‑unchanged fields. A fixed browser renderer then reproduces the screen exactly whenever the DOM is a sufficient statistic — empirically the case here — so **the renderer need not be learned** for this app class. (REPORT §33, §35.)

### 6.6 Cascade SFT: an in‑distribution win and an honest negative transfer

Training a cascade adapter (`cwm_cascade`) on `ui_dom`+`ui_tick` step‑over data **improves in‑distribution** markedly:

| Audited (n=96) | base exact | `cwm_cascade` exact |
|---|---:|---:|
| overall | 0.823 | **0.958** |
| `ui_dom` (delta‑exact) | 0.812 | **0.938** |
| `ui_tick` (delta‑exact) | 0.875 | **0.979** |

The win survives the metric audit and **extrapolates** beyond the trained element range (n=10–15 elements: 0.98) and **across app families** (a held‑out `togglelist` app with a different schema improves 0.44→0.56 under `cwm_heldapp`, which never saw it). **However**, the same narrow adapter **negatively transfers to real JavaScript**: on the real vanilla‑JS form‑validator, full‑trace exact drops **0.75 → 0.35** (changed‑field 0.75→0.66). This is a real result, not a harness artifact (confirmed under the fair full‑trace harness, 0 unparsed). The diagnosis (error‑audited) is *cascade‑abstraction non‑transfer*, not schema intrusion; the known fix — mixed‑corpus/JS‑flavored replay (the §24 forgetting remedy) — is identified but not yet executed. (REPORT §35–§36.)

### 6.7 Real‑app world model and the pixel/video capstone

Filtering TodoMVC to small states, **base CWM achieves exact 1.0** across all seven action types — delegation/helper calls are *not* the blocker. The earlier "real apps fail" numbers were harness artifacts: one‑shot step‑over cannot capture helper‑delegating dispatch (returns `None`), and full‑trace *truncates* without an adequate token budget (vanilla 0.16→**0.75** once the budget is raised). The honest picture is **positive**: base CWM executes real‑app handlers well when traces fit the budget.

**Capstone (REPORT §39).** Using the *real* TodoMVC reducer as context, we free‑roll a scripted user session, predicting each next state by full‑trace (CWM executes `cloneTodos`/`nextId`/`dispatch`), feeding predictions back in, and rendering each to a TodoMVC‑styled frame:

| Session | actions | per‑step exact |
|---|---|---:|
| Short | add·add·toggle·add·setFilter·toggle·setFilter·clearCompleted | **8/8** |
| Stress | 16 actions, ≤6 todos, incl. **edit** + **delete** (+id‑gaps) + filters | **16/16** |

Both run with **zero drift**; the rendered videos show a recognizable TodoMVC UI (checkboxes, strikethrough, "N left" counter, all/active/completed filters) evolving faithfully. The stress run first scored 14/16; both misses were diagnosed (deterministic re‑probe) as **token‑cap truncation** (the same `delete` is perfect at budget 6000 but truncates at 3072), and budget scaling restored 16/16. This composes the full stack — real app code → execution‑free FDM → free‑roll under input → real‑browser video — with **no engine in the loop**.

### 6.8 Object‑state, on‑policy imitation, and arithmetic (supporting studies)

- **Object‑state observability (φ‑expansion).** SFT teaching per‑frame object‑attribute rendering lifts oop free‑roll **0.02 → 0.93**; mixed‑corpus replay eliminates catastrophic forgetting of a held‑out long mode (multientity 0.68 → 1.0) while preserving oop. The Python‑only fix **transfers across languages** (it repaired a C++ class trace) — conceptual skills need not be taught per language. (REPORT §22, §24, §28.)
- **On‑policy (DAgger).** For oop, gold‑prefix and drift‑prefix DAgger are **identical** (free‑roll 0.9324) because the residual is a *structural* φ‑render slip, not drift. On‑policy data helps only for *drift‑induced* error. (REPORT §25.)
- **Arithmetic is a capability hole, not drift‑fixable by SFT/RL here.** Teacher‑forced long‑arithmetic is 0.726 but free‑roll is 0.141 (a +0.60 drift gap); gold SFT does not rescue it (0.143), and outcome‑RL cannot bootstrap it because CWM's per‑step errors are *confident* (near‑zero entropy, no exploration to reinforce). The lever is tool‑use/scratchpad for the per‑step computation, not more SFT/RL. (REPORT §26–§27.)

---

## 7. Analysis, Ablations, and Threats to Validity

### 7.1 Measurement artifacts repeatedly masqueraded as model failures

A central lesson is methodological. Three independent times, a "dramatic CWM failure" reduced to a harness or metric problem:

1. **Token‑cap truncation.** Long traces silently exceed `max_tokens`, the output is cut mid‑JSON, the parser fails, and the step scores wrong. Fixes: budget scaling with state size, and treating unparsed outputs distinctly. This explained the K=10 game "0.0", the vanilla "0.16", and the stress‑video "14/16."
2. **Copy‑inflated field‑F1.** Because most DOM fields are unchanged per step, predicting "no change" scores field‑F1 ≈0.907. We therefore report changed‑field accuracy, delta‑exact, and an explicit copy baseline; the SFT win and the base competence both survive these honest metrics.
3. **Extraction‑mode confound.** One‑shot step‑over cannot represent helper‑delegating handlers (returns `None`); full‑trace can but truncates. Reporting the right mode per app, with adequate budget, removes the apparent "JS cascade gap."

These corrections were driven by an adversarial `gpt‑5.5` rubber‑duck reviewer and are the reason we trust the positive results.

### 7.2 Statistical credibility

The flywheel gain is supported by a program‑level **bootstrap CI that excludes 0** for IDM−FDM₀, while IDM−oracle's CI **includes 0** (indistinguishable from oracle). Drift studies use the **rollout as the unit** (not the frame) with bootstrap CIs over 16–24 rollouts; capability (SFT, +0.18) and drift‑control (re‑grounding, +0.12) **compose** to +0.31 (base 0.44 → SFT+reground 0.75) with non‑overlapping intervals, and the re‑grounding knee sits at **k≈2** (recovering ≈83% of the drift gap). Memoryless apps (e.g. tab‑select, which overwrites state) self‑heal from drift; accumulating apps (toggle‑list/counter) require re‑grounding, so we use accumulating apps for honest drift demonstrations.

### 7.3 Threats to validity

- **Memorization.** Public games are in pretraining; we use bespoke/rule‑mutated worlds so correct predictions require reading the provided code. Real‑app results (TodoMVC/vanilla) *are* potentially memorizable, but the *interactive free‑roll* and *id‑gap/edit/delete* stress test behavior that recall alone would not reliably produce.
- **Scale of evaluation.** Several held‑out evaluations are n=40–120 transitions; we mitigate with CIs and multiple seeds but do not claim large‑sample precision.
- **Renderer assumption.** The "fixed renderer suffices" claim holds only where the logic state is a sufficient statistic for the screen; apps with hidden visual state (animations, canvas drawing not derivable from state) would need a learned renderer.
- **Language coverage.** The Python‑only tracer cannot produce SFT targets for JS, so the JS negative transfer cannot yet be fixed by the same pipeline; this is an open engineering gap, not a refuted hypothesis.

---

## 8. Discussion

### 8.1 What is the value‑add over just running the code?

The value is **leverage where execution is expensive or awkward**: multi‑tick game engines, heavyweight or game‑specific environments that each need bespoke wrappers, and *partial* probing during testing (one function, one component) without standing up the whole system. CWM provides a *uniform* `(state, action) → state'` interface across programs, and — uniquely — runs **backwards** (IDM) to recover which action produced a change, enabling self‑labeling. The pixel layer makes the predicted, code‑grounded state **human‑legible**: the product is *predicted behavior*, with pixels as the read‑out and the eval oracle.

### 8.2 Why ground in symbolic state rather than pixels end‑to‑end

A pixels‑in/pixels‑out model would need a perception encoder and a pixel decoder that are *themselves* code‑aware (the same frame can map to different states depending on code, and the same state renders differently across apps). Grounding in the **symbolic state** — the interpretable latent — sidesteps this: CWM already produces state, a real renderer already produces pixels, and both halves are teacher‑forceable against an engine. This is the pragmatic route to the pixel north‑star, and the capstone demonstrates it end‑to‑end on a real app.

### 8.3 When does finetuning help versus hurt?

A clear picture emerges. SFT is **decisive for compressed/OOD regimes** the base cannot do (the one‑shot game tick: 0.017→0.692; object‑state observability: 0.02→0.93) and powers the flywheel. SFT is **unnecessary and can be harmful** in regimes the base already handles (small real apps: base=1.0; narrow cascade SFT regresses real JS 0.75→0.35). The operational rule: **use the base for in‑distribution real‑app dynamics; reserve SFT for abstractions/compression the base lacks, and protect against forgetting with mixed‑corpus replay.**

---

## 9. Limitations

1. **JS negative transfer is unresolved** (0.75→0.35 under `cwm_cascade`); the fix (JS‑flavored/diverse replay) is blocked by the Python‑only tracer for SFT‑target generation.
2. **Hard‑arena self‑labeling collapses** (K≥6); progress there requires oracle/engine labels and a K‑curriculum.
3. **Arithmetic remains a capability hole** under SFT and outcome‑RL; it needs tool‑use/scratchpad augmentation.
4. **Throughput** is low (≈150 tok/s, 4×A6000 without NVLink, tp=4), limiting evaluation scale and rollout length.
5. **Renderer sufficiency** is app‑class‑dependent; visually stateful apps would need a learned decoder.
6. **Adapter‑switching** is per‑process under vLLM 0.23, complicating multi‑adapter comparisons (mitigated by one adapter per process).

## 10. Future Work

- **Fix JS negative transfer** via mixed‑corpus replay spanning Python *and* hand‑constructed JS step‑over anchors; evaluate exclusively under the full‑trace harness.
- **Scale states/traces** with step‑over abstraction + budget scaling so large real apps (long todo lists, multi‑component pages) fit the context window; quantify the abstraction/re‑grounding trade‑off at scale.
- **Cross‑game generalization**: a second game type to test whether the step‑over+flywheel recipe transfers across game families.
- **Engine‑as‑oracle for hard arenas** plus a difficulty curriculum, per the §32.10 recipe.
- **Perception module** (`f→s`) trained against the renderer's `(s, f)` pairs to close the pixels‑in loop, completing the `frame_in → … → frame_out` cycle.
- **Test generation**: use the IDM and execution‑free FDM to enumerate state/action transitions that drive a program toward failure, realizing the original testing motivation.

## 11. Conclusion

We extended a frozen, code‑trained LLM (CWM) from passive trace prediction into an **interactive, render‑grounded world model**. On games, a step‑over SFT installs a one‑shot tick FDM (0.017→0.692), an IDM is derived for free by forward search, and a forward↔inverse **flywheel** bootstraps action‑conditioned dynamics from **unlabeled** trajectories to **oracle parity** (0.525→0.683) with rigorous controls and a stable two‑round plateau — with a clearly mapped collapse boundary and an oracle‑label escape. On GUIs, base CWM is already a strong DOM‑state predictor (0.823), and composed with a real browser renderer it produces **faithful UI video**, culminating in a real‑TodoMVC free‑roll at **8/8** and **16/16** exact. Throughout, disciplined measurement repeatedly revealed that apparent failures were artifacts — while one genuine negative transfer is reported honestly. The destination is pixel‑level interaction; this work establishes the symbolic‑state spine that makes it tractable, and demonstrates it end‑to‑end on a real application.

---

## References (selected)

- Meta FAIR. *Code World Model (CWM).* 2025. <https://github.com/facebookresearch/cwm>
- D. Ha, J. Schmidhuber. *World Models.* NeurIPS, 2018.
- D. Hafner et al. *Dream to Control / DreamerV3.* 2019–2023.
- D. Pathak et al. *Curiosity‑driven Exploration by Self‑supervised Prediction (ICM).* ICML, 2017.
- S. Ross, G. Gordon, D. Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No‑Regret Online Learning (DAgger).* AISTATS, 2011.
- C. Gulcehre et al. *Reinforced Self‑Training (ReST).* 2023. · Z. Shao et al. *DeepSeekMath (GRPO).* 2024.
- T. Shi et al. *World of Bits / MiniWoB.* ICML, 2017. · TodoMVC reference app, <https://todomvc.com>.

## Appendix A — Adapter catalogue
See `ADAPTERS.md` (15 LoRA adapters, grouped by axis, each linked to its REPORT section) and the Hugging Face model card at <https://huggingface.co/nmk-kun/cwm-extended-adapters>.

## Appendix B — Reproducibility
`results/REPORT.md` §38 lists exact commands for every probe/eval. Environments: `requirements.txt` (vLLM/inference/render) and `requirements-train.txt` (PEFT/SFT). The capstone: `python todomvc_video.py facebook/cwm --tp 4 [--stress]`. Raw per‑experiment numbers are in `results/*.json` (91 files).

## Appendix C — Section index to the experimental log
The narrative log `results/REPORT.md` records the work in execution order (§0–§39), including dead ends and corrections. Key anchors: §8 (real‑CWM trace depth), §29 (game‑tick state tracking is a strength), §30 (step‑over SFT), §31 (IDM by search), §32 (the flywheel + controls + envelope), §33–§36 (UI/DOM render‑FDM + cascade SFT + negative transfer), §37 (rendered video + powered drift), §34.6–§34.8 (real‑app breakthrough + budget fix), §39 (real‑app video capstone).
