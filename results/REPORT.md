# Overnight probe — extending CWM to interactive program/game world-modeling

**Run by:** autonomous overnight session for @namak-kun. **No commits made.**
**Date:** 2026-06-26 (overnight).
**Code:** `wm_probe/` (sibling to the `cwm/` clone). **Raw numbers:** `wm_probe/results/*.json`.

---

## 0. TL;DR (read this)

We tested the *core hypothesis* of your idea — **"given the code as context, predict how a
program's symbolic STATE evolves under user INPUT, without running it"** — with a controlled
experiment designed to defeat memorization. **Phase 1** used Qwen2.5-Coder {7B,14B,32B} as a CWM
stand-in. **Phase 2 (added after access was granted) tests the REAL `facebook/cwm` 32B in its native
execution-trace format — see §8, and read it first: it's the strongest evidence.**

**Phase-2 headline (real CWM):** CWM's per-step state tracking is **essentially perfect** — given a
correct history it predicts the next execution frame correctly to **depth 247** (deepest tested),
and it **free-rolls entire programs correctly up to ~106 frames** (vs the Qwen stand-in drifting by
~step 10–20). This *confirms the central diagnosis*: the bottleneck is **compounding error in free
rollout**, not per-step capability — and CWM, purpose-trained on traces, pushes that wall much
further out than a prompted chat model.

**HEADLINE (§32 — the FDM↔IDM flywheel, read this first):** CWM **bootstraps action-conditioned game
dynamics from UNLABELED state sequences**. Self-labeling via FDM-as-IDM lifts held-out per-tick 0.52→0.68
(=true-action oracle; CI excludes 0) and single-step true-action 0.48→0.81; genuinely action-conditioned
(wrong-action swap_acc 0.0, distinct prediction per action); stable over 2 rounds (no collapse, margin
filtering=99% labels); gain lands on hardest side-effects (contact/stomp/death). Boundary: self-labeling
collapses at K8 (IDM=chance) but oracle SFT breaks it (0.28→0.37) → scale via engine-oracle+curriculum.

**Latest (§22→§25) — capability gain, forgetting SOLVED, DAgger unnecessary for oop:**
- §22: a 125M LoRA SFT on φ-expanded traces turns *collapsed* oop into free-rollable (**student 0.02→0.80**),
  but narrow oop-only SFT **forgot** a held-out solved mode (multientity **1.00→0.68**).
- §24: **mixed-corpus SFT** (oop + a short same-pattern replay proxy of the long held-out mode) **eliminates
  the forgetting (multientity 0.68→1.00) while improving oop (0.80→0.83)**. Gold SFT lifts oop free-rollout
  **0.02→0.93** (held-out) — correct-prefix SFT alone fixes most compounding drift.
- §23: tried the **paper's OPSD code** (privileged self-distillation); **refuted on CWM** — CWM ignores a far
  gold preamble (attends only to the local prefix), so OPSD-literal privilege is vacuous.
- §25: **DAgger A/B** (gold-prefix vs drift-prefix, matched) — on **oop**: **identical 0.9324**; on-policy adds
  nothing because the residual is a **structural φ-rendering miss** (drops a `-999` sentinel attr), not drift.
- §26: repeated the A/B on a **drift-heavy** mode (**arithmetic**, value chains): **gold SFT FAILS here**
  (base 0.19 → gold-SFT **0.14**, vs oop's 0.02→0.93), and single-shot DAgger from the weak base doesn't
  rescue it (resync wall: recovery targets non-computable from wrong prefixes). **Net: gold SFT's free-roll
  fix is MODE-DEPENDENT** — sufficient where drift is low (oop), insufficient where it compounds
  (arithmetic), which needs *iterative* DAgger or RL. (OPSD-literal stays refuted throughout.)

Five Phase-1 findings (stand-in; trends corroborated by CWM):

1. **Code-conditioned state prediction is REAL, not recall.** On procedurally-generated games with
   *meaningless* variable/action names, the model can only predict the next state by *reading the
   provided code* (prior-only baseline ≈ 0.05). It does: one-step exact-match 0.56–0.66.

2. **The "follows-the-name-not-the-code" collapse (your NitroGen left/right ghost) is mostly a
   CAPACITY problem.** With deliberately *misleading* names (action `"LEFT"` wired to move right),
   one-step exact-match scales **7B 0.37 → 14B 0.62 → 32B 0.70** (counterfactual action-accuracy
   **0.49 → 0.79 → 0.83**), approaching the neutral-naming ceiling (~0.78). Scale lets the model
   suppress the semantic prior and follow code-mediated causality — **env-free**.

3. **The real wall is long-horizon DRIFT, and scale does NOT fix it.** In closed-loop rollout
   (model eats its own predictions), exact-match@20 collapses to ~0.04–0.17 even though the *same
   model* teacher-forced is 0.33–0.75. Per-step ability is fine; **compounding error** destroys
   multi-step rollout. Scale buys a *longer coherent horizon* (first invariant violation step ~9→16)
   but not closed-loop correctness. **The practical fix works (§3.6): periodically re-ground from the
   real engine — even sparse re-grounding recovers most accuracy, and bigger models need it less.**


4. **"Bigger programs" degrades gracefully but really degrades.** Exact-match vs program size
   (code lines / #actions / grid): there is a soft cliff past ~6 actions / ~73 lines; 32B degrades
   more slowly than 7B but both fall (32B: 0.72→0.48→0.35→0.22 across complexity 1–4).

5. **The renderer mostly should NOT be learned — it should be FED more state.** A pure
   `render(state)` reproduces pixels exactly (0 error). It only fails when the renderer secretly
   depends on hidden state (animation phase, history-dependent camera/trail). Crucially, **logging
   that state restores exactness** — and a learned renderer couldn't invent a hidden phase/camera
   from a single frame either. So your intuition ("renderer may need to be learned") is really
   **"the symbolic state must be a sufficient statistic for pixels"** — an *observability* problem,
   not a missing neural component (except the genuine no-engine / perceptual case).

**Bottom line for the idea:** the *state-tracing-under-input* half is promising and **scales
env-free** for per-step prediction; the hard, unsolved part is **long-horizon rollout stability**
(exactly your NitroGen compounding-error concern). That's where the research effort should go, not
into a learned renderer.

---

## 1. What was actually tested (and what was NOT)

**Tested (a fair proxy):** *Given source code + current symbolic state + an input action, can a
strong code LLM predict the next state — by reading the code, not recalling a known game?* Plus how
that degrades over horizon, stochasticity, and program size; plus whether pixels need a learned
renderer.

**NOT tested / caveats (be honest):**
- This is **not CWM**. CWM was *mid-trained* on execution-trace tokens in a dedicated format; we
  *prompt* a general code LLM. A CWM-style trained model would likely be **better** at this. So our
  numbers are a *lower bound / trend indicator*, not CWM's ceiling. The harness is built so real CWM
  can be dropped in (swap the model behind `models/`).
- Our worlds are **toy** symbolic games. They assume the hard "extract canonical symbolic state +
  update + render" decomposition already exists — which for real engine/GPU games is itself the
  per-game-wrapper problem you wanted to avoid. We tested the *dynamics-modeling* part assuming that
  decomposition.
- Greedy decoding; n=24–40 games/condition. Trends are clear; exact decimals are noisy at ±0.05.

---

## 2. Setup

- **Generator (`worlds/gridgen.py`)** — the key anti-memorization trick: it *emits Python source*
  that is **simultaneously the ground truth (we `exec` it) and the model's context.** Rules are
  randomly composed per seed, so recall can't help — only code-reading. Knobs:
  - `naming_mode`: `semantic` (LEFT decreases x), `random` (labels `A,B,OP1…`), `misleading`
    (LEFT wired to move *right*; field `color` is really position). Misleading = the NitroGen probe.
  - `boundary` clamp/wrap/bounce, mode-gated effects, stochastic spawn, `complexity` scaling.
- **Metric honesty:** field-accuracy is nearly useless here (a no-change **Copy baseline scores
  ~0.86 field-acc** because most fields are static per step). The real signal is **exact-match** and
  **beating Copy** (Copy exact ≈ 0.05–0.11).
- **Baselines:** Copy (no-change), code-OFF (prior-only), Oracle (pipeline sanity = 1.0).
- **Models:** Qwen2.5-Coder-{7B,14B,32B}-Instruct via vLLM on 4× A6000.

---

## 3. Results

### 3.1 One-step prediction — code conditioning & the misleading-name collapse
Exact-match, code shown (prior-only / code-OFF in parens). Copy floor ≈ 0.05.

| naming | 7B | 14B | 32B | code-OFF (prior) |
|---|---|---|---|---|
| semantic | 0.61 | 0.72 | 0.78 | 0.25–0.29 |
| random (neutral names) | 0.56 | 0.62 | 0.66 | 0.01–0.06 |
| **misleading** (NitroGen probe) | **0.37** | **0.62** | **0.70** | 0.02 |

Counterfactual (same state, all actions) — misleading **action-accuracy / action-sensitivity**:
7B 0.49 / 0.96 → 14B 0.79 / 0.99 → 32B 0.83 / 1.01.
*Interpretation:* random ≫ prior ⇒ genuine code-reading. misleading rises sharply with scale ⇒ the
prior-collapse is a **capacity** limitation, not a fundamental env-free wall (for one step).

### 3.2 Rollout drift — the real wall
Free (closed-loop) vs teacher-forced, horizon 20.

| model | naming | free field@20 | free exact@20 | 1st invariant viol. | TF exact@20 |
|---|---|---|---|---|---|
| 7B | semantic | 0.79 | 0.08 | step 13 | 0.62 |
| 7B | misleading | 0.63 | 0.00 | step 9 | 0.29 |
| 32B | semantic | 0.86 | 0.17 | step 16 | 0.75 |
| 32B | random | 0.81 | 0.17 | never (20 steps) | 0.71 |
| 32B | misleading | 0.76 | 0.04 | step 16 | 0.33 |

*Interpretation:* the **teacher-forced vs free gap is the compounding-error wall.** Scale slows the
decay and keeps states on-manifold longer, but exact closed-loop prediction still collapses by step
20. **This is the bottleneck**, and it matches your NitroGen finding precisely.

### 3.3 Stochasticity — epistemic vs aleatoric (handled correctly)
Spawn-bearing steps only; never punish exact-match on hidden RNG.

| condition | metric | 7B | 14B | 32B |
|---|---|---|---|---|
| RNG **revealed** | exact | 0.23 | 0.31 | **0.69** |
| RNG **hidden** | exact | 0.06 | 0.05 | 0.15 |
| RNG hidden | **deterministic-field acc** | 0.92 | 0.95 | 0.95 |
| RNG hidden, sampled k=6 | reachable-set coverage (items) | 0.34 | 0.43 | 0.53 |

*Interpretation:* (a) hidden randomness rightly tanks exact-match but the **deterministic part stays
~0.95** — the epistemic/aleatoric split is real and measurable. (b) *Even revealed* randomness needs
the model to correctly **execute** the rng mechanics — also a capacity thing (0.23→0.69 with scale).
(c) uncertainty coverage is weak but improves with scale.

### 3.4 Complexity — "can we go bigger?"
Exact-match vs program size (complexity 1→4 = ~50→89 code lines, 4→7 actions, grid 43→152 cells).

| complexity | 7B | 14B | 32B |
|---|---|---|---|
| 1 | 0.47 | 0.51 | 0.72 |
| 2 | 0.49 | 0.55 | 0.48 |
| 3 | 0.18 | 0.40 | 0.35 |
| 4 | 0.14 | 0.22 | 0.22 |

*Interpretation:* graceful but real degradation; soft cliff past ~6 actions / ~73 lines. Bigger
models degrade more slowly but all decline — "bigger programs" is a genuine axis of difficulty.

### 3.5 Renderer sufficiency (no LLM) — should the renderer be learned?
Mean abs pixel error of `render(logic_state_only)` vs the true frame:

| renderer | own-render error | error if hidden state is LOGGED |
|---|---|---|
| pure function of state | **0.0** | 0.0 |
| + hidden animation phase | 1.95 | **0.0** |
| + history-dependent camera | 8.68 | **0.0** |
| + history-dependent trail | 1.19 | **0.0** |

*Interpretation:* **own-renderer suffices iff the logged symbolic state is a sufficient statistic for
pixels.** Failures are an *observability* problem (log the phase/camera/history), not a missing
neural renderer — a learned renderer can't invent hidden state from one frame either. A learned
renderer is only truly required for the **no-engine / irreducibly-perceptual** case (canvas/WebGL,
emergent visuals) — the genuine "Regime 2".

### 3.6 Drift mitigation — periodic RE-GROUNDING (demonstrates the fix)
Since closed-loop drift is the wall, run the real engine every `k` steps to reset the predicted
state. Mean exact-match over a 20-step horizon (random naming) vs **engine-call rate (1/k)**:

| engine calls / step | k | 7B mean-exact | 32B mean-exact |
|---|---|---|---|
| 1.0 (teacher-forced) | 1 | 0.59 | 0.77 |
| 0.5 | 2 | 0.45 | 0.69 |
| 0.33 | 3 | 0.45 | 0.64 |
| 0.2 | 5 | 0.32 | 0.55 |
| 0.1 | 10 | 0.22 | 0.48 |
| 0.0 (pure free) | ∞ | 0.18 | 0.37 |

*Interpretation:* re-grounding **works** and is tunable — even *occasional* engine calls extend the
usable horizon a lot, and **a bigger model needs the engine less often** (32B at 10% engine-calls
≈ 0.48 ≈ 7B at 100% ≈ 0.59). So the practical path to long-horizon prediction is **scale + sparse
re-grounding / engine-in-the-loop**, not pure free rollout. This is the concrete, demonstrated
answer to the drift problem.

---

## 4. What this means for your idea

- The **"world model that traces state under input"** half is the right bet and, encouragingly,
  **per-step code-conditioned prediction (incl. left/right-style counterfactuals) scales env-free** —
  a real, hopeful update on "we can't escape envs." At 32B the misleading-name collapse is largely
  gone *without* any environment.
- The **"rendered model"** should, for engine-available cases, be the **game's own `render`** fed a
  **sufficient symbolic state** — not a learned net. Learned rendering is a separate, later, no-engine
  track. Your intuition that it "needs to be learned" actually points at **state observability**.
- The **dominant open problem** is **long-horizon rollout stability** (compounding error). This is
  where to spend effort, and where environments/RL/verification likely re-enter — to *re-ground* the
  rollout, not to learn per-step dynamics.

## 5. Recommended next steps (in priority order)
1. **Attack drift directly** (highest value): **[STARTED — see §3.6, re-grounding works]** train on
   closed-loop rollouts (DAgger-style), error-correcting decoding, or RL with the engine as verifier.
   Headline metric = horizon-to-first-violation / engine-calls needed for a target accuracy.
2. **Plug in real CWM** when weights are available (harness is ready) — quantify the mid-training
   gap vs prompted Qwen.
3. **Bridge to a real engine-available domain** (the cleanest is a JS/canvas or pygame game with an
   explicit `state`/`update`/`render`) to validate outside the synthetic generator.
4. **Stochastic interface**: have the model emit `{deterministic_fields, random_support}` explicitly
   rather than a single next-state (improves the hidden-RNG case).
5. Only then: a learned renderer, scoped to the genuine no-engine/perceptual case.

## 6. Reproduce / file map
```
wm_probe/
  worlds/gridgen.py      # procedural generator (emitted source = ground truth = context)
  models/                # base baselines, llm.py (transformers), vllm_model.py, factory.py
  metrics.py  eval.py    # metrics + trajectory/one-step/counterfactual/rollout
  run_main.py            # exp1 one-step + counterfactual + code-ablation
  run_rollout.py         # exp2 drift (free vs teacher-forced)
  run_stoch.py           # exp3 stochasticity (revealed/hidden/sampled)
  render_probe.py        # exp4 renderer sufficiency
  run_complexity.py      # exp5 program-size sweep
  run_reground.py       # exp6 drift mitigation via periodic engine re-grounding
  run_all.py            # driver (one model load -> all exps); summarize.py consolidates
  results/*.json         # all raw numbers
# env: .venv (transformers) and .venv_vllm (vLLM). Run vLLM jobs with .venv_vllm/bin on PATH (ninja).
```

## 7. Decisions made autonomously (review)
- No API keys → local Qwen2.5-Coder as CWM stand-in (D1). vLLM for throughput; ~20–70 preds/s (D4).
- Adopted the rubber-duck's redesign: ONE procedural weird-gridworld instead of a game suite; exact-
  match (not field-acc) as the metric; misleading-naming as the central probe (D6).
- Added 14B as a scaling midpoint; parallelized 7B/14B/32B across the 4 GPUs (D5).
- **Did not** build a learned renderer (renderer-sufficiency probe showed it's premature).

---

## 8. PHASE 2 — the REAL CWM (facebook/cwm 32B), native trace format

After access was granted we downloaded `facebook/cwm` (61 GB, `CwmForCausalLM`, supported natively by
vLLM 0.23) and tested it in **its actual trained capability**: autoregressive **execution-trace
prediction** using CWM's special trace tokens (`<|trace_context_start|>`, `<|frame_sep|>`,
`<|call_sep|>`, `<|line_sep|>`, `<|return_sep|>`, `<|action_sep|>`, …). This is far more faithful
than the chat-JSON prompting used for the Qwen stand-ins: we give CWM the source code as context and
it emits frames = `(event, locals-as-JSON-diff, source-line, [return value])`, exactly as in the
repo's `demos/cwmdbg.py`. Ground truth comes from really executing the program under `sys.settrace`.

**Validation:** CWM perfectly reproduces the tech-report demo (`count_letters` over "strawberry",
30 frames, correctly counts the single 'a'). Our scorer marks known-correct traces as 1.0.

### 8.1 The question: how far can CWM track state?

We measured two regimes on parametric programs whose trace length grows with N (a modular-arithmetic
accumulator, a grid-walk with bounds + score, and a stack/list machine):

**A. Teacher-forced next-frame prediction at depth D** (given the *true* trace prefix of length D,
predict frame D). This isolates per-step tracking from compounding error.

| program | frames tested | first value-fail depth | frame-accuracy (all depths) |
|---|---|---|---|
| counter N=40 | 167 | none | **1.000** |
| grid N=30 | 247 | none | **1.000** |
| list N=40 | 189 | none | **1.000** |

→ **CWM predicts the next execution state PERFECTLY out to depth 247** (the deepest we tested),
including arithmetic (`(a*3+1)%100`, interacting accumulators), branch selection, bounds clamping,
and stack mutations. Per-step state tracking is essentially **solved**; depth/long-context is *not*
the limiter when the history is correct.

**B. Free rollout** (CWM generates the WHOLE trace from just the source; eats its own predictions —
the realistic "predict execution without running it" setting).

| program | N | true frames | result |
|---|---|---|---|
| counter | 12 | 55 | **fully correct (EOS at exact true length)** |
| grid | 12 | 106 | **fully correct** |
| list | 12 | 58 | **fully correct** |
| counter | 30 | 127 | **fully correct** (§16, clean run) |
| grid | 24 | 198 | **fully correct** (§16, clean run) |
| list | 30 | 142 | **fully correct** (§16, clean run) |

→ **CWM free-rolls entire programs correctly to ~200 frames** with NO drift on this deterministic
symbolic class. **CORRECTION (see §16):** an earlier draft of this section claimed divergence "around
N≈16–24" — that was a **token-cap truncation artifact** (the rushed sweep capped output at 2500
tokens, so longer traces were *cut off*, not *diverging*). A clean run with GT-proportional caps shows
free rollout is **perfect to 198 frames** (grid N=24). The genuine drift wall is therefore *much*
farther out than first reported — see §16 for the corrected picture and its consequence for training.

### 8.2 Why this matters / interpretation

- **Direct, faithful answer to "how far can it track state":** *Per-step, ~unbounded (≥247 frames).
  Free-rolling on its own, ~100 frames before compounding error bites.* CWM (purpose-trained on
  traces) is **dramatically better** than the chat-prompted Qwen stand-in (which drifted by ~step
  10–20), but exhibits the **same qualitative wall** — confirming the Phase-1 diagnosis is real and
  model-agnostic, just pushed much further out by training.
- This is the **strongest possible support for the project framing**: the learned component you want
  (symbolic state dynamics) genuinely works per-step and for ~100-frame horizons env-free; the open
  problem is **rollout stabilization** (re-grounding / DAgger / RL-with-verifier from §5), now with a
  much higher starting baseline than the stand-in suggested.

### 8.3 Honest caveats / hardware note
- The 4×A6000 box has **no NVLink**, so tensor-parallel (tp=4) **decode** of a *single long sequence*
  is communication-bound (~a few tok/s). Long full-trace free rollouts are therefore slow; we used
  **batched, prefix-cached, short generations** (teacher-forced depth probes) to get fast, clean
  signal, and small-N free rollouts for the closed-loop view. Pinning the exact N≥16 free-rollout
  accuracy curve is future work (cheap on an NVLink box, or via the PyTorch-DCP weights + their
  fastgen server).
- Programs are small symbolic state machines; "extract symbolic state + render" is assumed (the same
  decomposition caveat as Phase 1).

### 8.4 Phase-2 file map
```
models/cwm_trace.py     # CWM native trace client (raw trace-token format) over vLLM
gt_trace.py             # sys.settrace ground-truth tracer + frame scorer + GT->CWM-frame encoder
run_cwm_track.py        # batched free-rollout whole-trace generation + scoring (how-far sweep)
run_cwm_depth.py        # batched TEACHER-FORCED next-frame prediction at depth (the clean signal)
smoke_cwm.py            # count_letters validation
results/cwm_depth.json  # teacher-forced (perfect to 247) + small-N free rollout (perfect to ~106)
# Run with .venv_vllm/bin on PATH (ninja); CWM weights at ~/.cache/huggingface/hub/models--facebook--cwm
```

---

## 9. PHASE 3 — OUT-OF-DISTRIBUTION programs (C, concurrency, ML)

"Does CWM only work with Python?" and "where does trace-prediction break?" Tested by feeding CWM
6 programs in its native trace format and checking whether it predicts the correct final return
value (ground truth = really running gcc / a Python subprocess). Results:

| program | lang | true | CWM | ✓ | failure mechanism |
|---|---|---:|---|:--:|---|
| C arithmetic loop | C | 56 | 56 | ✅ | — (traces C correctly!) |
| C array / indexing | C | 8 | 8 | ✅ | — (tracks `arr`, `max`, indexing) |
| numpy small (3-vec) | py | 12 | 12 | ✅ | — (tracks `array([2,4,6])`) |
| numpy big (10×10 sum) | py | 5445 | *(the array)* | ❌ | **native-code wall** |
| multiprocessing Pool.map | py | 30 | *(a ForkContext obj)* | ❌ | **concurrency / library descent** |
| threading (2 threads) | py | 6 | *(none)* | ❌ | **concurrency / library descent** |

### 9.1 Headline: CWM is NOT Python-only — simple C transfers
Despite being mid-trained on *Python* execution traces, CWM **correctly traces simple C**: it ran the
arithmetic loop to the right answer (a: 1→3→7→15→31, b→56) and handled C **array indexing + branch**
(`if (arr[i] > max)` correctly skipped when false) to return 8. The trace format is structurally
language-agnostic and CWM's pretraining gives it C semantics. So C (and likely other imperative langs)
are in reach for simple programs — though we did not test pointers/UB/manual memory at depth.

### 9.2 The dominant OOD failure mode: descent into native/library internals
The three failures share **one mechanism**: when execution leaves plain user-level logic, CWM tries to
**trace *into* the library frame-by-frame** instead of treating the call as an opaque operation:
- **multiprocessing** → it descends into `Pool()` / `<multiprocessing.context.DefaultContext>` setup
  and never models the parallel workers; returns a context object instead of `30`.
- **threading** → it descends into `Thread.__init__`; a single *sequential* trace **cannot represent
  concurrent execution**, so no coherent result emerges.
- **numpy big** → it *correctly materializes* the 10×10 array, then hits `umr_sum` — a **compiled C
  ufunc** — which it cannot execute, so it returns the array itself rather than the scalar `4950`.
  (Small arrays work because the values are few enough to track/compute symbolically.)

### 9.3 Two fundamental walls (beyond compounding error)
1. **Native-code wall.** Computation that happens in compiled C/CUDA (numpy/torch kernels, game
   physics/render engines) is **opaque** to a Python-trace model. Tractable only when small enough to
   "mentally" simulate. This is *not* fixable by scale — the bytes simply aren't in the trace.
2. **Concurrency wall.** Parallel / scheduler-dependent execution (threads, processes, async) is **not
   expressible as a single sequential trace** and **not determined by source code alone** (interleaving
   depends on the OS scheduler) — the same aleatoric boundary as hidden RNG (§3.5), but structural.

### 9.4 What this means for the games plan (directly relevant)
A real game's heavy lifting (physics, rendering, audio) usually runs in **compiled/native code**, and
often **concurrently** (render thread, physics thread). CWM will faithfully trace the **Python game
*logic*** but hit the native-code wall at the engine boundary and the concurrency wall at threads.
→ This *confirms the symbolic-state framing*: model the game-logic layer as traceable code, and treat
engine/native calls as **opaque oracles you actually run** (the renderer/physics step), exactly the
"render with the real engine, predict only the symbolic dynamics" design. CWM's strengths and walls
line up with that split. (NitroGen's `nitrogen/eval/envs/` has ~40 FOSS games — most do their logic in
C/Lua/compiled engines, so the cleanest CWM targets are games whose *logic* is in Python or simple C.)

### 9.5 File map
```
run_ood.py              # the OOD probe (gcc/py-subprocess ground truth + CWM free-roll + final-return check)
results/cwm_ood.json    # the table above
```

---

## 10. PHASE 4 — working at an ABSTRACTION level (step-over)

The §9 failures came from CWM *descending into* library internals. CWM's trace machinery has the
native lever to avoid that: **step-over** (the debugger's `next`) — when a call is encountered, predict
its RETURN value directly as an opaque black box instead of tracing the body. We implemented a
"depth-1" policy: whenever CWM descends, force the next frame to be that call's return. (This is the
same operation you'd use to model a **game tick**: predict `update(state,input)→state'` in one shot.)

| program | true | step-over result | ✓ | what happened |
|---|---:|---|:--:|---|
| nested user fn | 19 | 35 | ❌ | mechanically fine, but **mis-computed** `helper(4)=30` (off-by-one: Σi² for 1..4 not 0..3) |
| multiprocessing | 30 | *(ran past budget)* | ⚠️ | **but abstracted the hard part correctly: `p.map(sq,[1,2,3,4]) → [1,4,9,16]`** ✅ |
| threading | 6 | **6** | ✅ | **abstraction RESCUED it** — stepped over `start()/join()`, predicted net `counter=6` |
| numpy small | 12 | 12 | ✅ | fine |

### 10.1 Answer: YES, we can work at an abstraction level — and it's the right tool for the walls
The headline positives are strong:
- **CWM correctly predicts opaque-call results.** Stepping over `p.map(sq, data)` it predicted exactly
  `[1, 4, 9, 16]` — i.e. it modeled a *parallel map* as a black-box function and got the right answer,
  bypassing the multiprocessing internals that sank §9.
- **Abstraction sidesteps the concurrency wall.** For threading it stepped over `start()/join()` and
  predicted the **deterministic final** `counter=6` — without modeling the (unpredictable) interleaving.
  This is the key trick: abstract away the nondeterministic *mechanism*, predict the deterministic
  *outcome*.

### 10.2 Two real caveats (the cost of abstraction)
1. **Abstracted predictions are learned GUESSES, not traced — so they can be wrong.** On the nested
   pure-Python case, step-over made `helper(4)=30` (an off-by-one; the true 14). Had CWM *traced* it
   line-by-line (§8) it would have been exact. So abstraction **trades exactness for reach**: only
   abstract what you cannot trace (native/concurrent code); trace what you can.
2. **The boundary policy matters.** Our naive single-level step-over didn't handle *nested* descent:
   the `with Pool(...)` teardown (`__exit__ → terminate → join`) pulled CWM back into library frames and
   it ran past budget — *even though it had already correctly produced `results=[1,4,9,16]`*. A
   recursive "step over anything that leaves user code" (or abstract only at explicitly marked
   boundaries) would fix this.

### 10.3 Synthesis — abstraction + re-grounding compose
This closes the loop with the rest of the report. The world model should be built at a **chosen
abstraction level** (line / function / tick / subsystem), where:
- things you **can trace** are modeled exactly (CWM is near-perfect there, §8);
- things you **can't** (native kernels, concurrency, the engine) are **stepped over** as opaque
  transitions — CWM gives a usable learned prediction (§10), but one that can err;
- therefore you **verify/correct** those abstracted predictions by occasionally running the real
  operation — i.e. **re-grounding / DAgger (§3.6, §5)**.

That triad — *trace what's tractable, abstract what isn't, re-ground to stay honest* — is the concrete
architecture for the "predict a codebase/game's output" goal, and it maps cleanly onto: model the
game **logic** as a traceable tick, treat the **engine/render/physics** as an opaque step you predict
*and periodically actually run*.

### 10.4 File map
```
run_cwm_abstract.py      # step-over (depth-1 abstraction) rollout, batched
results/cwm_abstract.json # the table above
```

---

## 11. PHASE 5 — multi-language + heavier ML battery

Tested CWM's trace prediction on 5 non-Python languages (harder C, C++, JS, Rust, Java) and heavier
ML (pure-Python matmul / NN-forward / gradient-descent, plus numpy). Metric = predict the entry
function's final RETURN value; ground truth = really compiling/running each (gcc/g++/node/rustc/java).

| program | lang | true | CWM | ✓ | note |
|---|---|---:|---|:--:|---|
| bit-ops + strlen | C | 11 | 11 | ✅ | `popcount(0xB7)`+`strlen` — bit twiddling works |
| recursion + ptr + struct | C | 28 | 8 | ❌ | **recursion blew up** (fib(7) → 105 frames, lost state) |
| STL vector + class | C++ | 120 | 204 | ❌ | C++ container/method handling wrong |
| explicit loop | JS | 55 | *(none)* | ❌ | no clean return (format/EOS issue) |
| map / reduce | JS | 30 | 30 | ✅ | higher-order array ops correct |
| iterator loop | Rust | 30 | 30 | ✅ | Rust transfers |
| array + branch | Java | 16 | 16 | ✅ | Java transfers |
| pure-Python matmul | py | 69 | 69 | ✅ | nested loops + 2-D state |
| pure-Python NN forward | py | 30 | 30 | ✅ | matmul + ReLU |
| pure-Python grad descent | py | 1233 | *(none)* | ❌ | **float accumulation over 12 updates** failed |
| numpy matmul `A@B` | py | 69 | 69 | ✅ | small native op OK |

**Score: 7/11 by strict final-value match — but trace inspection (§11.1) shows 2 of the 4 "failures"
are HARNESS ARTIFACTS where CWM actually tracked state correctly. True capability ≈ 9/11.**
(Languages: Rust✅ Java✅ C-bits✅ JS-mapreduce✅ JS-loop✅* ; C-recursion❌ C++❌.
ML: pure matmul✅ NN-forward✅ numpy-matmul✅ grad-descent✅* ; * = correct trace, mis-scored.)

### 11.1 Trace inspection corrects the scorecard (important)
Dumping the failure traces (`run_battery.py ... <names>`) revealed:
- **JS explicit loop — actually CORRECT.** The trace tracks `total` 1→5→14→30→55 and emits
  `RETURN total = 55`. Our extractor wrongly grabbed the outer `console.log(...)` return (None). CWM
  traced JS perfectly; the "failure" was a measurement bug.
- **Gradient descent — actually CORRECT (as far as it ran).** The floating-point updates are exact
  (`w: 0.0 → 0.02 → 0.0984 …`, `err = -3.96`, …) — CWM tracks evolving floats precisely. It only
  failed to emit a final value because the long trace (12 updates × ~4 lines, big per-frame locals)
  hit the 2000-token cap before `return`. A budget artifact, not a capability gap.
- **C++ — genuine failure.** CWM kept the `Acc` object **opaque** (`a: 'Acc()'`), never traced into
  `a.add()` to update the private member `total`, then **confabulated** `get() = 204` (true 120).
  Encapsulated object state behind method calls isn't tracked unless it steps *into* the methods.
- **C recursion — genuine failure.** `fib(7)`'s recursive call tree exploded the trace (~102 frames)
  and CWM lost the accumulated value (returned 8 = just the `Point` part). Call-tree *depth* triggers
  the §8 compounding-error wall.

### 11.2 Findings (revised)
- **Language transfer is strong and largely zero-shot.** Rust, Java, JS (both loop & map/reduce), and
  bit-twiddling C all traced correctly. CWM's execution model generalizes well beyond Python for
  straightforward imperative code. (Ground truth here is the final value; intermediate-frame fidelity
  for non-Python is spot-checked via the dumps, e.g. JS `total` progression was exactly right.)
- **Float tracking works** (grad descent floats were exact) — earlier doubt was a budget artifact.
- **The genuine walls are specific and match predictions:** (1) **deep recursion / call-tree depth**
  (state loss over a big trace), and (2) **encapsulated state behind opaque method/container calls**
  (C++ `Acc`/STL) — the §9 library-descent / §10 abstraction-reliability issue, now in C++.

### 11.3 Implication
CWM is a **strong multi-language base** — several languages work zero-shot and floats/ML-logic track
well. Targeted SFT/DAgger should focus on the two real gaps: **deep recursion** and **OOP/container
encapsulation** (teach it to trace into methods, or to predict encapsulated side-effects reliably).
Heavier numeric ML hits the native-code wall (§9) and is best handled by **abstraction** (§10).
Also: our measurement harness should extract the *entry function's* return (not the outermost) and use
GT-proportional caps — two of four "failures" were ours, not CWM's.

### 11.4 File map
```
run_battery.py            # multi-language + ML battery; pass names as 3rd arg to dump failure traces
results/cwm_battery.json  # strict scorecard;  results/cwm_battery_fail.json = inspected failures
```

---

## 12. PHASE 6 — Lua (game scripting) + HTML/DOM state

Two targeted asks: **Lua** (the dominant game-scripting language) and **HTML/DOM** (where "state" is
ambiguous). Key framing the user nailed: **HTML alone is static markup — no execution state.** A
page's dynamic state is the **DOM tree**, mutated by JS. So we test (a) DOM modeled as *symbolic
state* (pure-JS, no browser) and (b) the *real* DOM via jsdom.

| program | lang | true | CWM (by trace) | ✓ |
|---|---|---:|---|:--:|
| numeric loop | Lua | 55 | 55 | ✅ |
| entity table update (`player.x/y`) | Lua | 30 | 30 | ✅ |
| table + closure counter | Lua | 10 | 10 | ✅ |
| mock DOM tree build | JS | 410 | 410 | ✅ |
| mock DOM state mutation | JS | 32 | 32 | ✅ |
| **real DOM (jsdom)** | jsdom | 60 | **64** | ❌ (close) |

**Genuine score 5/6** (by trace inspection — see note). The only miss is the *real browser DOM*.

### 12.1 Findings
- **Lua works great — all 3, including game-flavored logic.** The entity-update test tracked the
  `player = {x, y, hp}` table correctly through a move list (`x: 0→3`, returned 30), and a closure-based
  counter accumulated correctly (10). **Excellent news for the games angle: Lua game scripts are
  squarely traceable.**
- **DOM-as-symbolic-state works perfectly.** When the DOM/app state is an ordinary object/tree
  (`{tag, children}`, `{count, items}`), CWM tracks every mutation exactly (410, 32) — it tracked
  `state.items` growing `['item-2','item-3']` and `count` incrementing. So "tracking webpage state" is
  fully viable **if you model the DOM as symbolic state**.
- **The real browser DOM is "semi-traceable".** On jsdom, CWM **descended into the DOM internals but
  modeled the API semantically** — `getElementById` returned `<ul id="list">`, `createElement` /
  `appendChild` produced sensible nodes — and landed **close (64 vs 60)**, just miscounting through the
  loop. This is a *softer* native wall than numpy's `umr_sum` (§9): the DOM API is well-known enough
  that CWM approximates it, whereas an opaque numeric kernel is a hard stop. Severity of the
  native-code wall scales with how well-known / symbolic the API is.

### 12.2 Answering "what would tracking state even mean for HTML?" and "jQuery?"
- **HTML state = the DOM tree.** Trace the JS that mutates it; predict the resulting tree/derived
  values. Symbolic-DOM modeling makes this exact; real-engine DOM is approximate (use the real
  browser as the oracle when you need exactness — the same trace/abstract/re-ground triad as §10).
- **jQuery (and any library):** since JS works, jQuery *usage* is fine, but tracing *into* jQuery is
  the §9 library-descent issue — handle it with **step-over abstraction** (§10): predict
  `$(...).text()` / `.append(...)` effects as opaque ops rather than tracing jQuery's source.

### 12.3 Harness note (recurring)
As in §11, strict scoring under-reported: 3 of these (Lua numeric/entity, DOM mutate) emitted the
**correct** `RETURN` (55/30/32, visible in the dumps) but `cwm_final_return` returned `None` — the
parsed `pred` dropped the entry return in those cases. **Trace inspection is authoritative; fix the
scorer to read the entry function's return.** Net: real capability here is 5/6, not the 1/6 a naive
read of the strict JSON would suggest.

### 12.4 File map
```
run_lua_dom.py            # Lua + DOM battery (lua5.4 / node / jsdom ground truth)
results/cwm_lua_dom.json  # strict scores (under-reports; see §12.3 + trace dumps in logs/cwm_lua_dom.log)
jsdeps/                   # local jsdom install for real-DOM ground truth
```

---

## 13. ASSUMPTIONS LEDGER (the grounding spine)

Per the directive "be sensible and grounded — note assumptions, see how/when they fail." Every claim
about extending CWM to codebases/games rests on these. Each row: the assumption, the phase that bears
on it, and the **observed** failure condition (✗ = already seen to fail; ⚠ = partially; ? = untested).

| # | Assumption | Fails when… | Status / evidence |
|---|---|---|---|
| A1 | **Symbolic state is a sufficient statistic** for the output we care about | hidden render state; native-backed values; nondeterminism | ⚠ §3.5 (anim/camera), §9 (numpy big), §9 (concurrency) |
| A2 | **Expert oracle is queryable at arbitrary student-visited states** (needed for DAgger/OPSD/RL & re-grounding) | the drifted predicted state is *invalid/unrunnable*; program has external side-effects that can't be reset mid-trajectory | ? — **critical to test before any on-policy training** |
| A3 | **Privileged teacher ≫ blind student** (the OPSD premise) | knowing the true state doesn't sharpen next-token enough to give a useful gradient | ? — **must verify empirically first** |
| A4 | **Per-step accuracy composes into long-horizon under correction** (re-ground/OPSD tames drift) | errors are correlated/systematic, or a single bad token breaks structure (parse), so 1/k correction doesn't recover | ⚠ §3.6 shows re-grounding helps; structural-break case untested |
| A5 | **Game logic is cleanly separable from the engine** ("trace logic, abstract engine") | logic and engine interleave (logic calls engine mid-tick and uses the result) | ✗ real love2d games: `love.update` calls `love.graphics/timer` → not standalone-traceable |
| A6 | **Free oracle ⇒ cheap training data** | instrumenting a real language/engine for traces *is* the per-game-wrapper problem (training-time) | ? — Python is free (CWM did this); other langs/engines unknown |
| A7 | **Training won't destroy general capability** (forgetting) | narrow trace SFT degrades broad code/reasoning | ? — **forgetting probe required**; motivates OPSD/KL-reg over hard SFT |
| A8 | **Target is deterministic given inputs** | RNG, time, concurrency, FP non-associativity, I/O | ✗ §3.5 hidden RNG, §9 threads/procs |
| A9 | **Language transfer holds** beyond Python | deep recursion; OOP/encapsulation; rich libraries | ⚠ §11: Rust/Java/JS/Lua ✓; C-recursion, C++-STL ✗ |
| A10 | **The measurement harness is faithful** | scorer reads wrong return / caps truncate correct traces | ✗ §11/§12: 2–3 "failures" were ours, not CWM's — **trust trace inspection** |

**Operating principle going forward:** every experiment states *which assumption it probes* and reports
hold/fail. The two most load-bearing untested ones — **A2 (oracle at arbitrary states)** and **A3
(privilege gap)** — gate the entire training program, so they are tested *first* and *cheaply* (small
model) before committing compute.

---

## 14. METHODOLOGY CORRECTION — measure CWM-specific claims on CWM (not a stand-in)

**Course correction (user-flagged, correct):** Phase-1 used Qwen2.5-Coder as a *stand-in* for CWM
*before we had CWM weights*. Now that we have CWM, continuing to test **CWM-specific** assumptions
(esp. A3, the OPSD privilege gap) on a stand-in is an **unjustified transfer** — and a *bigger/better*
Qwen (3.5-397B-A17B, Qwen3-Coder-Next-80B all exist as of 2026-02) does **not** fix it: the gap isn't
that the stand-in is weak, it's that **no Qwen is trace-mid-trained like CWM**. So stand-in results
describe "a strong code LLM prompted to predict state," not CWM's trained trace behavior.

**Policy from here:** CWM-specific claims are measured on **CWM** (we have it; inference is cheap).
A small model (1.5B) is retained ONLY as *training-loop plumbing* (does the trainer run / loss move),
never as a source of findings — and clearly labelled as such.

**A3 is in fact already evidenced on CWM (§8), re-read correctly:** teacher-forced (privileged: true
prefix) ≈ perfect to depth 247, while free rollout (blind: own drift) collapses at N≈16. *That gap is
the privilege gap, measured natively.* §14's probe (`run_assumptions_cwm.py`) refines it into a
matched-depth gap curve and adds the A2 drift-validity metric — both on CWM.

**Training-model choice (deferred, faithful options):**
- **LoRA on CWM-32B directly** — the faithful path (no transfer gap). Feasible on 4×A6000 (frozen bf16
  weights ~64 GB across 4 GPUs ≈ 16 GB/GPU + LoRA optimizer + checkpointed activations) via PEFT +
  FSDP/DeepSpeed; needs verifying PEFT target-modules on `CwmForCausalLM` and modest seq/batch.
- A better open model is fine for *general* world-model questions but **not** for CWM-specific ones.

Updated ledger status: **A3 → ⚠/✓ on CWM (large gap, §8 + §14)**; **A2 → measuring on CWM (§14)**.

---

## 15. TRACK A — real game-tick trace (self-contained Lua arena) + A5 confirmation

A realistic arena game **in Lua** (the game-scripting language): player + chase-AI enemies, collisions
(stomp = +score, enemy-contact = −hp), deterministic scheduled spawns, bounds, game-over. CWM traces
the whole multi-tick `simulate()` and we compare predicted per-tick state to the real `lua5.4` run.

**Result:** CWM produced a **217-frame** trace and tracked the game **partially**:

| game tick | CWM player state | ground truth | |
|---|---|---|---|
| 1 | x=4 y=3 hp=5 | x=4 y=3 hp=5 | ✅ |
| 2 | x=4 y=4 hp=5 | x=4 y=4 hp=5 | ✅ |
| 3 | x=5 y=4 **hp=5** | x=5 y=4 **hp=4** | ❌ (missed an enemy-contact −hp) |

- **Position/score tracked perfectly; the miss is a multi-entity interaction.** CWM kept the player's
  `x,y` exactly across ticks but **dropped an `hp` decrement** that happens deep inside the enemy
  loop (an enemy's chase step landing on the player). It tracks the *salient* actor (player movement)
  and loses a *side-effect buried in a secondary loop over many entities*.
- **This is a new, concrete failure flavor for A1:** symbolic state is sufficient *in principle*, but
  CWM's attention to it degrades for **effects distributed across many sub-entities** within one tick —
  not a horizon problem (it's only tick 3), but a **within-tick complexity / salience** problem.
- **A5 CONFIRMED (engine-interleaved logic is not standalone-traceable).** The companion test: a real
  `love.update(dt)` snippet (`love.keyboard.isDown`, `love.graphics.getWidth`, `love.math.random`)
  **fails under `lua5.4`**: `attempt to index a nil value (global 'love')`. Real engine games
  interleave logic with engine calls, so the clean "trace the logic" path requires either a
  self-contained logic layer or stepping over the engine calls (§10) with the engine as oracle.

**Implication for the games plan:** Lua game *logic* is traceable and CWM tracks the primary game
state well, but (a) **within-tick multi-entity side-effects** are a real accuracy gap (candidate for
SFT/OPSD), and (b) **engine calls must be abstracted/oracled** (A5) — you can't trace `love.*`. The
viable target is **self-contained or logic-separated** game code, with the engine as an opaque step.

```
run_lua_game.py            # arena game + A5 love2d check;  results/cwm_lua_game.json
```

---

## 16. CWM-NATIVE A2/A3 — and a major correction (no drift; OPSD premise fails *in the good way*)

Measured on **CWM directly** (per §14). Deterministic symbolic programs, free rollout vs teacher-forced,
by depth band:

| depth band | A3 teacher (true prefix) | A3 student (free/drifted) | privilege GAP | A2 valid |
|---|---|---|---|---|
| 1–10 | 1.00 | 1.00 | **0.00** | ~1.0* |
| 11–25 | 1.00 | 1.00 | **0.00** | 1.00 |
| 26–50 | 1.00 | 1.00 | **0.00** | 1.00 |
| 51–100 | 1.00 | 1.00 | **0.00** | 1.00 |
| 100+ | 1.00 | 1.00 | **0.00** | 1.00 |

Per program: counter N30 (127 frames), grid N24 (**198 frames**), list N30 (142 frames) — **free
rollout fully correct, frame-for-frame, on all three.** (*A2 "first invalid at 0" is just the entry
CALL frame's empty locals — a metric quirk; real validity ≈ 1.0.)

### 16.1 The correction
Earlier (rushed) runs suggested free rollout "diverges at N≈16". **That was wrong — a token-cap
truncation artifact** (A10: measurement faithfulness, the recurring offender). With proper caps, **CWM
free-rolls these programs perfectly to ~200 frames with zero drift.** CWM is therefore *substantially
better* at long-horizon free rollout than this report previously stated.

### 16.2 Consequence for training (this changes the plan — grounded)
- **A3's privilege-gap premise FAILS on this class — in the good way:** the gap is 0 because the
  **student is already perfect**, not because the teacher is weak. **OPSD/DAgger/re-grounding have no
  gradient where CWM already succeeds.** Training there would only risk catastrophic forgetting (A7)
  for no benefit.
- **Therefore: do NOT train on what works.** Training value exists ONLY where CWM genuinely fails —
  and we have a precise, grounded failure catalog from the other phases:
  1. **deep recursion** (§11, C `fib`),
  2. **OOP/encapsulation** (§11, C++ `Acc` opaque),
  3. **within-tick multi-entity side-effects** (§15, missed enemy-contact −hp),
  4. **native-code & concurrency** (§9 — but these want **abstraction** §10, not tracing).
- **The right next measurement** (before any training): re-run the A3 privilege-gap probe on the
  **failure cases** (recursion / multi-entity / encapsulation). If the privileged teacher *does* beat
  the drifted student *there*, OPSD has real signal exactly where it's needed. If even the teacher
  fails, the gap is a *capability* hole → SFT/RL, not distillation.

### 16.3 Ledger updates
- **A3 (privilege gap):** ✗ on easy programs (gap 0, student perfect) → **must be re-tested on failure
  cases**; that is where the OPSD premise lives or dies.
- **A2 (oracle at arbitrary states):** ✓ so far — free-rollout frames are ~100% structurally valid,
  so an oracle *could* relabel them (when drift exists).
- **A8 (determinism) / horizon:** the long-horizon drift wall is **much farther out than §8 implied**;
  for deterministic symbolic programs, ~200 frames is clean.

```
run_assumptions_cwm.py     # CWM-native A2/A3 probe;  results/cwm_assumptions.json
trace_dataset.py           # CWM-format trace data from the free oracle (A6 holds for Python)
# LoRA-on-CWM is mechanically ready: peft 0.19.1, CwmConfig recognized (64L/6144h), llama target modules.
```

---

## 17. DATASET DESIGN (targeted at failure modes, not easy cases)

Given §16 (CWM already free-rolls easy deterministic programs perfectly), the training dataset must
**concentrate on the failure modes** — training on easy cases gives no gradient and risks forgetting.
The §18 privilege-gap probe decides the *method* per mode; the dataset provides the *data* either way.

**Source of truth = the free oracle** (real interpreters: `sys.settrace` for Python, debuggers for
other langs). Each example = (source as context, true execution trace in CWM token format, loss mask
on the trace only). `trace_dataset.py` already does this for Python; A6 (free oracle ⇒ cheap data)
holds there.

**Composition (failure-weighted):**
| bucket | why | source |
|---|---|---|
| deep recursion (varied depth/branching) | §11 fib failure | Python recursion generators |
| OOP / encapsulated state via methods | §11 C++ Acc failure | Python classes w/ method-mutated state |
| multi-entity within-tick side-effects | §15 arena hp miss | Python game-tick sims (N entities) |
| long arithmetic / float accumulation | §11 grad-descent (budget) + long horizon | interacting-accumulator generators |
| a small fraction of "easy" (anti-forgetting replay) | preserve what works (A7) | mixed simple programs |

**Open dataset questions (to resolve empirically, grounded):**
1. **Method per bucket** = decided by §18: drift-induced buckets → OPSD/DAgger-style (student rollouts +
   oracle relabels); capability-hole buckets → SFT (true traces) or RL (verifiable reward).
2. **Forgetting control** (A7): replay-mix ratio of easy/general data; LoRA vs full; KL-to-base.
3. **Multi-language**: Python is free; other langs need a tracer. Start Python-only (covers all 4
   failure modes), add langs only if §11 language gaps justify the tracer-build cost (A6 for non-Python
   is untested).

**Verifiable rewards (for the RL option)** — all cheap and exact: final-return match, per-frame exact
match vs oracle, EOS-at-true-length, invariant satisfaction (in-bounds, schema-stable), parse-validity.

```
trace_dataset.py           # Python -> CWM-format (source ctx + masked trace);  A6 holds
run_privilege_gap.py       # §18 decider: per-failure-mode OPSD-vs-SFT-vs-RL;  results/cwm_privilege_gap.json
```

---

## 18b. NEW STRUCTURAL LIMIT (found while building §18 probe): trace length vs context window

The privilege-gap probe initially crashed: a teacher prefix for `multientity_sideeffect` (205 frames)
serialized to **16,537 tokens > 16,384 context**. This is a *real, grounded* limit worth logging:

- **Deep traces consume the context window fast** — each frame carries its locals (and CWM's diff
  encoding still grows the prefix). A ~200-frame trace can exceed a 16k window; CWM's full window is
  131k, but long/****deep traces will hit it eventually. → **trace length is itself a horizon limit**,
  independent of accuracy. Mitigations: CWM's diff-locals (already used), sliding the context, or
  **abstraction (§10)** to compress sub-computations into single opaque steps.
- This is a concrete instance of why **abstraction isn't just for native code** — it's also a
  *context-budget* tool: stepping over a sub-call replaces many internal frames with one return frame.

Fix applied: raised probe context to 24k + skip any probe whose prefix exceeds budget (so depth
coverage is honest rather than crashing). Re-running §18 with this guard.

---

## 19. PRIVILEGE-GAP RESULTS — the method decider (CWM native, on failure cases)

The §18 probe, measured on CWM: at each depth, predict frame d from the **true** prefix (teacher) vs
CWM's **own free-rollout** prefix (student), both scored vs truth.

| failure mode | teacher (privileged) | student (own drift) | gap | free-rollout frame-acc | diagnosis |
|---|---:|---:|---:|---:|---|
| **oop_encapsulation** | 0.95 | **0.02** | **+0.93** | 0.03 | drift-induced (root cause = hidden state) |
| **long_arithmetic** | 0.73 | 0.13 | **+0.60** | 0.14 | drift **+** capability ceiling |
| recursion_fib | 0.88 | 0.88 | 0.00 | 0.88 | mild capability ceiling, no drift |
| multientity_sideeffect | 0.99 | 1.00 | −0.01 | 0.99 | frame-fine; rare consequential miss |

**This cleanly discriminates four *different* failure types — and each wants a different fix:**

1. **`oop_encapsulation` — huge gap (0.95 vs 0.02), but the ROOT CAUSE is OBSERVABILITY (A1), not just
   drift.** Trace inspection: the object's mutated state is **never visible** — `self` is always
   rendered `<Acc object at 0x…>`, so `self.total`/`self.count` accumulating across `add()` calls are
   *hidden*. The privileged teacher stays aligned (fed truth) at 0.95; the student, with no anchor for
   the hidden state, diverges early → 0.02. **Primary fix = REPRESENTATION**: change the trace
   observation function φ to **expand object attributes** (`self = {total: 81, count: 1, maxv: 9}`),
   making the state a sufficient statistic. OPSD is then a secondary aid. *This is a data/φ fix that
   may remove the need to train at all here.*
2. **`long_arithmetic` — drift + capability.** Gap 0.60 (OPSD has signal) **and** teacher ceiling 0.73
   (even privileged, CWM mis-computes ~27% of `(a*3+b)%97`-style steps). → **OPSD to fix drift + SFT/RL
   to lift the per-step arithmetic capability.** Pure distillation won't reach the 0.27 the teacher
   itself misses.
3. **`recursion_fib` — mild capability ceiling, no drift.** teacher = student = 0.88 (gap 0). OPSD has
   nothing to distill (student already = teacher). → low priority; light SFT at most. (Note Python fib
   = 0.88 is far better than §11's C `fib` failure — language matters.)
4. **`multientity_sideeffect` — frame-accurate but outcome-fragile.** Frame-acc ~0.99, yet a *single*
   missed within-tick side-effect (the §15 hp decrement) corrupts the FINAL answer. → frame-level
   training won't prioritize the rare-but-critical frame; this wants **outcome-level reward (RL on the
   final value / invariant)**, not more imitation.

### 19.1 The decision (evidence-based, per mode)
| fix | applies to | why |
|---|---|---|
| **Representation (expand φ: object attrs, etc.)** | oop_encapsulation | hidden state → make it observable (A1) |
| **OPSD / on-policy distillation** | oop (residual), long_arithmetic | large teacher≫student gap = drift the teacher can correct |
| **SFT / RL capability** | long_arithmetic, recursion | teacher itself < 1.0 → a capability hole distillation can't fill |
| **RL outcome reward** | multientity | rare consequential miss invisible to frame-level loss |

**Headline:** OPSD's premise is **validated where it should be** (oop gap +0.93, arithmetic +0.60) and
correctly **absent where it shouldn't apply** (recursion gap 0, multientity gap 0). The probe is a
genuinely useful *method-selection diagnostic*. And the biggest single lever surfaced is **not a
training method at all — it's the trace representation (φ)**: exposing hidden object state likely fixes
the most dramatic failure outright.

### 19.2 Dataset consequence
The dataset must carry, per example, **both** the standard trace **and** an **observability-expanded
trace** (object attrs unpacked) so we can A/B whether φ-expansion alone closes the oop gap before
spending training compute. `trace_dataset.py` + `failure_buckets.py` generate the programs; next step
is the φ-expansion variant + the OPSD/SFT data builders.

```
run_privilege_gap.py        # the decider;  results/cwm_privilege_gap.json
failure_buckets.py          # varied failure-mode program generators (recursion/oop/multientity/arith/easy)
```

---

## 20. φ-EXPANSION A/B — HYPOTHESIS REFUTED (φ is baked into the model)

§19 hypothesized the oop_encapsulation failure (state hidden behind `<Acc object>`) could be fixed
*training-free* by expanding the trace observation function φ to show object attributes. **We built it
and tested it on CWM. The hypothesis is refuted.**

| program | φ | teacher | student | gap | note |
|---|---|---:|---:|---:|---|
| oop_encapsulation | standard | 0.95 | 0.02 | +0.93 | baseline |
| oop_encapsulation | **expanded** | **0.80** | 0.02 | +0.78 | teacher got **worse**; student unchanged |
| multientity | standard | 0.99 | 1.00 | −0.01 | control |
| multientity | expanded | 0.99 | 1.00 | −0.01 | control unaffected |

(φ-expansion verified to work as intended: `self` renders `Acc({'total': 9, 'count': 1})` — the
accumulating state IS now visible in the trace.)

### 20.1 Why it failed (the grounded lesson)
- **The expanded φ is OUT-OF-DISTRIBUTION for CWM.** CWM was mid-trained on Python traces where objects
  render with the standard repr (`<Acc object at 0x…>`). Feeding it the *expanded* format at inference
  gives it a representation it never learned → it predicts the (now-OOD) frames **worse** (teacher
  0.95 → 0.80), rather than exploiting the newly-visible state.
- **The observation function φ is part of what the model LEARNED, not a free knob you swap at
  inference.** This is a real, generalizable constraint: you cannot improve a trace world-model by
  changing its serialization format at test time — the format itself is in-distribution-locked.

### 20.2 Corrected conclusion (and it *strengthens* the training case)
- The §19 diagnosis stands — the oop failure's **root cause is observability** (hidden object state).
- But the **fix is NOT training-free.** To expose that state you must **train CWM on expanded-φ
  traces** (SFT), so the new format becomes in-distribution. Then the now-visible state should be
  trackable and the gap should close *for real* (student rising, not teacher falling).
- So oop_encapsulation needs **training after all** — specifically **SFT on a φ-expanded corpus** — and
  this is now the concrete, evidence-backed recommendation (not a speculative "maybe no training").
- **Supersedes §19.1/§19.2's "may remove the need to train at all here":** that was wrong; φ-expansion
  requires retraining to be usable.

### 20.3 Meta (the directive paying off)
This is exactly "note assumptions and see when they fail." The assumption **"φ-expansion is a
training-free representation fix"** FAILED via train/inference distribution mismatch — a clean, useful
negative result that redirects the plan: **the φ-expanded trace becomes a TRAINING TARGET (SFT), not an
inference-time swap.** The dataset (§17) should therefore emit expanded-φ traces *as the SFT signal*
for the encapsulation bucket.

```
run_phi_expansion.py        # the A/B test;  results/cwm_phi_expansion.json
gt_trace.py: trace_program(..., expand_objects=True)   # the φ-expansion (now an SFT target, not a knob)
```

---

## 21. LoRA-on-CWM TRAINING — infrastructure validated, first SFT (φ-expanded oop)

Per §14/§20, the faithful path is **LoRA on CWM directly**, and §20 showed the φ-expanded format must be
**trained in** (it's OOD at inference). Built and validated the full pipeline:

- **Data** (`build_sft_data.py`): 400 φ-expanded oop traces + 400 standard (control), tokenized to
  CWM format with loss masked to the trace (not the source). Verified the expanded SFT target shows
  `self = Acc({'total': 57, 'count': 8, 'maxv': 10})` — the previously-hidden state is now the
  supervision signal.
- **Trainer** (`train_lora_cwm.py`): custom loop, `device_map="auto"` (naive pipeline across 4×A6000,
  fits the 64 GB bf16 base ≈ 19–25 GB/GPU), gradient checkpointing, LoRA r=16 on the 7 Llama-style
  linear modules. **Smoke test passed**: loads in 19 s, **124.8 M trainable params (0.38 %)**,
  loss-down, **saves a valid adapter** (499 MB).
- **Eval path**: `CWMvLLM(lora_path=…)` serves base+adapter via vLLM LoRA; `run_phi_expansion.py --lora`
  re-runs the teacher/student gap probe on the trained model.

**First SFT (running):** φ-expanded oop bucket, 50 steps, LoRA r=16. Throughput ≈ 30–60 s/step on the
no-NVLink box (naive pipeline = one GPU active at a time; acceptable for adapter-scale SFT). Starting
loss ≈ 0.035 (CWM already knows most of the format; the OOD part is the expanded object state).

**The decisive test (next):** re-run §20's A/B on the LoRA model. Hypothesis (corrected, §20): training
the expanded φ in should now make the **student rise** (not the teacher fall) — i.e. the encapsulation
gap closes *for real* because the once-hidden state is both visible *and* in-distribution.

```
build_sft_data.py     # phi-expanded / standard SFT corpora -> data/*.jsonl
train_lora_cwm.py     # LoRA SFT on CWM-32B (device_map=auto, grad ckpt) -> adapters/
models/cwm_trace.py   # CWMvLLM(lora_path=...) for vLLM LoRA eval
```

---

## 22. LoRA A/B RESULT — the observability fix WORKS when trained in (and forgetting is real)

Ran §20's teacher/student gap probe on the **LoRA-on-CWM** model (adapter trained on the φ-expanded
oop bucket, §21) vs the base-model baseline. vLLM served base+adapter correctly (`PunicaWrapperGPU`,
`max_lora_rank=32`). Two findings — one confirmation, one cautionary — and together they make the
method choice concrete.

| program | φ | teacher | student | gap | baseline student | Δ student |
|---|---|---:|---:|---:|---:|---:|
| oop_encapsulation | standard | 0.92 | 0.02 | +0.90 | 0.02 | 0 (adapter trained on *expanded*, so standard untouched) |
| **oop_encapsulation** | **expanded** | 0.78 | **0.80** | **−0.02** | 0.02 | **+0.78** ✅ |
| multientity | standard | 0.99 | **0.68** | +0.31 | **1.00** | **−0.32** ⚠️ |
| multientity | expanded | 0.99 | **0.68** | +0.31 | **1.00** | **−0.32** ⚠️ |

### 22.1 Finding 1 — §20's corrected hypothesis is CONFIRMED (the fix is real)
Training the expanded-φ representation *in* lifts the oop free-rollout **student 0.02 → 0.80** and
**closes the gap +0.90 → −0.02**. Critically this happens the *right way* — **the student RISES to the
teacher's level (≈0.78–0.80), the teacher does not fall**. Contrast with §20's training-free swap,
where expanded-φ was OOD and made the *teacher* drop to 0.80 while the student stayed at 0.02. So:

- The §19 diagnosis (oop failure = **observability**, state hidden behind `<Acc object>`) was right.
- The §20 correction (the fix is **not** training-free; expanded-φ must be an **SFT target**) was right.
- **And now it is demonstrated end-to-end on the real CWM**: a 125 M-param (0.38 %) LoRA, ~50 steps,
  makes a once-collapsed failure mode free-rollable at 0.80. *The representation can be trained in.*

This is the first **positive capability gain** in the whole study — every prior result characterized
where CWM works or fails; this one *moves* a failure into the working set, env-free, via cheap SFT.

### 22.2 Finding 2 — narrow SFT REGRESSES a held-out, already-solved mode (forgetting is real)
The same adapter **degrades multientity** — a case the base model solved *perfectly* (student 1.00,
gap −0.01) — down to **student 0.68, gap +0.31**. multientity has no objects, so its standard and
expanded traces are identical, and both regress identically: this is not a φ-format effect, it is the
adapter shifting the model's distribution. The oop-only LoRA over-specialized and **forgot** unrelated
trace dynamics.

This is **direct empirical support for @namak-kun's standing instinct** — *"SFT has problems with
destroying old knowledge"* — which is exactly why they raised **OPSD / softer regimes** over naive
SFT. We now have the evidence, not just the worry: narrow trace-SFT buys a new capability **and** pays
a forgetting tax on held-out behavior.

### 22.3 The grounded takeaway → method choice
- **Direction is correct:** the privilege-gap probe (§19) → φ-expansion-as-SFT-target (§20) →
  LoRA SFT (§21) pipeline *closes a real gap the right way*. The method-selection diagnostic paid off.
- **Naive narrow SFT is not the final method:** it forgets. Two concrete fixes, both now justified by
  the data (not speculation):
  1. **Mixed / replay corpus** — train on the failure bucket **plus** a sample of already-working
     traces (multientity, easy, list/grid) so the adapter can't drift off them. Cheapest fix; test next.
  2. **KL-anchored on-policy distillation (OPSD)** — keep the student close to the base on its own
     rollouts via a privileged-teacher KL, which structurally resists forgetting where the base is
     already right (gap ≈ 0 → KL ≈ 0 → no pressure to change). This is the user's proposed method and
     the forgetting result is its motivation.
- **Net:** the §13 ledger entry "SFT may destroy old knowledge" moves from *assumption* to
  **measured fact** (−0.32 on a held-out mode), and the next experiment is **mixed-corpus SFT vs OPSD**
  on the *same* oop gap, scored on **both** oop (does the gain survive?) **and** multientity (does the
  forgetting disappear?).

```
run_phi_expansion.py --lora adapters/cwm_oop_expanded   # this A/B;  results/cwm_phi_expansion_lora.json
results/cwm_phi_expansion.json        # base baseline (oop std 0.95/0.02, expanded 0.80/0.02; multient 0.99/1.00)
results/cwm_phi_expansion_lora.json   # LoRA (oop expanded 0.78/0.80 gap closed; multient 0.99/0.68 regressed)
```

---

## 23. OPSD-ON-CWM — using the paper's code (dep resolution, port, viability-first)

Per @namak-kun: prefer the **paper's actual code** over a reimplementation, do the privileged
same-weights teacher (path a), and also try a separately-SFT'd teacher (path b). The chosen method is
**OPSD** (On-Policy Self-Distillation, Zhao et al., *Self-Distilled Reasoner*, arXiv:2601.18734) —
official repo `ConstantWangheng/OPSD`. (Note: distinct from **RLSD** = arXiv:2601.20802 *RL via
Self-Distillation* (Hübotter et al.), the RL+self-distill paper, and from **SDPO**, a third method.)

### 23.1 OPSD mechanism (read from their code)
- `OPSDTrainer(SFTTrainer)`: one model is student+teacher via different CONTEXT. Student prompt = problem;
  teacher prompt = problem + GOLD SOLUTION + transition. Student GENERATES a completion (on-policy);
  both forward on `[own_prompt ++ same student completion]`; loss = generalized **JSD** (or
  `use_tinker_loss` = sampled-token reverse-KL PG) over the completion, aligned by prompt lengths.
- **`fixed_teacher + use_peft` (recommended):** teacher = base with **LoRA adapter DISABLED**
  (`model.disable_adapter()`), student = base+LoRA. Anti-forgetting is *claimed* structural: where the
  base is already right, teacher≈student → JSD≈0 → ~no gradient. **This is exactly path (a).**

### 23.2 Dependency conflict — RESOLVED (the pinned env can't load CWM)
- OPSD pins `transformers==4.57.3`, `vllm==0.11.0`. **CWM's `modeling_cwm.py` first appears in
  `transformers v5.0.0`** (404 at all v4.57.x; config `model_type=cwm`, `CwmForCausalLM`, interleaved
  `layer_types`+`sliding_window`). vllm cwm also absent at 0.11.0. **So the paper's pinned env literally
  cannot load CWM.**
- **But** `trl` keeps `experimental.gold` (GOLDConfig/GOLDTrainer) through **main**, and **trl 1.7.0**
  requires `transformers>=4.56.2` (only `!=5.1.0`) and **`vllm>=0.14.0,<=0.23.0`** — our working stack
  (transformers **5.12.1** + vllm **0.23.0**, which load CWM) sits *inside* that range. So a single
  viable env exists; **we run the paper's actual trainer, not a reimplementation.**
- Installed `trl==1.7.0 --no-deps` into the training `.venv` (torch 2.12 + transformers 5.12.1 + peft
  0.19.1 + accelerate 1.14.0); transformers/torch unchanged. Patched **3 API drifts** (0.26.2→1.7.0):
  `truncate_dataset` (→`trl.experimental.utils`), `VLLMClient` (→`trl.generation.vllm_client`),
  `empty_cache`/`DataCollatorForChatML` (moved). **`opsd/opsd_trainer.py` now imports cleanly →
  `OPSDTrainer` available** in a CWM-capable process.

### 23.3 Data-quality bug fixed (φ-expansion address leak)
`gt_trace._expand_value` only expanded objects with a **non-empty** `__dict__` (`and d`). A
freshly-constructed object at its `__init__` *call* frame has `{}` → fell through to `repr()` →
non-deterministic `<Acc object at 0x..>` leaked into the gold trace (unlearnable address tokens +
inconsistent φ). Fixed to expand empty dicts too → `Acc({})`. Verified: **0 address leaks**, `self`
renders `Acc({})` at init then `Acc({'total':..,..})` as state accrues. Improves all SFT/OPSD targets.

### 23.4 Viability-FIRST (rubber-duck, gpt-5.5): probe before training
The duck flagged a **critical subtlety (Q4)**: OPSD's anti-forgetting needs `p_teacher(·|ctx) ≈
p_student(·|ctx)` where the base is right. But OPSD-literal gives the teacher a DIFFERENT context (gold
in-prompt) even on already-solved modes → nonzero JSD → **forgetting reintroduced, defeating the
purpose.** Also CWM is non-instruction + traces are deterministic, so "here's the trace, now re-trace"
is OOD. Verdict: **do not train blind — run a frozen-teacher viability probe first.**
`run_opsd_probe.py` measures gold next-frame accuracy under (1) teacher-forcing [non-privileged],
(2) Option-1 gold-retrace [privileged], (3) retrace with WRONG gold [ablation], on **oop** (capability)
and **multientity** (forgetting control). Gates: viable iff retrace≥tf on oop, retrace≈tf on
multientity, and retrace≫wrong-gold. If it fails → **hybrid**: gold-prefix capability loss +
same-context base-consistency (disable_adapter teacher, NO privilege) replay for anti-forgetting.

```
opsd/opsd_trainer.py     # the paper's OPSDTrainer, ported to trl 1.7.0 / transformers 5.12 / CWM
build_opsd_data.py       # {source, ctx_ids, gold_ids} rows (phi-expanded), collator-agnostic
run_opsd_probe.py        # frozen-teacher VIABILITY probe -> results/opsd_viability_probe.json (VERDICT below)
```

### 23.5 VIABILITY PROBE RESULT — Option-1 refuted; CWM ignores far-context privilege
`run_opsd_probe.py` (φ-expanded, frozen base, NO training):

| bucket | tf (non-priv) | retrace (Option-1 priv) | retrace_x (wrong gold) | n |
|---|---:|---:|---:|---:|
| oop_encapsulation | 0.973 | 0.973 | 0.973 | 75 |
| multientity | 1.00 | 1.00 | 1.00 | 53 |

**`retrace == tf == retrace_x` everywhere.** Putting the full gold trace (or a WRONG program's gold) in
the teacher's preamble has **zero effect** on next-frame prediction — CWM attends to the **local frame
prefix** and ignores a far preamble. So OPSD-literal "gold-in-teacher-prompt" privilege is **vacuous on
CWM** (the `real_signal retrace≫wrong-gold` gate fails). **VERDICT: Option-1 not viable → hybrid.**
This also *empirically* confirms the duck's Q4: since the privileged context doesn't change the
distribution even where we'd want it to, it certainly can't be relied on — and on solved modes the
(equal) distributions mean the privilege adds nothing either way. Privilege for CWM must be the **local
correct prefix** (Option-2), not a far solution-in-context.

**Incidental but important:** teacher-forcing on expanded-oop measured **0.973 here vs 0.80 in §20/§22**
— because §23.3 fixed the φ address-leak (non-deterministic `<obj at 0x..>` tokens were depressing it).
Re-measuring the base free-rollout student on fixed-φ oop next (the §22 numbers were on leaked data).

### 23.6 Re-grounding: φ-fix is marginal on the hard oop; drift is still the wall
Re-ran the §22 base A/B on the *fixed* FAILURE_PROGRAMS oop with the φ address-leak fix:

| φ | teacher (§22 leaked → now fixed) | student free-roll |
|---|---|---|
| standard | 0.949 → 0.949 | 0.017 (unchanged) |
| expanded | 0.797 → **0.814** | **0.017** (unchanged) |

The leak fix lifts the *teacher* only +0.017 on this (harder) program — the 0.973 in §23.5 was an easier
`gen_oop` set. **The free-rollout student stays 0.017: compounding drift, not φ noise, is the wall, and
training is still required.** §22's conclusion (fix oop by training expanded-φ in) stands. With OPSD-literal
refuted (§23.5), the path is **mixed-corpus SFT** (capability + replay) then **+ base-anchoring** if replay
under-delivers — both directly target §22's measured forgetting (multientity 1.00→0.68).

---

## 24. MIXED-CORPUS SFT — forgetting SOLVED, oop capability improved (path 1 works)

Per §23.5 (OPSD-literal refuted) and §23.6 (drift is the wall), ran the user's **path 1 = mixed-corpus
SFT**: same 50-step/LoRA-r16 budget as the §22 oop-only run, but the corpus is **65% oop (capability) +
35% replay** of short working traces — including `gen_multientity_short`, a short (≈30-frame) proxy of the
multientity pattern, because the real multientity trace (11k+ tokens) is too long to train on and stays
**held-out** for eval. (φ address-leak fixed, §23.3.)

| metric | §22 oop-only LoRA | **§24 mixed LoRA** |
|---|---:|---:|
| oop expanded **student** (free-roll gap probe) | 0.797 | **0.831** |
| oop expanded **free_acc** (full free rollout) | ~0.02→0.80 | **0.833** |
| multientity **student** (HELD-OUT, the §22 victim) | **0.678** (forgot) | **1.000** (retained) |
| oop gap (teacher−student) | +0.78→closed | −0.017 (closed, student≥teacher) |

**Two clean wins at once:**
1. **Forgetting ELIMINATED.** The held-out multientity (regressed to 0.678 by §22's narrow SFT) is back
   to **1.000** — a short same-pattern replay proxy fully preserves a *longer held-out instance* of the
   mode. Replay-based retention generalizes across trace length here.
2. **Capability improved, not just survived.** Despite oop being only 65% of the data, oop student went
   **0.797 → 0.831** (vs the oop-only run) — fixed-φ data + corpus diversity helped rather than hurt.

**Key scientific takeaway (answers duck Q2):** *gold-only* SFT (correct-prefix CE) lifts oop FREE-ROLLOUT
from **0.02 → 0.83** — so most of the compounding-drift failure is fixable without any on-policy data; the
model staying on-manifold + φ exposing object state is enough to largely stop the drift. The residual
(0.83, not 1.0) is the open question §25 tests with DAgger (does training on the student's OWN drifted
states beat more gold data?).

```
failure_buckets.gen_multientity_short   # short multientity-pattern replay proxy (held-out stays long)
build_sft_data.py --buckets oop:0.5,multientity_short:0.25,arithmetic:0.12,recursion:0.08,easy:0.05 --expand
adapters/cwm_mixed_expanded/            # the mixed-SFT adapter
results/cwm_phi_mixed_lora.json         # oop 0.831 / multientity 1.000
```

---

## 25. DAgger A/B (gold-prefix vs drift-prefix) — on-policy adds NOTHING for oop; residual is structural

The synthesis duck reframed the real fork as the **input-state distribution** (gold targets either way):
gold-prefix SFT vs **drift-prefix DAgger** (train on the student's OWN drifted states with gold recovery
targets). Built the matched ablation (`build_dagger_data.py`): same 60 oop programs, same depths, same gold
target frame per depth — **only the prefix differs**. Drift prefixes = rollouts of the §24 mixed-SFT student
(0.83, mild realistic drift); verified prefixes differ from gold in **86%** of rows, targets identical, and
the recovery target is diff-encoded against the *drifted* prev (so `..` references the visible state). Two
fresh LoRAs, matched 150-step budget: `cwm_dagger_gold` (Arm A) and `cwm_dagger_drift` (Arm B).

**Eval gotcha (cost real compute): vLLM 0.23 does NOT switch LoRA adapters by per-request `LoRARequest` id
within one engine** — it applies whatever adapter the engine was *initialized* with (`lora_request=None`
does correctly disable to base). A multi-adapter-in-one-session eval gave 3 identical rows; verified with a
token-identity probe; fixed by evaluating **one adapter per process**. (Memory stored.)

**Held-out free-rollout (gen_oop, seed=999, disjoint from training; novel values = duck's history-use check):**

| adapter | free-roll frame acc | fully-correct | min/max |
|---|---:|---:|---:|
| base | 0.033 | 0% | 0.025/0.041 |
| mixed-SFT (§24, whole-trace) | **0.9324** | 0% | 0.902/0.957 |
| Arm A — gold per-frame | **0.9324** | 0% | 0.902/0.957 |
| Arm B — drift per-frame (DAgger) | **0.9324** | 0% | 0.902/0.957 |

**All three are IDENTICAL to 4 decimals (same min/max).** On-policy/DAgger does **not** beat gold SFT here.

### 25.1 Why: the residual is a STRUCTURAL φ-rendering error, not drift
`run_residual_diag.py` shows the gold model's errors are at the **same frames in every program** (frame 6,
14, ...) with the **same mistake**:
```
GT  : return  self.maxv = -999  | self = Acc({'total': 0, 'count': 0, 'maxv': -999})
PRED: return  self.maxv = -999  | self = Acc({'total': 0, 'count': 0})        # drops maxv:-999
```
The model consistently **omits the `maxv: -999` attribute** when rendering the freshly-constructed object at
the `__init__` return frame (and the analogous early frame). This is a **representation-fidelity residual at
fixed early positions** — NOT compounding drift (which would corrupt *late* frames). Since the error is the
same regardless of training prefix, all three adapters share it → identical 0.9324.

### 25.2 Grounded conclusion (answers duck Q2/Q4)
- **Gold-only SFT keeps the model ON-MANIFOLD for oop** → free-rollout 0.02→0.93 with no drift to speak of.
- **On-policy/DAgger therefore adds nothing here**: there are no drift-induced error states to recover from;
  the only residual is a fixed φ-rendering miss. On-policy data can fix *drift*, not *representation*.
- So for oop, the user's path-1 (SFT) is not just sufficient but **saturating**; DAgger is unnecessary. The
  remaining 0.07 is a φ-fidelity bug (drop of the `-999` sentinel attr), fixable by targeted representation
  supervision — a *different* lever than input-state distribution.
- **Caveat / scope:** this refutes DAgger>SFT *for oop, which has little free-rollout drift*. It does NOT
  settle modes with genuine compounding drift (longer/among-entity traces); there DAgger could still help.
  The matched A/B harness (`build_dagger_data.py` + `run_freeroll_eval.py`) is ready to test those.

```
build_dagger_data.py --mode {gold,drift}   # matched per-frame ablation data
run_freeroll_eval.py --lora <adapter>      # held-out free-roll (ONE adapter/process; vLLM switch bug)
run_residual_diag.py                       # first-divergence characterization -> structural maxv drop
results/freeroll_gold.json / freeroll_drift.json   # 0.9324 == 0.9324
```

---

## 26. DAgger on a DRIFT-HEAVY mode (arithmetic) — gold SFT FAILS here; the mode-dependence is the point

§25 found oop is saturated (gold SFT fixes free-roll, no drift headroom). To test where drift GENUINELY
compounds, repeated the matched A/B on **arithmetic** (`a=(a*3+b)%m` chains — one wrong value propagates to
all later frames). Base free-rolls arithmetic at **0.187**, and §24's mixed-SFT (12% arithmetic) at **0.183**
(no help) — so there is large free-rollout headroom from compounding drift, unlike oop.

| adapter (held-out arithmetic, seed 999) | free-roll frame acc |
|---|---:|
| base | 0.187 |
| §24 mixed-SFT (whole-trace, 12% arith) | 0.183 |
| Arm A — **gold** per-frame SFT (100% arith) | **0.143**  ← *worse than base* |
| Arm B — **drift** per-frame SFT (DAgger from base) | 0.180  (≈ base) |

### 26.1 The contrast that matters
- **Gold SFT FIXES oop (0.02→0.93) but FAILS arithmetic (0.19→0.14).** gold/correct-prefix SFT only ever
  trains on on-manifold states; it works when free-rollout stays on-manifold (oop, low drift) but does
  nothing — even hurts — when free-rollout drifts off-manifold (arithmetic, compounding value error). This
  is *exactly* the duck's Q2: **gold SFT cannot fix free-rollout drift on a mode that actually drifts.**
- **DAgger from a WEAK student hits the resync wall.** Arm B's drift prefixes came from the base (free-roll
  0.19 = heavily wrong), so the gold recovery target at frame d is NOT computable from the visible (wrong)
  prefix. Training loss never converged (**plateau ≈0.29 vs gold's ≈0.04**) — the objective is largely
  unfittable; the model can only memorize position→value, which doesn't transfer (held-out 0.18 ≈ base).
- Note **drift (0.18) > gold (0.14)** here — the opposite ordering from oop. On a drift-heavy mode, training
  on (even garbage) on-policy states is *less harmful* than over-fitting correct-prefix per-frame targets,
  but neither actually fixes the drift.

### 26.2 Grounded conclusion — when does on-policy matter?
- **Gold SFT's free-rollout fix is MODE-DEPENDENT:** sufficient (saturating) where drift is low (oop),
  insufficient (even harmful) where drift compounds (arithmetic). There is no single "SFT solves it" answer.
- **Naive single-shot DAgger needs a decent base policy.** From a 0.19 student the drift is too heavy and
  recovery targets are non-computable (resync pathology) — so a *single* DAgger round from base doesn't
  rescue arithmetic. Proper DAgger must be **iterative** (bootstrap mild drift from a progressively better
  student) or paired with whole-trace coherence; pure RL (reward = trace match) is the other lever.
- This is the honest limit of the SFT/DAgger toolkit on compounding-drift value computation — and it
  cleanly delineates *where* the user's DAgger intuition pays off: drift-heavy modes, but only iteratively.

(Whole-trace 100%-arith SFT control to rule out the per-frame/no-EOS confound — see §26.3.)

### 26.3 Whole-trace control — per-frame was a partial confound, but arithmetic stays hard
Trained a dedicated **whole-trace** 100%-arith SFT (with EOS, coherent traces — like §24's oop recipe):

| arithmetic adapter (held-out) | free-roll |
|---|---:|
| base | 0.187 |
| per-frame gold (Arm A) | 0.143 |
| per-frame drift (Arm B) | 0.180 |
| **whole-trace SFT** | **0.243** |
| *(oop whole-trace SFT, for contrast)* | *0.93* |

- **Per-frame/no-EOS WAS a partial confound:** whole-trace (0.243) beats per-frame gold (0.143, which hurt)
  — the per-frame format degrades free-rollout (no learned stop / coherence). So part of Arm A's failure was
  format, not just gold-vs-drift.
- **But even proper whole-trace arith SFT only reaches 0.243** (+0.06 over base) — vs oop's 0.93. So the core
  conclusion stands and is now confound-controlled: **correct-prefix SFT lifts oop free-roll 0.02→0.93 but
  barely moves arithmetic 0.19→0.24.** Arithmetic's compounding value-drift is genuinely not fixable by SFT
  on correct prefixes; it needs on-policy exposure to the model's own drift — and single-shot per-frame
  DAgger from a weak base can't deliver it (resync wall, §26.1). **Iterative DAgger or RL is the real lever
  for high-drift modes.** This is the grounded boundary of what SFT achieves for CWM free-rollout.

```
results/freeroll_arith_gold.json (0.143) / freeroll_arith_drift.json (0.180) / freeroll_arith_mixed.json (0.183)
build_dagger_data.py --buckets arithmetic:1.0 --mode {gold,drift}   # matched drift-heavy A/B
```

---

## 27. RL VIABILITY — arithmetic is a CAPABILITY CEILING (outcome-RL can't bootstrap), not drift

The user asked for an RL solution (for the drift-heavy / residual failures SFT+DAgger couldn't fix).
Before building GRPO, ran the gating probe (`run_rl_viability.py`): at sampling temperature, does CWM
produce rollouts with (a) exploitable reward **spread** (GRPO signal) and (b) any **fully-correct**
rollouts to reinforce (ReST/rejection-sampling signal)?

**Arithmetic (base, T=0.8, G=8, 12 programs = 96 rollouts):**
| metric | value |
|---|---:|
| greedy reward | 0.198 |
| sampled group mean | 0.190 |
| **best-of-G mean** | **0.199** (≈ greedy) |
| headroom best-of-G − greedy | **0.001** |
| frac groups with any spread | 0.33 |
| **fully-correct rollouts** | **0 / 96** |

**Both GRPO and ReST are DEAD for arithmetic.** Temperature gives ~no spread (best-of-8 ties greedy),
and the model **never** samples a correct trace in 96 tries. This is the signature of a **capability
ceiling, not an exploration/drift problem**: the model is *confidently, deterministically wrong* at
computing the chained modular values `(a*3+b)%97` over ~23 steps — it's being asked to be a calculator,
and it can't, so there is nothing for outcome-RL to reinforce (RL needs the policy to *sometimes*
succeed). Even frame-level dense reward won't help: best-of-G frame-acc ties greedy → no better
computation is hiding in the sample distribution.

### 27.1 Grounded reframing of where RL helps
- **Outcome-RL (GRPO/ReST) can only amplify success the model can already sometimes produce.** For a
  raw-arithmetic capability hole (0 successes), it cannot bootstrap. This is a real, honest boundary of
  the SFT→DAgger→RL toolkit on CWM.
- **The right RL target is the opposite regime: a mode where the model is ALREADY CLOSE and sometimes
  samples a perfect trace** — e.g. the **oop residual** (mixed-SFT at 0.93, gap = the `maxv:-999`
  rendering slip). There, perfect rollouts should exist in the sample → ReST/GRPO can push 0.93→~1.0.
  Probing that next (oop + mixed-SFT adapter).
- **Architectural note (connects to the project's thesis):** raw multi-step arithmetic is exactly the
  kind of thing a code world model should *delegate to actual execution / a tool*, not predict — a
  natural boundary between "predict state evolution & control flow" (the model's strength) and "be a
  calculator" (better executed). Tool-use / scratchpad is the lever for arithmetic, not outcome-RL.

```
run_rl_viability.py   # reward-spread + perfect-rollout-existence gate -> results/rl_viability_*.json
```

### 27.2 The oop residual is ALSO RL-resistant — CWM errors are CONFIDENT (no exploration)
Tested the more promising RL target — the oop residual (mixed-SFT at 0.93, gap = the `maxv:-999`
rendering slip) — sweeping temperature **0.8 / 1.2 / 1.5** (G=8, 12 programs):

| T | best-of-G | reward spread | perfect rollouts |
|---|---:|---:|---:|
| 0.8 | 0.9308 | 0.0 | 0/96 |
| 1.2 | 0.9308 | 0.0 | 0/96 |
| 1.5 | 0.9308 | 0.0 | 0/96 |

**Identical reward (std 0.0) at every temperature** — even 1.5. A direct sampling diagnostic confirms this
is *not* a sampling bug: rollouts ARE token-distinct (6/6 at T=1.2), but they **diverge only very late**
(first difference at token ~398/400) — **CWM trace generation has near-zero entropy early and high entropy
only at the tail.** The model's *errors* (the `maxv` drop at frame 6, and the arithmetic miscomputations)
live in the **high-confidence early/shared region that sampling never varies** — the model is
*confidently wrong*, not uncertain.

### 27.3 The real conclusion — outcome-RL needs UNCERTAIN errors; CWM's are CONFIDENT
- **Outcome-RL (GRPO/ReST) can only fix errors the model is UNCERTAIN about** (so sampling surfaces a
  correct alternative to reinforce). CWM's residual errors are **confident and deterministic** — same
  wrong token every sample — so there is **no exploration signal**, at any temperature, for either the
  oop `maxv` slip (confident representational error) or arithmetic (confident miscomputation).
- This is a **general property of doing RL on a peaked trace world-model**, and it cleanly separates the
  levers:
  - **Confident representational error (oop `maxv`)** → fix with **targeted SFT** (supervise the correct
    rendering); RL is the wrong tool (nothing to explore).
  - **Confident capability hole (arithmetic)** → needs **tool-use / scratchpad / actual execution**, not
    outcome-RL (the model can never sample a correct trace to reinforce).
  - **RL would help only a mode that is *stochastically* wrong** — where the model sometimes nails the full
    rollout and sometimes drifts. We have not found such a mode in CWM yet: the solved ones are
    deterministic-correct, the unsolved ones are deterministic-wrong.
- **Net for the user's "this'll take RL":** grounded result is the opposite — *for the failures we have,
  RL is NOT the right lever.* The oop `maxv` residual is a 1-token SFT fix; arithmetic is a capability/tool
  boundary. RL on CWM trace prediction needs a genuinely **exploration-limited (stochastic-drift)** failure
  to be useful, and identifying/【constructing】one is the prerequisite for any GRPO work.

### 27.4 CORRECTION — arithmetic is DRIFT-dominated (teacher-forced 0.73), not a flat capability ceiling
The user asked: did our SFT kill arithmetic, or was it bad anyway? Two clean checks settle it:

**(a) SFT did NOT kill it (it was bad already).** On identical held-out programs, base free-roll = 0.1825,
the §24 mixed-SFT adapter = **0.1826** — exactly neutral. (The 0.143 in §26 was the per-frame-format arm;
whole-trace arith SFT recovered to 0.243.) So the mixed-SFT we ship did not regress arithmetic at all.

**(b) But "bad" = DRIFT, not a flat capability ceiling.** Teacher-forced vs free-roll on base
`long_arithmetic`:

| | accuracy |
|---|---:|
| teacher-forced (next frame from CORRECT prefix) | **0.726** |
| free-roll (own prefix) | 0.141 |
| **drift gap** | **+0.595** |

Given a *correct* history, base CWM predicts the next arithmetic frame **73%** of the time — it CAN do
each step. Free-roll collapses to 0.14 because the ~27% per-step miscomputations **compound** over ~73
frames (0.73⁷³ ≈ 0). So §27.1's "capability ceiling" label was too strong: arithmetic is
**drift-dominated, rooted in a per-step computation imperfection (the 27%).** This also *explains* the RL
probe's 0/96 perfect + no spread: a long horizon makes a fully-correct sampled rollout essentially
impossible, and the per-step errors are confident (committed wrong values), so neither ReST (no successes)
nor GRPO (no spread) gets signal — the conclusion that *outcome-RL can't bootstrap this* still holds, but
the reason is **horizon × confident per-step error**, not an inability to compute a single step.

Implication for the lever: improving arithmetic means raising the **per-step** computation accuracy (the
27%), which gold-prefix SFT can't (the model is already trained on correct traces) — so it's the
**calculator/tool-use** boundary (delegate the arithmetic to execution / scratchpad), or process-level
supervision, not outcome-RL and not more whole-trace SFT.

```
results/cwm_arith_teacherforced.json   # base long_arithmetic: teacher 0.726 / free-roll 0.141 (drift +0.60)
```

---

## 28. CROSS-LANGUAGE CHECK — Python-φ SFT did NOT narrow CWM; the oop fix TRANSFERS to C++

The user worried that SFT on Python φ-expanded traces might **narrow** CWM cross-language (the
"everything is language-specific" failure mode). Ran the multi-language battery (`run_battery.py --lora`)
on base vs the §24 mixed-SFT adapter, final-value match:

| program | lang | base | mixed-SFT adapter |
|---|---|---|---|
| Cpp_vector_class (class Acc{total}) | cpp | ✗ (204) | **✓ (120)** ← FIXED |
| JS_loops | js | ✗ (None) | **✓ (55)** ← fixed |
| JS_map_reduce | js | ✓ (30) | ✗ (None, over-gen 53 fr) ← regressed |
| C_bits_string | c | ✓ | ✓ |
| C_recursion_ptr_struct | c | ✗ | ✗ (pre-existing) |
| Rust_iter | rust | ✓ | ✓ |
| Java_array | java | ✓ | ✓ |
| ML_matmul / nn_forward / numpy_matmul | py | ✓ | ✓ |
| ML_grad_descent_pure | py | ✗ | ✗ (pre-existing) |
| **total match** | | **7/11** | **7/11** |

### 28.1 Findings (the worry is largely allayed — and there's a transfer WIN)
1. **No broad cross-language forgetting.** Both 7/11; Rust/Java/C/ML all unchanged. Python-only φ-SFT did
   NOT degrade CWM's zero-shot tracing of other languages. The SFT was *not* narrowing.
2. **The oop object-state fix TRANSFERS across languages.** `Cpp_vector_class` — a C++ `class Acc{ int
   total; add(); get(); }`, the *same hidden-object-state pattern* we trained on in Python — went from
   **WRONG (204) → CORRECT (120 = 2²+4²+6²+8²)** under the adapter. The skill "track object member state
   through method calls" learned on Python `Acc` generalized to C++ `Acc`. This is the **direct answer to
   "does the oop stuff expand to multiple languages?" — YES**, at least Python→C++, because CWM's trace
   format and the object-state concept are language-general, not syntactic. (JS_loops also fixed.)
3. **One minor regression:** `JS_map_reduce` over-generates (29→53 frames → wrong final value). A small
   format-drift cost, not a systematic language break.

### 28.2 Why this matters (connects to the project thesis)
This is the **opposite** of the NitroGen "everything is game/engine-specific" wall: a *single* Python SFT
improved C++ object-state tracking for free. CWM's shared trace representation makes the learned
capability **cross-lingual**, so we likely do NOT need per-language SFT for conceptual skills like
object-state observability — one language's supervision transfers. (Caveat: one C++ example; broaden with
more cross-language OOP cases to quantify transfer rate. The lone JS regression says mix in a little
multi-language replay if pushing harder.)

```
run_battery.py --lora <adapter>   # multi-language final-value battery -> results/cwm_battery[_lora].json
```

---

## 29. GAME-TICK OUTCOME — the game-relevant target; clean metric shows tracking is largely SOLVED

Per the user "fix the relevant problems," targeted the game-prediction-relevant failure (§15): within-tick
multi-entity side-effects (player moves [salient] + enemies chase + stomp `+score`/contact `-hp` buried in
entity loops). Built a self-contained Python game-tick generator (`game_tick.py`: player{x,y,hp,score} + K
enemies, deterministic) with an **OUTCOME** metric (final player state), since frame-accuracy masks a
dropped side-effect.

### 29.1 Two metric CONFOUNDS found and removed (the failure was mostly my measurement)
Iterating the metric exposed that the apparent failure was largely an artifact of how I encoded the outcome:

| outcome encoding | base outcome_acc | what it actually measured |
|---|---:|---|
| `hp*1000+score*100+x*10+y` (arithmetic checksum) | **0.00** | the model's **arithmetic** weakness (§27), not tracking |
| `[hp,score,x,y]` (list) | 0.70 | + a **list-ordering** parse confound (`[7,10,5,5]`→`[5,5,7,10]`) |
| `{hp,score,x,y}` (dict, order-independent) | **0.95** | **the actual state tracking** |

With the clean dict metric (n=20, K=3–5, T=2–3): **outcome_acc 0.95, frame_acc 0.98**, per-component
**hp 0.95, score 1.0, x 1.0, y 1.0**. So **CWM tracks the game tick almost perfectly — including the
buried `hp` side-effect** — and the §15 hp-miss is real but **rare** (1/20). The dramatic "0.0" was an
arithmetic-checksum confound in my own metric, not a salience failure. (Same lesson as §16's "no drift"
and §25's oop residual: trust the trace, and make the metric measure the thing you mean.)

### 29.2 This is good news for the games thesis
The game-prediction-relevant skill — track player + many enemies + within-tick stomp/contact side-effects
across ticks — is **largely already solved** by base CWM (0.95), *measured cleanly*. The one soft spot is
`hp` (the contact side-effect), consistent with §15. Whether that degrades with **entity count** (the §15
salience hypothesis) is being stress-tested next (K=10). If it holds, game-state tracking is a CWM
strength, not a gap — and the lever for game/GUI prediction is the **engine-as-oracle / abstraction**
(A5, §10,§15) for engine-interleaved code, not more state-tracking SFT.

```
game_tick.py        # player+K-enemy game-tick generator (+ short variant for SFT)
run_gametick.py     # OUTCOME-level eval (dict state metric) + per-component; --kenemies stress
results/gametick_base.json  # clean: outcome 0.95 / frame 0.98 / hp 0.95
```

### 29.3 Entity-count stress (K=10) — SOLVED once the token cap is removed (the failure was the harness)
Stress-tested the §15 salience hypothesis (does tracking degrade with many within-tick entities?) at
**K=10 enemies**. First run looked like a dramatic failure — outcome 0.0, frame 0.78, `len_ok=0`,
`cwm=None` everywhere — BUT every trace was **truncated** (`n_pred≈213/272`), and frame_acc 0.78 ≈ 213/272:
it was **entirely the 16k-token generation cap**, not the model. K=10 traces need ~22k tokens.

Re-ran with `max_tokens` raised to 30k (and `max_model_len` 32768):

| K (enemies) | outcome_acc | frame_acc | components (hp/score/x/y) |
|---|---:|---:|---|
| 3–5 | 0.95 | 0.98 | 0.95 / 1.0 / 1.0 / 1.0 |
| **10 (proper cap)** | **1.00** | **1.00** | **1.0 / 1.0 / 1.0 / 1.0** |

**At K=10, CWM tracks the game PERFECTLY** (12/12 games, 264–283 frames each, every buried hp/score
side-effect across 10 entities correct). **The §15 within-tick multi-entity salience failure does NOT
reproduce on a clean Python game-tick** — it was Lua-harness-specific, a rare event, or itself a parse
artifact.

### 29.4 The honest conclusion — game-state tracking is a CWM STRENGTH, not the gap
Every "game-tick failure" in this section was **my measurement**, not the model: arithmetic-checksum
encoding (0.0), list-order parse (0.7), token cap (0.0 at K=10). Removed, base CWM is **0.95–1.0** on
multi-entity game-tick state tracking including the buried within-tick side-effects. So:
- **Do NOT SFT to "teach game-state tracking" — it's already solved.** (The game-tick SFT corpus built here
  is unnecessary for this; kept only as replay material.)
- **The real levers for game/GUI prediction are the ones already identified, NOT state-tracking:**
  1. **Context/generation budget** — long game rollouts blow the token/context window (K=10,T=2 ≈ 22k
     tokens). This is *the* structural limit. Lever: **abstraction / step-over (§10)** to compress the
     trace, or periodic **re-grounding**.
  2. **Engine-interleaved code (A5, §15)** — `love.*` / engine calls aren't standalone-traceable. Lever:
     **engine-as-opaque-oracle**, step over engine calls.
- This is a strong, encouraging result for the project thesis: the *learned* part (symbolic state dynamics
  under input) is something CWM **already does well**, even for many-entity game ticks. The remaining work
  is **systems/representation** (trace length, engine boundary), not a model capability gap.

```
results/gametick_base_hard.json   # K=10, proper cap: outcome 1.0 / frame 1.0 (was 0.0 under the 16k cap)
```

---

## 30. TICK-LEVEL ABSTRACTION (step-over) — CWM can't one-shot a game tick; teaching it (SFT) is the lever

The structural limit for game prediction is **trace length** (§29.4): a K=10,T=2 game tick is ~22k tokens
of line-level trace; long games blow the context window. The natural fix is **tick-level abstraction** —
model `state' = step(state, action)` as ONE opaque transition (step-over, §10), compressing ~130
line-frames/tick → a few. This is also the `s_{i+1}|s_i` world-model unit the user asked about. Built
`run_gametick_abstract.py` (step-over rollout) + exact step-over ground truth (`gt_trace.trace_program(
stepover_depth=1)`).

### 30.1 Base CWM CANNOT predict a tick in one shot (verified, not a harness artifact)
| | line-level full trace (§29) | tick-level step-over (this) |
|---|---:|---:|
| state accuracy | **1.00** (K=10) | **per-tick 0.017** |
| frames per game | ~150–280 | ~14–18 (**~10x compression**) |

A frame dump confirms the step-over **mechanism works** (clean CALL→forced-RETURN, full state predicted),
so 0.017 is **real**. What CWM gets wrong in one shot:
```
tick0 truth : player{x4,y3,hp8,score0}  enemies[{6,5},{5,4},{4,3},{2,6},{5,3}]
CWM one-shot: player{x4,y3,hp8,score10} enemies[{7,4},...]   # +10 stomp HALLUCINATED; enemy chase WRONG
```
It nails the **salient** player x/y/hp but **cannot compute the K-enemy chase + stomp side-effect without
tracing the interior**. This is §15's within-tick multi-entity salience failure — and it is **only exposed
by abstraction**: line-by-line CWM computes each enemy update and is perfect (1.0); forced to predict the
whole tick at once, it can't (0.017). So the answer to "can CWM do `s_{i+1}|s_i` for a game tick?" is
**no, untrained** — it is a step-by-step computer, not a one-shot state-transition predictor.

### 30.2 The lever: SFT the abstraction in (the φ-expansion playbook, for ticks)
Like §20-21 (φ-expansion must be a trained-in target, not an inference swap), the **tick transition must
be SFT'd in**. Built a step-over SFT corpus (`build_sft_data.py --stepover 1`: main + each step() as an
opaque CALL→RETURN whose target is the exact post-tick state; 500 examples, ~1.6k tokens each — compact
*because* abstracted) and trained a LoRA. **Decisive question (result next):** does SFT teach CWM the
one-shot tick (per-tick acc 0.017 → high)? If yes → **compressed (~10x) AND accurate game-state
prediction**, which is the scalable game-world-model unit and directly unlocks long-horizon game/GUI
prediction within the context budget. If no → the multi-entity tick genuinely needs the interior, and the
lever is **re-grounding** (trace a few ticks, reset) instead.

```
gt_trace.trace_program(..., stepover_depth=1)   # exact step-over ground truth
run_gametick_abstract.py                        # tick-abstraction eval (per-tick state acc + compression)
build_sft_data.py --stepover 1                  # tick-transition SFT corpus
results/gametick_abstract_base.json             # base: per-tick 0.017 @ ~10x compression
```

### 30.3 RESULT — SFT teaches the one-shot tick (0.017 → 0.69); compressed + accurate game prediction
Trained a LoRA on the step-over corpus (80 steps, 500 examples), evaluated on **held-out** games (seed 999,
disjoint from training):

| metric (held-out) | base | **step-over SFT** |
|---|---:|---:|
| per-tick state acc (full state: player + all K enemies) | 0.017 | **0.692** |
| all-ticks-correct (whole game rollable forward) | 0.00 | **0.55** |
| compression vs line-level | 9.6× | 9.6× (preserved) |

**A 40× jump.** CWM **can** be taught to model a game tick as one opaque transition — it did NOT
fundamentally need the line-level interior, it just had to be **trained on the abstraction** (exactly the
§20-21 φ-expansion lesson: the abstraction is a trained-in target, not an inference-time swap). The loss
converged cleanly (0.044→0.002), confirming the one-shot multi-entity tick is *fittable* (contrast the
drift-arith plateau, §26).

**Failure pattern now** = (a) harder configs (K=5 ∧ T=3) and (b) **tick-level drift** within the
abstracted rollout (`[True,True,False]` — an early tick error compounds). But this is drift over **~3 tick
steps**, not ~150 line-frames — a *vastly* shorter horizon, so re-grounding every few ticks trivially
keeps it on track (the §10.3 triad: trace/abstract/re-ground).

### 30.4 Why this matters — the scalable game-world-model unit, demonstrated
This is the most direct hit on the project's north star ("given an input, see how the state evolves"):
- **Compressed:** ~10× fewer tokens/tick → long-horizon games now fit the context window (the §29.4
  structural limit is lifted for the abstracted representation).
- **Accurate:** 0.69 per-tick / 0.55 whole-game after a *first* 80-step LoRA — clear headroom with more
  data/steps.
- **It's the `s_{i+1}|s_i` transition the user asked about** — and SFT (not even RL) suffices to instill
  it, because the target is well-defined (no exploration needed, unlike §27's confident-error RL wall).
- **Composes with re-grounding:** where the abstracted rollout drifts (a few % per tick), periodically run
  the real `step()` (engine/oracle) to reset — trace-what's-tractable / abstract-what-isn't / re-ground.

Concrete recommendation: **the game-world-model is a step-over-SFT'd CWM that predicts tick transitions in
one shot, re-grounded every k ticks.** Next: scale the step-over corpus (more K/T variety, more steps) to
push 0.69→~0.9, add re-grounding to the rollout eval, and test transfer to the Lua arena.

```
adapters/cwm_gametick_stepover/         # the tick-transition LoRA
results/gametick_abstract_sft.json      # held-out: per-tick 0.692 / all-ticks 0.55 @ 9.6x compression
```

---

## 31. CWM AS AN IDM (inverse dynamics) — derived FREE from the FDM via forward search

User question: can a trained CWM be an **IDM** for a game? (And the apt catch: IDM is harder.) Resolved
both conceptually and empirically.

### 31.1 The framing
- What §30 built is a **Forward Dynamics Model (FDM)**: `(s_i, a_i) → s_{i+1}` — CWM's native mode (a
  forward execution simulator).
- An **IDM** is the inverse `(s_i, s_{i+1}) → a_i`. Harder *for CWM directly* (against the grain;
  inverting execution) and intrinsically ill-posed (aliasing: a wall-blocked move and a no-op give the
  same `s'`).
- **Key insight:** with an accurate FDM + a small discrete action space, you get the IDM for free via
  **forward search** — never train an IDM:
  `IDM(s, s') = argmin_a  distance( FDM(s, a),  s' )`.

### 31.2 Result — forward-search IDM is 100% even though the FDM is 0.69
Ran the step-over-SFT'd CWM (§30) as an IDM over held-out single-tick transitions (n=24, action set
{U,D,L,R}; each transition = 4 forward FDM queries):

| IDM rule | accuracy |
|---|---:|
| exact-match (FDM predicts `s'` perfectly for the true action) | 0.42 |
| **ranking (FDM ranks the true action's prediction CLOSEST to `s'`)** | **1.00** |

The detail is the punchline: even when the FDM's full-state prediction is slightly wrong (off by 1–3
components — a mispredicted enemy), **the true action's prediction is still the closest of the four**, every
time. So:
- **IDM via forward search is far MORE FORGIVING than the FDM.** It needs only the true action's
  next-state to be *nearer* than the 3 wrong actions — not perfect. A 0.69/tick FDM yields a **1.00**
  IDM here, because actions are well-separated in their effects (different player move = different state).
- **No IDM training needed.** The FDM adapter we already have *is* the IDM, queried 4× + argmin.
- Aliasing is handled cleanly: the metric counts any action that *truly* produces `s'` (the `true_set`),
  so genuine ambiguity isn't penalized.

### 31.3 Scope & implication
- This is at the **symbolic-state** layer (state dicts), not pixels. CWM is not a vision model (the
  project's day-1 finding). A real-game IDM from pixels needs a **perception bridge** (frame→state);
  CWM supplies the **dynamics** layer, not perception.
- Within that scope it's a strong result: **one step-over-SFT'd CWM is simultaneously a forward world
  model AND (via forward search) an inverse dynamics / action-labeling model** — the VPT use-case
  (label unlabeled gameplay with actions) at the symbolic level, for free.

```
game_tick.gen_one_tick_src / real_step   # single-tick FDM query + ground-truth transition
run_idm_search.py                        # FDM-as-IDM via forward search
results/idm_search_sft.json              # exact 0.42 / ranking 1.00 (n=24)
```

### 31.4 CORRECTION — IDM recovery is ~0.72 on a representative set, not 1.00 (the n=24 was easy)
The §31.2 "1.00" was an n=24 slice. Re-measured on **490 transitions** (§32 flywheel labeling): aliasing-aware
action recovery = **0.7245**, with a clear event-type structure:

| transition type | IDM recovery | n |
|---|---:|---:|
| move_only | 0.76 | 232 |
| stomp (+score) | 0.76 | 74 |
| death (enemy dies) | 0.76 | 74 |
| **contact (−hp, buried side-effect)** | **0.66** | 216 |

So the forgiving forward-search IDM is **strong but not free of error** — and the error concentrates exactly
on the hard **within-tick hp-contact** side-effect (the §15 salience locus), as the rubber-duck predicted
(the state-distance metric is dominated by player position, so it under-discriminates the buried hp change).
**34% of transitions have zero forward-search margin** (ties — the FDM can't tell the actions apart there).
Net: IDM-via-FDM is a usable but **noisy** labeler (~28% wrong), which is exactly the input quality the
flywheel (§32) must survive.

---

## 32. THE FDM↔IDM FLYWHEEL — self-improvement WITHOUT new action labels (ROUND 1 = WIN)

**The user's target:** close the loop. Use the FDM-as-IDM (§31) to label UNLABELED gameplay trajectories
with actions, fold those self-labeled (state, action→state′) examples back into FDM training, and measure
whether per-tick accuracy climbs **without any new human/oracle labels**. If it does, CWM can bootstrap its
own game-dynamics capability from raw state sequences.

**Rubber-duck (gpt-5.5) verdict — NOT circular:** this is **latent-action semi-supervised learning**. The
training *target* is the **OBSERVED next state** s_{i+1} (real gameplay, always correct); the IDM only infers
the *action label* that conditions the input. So a wrong action label is paired with the **true** outcome —
it never injects a hallucinated dynamics target. The duck mandated: (1) a SINGLE tightly-controlled round
first; (2) a 3-way test with an **oracle control**; (3) **observed-target** serialization (never re-trace);
(4) margin filtering as the guardrail for later rounds; (5) multi-round only after round 1 is proven.

### 32.1 Design (the decisive 3-way)
- **Unlabeled trajectories:** seed 4321, `gen_game_tick` defaults, 490 single-tick transitions. Actions hidden.
- **IDM labeling** (`build_flywheel_data.py`, FAST path = one `gen_full_trace_batch` per candidate, not the
  24-iter `batched_stepover`): forward-search each transition over the action set, pick the action whose
  FDM-predicted next state is closest to the observed next state. Recovery = **0.7245** (§31.4 table).
- **Two training arms, IDENTICAL except the action label:**
  - **FDM_IDM** = 490 self-labeled (IDM action) + 245 multi-tick replay (format anchor).
  - **FDM_oracle** = the SAME 490 trajectories with **true** actions + the same 245 replay.
  Both continue-train from **FDM_0** (= `cwm_gametick_stepover`, the §30 step-over FDM) for 60 steps, lr 1e-4.
- **Held-out eval:** seed **999** (disjoint from train seed 4321), n=40, `run_gametick_abstract.py` per-tick
  state accuracy. ONE adapter per vLLM process (vLLM-0.23 LoRA-switch bug).

### 32.2 RESULT — the self-labeled flywheel matches the oracle and both beat the baseline

| arm | per_tick_state_acc | all_ticks_correct | Δ vs FDM_0 |
|---|---:|---:|---:|
| FDM_0 (baseline, no flywheel) | 0.525 | 0.375 | — |
| **FDM_IDM (self-labeled, 0 new labels)** | **0.683** | **0.525** | **+0.158** |
| FDM_oracle (true actions, control) | 0.679 | 0.525 | +0.154 |

**Decision rule (pre-registered): WIN = FDM_IDM > FDM_0 AND FDM_IDM ≈ FDM_oracle. → SATISFIED.**
- FDM_IDM (0.683) **> FDM_0 (0.525)** by **+0.158** per-tick (all_ticks 0.375→0.525, +15% of programs).
- FDM_IDM (0.683) **≈ FDM_oracle (0.679)**, Δ=+0.004 — statistically identical; IDM even nudges ahead.

**Not a harness artifact:** 0/40 programs hit the frame cap, 0/40 under-predicted ticks, abstract-frame
counts identical across all three arms (only state CONTENT differs). The gain is broadly distributed —
IDM improved **14/40** programs and regressed only **3/40** (oracle improved 13/40) — not an outlier.

### 32.3 WHY 28% IDM label noise does NOT hurt (the crux)
Despite the IDM mislabeling ~28% of actions, FDM_IDM fully matches FDM_oracle. Two reasons, both structural:
1. **The target is the observed true next state**, correct independent of the action label. A wrong label can
   only make the *conditioning* slightly off; it never teaches a false transition outcome.
2. **The errors concentrate where actions are near-equivalent.** 34% of transitions are **zero-margin ties**
   (the FDM cannot separate the actions because they yield ~identical next states), and the worst event type
   is **contact/−hp (0.66)** whose buried side-effect barely moves the player-position-dominated distance
   metric. In exactly those cases the mislabeled (action, observed-state′) pair is **still ~dynamically
   consistent**, so it is benign. The **high-margin, clearly-distinguishable** transitions — the ones that
   actually teach dynamics — are recovered accurately.

⇒ The flywheel is **robust to a noisy IDM** *because* the noise lives on the indistinguishable transitions.
This is the mechanistic reason latent-action self-labeling works here, not just an empirical fluke.

### 32.4 What this establishes
- **CWM can bootstrap its own game-dynamics capability from UNLABELED state sequences** — +0.158 per-tick with
  **zero new action labels**, recovering 100% of the oracle's data-scaling gain. The two training axes
  (more trajectories; better labels) are *both* captured by the self-supervised loop.
- The earlier worry (§31.4: "IDM is only 0.72, the 28% noise may sink the flywheel") is **resolved** — the
  noise is concentrated on benign ties, so round 1 survives it without margin filtering.
- This is the first **self-improvement** result of the study (prior gains all needed gold/oracle SFT targets).

### 32.5 Caveats & what's NOT yet shown
- **Round 1 only.** Multi-round (relabel with FDM_IDM, retrain) risks **compounding collapse** as pseudo-label
  errors feed back. Round 2 must add **margin filtering** (drop the 34% zero-margin ties: keeps ~66% at higher
  recovery) and **stop-if-IDM-acc-drops** auditing.
- n=40, single seed; the +0.158 is ~150 tick-decisions and 21 vs 15 fully-correct programs. Real but modest n.
- FDM_0's absolute number fell 0.69(n=20)→0.525(n=40): seed-999's extra 20 programs are harder; the flywheel
  arms recover to ~0.68 = it mainly **buys robustness/coverage** the 500-example §30 SFT lacked.

```
build_flywheel_data.py                          # IDM-label unlabeled traj + oracle control + event/margin stats
data/flywheel_{idm,oracle}_r1_mixed.jsonl       # 735 each (490 labeled + 245 replay), differ ONLY in action label
adapters/cwm_fdm_idm_r1 / cwm_fdm_oracle_r1      # the two continue-trained arms (from cwm_gametick_stepover)
run_flywheel_eval.sh                             # 3-way eval driver (one vLLM process per adapter)
results/flywheel_eval_{fdm0,fdm_idm,fdm_oracle}_n40_s999.json
```

### 32.6 CONTROLS — the win is real AND it is genuinely action-conditioned (not "action ignored")
A rubber-duck (gpt-5.5) flagged the decisive alternative explanation: IDM≈oracle and "28% noise harmless"
would ALSO be explained if the FDM mostly predicts next-state from code+current-state and **near-ignores the
action**. Two cheap controls (no new training) settle it.

**(A) Program-level paired bootstrap / sign test** on the existing n=40 multi-tick eval (the honest unit is
the program, not the tick — ticks within a program are correlated):

| contrast | mean Δ per-tick | 95% CI (10k program-bootstrap) | sign test |
|---|---:|---|---:|
| **IDM − FDM_0** | **+0.158** | **[+0.029, +0.292] — excludes 0** | 14W/3L/23T, p=0.013 |
| ORACLE − FDM_0 | +0.154 | [+0.025, +0.283] — excludes 0 | 13W/3L/24T, p=0.021 |
| IDM − ORACLE | +0.004 | [−0.158, +0.167] — **includes 0** | 10W/9L/21T, p=1.0 |

⇒ the flywheel's improvement over baseline is **statistically credible**; IDM-vs-oracle is genuinely
indistinguishable (the eval is not powered to separate them — exactly as expected if labels are good).

**(B) Action-intervention sensitivity** (`run_action_sensitivity.py`, n=120 held-out single-tick
transitions, seed 999). For each transition we run the trained FDM under **all four actions** and compare to
ground truth. `swap_acc` = how often a WRONG action's predicted state matches the observed (true-action)
outcome; `pred_div` = # distinct predictions across the 4 actions (1=ignores action, 4=fully sensitive):

| arm | true_acc | **swap_acc** | true−swap | **pred_div** | contact(−hp) | stomp | death | move_only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FDM_0    | 0.483 | **0.000** | 0.483 | **4.0/4** | 0.279 | 0.312 | 0.312 | 0.776 |
| **FDM_IDM**   | **0.808** | **0.000** | 0.808 | **4.0/4** | **0.754** | **0.812** | **0.812** | 0.837 |
| FDM_oracle | 0.783 | 0.000 | 0.783 | 4.0/4 | 0.656 | 0.875 | 0.875 | 0.918 |

Three things are now nailed down:
1. **The FDM is genuinely action-conditioned, not action-ignoring.** `swap_acc = 0.000` everywhere (a wrong
   action NEVER reproduces the observed outcome) and `pred_div = 4.0/4` (a *distinct* prediction per action).
   The "action barely matters" alternative is **refuted** — every state is action-separable (gt_div 4/4) and
   the model tracks that separation.
2. **The flywheel improves action-conditioned dynamics to ~oracle level:** single-step true_acc
   **0.483 → 0.808 (IDM) ≈ 0.783 (oracle)**, a +0.32 jump on a more-powered n=120 metric than the 40-program
   eval. IDM ≈ oracle holds here too.
3. **The gain is concentrated on exactly the hard buried side-effects** FDM_0 was missing — contact/−hp
   0.279→0.75, stomp 0.312→0.81, death 0.312→0.81 — i.e., the §15 salience locus. The flywheel did not just
   add generic data; it taught the within-tick hp/score side-effect dynamics, action-conditioned.

**Verdict (round 1, fully controlled): the FDM↔IDM flywheel is a real, statistically-credible,
action-conditioned self-improvement** — CWM bootstraps game-dynamics capability from UNLABELED state
sequences, recovering the oracle's gain despite a 0.72 IDM labeler, with the benefit landing on the hardest
transitions. Round 2 (margin-filtered, fresh trajectories) is now justified.

```
run_action_sensitivity.py                        # action-intervention control (true vs swapped action)
results/action_sens_{fdm0,fdm_idm,fdm_oracle}_n120_s999.json
results/flywheel_eval_*  + program bootstrap     # §32.6(A) statistical bound
```

### 32.7 ROUND 2 — does it compound, collapse, or plateau? → STABLE PLATEAU (no collapse)
Round 1 won. The duck's mandated next test: iterate one more round with the **collapse guardrails** and see if
self-labeling compounds (climbs), collapses (pseudo-label feedback rot), or plateaus (data/budget ceiling).
- **Labeler = FDM_IDM** (the round-1 model, not FDM_0) → tests self-reinforcement.
- **Margin filtering ON** (`margin_min=1`, drop the 38% zero-margin ties): kept **495/795** labels at
  **0.992 recovery** (vs 0.717 unfiltered) — filtering purifies the labeler from 72% → 99% as predicted.
- Both r2 arms continue-train from FDM_IDM; fresh seed 7777; size-matched (495 + 245 replay), eval seed 999.

| arm | multitick per_tick | all_ticks | single-step true_acc | swap | pred_div |
|---|---:|---:|---:|---:|---:|
| FDM_IDM (round 1) | 0.683 | 0.525 | 0.808 | 0.0 | 4/4 |
| **FDM_IDM_r2** | **0.696** | **0.625** | 0.775 | 0.0 | 4/4 |
| FDM_oracle_r2 | 0.692 | 0.575 | 0.750 | 0.0 | 4/4 |

- **No collapse.** Round 2 ≈ round 1 (0.696 vs 0.683), all-ticks even up (0.525→0.625, 21→25/40 programs).
  Two stacked self-labeled rounds are **stable** — the compounding-collapse risk did not materialize, which is
  exactly the guardrail's job (99%-clean filtered labels can't rot the model).
- **IDM ≈ oracle again** (0.696 vs 0.692; 0.775 vs 0.750) — IDM marginally ahead, confirming label quality.
- **Plateau, not climb.** Round 1 already hit the oracle ceiling, so round 2 saturates there. To climb further
  the lever is harder/larger trajectories (more T, more K, novel rules), not more rounds at this difficulty.
- Action-conditioning fully preserved across both rounds (swap=0, div=4/4). Margin filtering is the right
  multi-round hygiene: 99% recovery vs 72% means the loop can be run repeatedly without degradation.

**FLYWHEEL ARC COMPLETE:** FDM₀ 0.483 → flywheel 0.808 single-step (≈oracle), stable over 2 rounds, genuinely
action-conditioned, statistically credible — CWM bootstraps action-conditioned game dynamics from unlabeled
state sequences. The remaining axis is difficulty/scale, not the loop itself.

### 32.8 DIFFICULTY CEILING — the flywheel gain holds/amplifies, but big arenas are the real wall
The only open lever (§32.7) was difficulty. Eval FDM_0 vs the best flywheel model (FDM_IDM_r2) on a HARD band
(K=6–8 enemies, T=4–6 ticks; full traces ~465 frames, 17.8× step-over compression), seed 999, n=30:

| arm | per_tick (easy K3-5/T2-3) | per_tick (HARD K6-8/T4-6) | all_ticks (hard) |
|---|---:|---:|---:|
| FDM_0 | 0.525 | 0.117 | 0.00 |
| FDM_IDM_r2 | 0.696 | **0.284** | 0.10 |

- The flywheel improvement **persists and slightly amplifies** with difficulty (+0.16 easy → +0.17 hard, 2.4×
  relative), so the self-labeled capability is not over-fit to easy configs.
- But absolute hard-band accuracy (0.28) is low: 8-enemy/6-tick ticks blow past the per-tick salience+horizon
  budget. This **confirms difficulty/scale is the genuine ceiling**, not the loop. The path is the §10.3 triad
  (abstract + SFT-the-flywheel + periodic re-grounding) plus harder/larger flywheel trajectories — future work.

**FULL FLYWHEEL STUDY DONE:** loop proven (r1 win), controlled (action-conditioned, statistically credible),
stable (r2 no collapse), and characterized (holds with difficulty; arena size is the wall). adapters: cwm_fdm_idm_r1/r2, cwm_fdm_oracle_r1/r2.

**Ceiling diagnosis (FDM_IDM_r2, hard K6-8 single-step):** true_acc 0.49 (vs 0.78 easy), swap=0/div=4/4 still.
So the hard wall is BOTH within-tick capability (0.49 from true prior, stomp/death 0.25 at 8 enemies) AND drift
(0.49→0.28 free over 4-6 ticks). Within-tick salience dominates → re-grounding alone caps at ~0.49; the lever
is harder flywheel training data (more K/T), confirming §32.7's "scale difficulty, not rounds." Action-
conditioning never breaks (swap=0, div=4/4) even hard. results/action_sens_idmr2_hardK.json.

### 32.9 FLYWHEEL OPERATING ENVELOPE — self-labeling COLLAPSES at hard difficulty (boundary found)
Tried to break the K8 ceiling by self-labeling HARD trajectories (K6-8, T4-6, labeler=FDM_IDM_r2, margin≥1):
**idm recovery 0.258 (≈ chance 0.25), frac_zero_margin = 1.0, n_kept = 0.** The forward-search IDM cannot
discriminate actions at hard difficulty — the FDM's hard-tick predictions are wrong/clustered, so all 4
actions tie. Margin filtering correctly keeps NOTHING. ⇒ **the flywheel only works where the FDM is already
good enough to forward-discriminate actions (easy-mid); it cannot bootstrap a regime it can't yet model.**
This is the precise ABANDON boundary the duck warned of, now mapped to difficulty. Breaking the K8 ceiling
needs oracle/engine labels (§A5) or a curriculum (grow K), not self-labeling.

### 32.10 BREAKING THE CEILING needs ORACLE labels (not self-labeling) — path confirmed
Hard-band (K6-8/T4-6) per-tick: FDM_0 0.117 → flywheel FDM_IDM_r2 0.284 → **hard-ORACLE SFT 0.369**
(all_ticks 0.1→0.2). Training on hard transitions WITH true labels climbs the K8 ceiling (+0.085 over the
easy-trained flywheel), while self-labeling there is impossible (§32.9, IDM=chance). ⇒ the flywheel's reach
ends where action-discrimination dies; beyond it the lever is engine-as-oracle labels (§A5) + a K-curriculum.
Self-labeling for easy-mid + oracle/engine for hard = the full scaling recipe. cwm_fdm_hardoracle.

**STUDY CLOSED:** flywheel works, is controlled, stable, holds with difficulty, has a mapped collapse boundary,
and the boundary is breakable with oracle labels — a complete, honest account of CWM as a self-improving,
action-conditioned game world model.

---

## 33. RENDERER/GUI AXIS — base CWM is ALREADY a strong UI-app world model (premise inverted)

The renderer-axis MVE asked: does base CWM crumble on one-shot UI-app state transitions the way it did on
game ticks (0.017 → SFT 0.69)? Built `ui_tick.py` (the GUI analog of `game_tick.py`): UI apps as deterministic
Python `dispatch(state, event) → state'` over a model-state dict (counter/todo/form/cart, a difficulty
gradient), oracle = exact Python exec, render(state)→DOM-JSON for the later pixel path. Probe
`run_uitick_probe.py` forces step-over of `dispatch` and scores exact-match + graded field-F1.

### 33.1 RESULT — base does NOT crumble on Python-native UI (it's CWM's home turf)
n=80, seed 999, scale 1:

| app | exact | field_f1 |
|---|---:|---:|
| counter | 1.00 | 1.00 |
| todo | 1.00 | 1.00 |
| form (cascading validation) | 0.70 | 0.975 |
| cart | 1.00 | 1.00 |
| **overall** | **0.925** | **0.994** |

**The premise inverts:** unlike the game (0.017), base CWM **already predicts one-shot UI-app transitions
near-perfectly**. Reason: these are simple **Python** dict mutations — exactly CWM's strength (§29 game-state
tracking was also 1.0; the game only crumbled because of K-entity within-tick salience, not because "ticks"
are hard). So "predict app behaviour from code" is **already YES** for Python-expressible apps. The only
scale-1 crack is `form` (0.70) — cascading validation flags.

### 33.2 SCALE is the lever (the UI analog of K enemies)
Re-probe at scale 4 (bigger apps: todo/cart lists grown ×~4), n=80, seed 999:

| app | scale 1 | scale 4 | what scales |
|---|---:|---:|---|
| counter | 1.00 | 0.95 | (nothing) |
| todo | 1.00 | 0.90 | list length (salience) |
| form | 0.70 | 0.60 | cascade depth |
| **cart** | 1.00 | **0.45** | **multi-item line+subtotal recompute** |
| **overall** | 0.925 | **0.725** | |

`cart` collapses to 0.45 — change one qty, must recompute ALL line totals + subtotal = the **multi-item
salience** failure, the direct UI analog of the game's buried stomp/contact side-effects. **Confound flagged:**
cart uses `qty*price` (CWM's separate arithmetic weakness, §27), so cart conflates salience with arithmetic;
`todo` at higher scale (toggle in a long list, NO arithmetic) is the clean salience probe (scale-8 running).

### 33.3 Implication for the project (honest)
- **The frontier is NOT "can CWM predict UI behaviour from code" — it already can for small Python apps.** The
  frontier is (a) **modality/distribution** (real JS/DOM, not Python — to be tested via jsdom on MiniWoB++/
  TodoMVC) and (b) **scale** (big apps/DOMs, where salience+budget bite, shown here). The value-add lives in
  generalising the *dynamics* skill to **unrun/novel code** + the **pixel boundary**, not in the symbolic core.
- We now HAVE a clean base-crumbles regime (scaled cart/form, overall 0.725) for the eval→train loop, plus a
  no-arithmetic version (scaled todo).

### 33.4 Data sources gathered (subagents) — see brainstorm/data/
- **MiniWoB++ (Farama, MIT)** — 100+ tiny self-contained HTML/JS tasks, seeded-deterministic, Gymnasium
  step-loop yields (DOM_before, action, DOM_after); the **real-JS/DOM** distribution-shift testbed. Needs
  Selenium+Chromium (or jsdom for DOM-only). [brainstorm/data/uibench_raw.md]
- **TodoMVC `javascript-es5/dist` (MIT)** — real vanilla-JS, no build, in-memory state, Playwright-drivable;
  first real-app target. Plus tiny apps (bradtraversy form-validator 5KB, fully deterministic).
- **Streamlit (`streamlit/demo-todo`)** — Python-native real apps: next render = pure fn of
  (source, session_state, action) → predict session_state transitions; best CWM fit but UI-capture is brittle.
  [brainstorm/data/realapps_raw.md]

```
ui_tick.py / run_uitick_probe.py   # GUI analog of game_tick + base one-shot transition probe (scale knob)
results/uitick_probe_base.json (0.925)  uitick_probe_base_scale4.json (0.725)  [scale8 running]
brainstorm/data/{uibench_raw,realapps_raw}.md  # MiniWoB++/TodoMVC/Streamlit recipes
```

### 33.5 REFINEMENT (scale-8) — the "scale crumble" is arithmetic/cascade, NOT salience
Re-probed at scale 8 (todo/cart lists ×~8 → 12–24 items), n=80, seed 777:

| app | scale 8 | diagnosis |
|---|---:|---|
| **todo (12–24 items)** | **0.95** | pure multi-item salience, NO arithmetic — **does NOT crumble** |
| counter | 0.90 | trivial |
| form | 0.65 | cascading boolean validation (deep conditional logic) |
| cart | 0.65 | `qty*price` line+subtotal = the §27 ARITHMETIC weakness (confound) |

**Key correction to §33.2:** the scale-4 cart collapse (0.45) was NOT multi-item *salience* — `todo` at 20+
items stays **0.95**, so CWM tracks long lists fine. The crumbles reduce to CWM's **pre-existing, known**
weaknesses: **arithmetic** (cart `qty*price`, §27 — SFT doesn't fix this per §26/§27) and **deep cascading
conditionals** (form). **Neither is UI-specific.** ⇒ Base CWM's *symbolic UI-dynamics core is essentially
solved*; there is no genuine UI-salience gap to SFT away.

### 33.6 DECISIVE IMPLICATION — the renderer-axis value-add is modality + pixels, not symbolic dynamics
Training on the scaled Python apps would mostly re-teach **arithmetic** (a known SFT-resistant hole) or form
cascades — not a clean win, and not the project's point. The ONE untested frontier that can genuinely crumble
base CWM and where SFT could genuinely win is **real JavaScript / DOM** (CWM is Python-trained; real JS + the
`document` API is the distribution-shift regime, cf. the §run_lua_dom native-call wall). **Next decisive eval:
the real-JS/DOM base probe** (jsdom on a real vanilla-JS app — no Selenium needed). Two outcomes, both
clarifying:
  - base crumbles on real JS/DOM → SFT to teach JS/DOM dynamics = the genuine, non-confounded renderer-axis win.
  - base holds on real JS/DOM too → CWM is already a *general* code-execution world model; the entire remaining
    contribution is the **pixel boundary** (render/perception) + **generalization to unrun code**, not dynamics.

---

## 34. REAL-APP UI transitions (4 subagent-harvested sources) — the genuine gap is CASCADING LOGIC, not JS

Fanned out 4 background subagents (no GPU) to harvest REAL app transitions to one contract
(`uidata/CONTRACT.md`); built a unified probe `run_uitrans_probe.py` (one CWM load, GPU-serialized).

### 34.1 Data harvested (303 verified real transitions)
| target | n | lang | how truth obtained | prompt_src |
|---|--:|---|---|---|
| todomvc | 83 | JS | real TodoMVC es5 model run in jsdom; extracted reducer proven == jsdom | yes |
| vanilla | 80 | JS | bradtraversy form-validator + movie-seat-booking, real handlers via jsdom | yes |
| streamlit | 56 | Python | real Streamlit rerun via `streamlit.testing.v1.AppTest`; session_state | yes |
| miniwob | 84 | — | MiniWoB++ tasks in jsdom: real (DOM_before, action, DOM_after) | raw (fallback) |
All self-consistent (every `prompt_src` re-executes to `truth_state`; independently re-verified).

### 34.2 HARNESS SAGA = the §29.1 lesson, again (extraction mode is a metric artifact)
Getting a trustworthy number took 4 harness modes — a cautionary tale:
- **full-trace @1.5k tok** → truncation → 139/219 unparsed (garbage).
- **full-trace @4k tok** → streamlit `copy.deepcopy` in the harvested code makes CWM trace the ENTIRE deepcopy
  recursion → ~3000 tok/trace → intractably slow + one prompt overflowed 8192 ctx and crashed the run.
- **depth-aware step-over** (trace main+dispatch, step over helpers) → correct idea but frame-by-frame is slow
  and a long accumulated trace overflowed context → crash.
- **one-shot step-over of dispatch, capped frames, robust py/JS-literal parse, de-bloated data** → WORKS.
Fix that unlocked it: strip the unnecessary `copy.deepcopy` from the streamlit prompt_src (re-verified: 56/56
still == truth) so CWM stops tracing stdlib. **Lesson re-confirmed: size the budget to the trace, abstract
library calls, and use an order-independent structural parser — never trust the first extraction.**

### 34.3 RESULT (one-shot step-over, n=219, seed as-harvested)
| target | exact | field_f1 | reading |
|---|---:|---:|---|
| **streamlit** (Python) | **0.893** | 0.986 | base aces real Python app transitions (= §33 home-turf) |
| **todomvc** (JS) | 0.771 | 0.771 | **all 64 parseable correct**; the 19 misses are HARNESS (one-shot can't capture a dispatch that delegates to `cloneTodos`/`nextId`), not capability |
| **vanilla** (JS form-validator) | **0.525** | 0.858 | **GENUINE** errors: CWM emits the DEFAULT control state instead of running the validation cascade |

### 34.4 Finding — the gap is CASCADING VALIDATION/CONDITIONAL logic, language-independent
- Base CWM **already handles** real-app state transitions where the logic is straightforward — Python
  (streamlit 0.89), simple JS reducers (todomvc parseable = 100%), Python-native (ui_tick 0.92, §33).
- The **one genuine crumble** is the **form-validator's cascading validation** (vanilla JS 0.53) — and it
  **matches the Python `form` weakness** (0.65-0.70, §33.5). So the real frontier is **deep conditional /
  validation cascades**, NOT "JS vs Python" distribution shift (todomvc JS is fine). Real JS is only
  marginally harder; the dominant factor is logic depth, consistent across languages.
- The **genuine SFT target** is therefore cascading-validation transitions (form-like) — base mispredicts by
  defaulting, a real, non-arithmetic, trainable gap (unlike the §33 arithmetic confound).
- `miniwob` 84 raw (DOM_before, action, DOM_after) triples are reserved for a future **DOM-prediction** probe
  (predict next DOM from JS source + current DOM — the pixel-boundary-adjacent task), which the one-shot
  reducer harness doesn't cover.

```
uidata/CONTRACT.md  uidata/{todomvc,vanilla,miniwob,streamlit}/   data/uitrans_*.jsonl
run_uitrans_probe.py (one-shot step-over + robust parse)   results/uitrans_probe_base_final.json
brainstorm/data/{uibench_raw,realapps_raw}.md (source recipes)
```

---

## 35. DOM-STATE render-FDM (the pixel-boundary-adjacent unit) + cascade SFT corpus

Two parallel threads: (A) build the render-state FDM probe (toward the pixel north-star); (B) subagents harvest
more cascading-validation data (the §34 genuine gap) for SFT.

### 35.1 DOM-state render-FDM — base CWM is a decent render-state predictor out of the box
`ui_dom.py`: unlike ui_tick (predict app MODEL-state), here the STATE **is** the canonical DOM tree and the
handler mutates the DOM directly (like real vanilla-DOM apps) — the render-state a browser then rasterizes to
pixels. Kept one-shot-able (flat `children`, single-loop dispatch, no recursion). Apps: tabs / accordion
(single-open) / togglelist+count / counter, n=3-8, with MULTI-ELEMENT DOM cascades (select a tab -> flip
aria-selected on ALL tabs + hidden on ALL panels = the DOM-space analog of game multi-entity salience).
`build_uidom_data.py` -> 100 verified transitions; probed with the unified one-shot step-over harness.

**Base CWM: exact 0.75, field_f1 0.986, 0 unparsed (n=100).** So CWM predicts DOM-state transitions well at the
field level (98.6%); the 25% non-exact are **multi-element cascade near-misses** (all shown fails are wide
`tabs` selects, f1 0.90-0.97 = one aria-selected/hidden element wrong). ⇒ the render-state FDM **works out of
the box**; the residual is the familiar **salience-at-scale** gap (wide cascades), a clean SFT/scale target —
and it is NOT arithmetic. This is the symbolic half of the pixel pipeline: DOM-JSON -> (browser) -> pixels.

### 35.2 Cascade SFT corpus harvested (the genuine §34 gap), 232 verified transitions
Two subagents harvested DEEP cascading-validation transitions (the language-independent gap from §34):
- `data/uitrans_cascade_js.jsonl` — 120 verified (node-exec) real JS multi-field/interdependent validators
  (80 input / 28 select / 12 toggle); cascade depth = many interdependent error/enable recomputations per action.
- `data/uitrans_cascade_py.jsonl` — 112 verified (python-exec, AppTest-equivalent) deep form/wizard validation,
  dependent departments, pricing/budget/coupon gates, submit recomputation.
Both self-consistent (every prompt_src re-executes to truth_state; independently re-verified 8/8 each).

### 35.3 Status — ready for the SFT train-and-win
We now have BOTH a clean base-crumbles regime AND the data to fix it:
- genuine gap: cascading validation (real JS vanilla 0.53 §34; deep cascades harvested here) + DOM multi-element
  cascade (uidom exact 0.75 §35.1) — both real, non-arithmetic, trainable.
- corpus: 232 cascade transitions (+ generable uidom/ui_tick) for a train/held-out split.
Next: build step-over SFT data from these, LoRA-train, eval base-vs-SFT on held-out cascade + the real vanilla
form-validator -> the clean renderer-axis train-and-win.

```
ui_dom.py  build_uidom_data.py  data/uitrans_uidom.jsonl  results/uidom_probe_base.json (0.75/0.986)
data/uitrans_cascade_{js,py}.jsonl (232)  uidata/cascade_{js,py}/
```

---

## 36. CASCADE SFT — the gap IS trainable in-dist, but narrow SFT NEGATIVELY transfers (the §22 redux)

Built step-over SFT for the cascading-validation gap (§34/§35). JS can't be traced by gt_trace (Python-only),
so trained on PYTHON cascades (ui_tick form/cart + ui_dom) and kept JS (vanilla form-validator, cascade_js)
ENTIRELY held out as a cross-language transfer test (the §28 design). 192 train ex, 48-step LoRA
(loss 0.056 -> 0.0003), `adapters/cwm_cascade`. Eval base vs SFT (one-shot step-over, n=248, 0 unparsed):

| target | base exact | SFT exact | Δ | reading |
|---|---:|---:|---:|---|
| **uitick** (held-out, Py form/cart) | 0.679 | **0.964** | **+0.285** | in-dist: gap trainable |
| **uidom** (held-out, Py DOM cascade) | 0.800 | **1.000** | +0.200 | in-dist: solved |
| cascade_js (JS deep cascade) | 0.783 | 0.750 | -0.033 | cross-lang: ~flat |
| **vanilla** (REAL JS form-validator) | 0.525 | **0.350** | **-0.175** | cross-lang: REGRESSED |

### 36.1 Reading — two findings, one positive one cautionary
1. **The cascade gap IS trainable.** In-distribution, SFT lifts the exact-match dramatically (uitick
   0.68->0.96, uidom 0.80->1.0). So base CWM's cascading-validation weakness (§34) is a real, fixable
   capability hole, not a ceiling. The clean "base-crumbles -> SFT-wins" demo holds **in-distribution**.
2. **Narrow SFT NEGATIVELY transfers to real JS.** The SAME task in JS (vanilla form-validator) got WORSE
   (0.53->0.35, field_f1 0.86->0.77). Training on the Python ui_tick form's specific state schema
   (fields/errors/can_submit) made CWM mis-apply it to vanilla's different schema (controls/status/message).
   This is **schema-overfitting / catastrophic forgetting** — the exact §22 failure (narrow oop SFT forgot
   multientity), now in the UI domain. (cascade_js barely moved: its schema differs more, less interference.)

### 36.2 The fix is known: mixed-corpus replay (§24)
§24 eliminated §22's forgetting by adding anti-forgetting REPLAY to the SFT. Apply the same here: mix the
cascade SFT with replay of diverse modes (easy/oop/multientity + a JS-flavored anchor) so the model learns the
cascade abstraction WITHOUT narrowing its schema/language distribution. Prediction: in-dist gains preserved,
vanilla regression removed. (Trying next: `adapters/cwm_cascade_mixed`.)

```
build_cascade_sft.py  build_uitick_data.py  data/sft_cascade_train.jsonl  adapters/cwm_cascade
results/uitrans_eval_{base,sft}.json
```

### 36.3 PRECISE DIAGNOSIS (error audit, gpt-duck #1) — NOT schema intrusion; cascade-abstraction non-transfer
Audited the SFT model's vanilla (real JS) failures. The duck's "schema-overfit/intrusion" hypothesis is
**refuted**: the SFT model emits vanilla's CORRECT schema (`controls/{status,message}`) with **zero** ui_tick
schema leakage. The actual error: it validates only the **touched** field and leaves the others at their
DEFAULT (`'Error message'/''`) instead of re-validating ALL fields — i.e. it does NOT run the full validation
cascade on JS. Base fails the same way (0.53); SFT defaults slightly MORE (field_f1 0.86->0.77). So the real
story is **language/protocol narrowing**: Python-cascade SFT improved Python execution but mildly degraded JS
execution fidelity (more defaulting), WITHOUT corrupting the schema. The cascade abstraction did not transfer
to JS. ⇒ fix is a JS-flavored anchor / schema+language-diverse replay (duck #2/#3), not just schema diversity.

### 36.4 ABSTRACTION evidence + the PIXEL render capstone (frame-as-generation works)
- **Not pure memorization (in-dist):** on FRESH unseen ui_dom instances (seed 555, different n, never trained),
  base 6/8 -> **SFT 8/8** exact — the SFT generalizes to fresh instances of the trained app families (learned
  the multi-element DOM cascade, not just memorized seeds). [data/uidom_demo.jsonl]
- **The pixel pipeline (dom_render.py): DOM-JSON -> HTML -> headless Chromium -> PNG.** CWM predicts the next
  render-state; a REAL browser rasterizes it. On a fresh tabs transition (select tab-4):
  - TRUTH render: tab T4 selected, "content 4".
  - **base** predicted render: T1 / "content 1" — WRONG (mispredicted the tab cascade).
  - **SFT** predicted render: T4 / "content 4" — CORRECT (pixel-identical to truth).
  This is the user's "frame as generation": the model's predicted next frame, rasterized, matches reality —
  and the SFT improvement is **visible in pixels**. results/render_predict/*.png (before/TRUE/PRED_{base,sft}).

```
dom_render.py (DOM-JSON->Chromium PNG)  render_predict_demo.py (CWM-predicted DOM -> pixels)
data/uidom_demo.jsonl  results/render_predict/  results/render_demo/   (playwright+chromium in .venv_vllm via uv)
```

---

## 37. RENDERED VIDEO — CWM free-rolls the render-state, the browser plays it back (the pixel north-star)

`render_rollout.py`: CWM free-rolls the DOM render-state — predict DOM_{i+1} from its OWN predicted DOM_i +
event_i (one-shot step-over), for a sequence of UI events — and a real headless browser rasterizes each
predicted DOM into a frame; the frames assemble into an animated GIF. This is the user's literal goal: *an
image/VIDEO of the app responding to input*, generated by the model. It also IS the DRIFT axis in render space.

### 37.1 Result — SFT turns a drifting rollout into a faithful video
Same app (tabs, 6 enemies... er, 6 tabs), same seed, same 6-click sequence, free-rolled:

| model | per-step exact (6 clicks) | rendered video |
|---|---|---|
| **base CWM** | `[T,T,T,F,F,F]` = **3/6** | drifts after 3 steps -> wrong tab/panel |
| **cwm_cascade SFT** | `[T,T,T,T,T,T]` = **6/6** | faithful 6-step UI video, no drift |

Final-frame pixels (after the 6th click; truth = tab T1 / "content 1"):
- **base** rendered T2 / "content 2" — WRONG (accumulated rollout drift).
- **SFT** rendered T1 / "content 1" — pixel-correct.

So: (1) **the render-state world model produces an actual rendered video** of a UI responding to a click
sequence (results/render_rollout/rollout_{base,sft}.gif); (2) **free-roll DRIFT is real in render space** (base
3/6, correct early then diverges — the §32.8/drift-axis pattern, now visible as a desyncing video); (3) **the
DOM-cascade SFT removes the drift here** (6/6), and the improvement is literally watchable. Re-grounding
(`--reground_k`) is wired for the cases SFT alone doesn't fix.

### 37.2 What this composes (the renderer-first recipe, demonstrated)
- RENDERER: DOM-JSON render-state -> browser pixels (dom_render.py), CWM never emits pixels. ✓
- The render-state FDM (§35) free-rolls into a VIDEO; SFT lifts per-step fidelity (§36) -> longer faithful
  rollouts; DRIFT re-grounding is the lever for the rest. All three axes meet in one rendered rollout.
- Honest scope: tabs is a clean single-select cascade; wider/multi-app rollouts + real-app (TodoMVC) videos +
  re-grounding curves are the next steps. But the end-to-end pixel pipeline the user asked for is WORKING.

```
render_rollout.py (free-roll DOM -> frames + GIF, --reground_k for drift-control)
results/render_rollout/rollout_{base,sft}.gif  + per-step frames + rollout_{base,sft}.json
```

### 37.3 DRIFT in render space — memoryless self-heals; accumulating needs re-grounding (capped by capability)
Two app types, free-rolled, expose the drift structure:
- **tabs (memoryless):** base 12-step free-roll = `[T,T,T,F,F,F,F,T,T,T,T,T]` 8/12 — drifts mid-roll then
  SELF-HEALS, because each `select` fully overwrites the state (no error accumulation). Memoryless render-state
  apps are drift-robust; re-grounding barely needed.
- **togglelist (accumulating):** base free-roll = `[F,T,F,F,F,T,T,F]` 3/8; re-grounding every k=3 ->
  `[F,T,F,T,T,T,T,F]` 5/8. Re-grounding recovers the drift component, but is CAPPED by base's single-step
  capability gap (togglelist's multi-element count-recompute is the §35 salience gap — base mispredicts even
  step-1). This is the §32.8 **capability-vs-drift decomposition, now in render space**: re-grounding (DRIFT
  lever) lifts 3/8->5/8; closing the rest needs the SFT (CAPABILITY lever, which made ui_dom 1.0). So a faithful
  long UI video = SFT (raise per-step ceiling) + re-grounding (kill residual drift) — both, as the recipe says.
results/drift2/rollout_base_tl_{k0,k3}.gif, results/drift_curve/rollout_base_k0.gif.

### 35.4 METRIC CORRECTION (gpt-duck) — field-F1 is COPY-inflated; use exact + changed-field
A copy baseline (predict next-state = current-state) exposes that **field-F1 is inflated by static DOM mass**:
- ui_dom: COPY field_f1 = **0.895** (exact 0.05), because only **8% of fields change** per transition
  (avg 2.6 of 32.9). So base CWM's 0.986 field-F1 is only +0.09 over copy — NOT the headline.
- ui_tick: COPY field_f1 0.669 (34% of fields change) — less inflated.
⇒ **The honest headline metrics are EXACT-match (ui_dom base 0.75 vs copy 0.05; SFT 1.0) and CHANGED-FIELD
accuracy** (did the model get the fields that actually change right?), NOT field-F1. The exact-match results in
§35-37 stand (copy=0.05, so exact is a real signal); field-F1 is demoted to a structural-sanity secondary.
Adding changed-field accuracy to run_uitrans_probe and re-scoring. (Also: the pixel demo is an end-to-end
human-inspectable artifact + a viable LLM-DOM/browser-raster DECOMPOSITION, NOT independent pixel-prediction
evidence — pred==truth => identical pixels by construction. Framed honestly going forward.)

### 35.5 AUDITED base-vs-SFT (honest metrics, fresh seed-222 data) — the win survives the metric audit
Re-ran with changed-field accuracy (only fields that change) + delta-exact (predicted change-SET == true
change-SET) + copy-baseline, on FRESH held-out instances (seed 222, n=96):

| metric | base uidom | SFT uidom | base uitick | SFT uitick |
|---|---:|---:|---:|---:|
| exact | 0.812 | **0.938** | 0.833 | **0.979** |
| changed-field acc | 0.875 | **0.969** | 1.000 | 0.990 |
| delta-exact | 0.812 | **0.938** | 0.875 | **0.979** |
| field_f1 (copy_f1) | 0.991 (0.907) | 0.998 (0.907) | 0.984 (0.70) | 0.998 (0.70) |

- **field_f1 is inflated** (uidom copy_f1=0.907) — demoted. But on the HONEST metrics the result stands: SFT
  lifts uidom delta-exact **0.81->0.94** and changed-field **0.875->0.97**, i.e. it gets the *changing* fields
  (the multi-element DOM cascade) right, not static mass.
- **Generalization (not memorization):** these are fresh seed-222 instances (train used seed 999/321); SFT
  still improves. (Stronger held-out-by-app-family / element-count-extrapolation tests are the next step,
  per gpt-duck.)
- **Base is genuinely competent** (uidom changed-field 0.875, uitick 1.0) — well above the copy baseline
  (changed-field copy = 0 by construction) — so the symbolic render-FDM is real, and SFT sharpens it.
results/audit_{base,sft}.json.

### 35.6 ABSTRACTION confirmed — the DOM-cascade SFT EXTRAPOLATES beyond its trained element range
The decisive memorization test (gpt-duck #4): train on ui_dom with 3-8 elements, test on **10-15** (more
elements than ever seen), honest metrics, fresh seed 333, n=48:

| | base | SFT | 
|---|---:|---:|
| exact | 0.750 | **0.979** |
| changed-field acc | 0.807 | **0.990** |
| delta-exact | 0.750 | **0.979** |

Base DEGRADES with more elements (changed-field 0.875 @n3-9 -> 0.807 @n10-15: more cascade/salience load). The
**SFT extrapolates** — 0.98 exact at element counts it NEVER trained on. This is strong evidence it learned the
multi-element DOM-cascade ABSTRACTION (flip aria-selected/hidden across ALL elements), not memorized small
instances. Combined with §35.5 (fresh-seed generalization, honest changed-field/delta metrics), the render-FDM
SFT result is real and abstraction-level, not a metric or memorization artifact. results/xl_{base,sft}.json.

**Renderer-axis status (honest):** the symbolic render-state FDM works and is SFT-improvable to ~0.98 on
synthetic single-loop DOM apps (changed-field metric, element-extrapolating); a real browser rasterizes
predictions to pixels and to a free-rolled video. NOT yet shown on real delegated-handler apps (TodoMVC/vanilla
one-shot is helper-blocked) — that real-app slice is the next credibility jump (gpt-duck #2).

### 34.5 REAL-APP slice via full-trace — FAILED (harness/truncation), real apps remain hard to measure cleanly
Attempted the gpt-duck #2 real-app credibility slice: full-trace mode (CWM executes the real reducer incl.
helpers; robust depth-0 entry-return extraction) on todomvc + vanilla, audited metrics:
- todomvc: exact 0.602, changed-field 0.614 (field_f1 0.602 < copy_f1 0.752 -> sub-copy = broken extraction)
- vanilla: exact 0.163, changed-field 0.188 (36/163 unparsed) — long validateAll cascades TRUNCATE at 4096 tok
  -> entry-return never reached -> garbage.
Full-trace is WORSE than the §34 one-shot step-over (todomvc 0.77 parseable-correct, vanilla 0.53) because real
reducer traces are long and truncate. **Conclusion: real delegated-handler apps are genuinely harness-hard to
measure** (one-shot can't capture helper-delegating dispatch -> None; full-trace truncates on long traces). The
honest real-app estimate stays at the §34 one-shot numbers. A clean real-app slice needs either depth-aware
step-over with a large context budget (the §34.2 attempt crashed on context overflow) OR inlined/self-contained
real handlers OR a model/render-state-split reducer — deferred as the real next-credibility-jump (not a quick
win). The SOLID, audited, abstraction-validated contribution stays the SYNTHETIC render-FDM (§35.5-35.6) + the
working pixel/video pipeline (§37).

### 37.4 POWERED drift study (gpt-duck #6) — capability × drift-control COMPOSE (16 rollouts, CIs)
Replaced the anecdotal single rollouts with 16 independent free-roll rollouts per config (rollout as the unit;
bootstrap CI over rollouts), togglelist (accumulating app), 8 steps. mean per-step exact-match accuracy:

| config | mean per-step acc | CI95 (over rollouts) | all-steps-correct |
|---|---:|---|---:|
| base, free-roll (k=∞) | 0.438 | [0.32, 0.56] | 0.00 |
| base, re-ground k=3 | 0.562 | [0.47, 0.66] | 0.00 |
| **SFT (cwm_cascade), free-roll** | 0.617 | [0.47, 0.77] | 0.19 |
| **SFT + re-ground k=3** | **0.750** | [0.64, 0.85] | 0.25 |

The two recipe levers ADD and COMPOSE:
- **Capability (SFT):** 0.438 -> 0.617 (+0.18) — raises the single-step ceiling.
- **Drift-control (re-grounding k=3):** 0.438 -> 0.562 (+0.12) — closes part of the rollout gap.
- **Both:** 0.438 -> **0.750** (+0.31) — base [0.32,0.56] and SFT+reground [0.64,0.85] are **non-overlapping**.
This is the §32.8 / drift.md capability-vs-drift decomposition, now POWERED and in render space: a faithful long
UI rollout needs BOTH the SFT (capability) and re-grounding (drift), exactly as the renderer-first recipe
predicts. (togglelist stays <1.0 even combined — a residual hard-state capability gap, the next SFT target.)
run_drift_stats.py, results/driftstats_{base,sft}_{k0,k3}.json.

### 35.7 CROSS-APP abstraction — the cascade skill transfers to a HELD-OUT app family (gpt-duck #4 hardest)
The strongest memorization test: train the SFT on tabs/accordion/counter (+ ui_tick form/cart) but HOLD OUT
togglelist entirely (cwm_heldapp), then eval on togglelist (different schema: data-done/count vs
aria-selected/hidden). Single-step FDM, n=16, fresh seed 444:

| model | exact | changed-field | delta |
|---|---:|---:|---:|
| base | 0.438 | 0.719 | 0.438 |
| **cwm_heldapp** (togglelist NEVER trained) | 0.562 | 0.781 | 0.562 |
| cwm_cascade (in-dist, trained WITH togglelist) | 0.625 | 0.812 | 0.625 |

The cascade SFT **transfers across app families**: held-out togglelist rises base 0.438 -> 0.562 (+0.124)
DESPITE never training on togglelist — capturing ~66% of the in-dist gain (cascade 0.625, +0.187). So the
multi-element-cascade skill generalizes to a NEW app with a DIFFERENT schema, not just to new instances/sizes
of trained apps. Combined with §35.6 (element-count extrapolation) and §35.5 (fresh-instance + honest metrics),
the render-FDM SFT abstraction is validated across **instances, element counts, AND app families** — the
strongest evidence the model learned the cascade abstraction, not templates. (togglelist caps ~0.625 even
in-dist single-step -> residual count-recompute capability gap, the next target.) results/heldapp_*.json.

### 37.5 RE-GROUNDING KNEE (powered, gpt-duck #6) — full k-sweep with CIs
Mapped the re-grounding knee on togglelist (base, 24 rollouts/point, 8 steps, bootstrap CI over rollouts):

| k (re-ground period) | mean per-step acc | CI95 |
|---|---:|---|
| 1 (teacher-forced = capability ceiling C) | 0.594 | [0.52, 0.67] |
| 2 | 0.573 | [0.50, 0.65] |
| 3 | 0.531 | [0.46, 0.60] |
| 5 | 0.479 | [0.40, 0.56] |
| ∞ (free-roll floor) | 0.469 | [0.39, 0.55] |

Capability ceiling C=0.594, free-roll floor 0.469, **drift gap D=0.125**. The curve descends monotonically; the
**knee is at k≈2** — re-grounding every 2nd step recovers **83%** of the drift gap (0.573 vs floor 0.469) at half
the oracle cost of teacher-forcing. togglelist is more CAPABILITY-bound than drift-bound (small D=0.125 because
single-step C is itself only 0.59) — consistent with §37.4 where the SFT capability lever (+0.18) dominated the
re-grounding lever (+0.12). So the recipe priority on this app is SFT first (raise C), re-grounding second
(close the modest D). The k≈2 knee = the renderer's render-budget: re-render ~half as often for a near-ceiling
faithful video. results/knee_base_k*.json.

---

## 38. REPRODUCIBILITY — exact commands for the renderer/pixel axis (§34-37)
Two venvs: `.venv` (train: torch/transformers/peft), `.venv_vllm` (infer: vllm 0.23 + playwright/streamlit via uv).
One adapter per vLLM process. Adapters: `cwm_cascade` (all ui_dom+ui_tick), `cwm_heldapp` (no togglelist).

```bash
# --- DATA (CPU) ---
python3 build_uidom_data.py  --per_app 25 --seed 999 --out data/uitrans_uidom.jsonl     # DOM render-FDM apps
python3 build_uitick_data.py --apps form,cart --per_app 70 --seed 321 --out data/uitrans_uitick.jsonl
.venv/bin/python build_cascade_sft.py facebook/cwm \
   --train_sources data/uitrans_uitick.jsonl,data/uitrans_uidom.jsonl --max_len 6000 \
   --out_train data/sft_cascade_train.jsonl --out_heldout data/uitrans_cascade_heldout.jsonl

# --- TRAIN (.venv, ~25min, 4xA6000) ---
.venv/bin/python train_lora_cwm.py facebook/cwm --data data/sft_cascade_train.jsonl \
   --out adapters/cwm_cascade --max_steps 100 --lr 1e-4 --grad_accum 8

# --- EVAL with AUDITED metrics (.venv_vllm; exact + changed-field + delta + copy_f1) ---
.venv_vllm/bin/python run_uitrans_probe.py facebook/cwm --tp 4 --data data/uitrans_uidom.jsonl \
   --out results/audit_base.json                                  # base
.venv_vllm/bin/python run_uitrans_probe.py facebook/cwm --tp 4 --lora adapters/cwm_cascade \
   --data data/uitrans_uidom.jsonl --out results/audit_sft.json   # SFT
#   --mode fulltrace  -> for delegated-handler real apps (todomvc/vanilla; note: truncates on long traces, §34.5)

# --- PIXELS (.venv_vllm; Playwright+Chromium) ---
.venv_vllm/bin/python dom_render.py --data data/uitrans_uidom.jsonl --n 4 --out_dir results/render_demo
.venv_vllm/bin/python render_predict_demo.py facebook/cwm --data data/uidom_demo.jsonl --n 8 \
   --lora adapters/cwm_cascade --tag sft --out_dir results/render_predict     # predicted DOM -> PNG
.venv_vllm/bin/python render_rollout.py facebook/cwm --app tabs --steps 6 --lora adapters/cwm_cascade \
   --tag sft --out_dir results/render_rollout    # free-roll -> rendered VIDEO (GIF); --reground_k N for drift

# --- POWERED DRIFT (.venv_vllm; rollout as unit, bootstrap CIs) ---
.venv_vllm/bin/python run_drift_stats.py facebook/cwm --app togglelist --n_roll 24 --steps 8 \
   --reground_k 3 --lora adapters/cwm_cascade --tag sft_k3 --out results/driftstats_sft_k3.json
```
Key results: §35.5 audited base->SFT (delta 0.81->0.94); §35.6 element extrapolation (n10-15: 0.98);
§35.7 cross-app transfer (held-out togglelist 0.44->0.56); §37.4-37.5 drift composition + k~2 knee.

### 34.6 REAL-APP BREAKTHROUGH — base CWM nails small TodoMVC (delegation is NOT the blocker, LENGTH is)
The §34.5 full-trace failure looked like real apps are hard. Decisive control: filter REAL TodoMVC to SMALL
states (<=3 todos -> short traces) and full-trace probe (n=32, all 7 action types):

| | exact | changed-field | delta | copy_f1 | unparsed |
|---|---:|---:|---:|---:|---:|
| **base CWM, small TodoMVC** | **1.000** | **1.000** | **1.000** | 0.753 | 0 |

**Base CWM PERFECTLY predicts real TodoMVC state transitions** — add/toggle/delete/edit/toggleAll/
clearCompleted/setFilter — executing the REAL reducer including its `cloneTodos`/`nextId` HELPER calls. So:
- **Helper-delegation is NOT the blocker.** CWM traces delegated real-app handlers correctly.
- **The blocker is TRACE LENGTH:** §34.5's todomvc 0.60 / vanilla 0.16 was pure TRUNCATION on long traces
  (big todo lists / 14-field cascades exceeding the token budget), not a capability or delegation failure.
- ⇒ the real-app credibility jump (gpt-duck #2) is ACHIEVED for states that fit the budget; scaling to large
  real apps is a TOKEN-BUDGET / abstraction problem (bigger ctx, step-over helpers, or state chunking) — the
  same §29.3/§30 lesson (size the budget to the trace; SFT the step-over abstraction), NOT a new capability wall.
This is the strongest real-app evidence in the study: CWM is a working state-transition predictor for a REAL
web app (TodoMVC) at small scale, out of the box. results/todomvc_small_base.json.

### 34.7 BUDGET FIX confirms it — vanilla 0.16 -> 0.75 (the "JS cascade gap" was largely a HARNESS artifact)
Re-ran vanilla (real JS form-validator) full-trace with an adequate token budget (4096 -> 7500):

| vanilla full-trace | exact | changed-field | delta | unparsed |
|---|---:|---:|---:|---:|
| budget 4096 (§34.5) | 0.163 | 0.188 | 0.163 | 36/80 |
| **budget 7500** | **0.750** | **0.750** | **0.750** | **1/80** |

Raising the budget lifts vanilla **0.16 -> 0.75** (36 -> 1 unparsed) — CONFIRMING §34.5 was TRUNCATION. So when
CWM actually EXECUTES the real JS validation cascade (full-trace, adequate budget) it scores **0.75**, far above
the one-shot step-over 0.53 (§34) and the truncated 0.16. **Major reframing:**
- The "vanilla/JS cascade gap" (§34/§36) is **largely a ONE-SHOT-abstraction + truncation HARNESS artifact**, NOT
  a fundamental capability gap. CWM CAN run the deep JS validation cascade when it traces it fully (0.75).
- The §36 "narrow SFT negatively transfers to JS" result was measured with the WEAK one-shot harness (0.53->0.35)
  — it should be re-measured with full-trace; the true JS capability is ~0.75, much closer to TodoMVC/Python.
- Combined with §34.6 (small TodoMVC 1.0), the honest real-app picture is POSITIVE: base CWM executes real-app
  handlers (TodoMVC + vanilla form-validator) well; the dominant lever is TOKEN BUDGET / trace-length
  abstraction (§29.3/§30), and one-shot step-over is just a lossy way to measure delegated/cascading handlers.
The residual vanilla 0.25 (real errors at adequate budget) is the genuine cascade difficulty — modest, and the
real (not artifact-inflated) SFT target. results/vanilla_bigbudget.json.

### 34.8 ...but the §36 JS NEGATIVE TRANSFER is REAL (not a one-shot artifact) — confirmed under full-trace
Re-measured cwm_cascade on vanilla with full-trace (the §34.7 fair harness):

| vanilla (real JS) | base | cwm_cascade SFT |
|---|---:|---:|
| full-trace exact | 0.750 | 0.350 |
| full-trace changed-field | 0.750 | 0.656 |

So even with the FAIR full-trace harness (CWM executes the cascade, 0 unparsed), the narrow Python-cascade SFT
**still degrades vanilla** (base 0.75 -> SFT 0.35 exact). The §36 negative transfer is REAL, not a one-shot
artifact — though the changed-field metric shows it's less catastrophic (0.75 -> 0.66: the SFT model still gets
most changed fields right but introduces more spurious errors, so EXACT drops more). Net honest picture:
- **Base CWM is a good real-app FDM** (TodoMVC 1.0 small, vanilla 0.75 full-trace) — §34.6/34.7.
- **Narrow Python-cascade SFT genuinely hurts real-JS execution** (vanilla 0.75->0.35) — §36 stands; the fix
  (JS-flavored/diverse replay, §36.2) is still needed and should be evaluated with full-trace going forward.
- The earlier vanilla numbers (one-shot 0.53; truncated full-trace 0.16) were BOTH artifact-depressed; the true
  base vanilla capability is 0.75. results/vanilla_sft_fulltrace.json.

---

## §39 — REAL-APP VIDEO CAPSTONE: CWM free-rolls the REAL TodoMVC reducer -> a UI video (8/8 exact)
The strongest north-star demo: a **model-generated video of a REAL web app responding to a full user session**,
generated end-to-end by CWM with NO engine in the loop.

**Setup** (`todomvc_video.py`): the REAL TodoMVC reducer (`todomvc_reducer.js.txt`: cloneTodos / nextId /
dispatch, ~1900 chars, model-state `{filter, todos:[{id,title,completed}]}`). A scripted 8-action user session:
`add "buy milk" -> add "walk dog" -> toggle 1 -> add "write report" -> setFilter active -> toggle 2 ->
setFilter completed -> clearCompleted`. For each action CWM predicts the next model-state via **full-trace**
(it executes the real reducer incl. the helper calls), and **free-rolls** (each prediction feeds the next step).
Each predicted state is rendered by the real headless-Chromium pipeline (`dom_render.render_many` +
`render_todomvc`, a TodoMVC-styled view) into a frame; frames -> GIF.

**Result:** per-step exact = **[True]×8 = 8/8**. CWM free-rolled the entire real-app session with ZERO drift —
every predicted state matched the real reducer's output (ground truth via `node`), including:
- the `nextId`/`cloneTodos` helper semantics on add,
- toggle flipping `completed` on the right id,
- `setFilter` changing only the view (active/completed correctly show/hide the non-matching todos in the render),
- `clearCompleted` removing the 2 completed todos (the completed-filter view then correctly empties).

**Artifacts:** `results/todomvc_video/todomvc.gif` + `step{0..8}_*_OK.png` (every frame `_OK`). The rendered video
shows a recognizable TodoMVC UI (checkboxes, strikethrough done items, "N left" counter, all/active/completed
filter pills) evolving faithfully under the user actions.

**Why this is the headline:** it composes the whole stack proven this session into the user's exact north-star —
"given app code, predict an image/video of the UI responding to input":
1. REAL app code (the actual TodoMVC reducer, not a Python twin) as context,
2. CWM as the execution-free FDM predicting state transitions (full-trace, §34.6-34.7),
3. free-roll over an action sequence (the interactive long-horizon axis),
4. real-browser render of each predicted state -> a video (the pixel axis, §37).
Base CWM (no SFT) suffices at this scale because small TodoMVC is squarely within its capability (§34.6 = 1.0).
The renderer is `render(state)` (no learned renderer needed — DOM-state is a sufficient statistic for the view,
§A/§35), and there is NO engine in the rollout loop — CWM IS the dynamics model. This is the cleanest
demonstration that the renderer/pixel extension of CWM works on a REAL application.

**Honest scope:** this is the regime where base CWM is strong (small states, short traces). The open frontier is
unchanged: (a) larger states / longer reducer traces need step-over abstraction + token budget (§29.3/§30), and
(b) the §36 narrow-SFT JS negative transfer (only relevant if SFT is used; base needs none here). The capstone
deliberately uses base CWM to show the un-finetuned model already free-rolls a real app faithfully at this scale.

### §39.1 — STRESS the real-app video (16 actions, 6 todos, edits/deletes) -> 16/16; the only "drift" was a token-cap artifact
To locate base free-roll's drift onset on the REAL app, `todomvc_video.py --stress` runs a harder 16-action
session (up to 6 todos, interleaved `add/toggle/EDIT/DELETE/setFilter/clearCompleted`, id-based addressing with
gaps after deletes). First pass (fixed `max_tokens=3072`): **14/16**, with the only two misses at step 11
(`toggle id6`) and step 12 (`delete id3`) — the 6-todo regime.

**Diagnosis (deterministic greedy re-probe, now deleted):** both misses were the recurring **§29.1 token-cap
truncation artifact**, NOT a dynamics error:
- `delete id3` @budget=3072 -> raw output truncated mid-JSON (rawlen 115, `...{'id': 2`), unparseable -> "wrong".
  @budget=6000 -> **good=True, exactly correct** (id3 removed, 1/2/4/5/6 remain). Pure budget.
- `toggle id6` from a clean canonical-key-order state @3072 -> **good=True** (perfect). The stress miss came from
  free-roll feeding the model's OWN output back, whose key order is `completed/id/title` (vs canonical
  `id/title/completed`); that longer/0-EOS-tilted serialization tipped the 6-todo trace over the 3072 cap.
  Both are HARNESS/representation effects (token budget + serialization), not a transition-logic failure.

**Fix + re-run:** scaled the per-step budget with state size (`max(3072, 1500 + 1100*len(todos))`, the §29.1
"size token caps to trace length" lesson). Re-running `--stress` -> **per-step exact = [True]×16 = 16/16, ZERO
drift.** Base CWM free-rolls the entire 16-action real-app session perfectly — including the `edit` (title
"buy milk"->"buy oat milk"), the `delete` with subsequent id-gaps, and every filter show/hide transition.

**Artifacts:** `results/todomvc_video_stress/todomvc.gif` + `step{0..16}_*_OK.png` (all OK);
`results/MORNING_todomvc_stress_16of16.png` (17-frame labeled montage). This both strengthens §39 (base handles
a non-trivial 6-todo real app, not just ≤3) and reproduces the project's STANDING LESSON once more: a "dramatic
CWM failure" (the 2 stress misses) was again my harness (token cap), not the model — and the open scaling lever
remains token-budget / step-over abstraction (§29.3/§30), exactly as predicted, with no dynamics wall in sight
at this scale.
