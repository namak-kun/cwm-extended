# Extending the Code World Model into a Self-Improving Game World Model

*A coherent summary of the `wm_probe/` study. Master log with every experiment: `results/REPORT.md`
(§0–§32, ~1850 lines). Raw numbers: `results/*.json`. All results are LoRA adapters on `facebook/cwm` 32B;
no commits were made to any repo.*

---

## Abstract

Meta's **Code World Model (CWM)** predicts the execution state of code frame-by-frame *without running it* —
a symbolic world model over programs. We extend it from "trace a program" into a **self-improving,
action-conditioned game world model**. Three steps: (1) teach CWM to predict an entire game **tick** in one
shot (a forward dynamics model, FDM, at ~10× compression); (2) derive an **inverse dynamics model (IDM)** for
free by forward-searching the FDM; (3) close a **flywheel** — use the IDM to self-label *unlabeled* gameplay
with actions, then retrain the FDM. The flywheel lifts held-out per-tick state accuracy **0.52 → 0.68,
matching a true-action oracle, with zero new action labels**, and single-step action-conditioned accuracy
**0.48 → 0.81**. We prove the gain is genuinely action-conditioned (not a state prior), statistically credible
(bootstrap CI excludes 0), and stable across two self-labeling rounds. We map the method's boundary: it
collapses on hard arenas where the FDM can't discriminate actions, but those are reachable with oracle labels.
**Net: CWM can bootstrap game dynamics from raw state sequences — no per-game engine, wrapper, or labels.**

---

## Positioning — how to describe this project (and why it's useful)

**One sentence.** *A model that predicts how an app/game **behaves** — what its UI/screen becomes when a user
acts — directly from the source code, and runs backwards to recover which action produced a change. The
pixels are the human-readable read-out of a predicted, code-grounded **symbolic state**.*

The product is **predicted behavior**, not images; rendering is the display layer and the eval oracle.

**The destination is pixel-level (frame-in → frame-out).** The end goal is a world model you drive in pixels:
feed a frame, take an action, watch generated frames evolve. The symbolic state (DOM / game-state) is the
**interpretable latent** that makes that tractable and faithful:

```
frame_in ─[perception: pixels→state]→ state ─[CWM FDM: (state,action)→state′]→ state′ ─[render/generate: state→pixels]→ frame_out
```

**Why this beats a black-box video world model (Genie/GameNGen-style diffusion).** Same pixel-in/pixel-out
interface, but the latent is a **code-grounded, inspectable symbolic state** — so predictions are *faithful*
(grounded in the actual program), *debuggable* (you can read the state), and *controllable* (counterfactual:
swap the action or the code and see the predicted divergence). A pure pixel→pixel model is none of these.

**The "why not just run it?" answer.** If you can cheaply run the app, render it — that's not the point. The
value is a **learned model** instead of an **execution**: it (1) works on code you can't/won't run (a PR diff,
a half-written component, no backend/deploy); (2) is **counterfactual** ("what does the *buggy* version render
vs. the correct one?" → divergence = candidate bug); (3) is a **fast surrogate** inside a test-gen / search /
agent loop (imagine "if I click here…" without thousands of slow real renders); (4) yields **free action
labels** — the IDM recovers the action between two observed states, turning raw UI/gameplay recordings into
auto-labeled demonstration data.

**Who it's useful to.**
- **Testing / QA** (the original goal): the predicted post-action state is a **test oracle** — diff the real
  app against it to flag regressions; or generate inputs predicted to break an invariant.
- **UI / game agents & RL** (the NitroGen pain, generalized): a learned world model + free inverse-dynamics
  labeling = a planning environment and an imitation-data pump, with **no hand-built per-app environment**.
- **Code review / understanding**: "what does this front-end change actually *do* to the rendered state?" — on
  code that isn't deployed.

**The research framing.** UIs are the cleanest testbed for the general question *can a code model predict
program behavior under input?* — the DOM **is** execution state, the browser gives **free exact ground truth**,
and it's **renderable** for human inspection. The method (one-shot step-over forward model, inverse model for
free via search, self-labeling flywheel) is **not DOM-specific**: DOM is the first instance of a general
`Engine` abstraction (games and arbitrary code are others).

**What not to claim (to stay credible).** Not "replaces the browser" (you *use* it as the oracle); not
"generates webpage images from scratch" today (it predicts *structure*; an engine draws it — learned pixel
generation is the staged stretch). Lead with **behavior prediction**; pixels are the read-out.

---

## 1. Motivation

The original goal: *given a codebase and an input, predict how its output evolves — in a multimodal sense* —
to (a) generate better test cases (the model knows where code fails) and (b) eventually render GUI/game output.
The driving pain came from game-RL work (NitroGen): **building game environments is hard because everything is
game-specific** — the wrapper, the reward, the integration. The hypothesis: a CWM-style model that predicts how
*execution state evolves under input* could sidestep per-game environment engineering — you give it the code as
context and read out the predicted state (and, ultimately, the predicted pixels).

This study tackles the **symbolic core** of that hypothesis on games, and characterizes exactly how far it goes.

---

## 2. Background: what CWM does

CWM is a 32B LLM post-trained to emit Python **execution traces**: given source + an entry call, it predicts
the sequence of execution frames (line-by-line variable states) that running the code *would* produce. It is a
**step-by-step symbolic interpreter**, not a one-shot input→output function.

Two facts from our Phase-2 probes frame everything that follows:
- **Per-step state tracking is essentially perfect** — given a correct history, CWM predicts the next frame
  correctly to depth **247** (deepest tested).
- **The bottleneck is compounding error in free rollout** — unrolled from scratch it stays correct for ~**106
  frames** then drifts. The wall is horizon/drift, not per-step capability.

This is the lever the whole extension pulls on: keep CWM doing what it's good at (per-step state) and attack the
horizon/abstraction problem.

---

## 3. Method: building a game world model on top of CWM

We use a controlled game-tick microworld (`game_tick.py`): a player `{x,y,hp,score}` plus *K* enemies; each
**tick** applies an action (move U/D/L/R), then resolves buried side-effects in an entity loop — **stomp**
(+score, enemy dies) and **contact** (−hp). This deliberately reproduces the real game-prediction failure mode:
a *salient* actor (player x/y) that's easy to track, plus *consequential side-effects hidden in a secondary
loop* that are easy to drop.

### 3.1 State tracking is already a strength (once measured cleanly)

Early "CWM can't track game state" results were **all metric artifacts** — an arithmetic-checksum return
(invoked CWM's separate arithmetic weakness → 0.0), a list-ordering confound (→0.70), and token-cap truncation
(→0.0). With an **order-independent dict-state metric** and an adequate token budget, CWM tracks game-tick state
(player + *K* enemies + within-tick stomp/contact across ticks) **near-perfectly: outcome 0.95 at K=3–5, and
1.0 at K=10**. *Lesson that recurs throughout: CWM's dramatic failures are usually harness/metric bugs.*

### 3.2 The forward dynamics model: one-shot tick via "step-over" abstraction

Tracking state line-by-line doesn't scale (a few ticks = hundreds of frames → blows the context window). Can
CWM predict a whole tick `s_{i+1} | s_i` in **one shot**? Base CWM **cannot** (per-tick 0.017) — it nails the
salient player x/y but can't compute the K-enemy chase + side-effects without tracing the interior.

The fix is the φ-expansion playbook: **SFT the abstraction in.** We added a "step-over" trace format
(`gt_trace.trace_program(stepover_depth=1)`: main lines + the whole `step()` as one opaque CALL→RETURN, ~10×
shorter) and trained an 80-step LoRA on it. Result: **per-tick 0.017 → 0.692, all-ticks-correct 0 → 0.55, at
9.6× compression** on held-out games. This adapter (`cwm_gametick_stepover`) is our **FDM₀** — a compressed,
accurate, one-shot game-tick predictor. *This is the scalable game-world-model unit.*

### 3.3 Inverse dynamics for free (forward search over the FDM)

An IDM answers "which action was taken between two observed states?" — the harder, RL-relevant direction. We get
it **without training a separate model**: for each candidate action, ask the FDM to predict the next state, and
pick the action whose prediction is closest to the observed next state (`run_idm_search.py`). On a
representative 490-transition set the forward-search IDM recovers the true action at **0.72**, with error
concentrated exactly on the buried **contact/−hp** transitions (0.66) and on **zero-margin ties** (34% — cases
where actions yield near-identical next states). A usable but **noisy (~28% wrong)** labeler.

### 3.4 The FDM↔IDM flywheel (the contribution)

Close the loop: take **unlabeled** gameplay (real state sequences, actions hidden), use the IDM to **infer the
action labels**, and fold the resulting `(state, action → observed-next-state)` examples back into FDM training.

This is **latent-action semi-supervised learning, not circularity** (validated by a rubber-duck check): the
training *target* is the **observed true next state** — always correct — while the IDM only infers the *action*
that conditions the input. A wrong action label is paired with a *true* outcome; it never injects a hallucinated
dynamics target. Design (`build_flywheel_data.py`): a decisive 3-way with an **oracle control** (same
trajectories + *true* actions) and a baseline (FDM₀), all eval'd on a disjoint held-out seed.

---

## 4. Results

### 4.1 The flywheel works — and matches the oracle with zero new labels

Held-out seed 999, n=40 multi-tick programs, per-tick state accuracy:

| arm | per-tick | all-ticks-correct |
|---|---:|---:|
| FDM₀ (baseline, no flywheel) | 0.525 | 0.375 |
| **FDM_IDM (self-labeled, 0 new labels)** | **0.683** | **0.525** |
| FDM_oracle (true actions, control) | 0.679 | 0.525 |

Pre-registered rule **WIN = FDM_IDM > FDM₀ AND FDM_IDM ≈ FDM_oracle → satisfied.** Despite a 0.72 / 28%-noise
labeler, the self-labeled arm **fully recovers the oracle's gain**. No truncation / missing-tick artifacts;
gain broadly distributed (improved 14/40 programs, regressed 3/40).

**Why the noise is benign:** the target is the observed true state (correct regardless of label), and the IDM
errors land on the 34% zero-margin **ties** + contact transitions where actions are near-equivalent — so a
mislabeled `(action, observed-state′)` pair is still ~dynamically consistent. The high-margin transitions that
actually *teach* dynamics are recovered accurately.

### 4.2 It is genuinely action-conditioned (the decisive control)

A rubber-duck flagged the key alternative: IDM≈oracle would *also* hold if the FDM mostly predicts next-state
from code+current-state and **ignores the action**. We tested directly (`run_action_sensitivity.py`, n=120
single-tick, seed 999): run the trained FDM under **all four actions** per state.

| arm | true-action acc | **swap-action acc** | **pred diversity** | contact(−hp) | stomp | death |
|---|---:|---:|---:|---:|---:|---:|
| FDM₀ | 0.483 | **0.000** | **4.0 / 4** | 0.279 | 0.312 | 0.312 |
| **FDM_IDM** | **0.808** | **0.000** | **4.0 / 4** | **0.754** | **0.812** | **0.812** |
| FDM_oracle | 0.783 | 0.000 | 4.0 / 4 | 0.656 | 0.875 | 0.875 |

- **`swap_acc = 0.000` and `pred_div = 4/4` everywhere** — a wrong action *never* reproduces the observed
  outcome and the model emits a *distinct* prediction per action. The "action ignored" hypothesis is
  **refuted**; the FDM is genuinely action-conditioned.
- The flywheel lifts single-step true-action accuracy **0.48 → 0.81 ≈ oracle 0.78**, with the gain
  **concentrated on exactly the hard buried side-effects** (contact 0.28→0.75, stomp/death 0.31→0.81) — the
  failure locus FDM₀ was missing. It didn't add generic data; it taught within-tick side-effect dynamics.

### 4.3 The win is statistically credible

Program-level paired bootstrap (the honest unit is the program, not the correlated tick), 10k resamples:

| contrast | mean Δ per-tick | 95% CI | sign test |
|---|---:|---|---:|
| **IDM − FDM₀** | **+0.158** | **[+0.029, +0.292] — excludes 0** | 14W/3L, p=0.013 |
| ORACLE − FDM₀ | +0.154 | [+0.025, +0.283] — excludes 0 | 13W/3L, p=0.021 |
| IDM − ORACLE | +0.004 | [−0.158, +0.167] — includes 0 | 10W/9L, p=1.0 |

The improvement over baseline is credible; IDM vs oracle is genuinely indistinguishable (as expected if labels
are good).

### 4.4 Stable over rounds (no compounding collapse)

Round 2: relabel a fresh batch with FDM_IDM (self-reinforcement), **margin-filter** (drop zero-margin ties),
retrain. Filtering purifies the labeler **0.72 → 0.99 recovery** (kept 495/795). Result: per-tick
**0.683 → 0.696 ≈ oracle 0.692**, all-ticks **0.525 → 0.625**, action-conditioning preserved (swap=0, div=4/4).
**Two stacked self-labeled rounds are stable** — no rot. Round 1 already hit the oracle ceiling, so round 2
*plateaus*; the lever to climb further is difficulty/scale, not more rounds. **Margin filtering is the key
multi-round anti-collapse guardrail.**

### 4.5 Operating envelope: where it breaks, and how to break the ceiling

- **Difficulty holds the gain:** on a hard band (K=6–8 enemies, T=4–6 ticks; ~465-frame traces), FDM₀ 0.117 →
  FDM_IDM_r2 **0.284** (+0.17, 2.4× relative). The self-labeled capability isn't overfit to easy configs.
- **But self-labeling collapses there:** trying to self-label hard trajectories, the IDM hits **chance (0.258),
  100% zero-margin ties, 0 kept** — the FDM is too weak to forward-discriminate actions, so every label is a
  tie. **The flywheel only bootstraps regimes the FDM can already partly model.** This is the method's boundary,
  mapped to difficulty.
- **The ceiling is breakable — with real labels:** hard-band SFT *with oracle actions* climbs **0.284 → 0.369**
  (all-ticks 0.1→0.2). So hard arenas are learnable; they just need oracle/engine labels + a curriculum, not
  self-labeling.

**Full scaling recipe:** flywheel (self-labeling) for easy–mid difficulty; engine-as-oracle labels + a
K-curriculum for hard.

---

## 5. What this buys you (benefits, tied to the original goals)

1. **No per-game environment engineering.** The flywheel bootstraps action-conditioned dynamics from *raw state
   sequences alone* — no engine, wrapper, reward, or human action labels. This directly dissolves the NitroGen
   pain point that motivated the project.
2. **A learned, queryable simulator.** FDM = forward model (plan/predict next state under an action); IDM =
   inverse model (recover actions, label data). Together they're the substrate for RL, planning, or search over
   a game — derived from one trace-trained LLM.
3. **Self-improvement.** This is the first result in the study where CWM *improves its own capability with no new
   supervision* (prior gains all needed gold/oracle SFT targets). The improvement lands on the hardest,
   most-buried dynamics.
4. **Toward better test cases.** A model that predicts *where and how* state diverges under input is exactly a
   model that knows where code can fail — the test-generation use case, now grounded in a measured
   forward/inverse model.

---

## 6. Limitations (honest)

- **Symbolic only.** Everything here is structured **state**, not pixels. The image/video-of-the-app half of the
  original wish is not yet built (see §7).
- **Microworld.** Results are on a controlled game-tick generator, not real game code/frames. It was designed to
  expose the right failure mode (salient-actor + buried-side-effect), but it is not a real engine.
- **Difficulty ceiling.** Absolute accuracy on big arenas (K=8, T=6) is low (~0.28–0.37); the self-labeling loop
  cannot reach there unaided.
- **Drift.** The hard-band wall is *both* within-tick capability (single-step 0.49) and rollout drift
  (0.49→0.28 over 4–6 ticks); long horizons still need re-grounding.
- **Modest n.** The headline +0.158 is ~150 tick-decisions / 40 programs, single seed (CI excludes 0, but n is
  small).
- **Adapter isolation.** vLLM 0.23 can't hot-swap LoRA adapters within one engine — every comparison runs one
  adapter per process (a tooling constraint, handled but worth knowing).

---

## 7. Next steps

**Now:** this writeup (done). **Then, in parallel** (the loop itself is proven; remaining work is scale + the
multimodal axis):

1. **Scale the symbolic world model** (`axis_scale`) — engine-as-oracle labels + K-curriculum to break the hard
   ceiling self-labeling can't reach; push step-over per-tick 0.69 → 0.9.
2. **Pixel/GUI axis** (`axis_gui`) — the originally-pitched multimodal half: bridge symbolic state → **rendered
   output**. **Start with GUI apps** (given app/HTML code, predict the rendered view and its response to input),
   *before* real games.
4. **Drift control** (`axis_drift`) — multi-tick rollout + periodic re-grounding for long horizons; separate the
   capability wall from the drift wall.

**Deferred:** **Real games on actual frames** via the stable-retro env (`axis_realgames`) — *"might be too
much"*; noted for later, after the GUI-app axis shows traction.

---

## Appendix: artifact map

**Adapters** (`adapters/`):
- `cwm_gametick_stepover` — **FDM₀**, the step-over one-shot tick model (0.69) and the IDM labeler.
- `cwm_fdm_idm_r1` / `cwm_fdm_idm_r2` — flywheel (self-labeled) FDM, rounds 1 & 2.
- `cwm_fdm_oracle_r1` / `cwm_fdm_oracle_r2` — oracle-label controls, rounds 1 & 2.
- `cwm_fdm_hardoracle` — hard-band oracle SFT (ceiling-break demo).

**Code** (`wm_probe/`):
- `game_tick.py` — the game-tick microworld (player + K enemies, stomp/contact side-effects).
- `gt_trace.py` — ground-truth tracer; `trace_program(stepover_depth=1)` gives the exact step-over abstraction.
- `run_gametick.py` / `run_gametick_abstract.py` — state-tracking eval / step-over (FDM) eval (`--lora`,
  `--kmin/kmax/tmin/tmax`).
- `run_idm_search.py` — FDM-as-IDM via forward search.
- `build_flywheel_data.py` — IDM-label unlabeled trajectories + oracle control + event/margin stats
  (`--margin_min`, `--kmin/kmax/...`).
- `run_action_sensitivity.py` — the action-conditioning control (true vs swapped action, pred diversity).
- `run_flywheel_eval.sh` / `run_r2_eval.sh` / `run_diff_sweep.sh` — eval drivers (one adapter per process).
- `train_lora_cwm.py` — LoRA trainer; `--init_adapter` continues from an existing adapter.

**Key results** (`results/`):
- `REPORT.md` §29–§32 — the full chronological record (read §32 for the flywheel).
- `flywheel_eval_*_n40_s999.json` — 3-way multi-tick eval (§4.1, §4.4).
- `action_sens_*_n120_s999.json` — action-conditioning control (§4.2).
- `flywheel_label_{r1,r2,hard,hard0}.json` — IDM labeling stats / margin filtering / collapse (§4.4–§4.5).
- `diff_hard_{fdm0,idmr2,hardoracle}.json` — hard-band difficulty + oracle-break (§4.5).
