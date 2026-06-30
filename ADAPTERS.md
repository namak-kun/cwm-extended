# Trained adapters — what each one is

_The 15 LoRA adapters are hosted on the 🤗 Hub: <https://huggingface.co/nmk-kun/cwm-extended-adapters>._
_This file mirrors that catalogue. Numbers link to `results/REPORT.md` sections._

## What each adapter is

### 🎮 Game world-model — the headline axis (§30, §32)
The "world model" thesis: predict a game's tick-by-tick state evolution from code, execution-free; then
bootstrap *action-conditioned* dynamics from **unlabeled** state sequences via a forward↔inverse flywheel.

| adapter | what it teaches | result | REPORT |
|---|---|---|---|
| **`cwm_gametick_stepover`** | One-shot game-tick transition `s_i → s_{i+1}` (player + K enemies + within-tick stomp/contact side-effects), via step-over SFT. This is **FDM₀**, the base the flywheel arms continue-train from. | per-tick state **0.017 → 0.692** | §30 |
| **`cwm_fdm_idm_r1`** | Flywheel **round 1**: continue-trained from `cwm_gametick_stepover` on **self-labeled** trajectories (FDM-as-IDM forward-search inverse dynamics — no action labels). | per-tick **0.525 → 0.683** (≈ true-action oracle; CI excludes 0) | §32 |
| **`cwm_fdm_idm_r2`** | Flywheel **round 2**: stacks a 2nd self-labeled round (margin-filtered → 99% label recovery). Stable plateau, **no collapse**. | per-tick **0.683 → 0.696** | §32.7 |
| `cwm_fdm_oracle_r1` | **Control** for `idm_r1`: identical recipe but **true-action oracle** labels. Confirms self-labeling ≈ oracle. | per-tick ≈ 0.679 | §32 |
| `cwm_fdm_oracle_r2` | **Control** for `idm_r2` (oracle round 2). | per-tick ≈ 0.692 | §32.7 |
| **`cwm_fdm_hardoracle`** | **Hard arena** (K6–8, where self-labeling collapses to chance): **oracle** SFT shows the hard ceiling *is* breakable with oracle/engine labels + a K-curriculum. | per-tick **0.284 → 0.369** | §32.10 |

### 🖼️ UI / DOM render world-model — the pixel axis (§35, §36)
State = canonical **DOM tree** (a sufficient statistic for the rendered pixels). These probe cascade/validation
logic and abstraction transfer.

| adapter | what it teaches | result | REPORT |
|---|---|---|---|
| **`cwm_cascade`** | Step-over SFT on UI DOM-cascade apps (`ui_dom` + `ui_tick`). **In-distribution win**, but a cautionary **negative transfer to real JS** — the main open SFT problem. | uidom exact **0.80 → 1.0**; real-JS vanilla **0.75 → 0.35** ⚠️ | §36 |
| **`cwm_heldapp`** | Same UI-cascade SFT but trained **without** the `togglelist` app, then evaluated on it (different schema) — an abstraction / cross-app held-out test. | togglelist exact **0.44 → 0.56** | §35.7 |

### 🧱 Object-state / φ-expansion — CWM trace-format studies (§22–§25)
| adapter | what it teaches | result | REPORT |
|---|---|---|---|
| **`cwm_oop_expanded`** | φ-expansion SFT teaching object-state observability (render object attributes each frame). | oop free-roll **0.02 → 0.93** | §22–24 |
| **`cwm_mixed_expanded`** | φ-expansion **+ mixed-corpus replay** to eliminate catastrophic forgetting of a held-out long mode. | held-out multientity **0.68 → 1.0** (oop preserved) | §22 |
| `cwm_dagger_gold` | OOP **gold-prefix** DAgger (matched 150-step budget). | free-roll **0.9324** | §25 |
| `cwm_dagger_drift` | OOP **drift-prefix** (on-policy) DAgger. Identical to gold → residual is a *structural* φ-render slip, not drift. | free-roll **0.9324** | §25 |

### ➗ Arithmetic drift studies — mostly neutral (a capability hole, not fixable by this SFT) (§26–27)
| adapter | what it teaches | result | REPORT |
|---|---|---|---|
| `cwm_arith_gold` | Correct-prefix (gold) per-frame SFT on long-arithmetic free-roll. | **0.143** (fails — compounding value drift) | §26 |
| `cwm_arith_drift` | Drift-prefix (single-round DAgger-style) SFT. | **0.180** ≈ base | §26 |
| `cwm_arith_wholetrace` | Whole-trace arithmetic SFT variant (same conclusion: needs tool-use/scratchpad, not SFT/RL). | ≈ base | §26–27 |

---

## Load an adapter (PEFT)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("facebook/cwm", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(
    base, "nmk-kun/cwm-extended-adapters",
    subfolder="cwm_gametick_stepover",   # <- any folder name from the tables above
)
```

> For vLLM-based inference (the project's harness), see `models/cwm_trace.py` and the `run_*.py` probes in the
> GitHub repo; each adapter is loaded in its **own** process (vLLM 0.23 has a multi-adapter-per-session bug).

## Which one do I want?
- **Game-tick prediction / the flywheel headline:** `cwm_gametick_stepover` → `cwm_fdm_idm_r1` → `cwm_fdm_idm_r2`.
- **UI/DOM render-FDM:** `cwm_cascade` (but note the real-JS regression; base CWM is often the better UI FDM — see REPORT §34, §39).
- **Object-state observability:** `cwm_oop_expanded` / `cwm_mixed_expanded`.
- The `oracle`, `dagger`, and `arith` adapters are **controls / ablations**, not deployment targets.

## License
Built on Meta FAIR's Code World Model; intended for **noncommercial research** use consistent with the FAIR
Noncommercial Research License. See <https://github.com/facebookresearch/cwm>.
