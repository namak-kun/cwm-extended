# Real-app UI-transition data CONTRACT (so one CWM probe consumes every target)

Each target subagent produces **`wm_probe/data/uitrans_<target>.jsonl`** (+ its harness under
`wm_probe/uidata/<target>/`). One JSON object per UI transition, with this schema:

```jsonc
{
  "target":       "todomvc",            // source name
  "lang":         "js",                  // "js" (node) or "python"
  "prompt_src":   "<self-contained program text>",  // see below — what CWM traces
  "entry":        "main",                // entry function carrying  '// << START_OF_TRACE' (js) or '# << START_OF_TRACE' (py)
  "action":       {"type":"toggle","id":2},          // the UI event (any JSON)
  "state_before": { ... },               // canonical current app/model state (JSON)
  "truth_state":  { ... },               // canonical NEXT state = result of REALLY running the app
  "source_app":   "<path or short note on the real app/handler this came from>"
}
```

## `prompt_src` — the program CWM step-over-traces
A SELF-CONTAINED program: the app's REAL update/handler logic (a `dispatch(state, action)->state` or the
app's reducer), then a `main()` that sets `state_before`, calls the handler ONCE with `action`, returns the
next state. Mirror `ui_tick.gen_one_event_src` / `game_tick.gen_one_tick_src`:

```js
// ---- real handler logic from the app (extracted/adapted) ----
function dispatch(state, action){ /* ...the app's real update logic... */ return state; }
function main(){  // << START_OF_TRACE
  let state = {/* state_before */};
  state = dispatch(state, {/* action */});
  return state;
}
console.log(JSON.stringify(main()));
```

## HARD REQUIREMENTS (self-consistency = the validity guarantee)
1. **`prompt_src` must EXECUTE** (`node` for js, `python3` for python) and return EXACTLY `truth_state`.
   Verify every row; **drop** any that don't. This is what makes the probe meaningful.
2. **`truth_state` must reflect the REAL app**, not a guess: derive `prompt_src`'s handler FROM the real app
   source (so running it == running the app). If you re-express handler logic, prove equivalence by running
   the real app (jsdom / Selenium / Streamlit AppTest) on a sample and checking equality.
3. **Canonical state**: order-independent — sort dict keys; keep list order only where semantic (e.g. todo
   order). Use the same canonicalization for `state_before` and `truth_state`.
4. **Budget**: keep `prompt_src` under ~6k tokens (small apps; trim unrelated code).
5. **Coverage**: aim for 40-100 transitions per target across VARIED actions and state sizes (include a few
   large states to probe scale).
6. **NO GPU / NO CWM.** You only build harnesses, harvest data, and verify by executing in node/python. The
   main agent runs the CWM probe afterward.

## Fallback if a target can't be a clean self-contained program
Deliver the raw `(state_before, action, truth_state, source_app)` triples anyway (omit/relax `prompt_src`) +
a short `uidata/<target>/NOTES.md` explaining the blocker. Partial real data still beats none.

## Targets & owners
- `todomvc`  — TodoMVC vanilla-JS es5 (`tastejs/todomvc`), via jsdom. Real MVC model.
- `vanilla`  — tiny bradtraversy apps (form-validator 5KB, etc.), via jsdom.
- `miniwob`  — MiniWoB++ tasks (Farama). TRY jsdom-first (tasks are self-contained HTML+core.js); Selenium only
  if necessary. Deterministic seeds only.
- `streamlit`— `streamlit/demo-todo` + counter, via `streamlit.testing.v1.AppTest` (headless, exposes state).
  Python-native: state = session_state; predict next session_state.
