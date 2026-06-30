"""UI-tick generator — the GUI analog of game_tick.py (REPORT §33 / renderer axis MVE).

A "UI app" is expressed as a deterministic Python update function `dispatch(state, event) -> state`
over an app MODEL-STATE dict (the §9 "clean ideal": predict the small model-state, not the full DOM;
a separate render(state)->DOM/pixels handles display). This mirrors game_tick EXACTLY:

  game_tick                         ui_tick
  -----------------------------     ---------------------------------
  state = {player, enemies}         state = app model-state (e.g. {items, filter, ...})
  action in {U,D,L,R}               event  = {"type": ..., ...}  (click/input/...)
  step(state, action) -> state'     dispatch(state, event) -> state'
  real_step (oracle)                real_dispatch (oracle, exact via Python exec)

Difficulty gradient (to locate WHERE base CWM crumbles, like the game multi-entity salience):
  counter  - trivial, single salient mutation                     (expect base OK)
  todo     - add/toggle/delete/filter + active-count recompute    (moderate)
  form     - per-keystroke CASCADING validation -> can_submit     (buried side-effects)
  cart     - change one qty -> recompute ALL line totals+subtotal (multi-item salience analog)

Oracle = real Python execution of the (single source-of-truth) dispatch source. Renderable: each app
has render(state)->DOM-JSON for the later pixel path (browser rasterizes), but the PREDICTION target
is the model-state (small, clean, code-grounded).
"""
from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Each app = a single SOURCE-OF-TRUTH `dispatch` source string (traced by CWM AND
# exec'd for the oracle), plus init_state / event-sampler / render.
# ---------------------------------------------------------------------------

_COUNTER_SRC = '''def dispatch(state, event):
    t = event["type"]
    if t == "inc":   state["count"] += state["step"]
    elif t == "dec": state["count"] -= state["step"]
    elif t == "reset": state["count"] = 0
    elif t == "set_step": state["step"] = event["value"]
    state["is_zero"] = (state["count"] == 0)
    return state
'''

_TODO_SRC = '''def dispatch(state, event):
    t = event["type"]
    if t == "add":
        state["items"].append({"id": state["next_id"], "text": event["value"], "done": False})
        state["next_id"] += 1
    elif t == "toggle":
        for it in state["items"]:
            if it["id"] == event["id"]:
                it["done"] = not it["done"]
    elif t == "delete":
        state["items"] = [it for it in state["items"] if it["id"] != event["id"]]
    elif t == "clear_completed":
        state["items"] = [it for it in state["items"] if not it["done"]]
    elif t == "set_filter":
        state["filter"] = event["value"]
    state["active_count"] = sum(1 for it in state["items"] if not it["done"])
    state["total_count"] = len(state["items"])
    return state
'''

_FORM_SRC = '''def dispatch(state, event):
    if event["type"] == "input":
        state["fields"][event["target"]] = event["value"]
    f = state["fields"]
    errors = {}
    if "@" not in f["email"] or "." not in f["email"]:
        errors["email"] = "invalid"
    if len(f["password"]) < 8:
        errors["password"] = "too_short"
    if f["confirm"] != f["password"]:
        errors["confirm"] = "mismatch"
    state["errors"] = errors
    state["can_submit"] = (len(errors) == 0)
    return state
'''

_CART_SRC = '''def dispatch(state, event):
    if event["type"] == "set_qty":
        for it in state["items"]:
            if it["id"] == event["id"]:
                it["qty"] = event["value"]
    elif event["type"] == "remove":
        state["items"] = [it for it in state["items"] if it["id"] != event["id"]]
    subtotal = 0
    for it in state["items"]:
        it["line"] = it["qty"] * it["price"]
        subtotal += it["line"]
    state["subtotal"] = subtotal
    state["n_units"] = sum(it["qty"] for it in state["items"])
    return state
'''


def _counter_init(rng):
    return {"count": rng.randint(0, 5), "step": rng.choice([1, 2, 3]), "is_zero": False}

def _counter_events(rng, state):
    evs = [{"type": "inc"}, {"type": "dec"}, {"type": "reset"},
           {"type": "set_step", "value": rng.choice([1, 2, 3])}]
    return rng.choice(evs)

def _counter_render(s):
    return {"tag": "div", "children": [
        {"tag": "span", "id": "count", "text": str(s["count"])},
        {"tag": "button", "id": "inc", "text": "+"},
        {"tag": "button", "id": "dec", "text": "-"}]}


_WORDS = ["milk", "eggs", "bread", "tea", "rice", "soap"]

_SCALE = {"n": 1}  # difficulty multiplier on item counts (the UI analog of K enemies)

def set_scale(n):
    _SCALE["n"] = max(1, int(n))

def _todo_init(rng):
    lo, hi = 1 * _SCALE["n"], 3 * _SCALE["n"]
    n = rng.randint(lo, hi)
    items = [{"id": i, "text": rng.choice(_WORDS), "done": rng.random() < 0.4} for i in range(n)]
    st = {"items": items, "next_id": n, "filter": "all", "active_count": 0, "total_count": n}
    return st

def _todo_events(rng, state):
    ids = [it["id"] for it in state["items"]]
    evs = [{"type": "add", "value": rng.choice(_WORDS)},
           {"type": "set_filter", "value": rng.choice(["all", "active", "completed"])},
           {"type": "clear_completed"}]
    if ids:
        evs.append({"type": "toggle", "id": rng.choice(ids)})
        evs.append({"type": "delete", "id": rng.choice(ids)})
    return rng.choice(evs)

def _todo_render(s):
    lis = [{"tag": "li", "class": (["done"] if it["done"] else []), "text": it["text"]} for it in s["items"]]
    return {"tag": "div", "children": [
        {"tag": "ul", "id": "list", "children": lis},
        {"tag": "span", "id": "active", "text": str(s["active_count"])}]}


def _form_init(rng):
    return {"fields": {"email": rng.choice(["a@b.com", "bad", "x@y", ""]),
                       "password": rng.choice(["short", "longenough1", ""]),
                       "confirm": ""},
            "errors": {}, "can_submit": False}

def _form_events(rng, state):
    tgt = rng.choice(["email", "password", "confirm"])
    vals = {"email": ["a@b.com", "bad", "user@site.org", "x@y"],
            "password": ["longenough1", "short", "abcdefgh"],
            "confirm": ["longenough1", "short", "abcdefgh"]}
    return {"type": "input", "target": tgt, "value": rng.choice(vals[tgt])}

def _form_render(s):
    ch = []
    for fld in ["email", "password", "confirm"]:
        ch.append({"tag": "input", "id": fld, "attrs": {"value": s["fields"][fld]}})
        if fld in s["errors"]:
            ch.append({"tag": "span", "class": ["err"], "text": s["errors"][fld]})
    ch.append({"tag": "button", "id": "submit", "attrs": {"disabled": (not s["can_submit"])}})
    return {"tag": "form", "children": ch}


def _cart_init(rng):
    n = rng.randint(2 * _SCALE["n"], 3 * _SCALE["n"])
    items = [{"id": i, "name": rng.choice(_WORDS), "price": rng.choice([2, 3, 5]),
              "qty": rng.randint(1, 3), "line": 0} for i in range(n)]
    return {"items": items, "subtotal": 0, "n_units": 0}

def _cart_events(rng, state):
    ids = [it["id"] for it in state["items"]]
    evs = []
    if ids:
        evs.append({"type": "set_qty", "id": rng.choice(ids), "value": rng.randint(0, 4)})
        evs.append({"type": "remove", "id": rng.choice(ids)})
    return rng.choice(evs) if evs else {"type": "set_qty", "id": 0, "value": 1}

def _cart_render(s):
    rows = [{"tag": "tr", "children": [
        {"tag": "td", "text": it["name"]}, {"tag": "td", "text": str(it["qty"])},
        {"tag": "td", "text": str(it["line"])}]} for it in s["items"]]
    return {"tag": "table", "children": rows + [
        {"tag": "tr", "id": "subtotal", "text": str(s["subtotal"])}]}


APPS = {
    "counter": {"src": _COUNTER_SRC, "init": _counter_init, "events": _counter_events, "render": _counter_render},
    "todo":    {"src": _TODO_SRC,    "init": _todo_init,    "events": _todo_events,    "render": _todo_render},
    "form":    {"src": _FORM_SRC,    "init": _form_init,    "events": _form_events,    "render": _form_render},
    "cart":    {"src": _CART_SRC,    "init": _cart_init,    "events": _cart_events,    "render": _cart_render},
}
APP_NAMES = list(APPS)


def gen_one_event_src(app: str, state: dict, event: dict) -> tuple[str, str]:
    """One-event program: start at `state`, apply a single `event`, return the new model-state.
    The FDM unit (mirrors game_tick.gen_one_tick_src)."""
    src = APPS[app]["src"] + f'''
def main():  # << START_OF_TRACE
    state = {state!r}
    state = dispatch(state, {event!r})
    return state

main()
'''
    return src, "main"


def gen_ui_episode(rng: random.Random, app: str = None, t_events: int = None, return_meta=False):
    """Multi-event episode (mirrors gen_game_tick): apply T events, return final model-state."""
    if app is None:
        app = rng.choice(APP_NAMES)
    T = t_events if t_events is not None else rng.randint(2, 3)
    init = APPS[app]["init"](rng)
    # sample events against the EVOLVING true state so ids/targets stay valid
    import copy
    st = copy.deepcopy(init)
    events = []
    for _ in range(T):
        ev = APPS[app]["events"](rng, st)
        events.append(ev)
        st = real_dispatch(app, st, ev)
    src = APPS[app]["src"] + f'''
def main():  # << START_OF_TRACE
    state = {init!r}
    for ev in {events!r}:
        state = dispatch(state, ev)
    return state

main()
'''
    if return_meta:
        return src, "main", {"app": app, "init": init, "events": events, "T": T}
    return src, "main"


def real_dispatch(app: str, state: dict, event: dict) -> dict:
    """Ground-truth single transition (exact, via Python exec of the app's own dispatch)."""
    import copy
    ns: dict = {}
    exec(APPS[app]["src"], ns)
    return ns["dispatch"](copy.deepcopy(state), event)


def ground_truth_states(app: str, meta: dict) -> list[dict]:
    """Per-event ground-truth model-states for an episode (for the multi-event eval)."""
    import copy
    state = copy.deepcopy(meta["init"])
    out = []
    for ev in meta["events"]:
        state = real_dispatch(app, state, ev)
        out.append(copy.deepcopy(state))
    return out


def render_dom(app: str, state: dict) -> dict:
    """Model-state -> canonical DOM-JSON (the later pixel path; browser rasterizes this)."""
    return APPS[app]["render"](state)


if __name__ == "__main__":
    rng = random.Random(0)
    for app in APP_NAMES:
        src, entry, meta = gen_ui_episode(rng, app=app, return_meta=True)
        gts = ground_truth_states(app, meta)
        print(f"--- {app}: T={meta['T']} final={gts[-1]} ---")
        # sanity: one-shot oracle matches
        s0 = meta["init"]; e0 = meta["events"][0]
        assert real_dispatch(app, s0, e0) == gts[0], f"{app} oracle mismatch"
    print("ui_tick.py OK: all apps deterministic, one-event oracle consistent")
