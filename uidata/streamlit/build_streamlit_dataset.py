from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "uitrans_streamlit.jsonl"

COUNTER_DISPATCH = r'''import copy, json

def dispatch(state, action):
    state = copy.deepcopy(state)
    t = action["type"]
    if t == "click_increment":
        state["count"] += state["step"]
    elif t == "click_decrement":
        state["count"] -= state["step"]
    elif t == "click_reset":
        state["count"] = 0
    elif t == "set_step":
        state["step_input"] = action["value"]
        state["step"] = state["step_input"]
    elif t == "set_label":
        state["label_input"] = action["value"]
        state["label"] = state["label_input"].strip()
    state["is_zero"] = (state["count"] == 0)
    return state
'''

FORM_DISPATCH = r'''import copy, json

def _validate(state):
    f = state["fields"]
    errors = {}
    if "@" not in f["email"] or "." not in f["email"]:
        errors["email"] = "invalid"
    if len(f["password"]) < 8:
        errors["password"] = "too_short"
    if f["confirm"] != f["password"]:
        errors["confirm"] = "mismatch"
    if not state["accepted_terms"]:
        errors["accepted_terms"] = "required"
    state["errors"] = errors
    state["can_submit"] = (len(errors) == 0)

def dispatch(state, action):
    state = copy.deepcopy(state)
    if action["type"] == "set_text":
        name = action["field"]
        state[f"input_{name}"] = action["value"]
        state["fields"][name] = state[f"input_{name}"]
    elif action["type"] == "set_checkbox" and action["field"] == "terms":
        state["input_terms"] = action["value"]
        state["accepted_terms"] = state["input_terms"]
    elif action["type"] == "set_checkbox" and action["field"] == "newsletter":
        state["input_newsletter"] = action["value"]
        state["newsletter"] = state["input_newsletter"]
    _validate(state)
    return state
'''

TODO_DISPATCH = r'''import copy, json

def _uid(n):
    return "00000000-0000-0000-0000-" + str(n).zfill(12)

def _sync_checkboxes(state):
    state["checkboxes"] = {todo["uid"]: bool(todo["is_done"]) for todo in state["todos"]}

def dispatch(state, action):
    state = copy.deepcopy(state)
    t = action["type"]
    if t == "add":
        state["new_item_text"] = action["text"]
        state["_uid_counter"] += 1
        state["todos"].append({"text": state["new_item_text"], "is_done": False, "uid": _uid(state["_uid_counter"])})
        state["new_item_text"] = ""
    elif t == "toggle":
        i = action["index"]
        new_value = action["value"]
        state["todos"][i]["is_done"] = new_value
    elif t == "delete":
        state["todos"].pop(action["index"])
    elif t == "delete_all_checked":
        state["todos"] = [todo for todo in state["todos"] if not todo["is_done"]]
    _sync_checkboxes(state)
    return state
'''

DISPATCH = {"counter": COUNTER_DISPATCH, "form": FORM_DISPATCH, "todo": TODO_DISPATCH}


def clean(v: Any) -> Any:
    if is_dataclass(v):
        d = asdict(v)
        d["uid"] = str(d["uid"])
        d["is_done"] = bool(getattr(v, "is_done", False))
        return {k: clean(d[k]) for k in sorted(d)}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): clean(v[k]) for k in sorted(v)}
    if hasattr(v, "hex") and v.__class__.__name__ == "UUID":
        return str(v)
    return v


def filtered(at: AppTest) -> dict:
    return clean(at.session_state.filtered_state)


def canon_counter(at: AppTest) -> dict:
    fs = filtered(at)
    keys = ["count", "step", "is_zero", "label", "step_input", "label_input"]
    return {k: fs[k] for k in keys}


def canon_form(at: AppTest) -> dict:
    fs = filtered(at)
    keys = ["fields", "errors", "can_submit", "accepted_terms", "newsletter", "input_email", "input_password", "input_confirm", "input_terms", "input_newsletter"]
    return {k: fs[k] for k in keys}


def canon_todo(at: AppTest) -> dict:
    fs = filtered(at)
    todos = fs["todos"]
    return {
        "_uid_counter": fs["_uid_counter"],
        "new_item_text": fs.get("new_item_text", ""),
        "todos": todos,
        "checkboxes": {todo["uid"]: bool(todo["is_done"]) for todo in todos},
    }


def prompt_src(app: str, state: dict, action: dict) -> str:
    return DISPATCH[app] + "\n" + f'''def main():  # << START_OF_TRACE
    state = {repr(state)}
    state = dispatch(state, {repr(action)})
    return state

if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":")))
'''


def run_prompt(src: str) -> dict:
    proc = subprocess.run(["python3", "-c", src], text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def add_row(rows: list[dict], app: str, at: AppTest, action: dict, apply_fn) -> bool:
    before = {"counter": canon_counter, "form": canon_form, "todo": canon_todo}[app](at)
    apply_fn()
    truth = {"counter": canon_counter, "form": canon_form, "todo": canon_todo}[app](at)
    src = prompt_src(app, before, action)
    predicted = run_prompt(src)
    if predicted != truth:
        return False
    rows.append({
        "target": "streamlit",
        "lang": "python",
        "prompt_src": src,
        "entry": "main",
        "action": action,
        "state_before": before,
        "truth_state": truth,
        "source_app": f"uidata/streamlit/{app}_app.py",
    })
    return True


def gen_counter(rows):
    at = AppTest.from_file(str(HERE / "counter_app.py")).run()
    actions = [
        {"type":"click_increment"}, {"type":"click_increment"}, {"type":"set_step", "value":3},
        {"type":"click_increment"}, {"type":"click_decrement"}, {"type":"set_label", "value":" clicks "},
        {"type":"click_reset"}, {"type":"set_step", "value":5}, {"type":"click_decrement"},
        {"type":"set_label", "value":"negative"}, {"type":"click_increment"}, {"type":"click_increment"},
        {"type":"set_step", "value":2}, {"type":"click_decrement"}, {"type":"click_reset"},
        {"type":"set_step", "value":1}, {"type":"click_increment"}, {"type":"set_label", "value":"done"},
    ]
    for action in actions:
        if action["type"] == "click_increment":
            fn = lambda at=at: at.button[0].click().run()
        elif action["type"] == "click_decrement":
            fn = lambda at=at: at.button[1].click().run()
        elif action["type"] == "click_reset":
            fn = lambda at=at: at.button[2].click().run()
        elif action["type"] == "set_step":
            fn = lambda action=action, at=at: at.number_input[0].set_value(action["value"]).run()
        else:
            fn = lambda action=action, at=at: at.text_input[0].set_value(action["value"]).run()
        add_row(rows, "counter", at, action, fn)


def gen_form(rows):
    at = AppTest.from_file(str(HERE / "form_app.py")).run()
    actions = [
        {"type":"set_text", "field":"email", "value":"bad"},
        {"type":"set_text", "field":"password", "value":"short"},
        {"type":"set_checkbox", "field":"terms", "value":True},
        {"type":"set_text", "field":"email", "value":"user@example.com"},
        {"type":"set_text", "field":"password", "value":"longenough"},
        {"type":"set_text", "field":"confirm", "value":"nope"},
        {"type":"set_text", "field":"confirm", "value":"longenough"},
        {"type":"set_checkbox", "field":"newsletter", "value":True},
        {"type":"set_checkbox", "field":"terms", "value":False},
        {"type":"set_text", "field":"email", "value":"x@y"},
        {"type":"set_text", "field":"email", "value":"x@y.org"},
        {"type":"set_checkbox", "field":"terms", "value":True},
        {"type":"set_text", "field":"password", "value":"abcdefgh"},
        {"type":"set_text", "field":"confirm", "value":"abcdefgh"},
        {"type":"set_checkbox", "field":"newsletter", "value":False},
        {"type":"set_text", "field":"email", "value":""},
        {"type":"set_text", "field":"email", "value":"a@b.com"},
        {"type":"set_checkbox", "field":"terms", "value":False},
    ]
    field_to_idx = {"email":0, "password":1, "confirm":2}
    for action in actions:
        if action["type"] == "set_text":
            idx = field_to_idx[action["field"]]
            fn = lambda idx=idx, action=action, at=at: at.text_input[idx].set_value(action["value"]).run()
        else:
            idx = 0 if action["field"] == "terms" else 1
            fn = lambda idx=idx, action=action, at=at: at.checkbox[idx].set_value(action["value"]).run()
        add_row(rows, "form", at, action, fn)


def gen_todo(rows):
    at = AppTest.from_file(str(HERE / "demo_todo_app.py")).run()
    add_texts = ["Read docs", "File taxes", "Call Sam", "Refill tea", "Plan trip", "Buy stamps", "Clean desk", "Mail card", "Book train", "Water basil", "Pack bag", "Charge phone"]
    actions = []
    for text in add_texts[:8]:
        actions.append({"type":"add", "text":text})
    actions += [
        {"type":"toggle", "index":0}, {"type":"toggle", "index":3}, {"type":"toggle", "index":7},
        {"type":"delete", "index":1}, {"type":"add", "text":add_texts[8]}, {"type":"toggle", "index":4},
        {"type":"delete_all_checked"}, {"type":"add", "text":add_texts[9]}, {"type":"add", "text":add_texts[10]},
        {"type":"toggle", "index":0}, {"type":"toggle", "index":5}, {"type":"delete", "index":2},
        {"type":"add", "text":add_texts[11]}, {"type":"toggle", "index":6}, {"type":"delete_all_checked"},
        {"type":"toggle", "index":0}, {"type":"toggle", "index":0}, {"type":"delete", "index":0},
    ]
    for action in actions:
        # Fill toggle action value from the current rendered checkbox/app state.
        if action["type"] == "add":
            fn = lambda action=action, at=at: (at.text_input[0].set_value(action["text"]), at.button[0].click().run())
        elif action["type"] == "toggle":
            idx = min(action["index"], len(at.checkbox) - 1)
            current = bool(at.checkbox[idx].value)
            action = dict(action, index=idx, value=(not current))
            fn = lambda idx=idx, action=action, at=at: at.checkbox[idx].set_value(action["value"]).run()
        elif action["type"] == "delete":
            idx = min(action["index"], len(at.checkbox) - 1)
            action = dict(action, index=idx)
            fn = lambda idx=idx, at=at: at.button[1 + idx].click().run()
        else:
            fn = lambda at=at: at.button[-1].click().run()
        add_row(rows, "todo", at, action, fn)


def main():
    rows: list[dict] = []
    gen_counter(rows)
    gen_form(rows)
    gen_todo(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": len(rows), "out": str(OUT)}, sort_keys=True))

if __name__ == "__main__":
    main()
