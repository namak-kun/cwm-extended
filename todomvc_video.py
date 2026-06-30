"""REAL-APP video: free-roll the REAL TodoMVC reducer with CWM, render each predicted state -> a UI video.

Base CWM nails small TodoMVC (§34.6, 1.0), so this free-rolls a real web app's UI under a sequence of user
actions (add/toggle/delete/filter), predicting each next model-state via FULL-TRACE (CWM executes the real
reducer incl. cloneTodos/nextId helpers), and a real browser renders each into a TodoMVC-styled frame. The
strongest north-star demo: a model-generated video of a REAL app responding to input.
"""
from __future__ import annotations
import argparse, json, copy, os
from models.cwm_trace import CWMvLLM
from run_uitrans_probe import full_trace_preds, robust_parse
from run_gametick_abstract import _norm
from dom_render import render_many

REDUCER = open(os.path.join(os.path.dirname(__file__), "todomvc_reducer.js.txt")).read()

CSS = """
 body{font:16px system-ui;background:#f5f5f5;margin:0;padding:18px}
 h1{color:#b83f45;font-weight:200;text-align:center;margin:0 0 8px}
 .app{max-width:380px;margin:auto;background:#fff;box-shadow:0 2px 6px #0002}
 ul{list-style:none;margin:0;padding:0}
 li{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid #ededed;font-size:18px}
 li.done span{color:#aaa;text-decoration:line-through}
 input[type=checkbox]{width:18px;height:18px}
 footer{display:flex;gap:8px;align-items:center;padding:8px 14px;color:#777;font-size:13px}
 .f{padding:2px 7px;border:1px solid transparent;border-radius:4px}
 .f.sel{border-color:#b83f45}
"""


def render_todomvc(model):
    todos = model.get("todos", [])
    flt = model.get("filter", "all")
    vis = [t for t in todos if flt == "all" or (flt == "active") == (not t.get("completed"))]
    lis = []
    for t in vis:
        lis.append({"tag": "li", "class": (["done"] if t.get("completed") else []), "children": [
            {"tag": "input", "attrs": {"type": "checkbox", "checked": bool(t.get("completed"))}},
            {"tag": "span", "text": str(t.get("title", ""))}]})
    active = sum(1 for t in todos if not t.get("completed"))
    fbtns = [{"tag": "span", "class": (["f", "sel"] if flt == f else ["f"]), "text": f}
             for f in ["all", "active", "completed"]]
    return {"tag": "div", "attrs": {}, "children": [
        {"tag": "h1", "text": "todos"},
        {"tag": "div", "class": ["app"], "children": [
            {"tag": "ul", "id": "list", "children": lis},
            {"tag": "footer", "children": [{"tag": "span", "id": "count", "text": f"{active} left"}] + fbtns}]}]}


def step_program(state, action):
    return REDUCER + f'''
function main(){{  // << START_OF_TRACE
  let state = {json.dumps(state)};
  state = dispatch(state, {json.dumps(action)});
  return state;
}}
console.log(JSON.stringify(main()));
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out_dir", default="results/todomvc_video")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--stress", action="store_true",
                    help="longer/wider session to locate base free-roll drift onset on a real app")
    a = ap.parse_args()
    # a scripted user session on a REAL TodoMVC
    state = {"filter": "all", "todos": []}
    actions = [
        {"type": "add", "title": "buy milk"},
        {"type": "add", "title": "walk dog"},
        {"type": "toggle", "id": 1},
        {"type": "add", "title": "write report"},
        {"type": "setFilter", "filter": "active"},
        {"type": "toggle", "id": 2},
        {"type": "setFilter", "filter": "completed"},
        {"type": "clearCompleted"},
    ]
    if a.stress:  # longer/wider: up to ~6 todos, edits/deletes interleaved -> probe drift onset
        actions = [
            {"type": "add", "title": "buy milk"},
            {"type": "add", "title": "walk dog"},
            {"type": "add", "title": "write report"},
            {"type": "add", "title": "call alice"},
            {"type": "toggle", "id": 2},
            {"type": "add", "title": "pay rent"},
            {"type": "toggle", "id": 4},
            {"type": "edit", "id": 1, "title": "buy oat milk"},
            {"type": "setFilter", "filter": "active"},
            {"type": "add", "title": "book flight"},
            {"type": "toggle", "id": 6},
            {"type": "delete", "id": 3},
            {"type": "setFilter", "filter": "completed"},
            {"type": "toggle", "id": 4},
            {"type": "setFilter", "filter": "all"},
            {"type": "clearCompleted"},
        ]
    m = CWMvLLM(a.model_path, tp=a.tp, max_model_len=12288, lora_path=a.lora)
    items, gt = [("step0_init", render_todomvc(state))], copy.deepcopy(state)
    ok = []
    import subprocess
    for i, act in enumerate(actions):
        # ground truth via node (the real reducer)
        prog = step_program(state, act)
        out = subprocess.run(["node", "-e", prog], capture_output=True, text=True, timeout=30)
        gt = json.loads(out.stdout.strip().splitlines()[-1])
        # CWM free-roll prediction (full-trace executes the real reducer)
        # budget scales with state size: bigger todo lists -> longer cloneTodos/filter traces (§29.1 token-cap lesson)
        budget = max(3072, 1500 + 1100 * len(state.get("todos", [])))
        raw = full_trace_preds(m, [{"prompt_src": prog}], max_tokens=budget)[0]
        pred = robust_parse(raw)
        good = (isinstance(pred, dict) and _norm(pred) == _norm(gt))
        ok.append(bool(good))
        if not isinstance(pred, dict):
            pred = gt
        state = pred   # free-roll: feed own prediction
        lbl = f"step{i+1}_{act['type']}_{'OK' if good else 'X'}"
        items.append((lbl, render_todomvc(pred)))
    paths = render_many(items, a.out_dir, css=CSS, width=420, height=320)
    try:
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in paths]
        imgs[0].save(os.path.join(a.out_dir, "todomvc.gif"), save_all=True,
                     append_images=imgs[1:], duration=1100, loop=0)
        print(f"GIF -> {a.out_dir}/todomvc.gif", flush=True)
    except Exception as e:
        print("gif skip", e)
    print(f"[todomvc-video] per-step exact={ok} ({sum(ok)}/{len(ok)}) frames -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
