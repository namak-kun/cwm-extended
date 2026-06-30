"""DOM-state FDM generator (renderer axis, REPORT §35) — the render-state world-model unit.

Unlike ui_tick (predict the app MODEL-state), here the STATE **is** the canonical DOM tree and the handler
mutates the DOM directly (like real vanilla-DOM apps). This is the render-state CWM should predict; a real
browser then rasterizes DOM->pixels (the pixel boundary). Predicting DOM transitions is the FDM unit closest
to the user's pixel north-star while staying symbolic/testable.

Design choices that keep it ONE-SHOT step-over-able (avoid the todomvc helper-delegation trap):
  - DOM is one container with a flat `children` list (tabs/items/panels) -> dispatch needs a SINGLE loop, no
    recursion, no helper calls -> self-contained dispatch CWM can one-shot.
  - Apps include MULTI-ELEMENT DOM cascades (select a tab -> update aria-selected on ALL tabs + hidden on ALL
    panels) = the DOM-space analog of the game multi-entity salience.

Apps: tabs, accordion (single-open), togglelist (+ live active-count text), counter (text-node update).
Oracle = exact Python exec of the dispatch source (single source of truth). Emits the data/uitrans_uidom.jsonl
CONTRACT so run_uitrans_probe.py consumes it directly.
"""
from __future__ import annotations
import random

_TABS_SRC = '''def dispatch(dom, event):
    sel = event["id"]
    idx = sel.split("-")[1]
    for node in dom["children"]:
        nid = node.get("id", "")
        if nid.startswith("tab-"):
            node["attrs"]["aria-selected"] = "true" if nid == sel else "false"
        elif nid.startswith("panel-"):
            node["attrs"]["hidden"] = (nid != "panel-" + idx)
    return dom
'''

_ACCORDION_SRC = '''def dispatch(dom, event):
    sel = event["id"]
    n_sel = sel.split("-")[1]
    was_open = "false"
    for node in dom["children"]:
        if node.get("id", "") == sel:
            was_open = node["attrs"]["aria-expanded"]
    new_open = "true" if was_open == "false" else "false"
    for node in dom["children"]:
        nid = node.get("id", "")
        if nid.startswith("hd-"):
            node["attrs"]["aria-expanded"] = new_open if nid == sel else "false"
        elif nid.startswith("pan-"):
            n = nid.split("-")[1]
            node["attrs"]["hidden"] = not (n == n_sel and new_open == "true")
    return dom
'''

_TOGGLELIST_SRC = '''def dispatch(dom, event):
    active = 0
    for node in dom["children"]:
        nid = node.get("id", "")
        if nid.startswith("item-"):
            if nid == event["id"] and event["type"] == "toggle":
                node["attrs"]["data-done"] = "false" if node["attrs"]["data-done"] == "true" else "true"
            if node["attrs"]["data-done"] == "false":
                active += 1
        elif nid == "count":
            node["text"] = ""
    for node in dom["children"]:
        if node.get("id", "") == "count":
            node["text"] = str(active)
    return dom
'''

_COUNTER_SRC = '''def dispatch(dom, event):
    val = 0
    for node in dom["children"]:
        if node.get("id", "") == "value":
            val = int(node["text"])
    if event["type"] == "inc": val += 1
    elif event["type"] == "dec": val -= 1
    elif event["type"] == "reset": val = 0
    for node in dom["children"]:
        if node.get("id", "") == "value":
            node["text"] = str(val)
        elif node.get("id", "") == "parity":
            node["text"] = "even" if val % 2 == 0 else "odd"
    return dom
'''


def _tabs_init(rng, n):
    sel = rng.randrange(n)
    ch = []
    for i in range(n):
        ch.append({"tag": "button", "id": f"tab-{i}", "attrs": {"aria-selected": "true" if i == sel else "false"}, "text": f"T{i}"})
    for i in range(n):
        ch.append({"tag": "div", "id": f"panel-{i}", "attrs": {"hidden": i != sel}, "text": f"content {i}"})
    return {"tag": "div", "attrs": {}, "children": ch}

def _tabs_event(rng, dom, n):
    return {"type": "select", "id": f"tab-{rng.randrange(n)}"}


def _accordion_init(rng, n):
    opened = rng.randrange(n + 1)  # n => none open
    ch = []
    for i in range(n):
        exp = (i == opened)
        ch.append({"tag": "button", "id": f"hd-{i}", "attrs": {"aria-expanded": "true" if exp else "false"}, "text": f"H{i}"})
        ch.append({"tag": "div", "id": f"pan-{i}", "attrs": {"hidden": not exp}, "text": f"body {i}"})
    return {"tag": "div", "attrs": {}, "children": ch}

def _accordion_event(rng, dom, n):
    return {"type": "click", "id": f"hd-{rng.randrange(n)}"}


def _togglelist_init(rng, n):
    ch = []
    active = 0
    for i in range(n):
        done = rng.random() < 0.4
        if not done:
            active += 1
        ch.append({"tag": "li", "id": f"item-{i}", "attrs": {"data-done": "true" if done else "false"}, "text": f"task{i}"})
    ch.append({"tag": "span", "id": "count", "attrs": {}, "text": str(active)})
    return {"tag": "ul", "attrs": {}, "children": ch}

def _togglelist_event(rng, dom, n):
    return {"type": "toggle", "id": f"item-{rng.randrange(n)}"}


def _counter_init(rng, n):
    v = rng.randint(0, 9)
    return {"tag": "div", "attrs": {}, "children": [
        {"tag": "span", "id": "value", "attrs": {}, "text": str(v)},
        {"tag": "span", "id": "parity", "attrs": {}, "text": "even" if v % 2 == 0 else "odd"},
        {"tag": "button", "id": "inc", "attrs": {}, "text": "+"}]}

def _counter_event(rng, dom, n):
    return {"type": rng.choice(["inc", "dec", "reset"])}


APPS = {
    "tabs":       {"src": _TABS_SRC,       "init": _tabs_init,       "event": _tabs_event},
    "accordion":  {"src": _ACCORDION_SRC,  "init": _accordion_init,  "event": _accordion_event},
    "togglelist": {"src": _TOGGLELIST_SRC, "init": _togglelist_init, "event": _togglelist_event},
    "counter":    {"src": _COUNTER_SRC,    "init": _counter_init,    "event": _counter_event},
}
APP_NAMES = list(APPS)


def real_dispatch(app, dom, event):
    import copy
    ns = {}
    exec(APPS[app]["src"], ns)
    return ns["dispatch"](copy.deepcopy(dom), event)


def gen_one_event_src(app, dom, event):
    src = APPS[app]["src"] + f'''
def main():  # << START_OF_TRACE
    dom = {dom!r}
    dom = dispatch(dom, {event!r})
    return dom

main()
'''
    return src, "main"
