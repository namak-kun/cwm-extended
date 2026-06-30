"""Battery 2: Lua (game-scripting) + HTML/DOM state.

Lua: the dominant game-scripting language (Solarus, LOVE, embedded engines).
HTML/DOM: HTML itself is static markup with no execution state. The dynamic
"state" of a page is the DOM TREE, mutated by JS. So we test:
  - pure-JS mock DOM (self-contained: can CWM track tree-structured state?)
  - real DOM via jsdom (does CWM model the real DOM API or hit the native wall?)

Metric = predict entry function's final RETURN value; ground truth = run it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

PROGRAMS = []

# ---- Lua: game-logic flavored -------------------------------------------------
PROGRAMS.append(("Lua_numeric", "lua", '''function compute()  -- << START_OF_TRACE
    local sum = 0
    for i = 1, 5 do
        sum = sum + i * i
    end
    return sum
end
print(compute())
'''))

PROGRAMS.append(("Lua_entity_update", "lua", '''function update()  -- << START_OF_TRACE
    local player = {x = 0, y = 0, hp = 100}
    local moves = {"right", "right", "up", "down", "right"}
    for _, m in ipairs(moves) do
        if m == "right" then player.x = player.x + 1
        elseif m == "left" then player.x = player.x - 1
        elseif m == "up" then player.y = player.y + 1
        elseif m == "down" then player.y = player.y - 1 end
    end
    return player.x * 10 + player.y
end
print(update())
'''))

PROGRAMS.append(("Lua_table_closure", "lua", '''function make_counter()
    local n = 0
    return function() n = n + 1; return n end
end
function compute()  -- << START_OF_TRACE
    local c = make_counter()
    local total = 0
    for i = 1, 4 do
        total = total + c()
    end
    return total
end
print(compute())
'''))

# ---- HTML/DOM: pure-JS mock DOM (no deps) -----------------------------------
PROGRAMS.append(("DOM_mock_tree", "js", '''function render() {  // << START_OF_TRACE
    let root = { tag: "ul", children: [] };
    let items = ["a", "bb", "ccc", "dddd"];
    for (let it of items) {
        root.children.push({ tag: "li", text: it });
    }
    let totalLen = 0;
    for (let c of root.children) {
        totalLen += c.text.length;
    }
    return root.children.length * 100 + totalLen;
}
console.log(render());
'''))

PROGRAMS.append(("DOM_mock_mutate", "js", '''function app() {  // << START_OF_TRACE
    let state = { count: 0, items: [] };
    let clicks = ["inc", "inc", "add", "inc", "add"];
    for (let ev of clicks) {
        if (ev === "inc") {
            state.count += 1;
        } else if (ev === "add") {
            state.items.push("item-" + state.count);
        }
    }
    return state.count * 10 + state.items.length;
}
console.log(app());
'''))

# ---- HTML/DOM: REAL DOM via jsdom -------------------------------------------
JSDOM_PATH = os.path.join(os.path.dirname(__file__), "jsdeps", "node_modules")
PROGRAMS.append(("DOM_jsdom_real", "jsdom", '''const { JSDOM } = require("jsdom");
function compute() {  // << START_OF_TRACE
    const dom = new JSDOM('<!DOCTYPE html><ul id="list"></ul>');
    const doc = dom.window.document;
    const ul = doc.getElementById("list");
    for (let i = 1; i <= 4; i++) {
        const li = doc.createElement("li");
        li.textContent = "item" + i;
        ul.appendChild(li);
    }
    return ul.children.length * 10 + ul.textContent.length;
}
console.log(compute());
'''))


def true_output(lang: str, src: str):
    with tempfile.TemporaryDirectory() as d:
        try:
            if lang == "lua":
                f = os.path.join(d, "p.lua"); open(f, "w").write(src)
                r = subprocess.run(["lua5.4", f], capture_output=True, text=True, timeout=60)
            elif lang == "js":
                f = os.path.join(d, "p.js"); open(f, "w").write(src)
                r = subprocess.run(["node", f], capture_output=True, text=True, timeout=60)
            elif lang == "jsdom":
                f = os.path.join(d, "p.js"); open(f, "w").write(src)
                env = dict(os.environ, NODE_PATH=JSDOM_PATH)
                r = subprocess.run(["node", f], capture_output=True, text=True, timeout=120, env=env)
            else:
                return None
            out = r.stdout.strip().splitlines()
            return int(out[-1]) if out and out[-1].strip().lstrip("-").isdigit() else (r.stdout.strip() or r.stderr.strip()[:80])
        except (subprocess.TimeoutExpired, ValueError):
            return None


def main_run(model_path, tp=4, dump=False):
    from models.cwm_trace import (CWMvLLM, Event, CALL_SEP, build_prompt, parse_full_trace, resolve_locals)
    from run_ood import cwm_final_return

    t0 = time.time()
    truths = {nm: true_output(lang, src) for nm, lang, src in PROGRAMS}
    print("ground truths:", truths, flush=True)

    # For CWM we strip the jsdom require line's noise but keep the source as-is.
    m = CWMvLLM(model_path, tp=tp, max_model_len=8192)
    print("== CWM loaded ==", flush=True)

    prompts = [build_prompt(m, src, [], force_event=Event.CALL) for _, _, src in PROGRAMS]
    caps = [2200] * len(PROGRAMS)
    gens = m.gen_full_trace_batch(prompts, caps)

    results = {}
    for (nm, lang, src), gen in zip(PROGRAMS, gens):
        pred = parse_full_trace(m, [CALL_SEP] + gen)
        cv = cwm_final_return(pred)
        tv = truths[nm]
        ok = (cv == tv) if isinstance(tv, int) else None
        results[nm] = {"lang": lang, "true": tv, "cwm": cv, "match": ok, "frames": len(pred)}
        print(f"[{nm:20} {lang:6}] true={tv}  cwm={cv}  MATCH={ok}  frames={len(pred)}", flush=True)
        if dump:
            for f in pred[:40]:
                print(f"    {f.event.name:7} {f.source_line.strip()[:48]:48} {resolve_locals(f)} arg={f.arg}")
            print()

    json.dump({"model": model_path, "results": results, "elapsed_sec": round(time.time()-t0, 1)},
              open("results/cwm_lua_dom.json", "w"), indent=2)
    print(f"\nsaved -> results/cwm_lua_dom.json ({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    dump = "--dump" in sys.argv
    main_run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 4, dump=dump)
