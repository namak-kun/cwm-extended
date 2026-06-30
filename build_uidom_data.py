"""Emit data/uitrans_uidom.jsonl (CONTRACT format) from ui_dom apps, for run_uitrans_probe.

Varies app + #elements (scale) + preroll so states are non-trivial; verifies every prompt_src executes to
truth_state. The DOM-state render-FDM probe set.
"""
import argparse, json, random, copy
from ui_dom import APP_NAMES, APPS, real_dispatch, gen_one_event_src


def make(rng, app, n, preroll):
    dom = APPS[app]["init"](rng, n)
    for _ in range(rng.randint(0, preroll)):
        dom = real_dispatch(app, dom, APPS[app]["event"](rng, dom, n))
    ev = APPS[app]["event"](rng, dom, n)
    nxt = real_dispatch(app, dom, ev)
    return dom, ev, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_app", type=int, default=25)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--nmin", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--preroll", type=int, default=3)
    ap.add_argument("--apps", default=None, help="comma-list to restrict apps (default: all)")
    ap.add_argument("--out", default="data/uitrans_uidom.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows, dropped = [], 0
    use_apps = a.apps.split(",") if a.apps else APP_NAMES
    for app in use_apps:
        for _ in range(a.per_app):
            n = rng.randint(a.nmin, a.nmax)
            dom, ev, nxt = make(rng, app, n, a.preroll)
            src, entry = gen_one_event_src(app, dom, ev)
            ns = {}
            try:
                exec(src.replace("main()\n", "__o=main()\n", 1), ns)
                if ns["__o"] != nxt:
                    dropped += 1
                    continue
            except Exception:
                dropped += 1
                continue
            rows.append({"target": "uidom", "lang": "python", "prompt_src": src, "entry": entry,
                         "action": ev, "state_before": dom, "truth_state": nxt,
                         "source_app": f"ui_dom.{app} (render-state FDM, n={n})", "app": app, "n": n})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} rows (dropped {dropped}) -> {a.out}")
    print("by app:", dict(Counter(r["app"] for r in rows)))
    print("n range:", min(r["n"] for r in rows), "-", max(r["n"] for r in rows))


if __name__ == "__main__":
    main()
