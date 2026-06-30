"""Emit ui_tick app transitions in CONTRACT format (for cascade SFT training + eval)."""
import argparse, json, random, copy
from ui_tick import APP_NAMES, APPS, gen_one_event_src, real_dispatch, set_scale


def make(rng, app, preroll):
    s = APPS[app]["init"](rng)
    for _ in range(rng.randint(0, preroll)):
        s = real_dispatch(app, s, APPS[app]["events"](rng, s))
    e = APPS[app]["events"](rng, s)
    return s, e, real_dispatch(app, s, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", default="form,cart")
    ap.add_argument("--per_app", type=int, default=60)
    ap.add_argument("--seed", type=int, default=321)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--preroll", type=int, default=3)
    ap.add_argument("--out", default="data/uitrans_uitick.jsonl")
    a = ap.parse_args()
    set_scale(a.scale)
    rng = random.Random(a.seed)
    rows = []
    for app in a.apps.split(","):
        for _ in range(a.per_app):
            s, e, nxt = make(rng, app, a.preroll)
            src, entry = gen_one_event_src(app, s, e)
            ns = {}
            try:
                exec(src.replace("main()\n", "__o=main()\n", 1), ns)
                if ns["__o"] != nxt:
                    continue
            except Exception:
                continue
            rows.append({"target": "uitick", "lang": "python", "prompt_src": src, "entry": entry,
                         "action": e, "state_before": s, "truth_state": nxt,
                         "source_app": f"ui_tick.{app}", "app": app})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} -> {a.out}; by app: {dict(Counter(r['app'] for r in rows))}")


if __name__ == "__main__":
    main()
