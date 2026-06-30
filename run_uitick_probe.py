"""Base-CWM one-shot UI-transition probe (renderer-axis MVE, REPORT §33).

The GUI analog of the game step-over probe (which gave base 0.017 -> SFT 0.69). For each
(model-state s, event e), force CWM to predict dispatch(s,e) in ONE shot (step over dispatch's
interior) and compare the predicted next model-state to the exact Python oracle.

Metrics:
  exact_match : predicted next-state == oracle next-state (canonical)         <- the key number
  field_f1    : graded leaf-level precision/recall over (path,value) pairs    <- partial credit
                (so a single buried wrong field is VISIBLE, not masked - the §29.1 lesson)
Per-app breakdown across the difficulty gradient (counter < todo < form < cart).
Hypothesis: base CWM crumbles on the cascading/multi-item apps (form/cart), establishing the gap.
One adapter per process (vLLM LoRA-switch bug).
"""
from __future__ import annotations

import argparse
import json
import random
from statistics import mean
from collections import defaultdict

from vllm import TokensPrompt
from models.cwm_trace import CWMvLLM, Event, FRAME_SEP, EOS, build_prompt, parse_frame
from run_gametick_abstract import _parse_state, _norm
from ui_tick import APP_NAMES, APPS, gen_one_event_src, real_dispatch, set_scale


def one_event_stepover(m, items, max_frames=10, max_tokens=1024, lora_kwargs=None, return_raw=False):
    """Force step-over of dispatch(); return each item's one-shot predicted next model-state (dict|None).
    If return_raw, return (pred, raw_arg_str) so the caller can robustly re-parse non-Python outputs."""
    sp = m.SP(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[FRAME_SEP, EOS])
    lk = lora_kwargs or {}
    st = [{"src": it["src"], "frames": [], "force": Event.CALL, "done": False, "pred": None, "raw": None}
          for it in items]
    for _ in range(max_frames):
        active = [s for s in st if not s["done"]]
        if not active:
            break
        prompts = [build_prompt(m, s["src"], s["frames"], force_event=s["force"]) for s in active]
        outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False, **lk)
        for s, o in zip(active, outs):
            prev_force = s["force"]
            f = parse_frame(m, list(o.outputs[0].token_ids) + [FRAME_SEP],
                            forced_event=s["force"], prev=s["frames"][-1] if s["frames"] else None)
            if f is None:
                s["done"] = True
                continue
            s["frames"].append(f)
            # STRUCTURAL capture: the RETURN we just forced (step-over of dispatch) IS the prediction,
            # regardless of whether its arg parses (parse it leniently; keep the raw for robust re-parse).
            if f.event == Event.RETURN and prev_force == Event.RETURN:
                s["raw"] = f.arg
                s["pred"] = _parse_state(f.arg)
                s["done"] = True
                continue
            if f.event == Event.CALL and len(s["frames"]) > 1:
                s["force"] = Event.RETURN          # descend into dispatch -> force its return (step-over)
            else:
                s["force"] = None
            # fallback capture: a dict-looking RETURN even if we didn't force it (e.g. main's own return)
            if f.event == Event.RETURN:
                a = _parse_state(f.arg)
                if isinstance(a, dict) and s["pred"] is None:
                    s["pred"], s["raw"] = a, f.arg
                    s["done"] = True
    if return_raw:
        return [(s["pred"], s.get("raw")) for s in st]
    return [s["pred"] for s in st]


def _flatten(obj, path=""):
    """Leaf (path,value) pairs for graded field-F1; lists indexed positionally."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{path}[{i}]"))
    else:
        out[path] = obj
    return out


def field_f1(pred, truth):
    if not isinstance(pred, dict):
        return 0.0
    P, T = _flatten(_norm(pred)), _flatten(_norm(truth))
    if not T:
        return 1.0 if not P else 0.0
    inter = sum(1 for k, v in T.items() if k in P and P[k] == v)
    prec = inter / max(len(P), 1)
    rec = inter / len(T)
    return 0.0 if (prec + rec) == 0 else round(2 * prec * rec / (prec + rec), 4)


def make_transition(rng, app, preroll_max=3):
    """init -> preroll k random oracle events -> a current state s; then sample one event e."""
    s = APPS[app]["init"](rng)
    for _ in range(rng.randint(0, preroll_max)):
        s = real_dispatch(app, s, APPS[app]["events"](rng, s))
    e = APPS[app]["events"](rng, s)
    s_next = real_dispatch(app, s, e)
    return s, e, s_next


def run(model_path, tp, n, seed, lora, out, preroll=3, scale=1):
    set_scale(scale)
    rng = random.Random(seed)
    samples = []
    per_app = max(1, n // len(APP_NAMES))
    for app in APP_NAMES:
        for _ in range(per_app):
            s, e, s_next = make_transition(rng, app, preroll)
            src, _ = gen_one_event_src(app, s, e)
            samples.append({"app": app, "s": s, "e": e, "s_next": s_next, "src": src})

    m = CWMvLLM(model_path, tp=tp, max_model_len=8192, lora_path=lora)
    print(f"== CWM {'+LoRA '+lora if lora else '(BASE)'} | UI one-shot transition probe n={len(samples)} ==", flush=True)
    preds = one_event_stepover(m, samples, lora_kwargs=m._gen_kwargs())

    by_app_exact, by_app_f1 = defaultdict(list), defaultdict(list)
    examples = []
    for smp, pred in zip(samples, preds):
        ex = (isinstance(pred, dict) and _norm(pred) == _norm(smp["s_next"]))
        f1 = field_f1(pred, smp["s_next"])
        by_app_exact[smp["app"]].append(1.0 if ex else 0.0)
        by_app_f1[smp["app"]].append(f1)
        if len(examples) < 8:
            examples.append({"app": smp["app"], "event": smp["e"], "exact": ex, "f1": f1,
                             "pred": pred, "truth": smp["s_next"]})

    res = {
        "model": model_path, "lora": lora, "n": len(samples), "seed": seed,
        "exact_match": round(mean([v for a in by_app_exact.values() for v in a]), 4),
        "field_f1": round(mean([v for a in by_app_f1.values() for v in a]), 4),
        "by_app": {a: {"exact": round(mean(by_app_exact[a]), 3),
                       "field_f1": round(mean(by_app_f1[a]), 3),
                       "n": len(by_app_exact[a])} for a in APP_NAMES},
        "examples": examples,
    }
    print(f"  OVERALL exact_match={res['exact_match']}  field_f1={res['field_f1']}", flush=True)
    for a in APP_NAMES:
        d = res["by_app"][a]
        print(f"    [{a:8s}] exact={d['exact']:<6} field_f1={d['field_f1']:<6} n={d['n']}", flush=True)
    for ex in examples[:4]:
        print(f"    e.g. {ex['app']} {ex['event']} exact={ex['exact']} f1={ex['f1']}", flush=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--preroll", type=int, default=3)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--out", default="results/uitick_probe_base.json")
    a = ap.parse_args()
    run(a.model_path, a.tp, a.n, a.seed, a.lora, a.out, a.preroll, a.scale)
