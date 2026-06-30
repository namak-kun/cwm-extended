"""Game-tick OUTCOME eval: expose the within-tick multi-entity salience failure that frame-accuracy
masks. For each game-tick program, free-roll CWM and measure:
  - outcome_acc  : CWM's predicted FINAL return == ground-truth return (THE game-relevant metric;
                   any dropped consequential side-effect -> wrong final hp/score -> wrong outcome)
  - frame_acc    : standard per-frame accuracy (expected HIGH -> demonstrates the masking)
  - len_ok       : CWM produced a full-length, well-formed trace (n_pred == n_gt)
One adapter per process (vLLM LoRA-switch bug, REPORT 25).
"""
from __future__ import annotations

import argparse
import json
import random
from statistics import mean

from vllm import TokensPrompt

from models.cwm_trace import (CWMvLLM, Event, EOS, CALL_SEP,
                              build_prompt, parse_full_trace, resolve_locals)
from gt_trace import trace_program, score_frame
from run_cwm_track import estimate_trace_tokens
from game_tick import gen_game_tick, true_return


def parse_state_return(arg):
    """CWM's final return is a dict {hp,score,x,y}. Parse to a {comp:int} dict (or None)."""
    comps = ("hp", "score", "x", "y")
    if arg is None:
        return None
    if isinstance(arg, dict):
        try:
            return {k: int(arg[k]) for k in comps if k in arg}
        except (ValueError, TypeError):
            return None
    s = str(arg)
    import re
    out = {}
    for k in comps:
        mobj = re.search(rf"['\"]?{k}['\"]?\s*[:=]\s*(-?\d+)", s)
        if mobj:
            out[k] = int(mobj.group(1))
    return out or None


def last_return_arg(frames):
    from models.cwm_trace import Event
    for f in reversed(frames):
        if f.event == Event.RETURN and f.arg is not None:
            return f.arg
    return None


def run(model_path, tp, n, seed, lora, out, k_enemies=None, t_ticks=None):
    rng = random.Random(seed)
    progs = []
    seen = set()
    while len(progs) < n and len(seen) < n * 12:
        src, entry = gen_game_tick(rng, k_enemies=k_enemies, t_ticks=t_ticks)
        if src in seen:
            continue
        seen.add(src)
        tv = true_return(src)            # ground-truth [hp, score, x, y]
        gt = trace_program(src, entry, expand_objects=False)
        if gt and tv is not None and len(gt) > 5:
            progs.append((src, entry, gt, tv))

    m = CWMvLLM(model_path, tp=tp, max_model_len=32768, lora_path=lora)
    print(f"== CWM loaded {'+LoRA '+lora if lora else '(base)'} | game-tick OUTCOME eval n={len(progs)} seed={seed} ==", flush=True)

    fps = [build_prompt(m, src, [], force_event=Event.CALL) for (src, _, _, _) in progs]
    maxcap = min(max(int(estimate_trace_tokens(m, gt) * 1.4) + 256 for (_, _, gt, _) in progs), 30000)
    print(f"  maxcap={maxcap} (longest gt {max(len(gt) for _,_,gt,_ in progs)} frames)", flush=True)
    sp = m.SP(temperature=0.0, max_tokens=maxcap, stop_token_ids=[EOS])
    outs = m.llm.generate([TokensPrompt(prompt_token_ids=fp) for fp in fps], sp, use_tqdm=False, **m._gen_kwargs())

    comp_names = ["hp", "score", "x", "y"]
    frame_accs, outcome_oks, len_oks = [], [], []
    comp_ok = {c: [] for c in comp_names}
    detail = []
    for (src, entry, gt, tv), o in zip(progs, outs):
        df = parse_full_trace(m, [CALL_SEP] + list(o.outputs[0].token_ids))
        nmin = min(len(gt), len(df))
        ok = [score_frame(gt[i], df[i], resolve_locals)["frame_ok"] for i in range(nmin)]
        denom = max(len(gt), len(df))
        frame_accs.append(sum(ok) / denom if denom else 0.0)
        cv = parse_state_return(last_return_arg(df))
        outcome_ok = (cv is not None and all(cv.get(k) == tv[k] for k in tv))
        outcome_oks.append(outcome_ok)
        len_oks.append(len(df) == len(gt))
        for c in comp_names:
            comp_ok[c].append(cv is not None and cv.get(c) == tv.get(c))
        detail.append({"truth": dict(tv), "cwm": cv, "outcome_ok": outcome_ok,
                       "frame_acc": round(frame_accs[-1], 3), "n_gt": len(gt), "n_pred": len(df)})

    res = {
        "lora": lora, "n": len(progs), "seed": seed,
        "outcome_acc": round(mean(outcome_oks), 4),
        "frame_acc": round(mean(frame_accs), 4),
        "len_ok_rate": round(mean(len_oks), 4),
        "component_acc": {c: round(mean(comp_ok[c]), 4) for c in comp_names},
        "detail": detail,
    }
    print(f"  OUTCOME_acc={res['outcome_acc']}   frame_acc={res['frame_acc']}   len_ok={res['len_ok_rate']}", flush=True)
    print(f"  component_acc (salient x/y vs buried hp/score): {res['component_acc']}", flush=True)
    print(f"  (frame_acc >> outcome_acc = the MASKED within-tick salience failure)", flush=True)
    for d in detail[:8]:
        print(f"    truth={d['truth']} cwm={d['cwm']} outcome_ok={d['outcome_ok']} frame_acc={d['frame_acc']} ({d['n_pred']}/{d['n_gt']}f)", flush=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--kenemies", type=int, default=None, help="fix enemy count (salience stress)")
    ap.add_argument("--tticks", type=int, default=None, help="fix tick count")
    ap.add_argument("--out", default="results/gametick_outcome.json")
    a = ap.parse_args()
    run(a.model_path, a.tp, a.n, a.seed, a.lora, a.out, a.kenemies, a.tticks)
