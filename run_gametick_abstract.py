"""TICK-LEVEL ABSTRACTION eval: can CWM model a game tick as ONE opaque transition?

Instead of tracing the ~130 interior line-frames of step(state,action) per tick, use STEP-OVER
(REPORT 10): when CWM descends into step(), force its RETURN so CWM predicts the whole new state in
ONE shot. This compresses ~130 frames/tick -> ~a few, letting long games fit the context window -- IF
the abstracted (one-shot) transition stays accurate. That is the s_{i+1}|s_i game-world-model unit.

Metrics:
  - per_tick_state_acc : fraction of ticks where CWM's predicted post-step state == ground truth
                         (full state: player hp/score/x/y + every enemy x/y/alive). THE key number.
  - final_outcome_acc  : last-tick player {hp,score,x,y} correct
  - compression         : abstract frames vs the full line-level trace length
One adapter per process (vLLM LoRA-switch bug).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from statistics import mean

from vllm import TokensPrompt

from models.cwm_trace import (CWMvLLM, Event, FRAME_SEP, EOS, CALL_SEP,
                              build_prompt, parse_frame, resolve_locals)
from gt_trace import trace_program
from game_tick import gen_game_tick, ground_truth_states


def _norm(obj):
    """Normalize a state-ish value to a comparable canonical form."""
    if isinstance(obj, dict):
        return {k: _norm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        try:
            import ast
            return _norm(ast.literal_eval(s))
        except Exception:
            return s
    return obj


def _parse_state(arg):
    """Parse a step() return arg (the new full state) into a dict, tolerant of repr/JSON."""
    if arg is None:
        return None
    if isinstance(arg, dict):
        return _norm(arg)
    s = str(arg).strip()
    import ast
    try:
        return _norm(ast.literal_eval(s))
    except Exception:
        try:
            return _norm(json.loads(s.replace("'", '"').replace("True", "true").replace("False", "false")))
        except Exception:
            return None


def state_eq(pred, truth):
    if not isinstance(pred, dict):
        return False
    p_pl, t_pl = pred.get("player"), truth["player"]
    if _norm(p_pl) != _norm(t_pl):
        return False
    p_en, t_en = pred.get("enemies"), truth["enemies"]
    if not isinstance(p_en, list) or len(p_en) != len(t_en):
        return False
    return all(_norm(a) == _norm(b) for a, b in zip(p_en, t_en))


def batched_stepover(m, items, max_frames=60, max_tokens=2048, lora_kwargs=None):
    sp = m.SP(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[FRAME_SEP, EOS])
    lk = lora_kwargs or {}
    states = [{"src": it["src"], "frames": [], "force": Event.CALL, "done": False} for it in items]
    for _ in range(max_frames):
        active = [s for s in states if not s["done"]]
        if not active:
            break
        prompts = [build_prompt(m, s["src"], s["frames"], force_event=s["force"]) for s in active]
        outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False, **lk)
        for s, o in zip(active, outs):
            f = parse_frame(m, list(o.outputs[0].token_ids) + [FRAME_SEP],
                            forced_event=s["force"], prev=s["frames"][-1] if s["frames"] else None)
            if f is None:
                s["done"] = True
                continue
            s["frames"].append(f)
            # STEP-OVER: when CWM descends into a call below the entry, force its return (opaque tick)
            if f.event == Event.CALL and len(s["frames"]) > 1:
                s["force"] = Event.RETURN
            else:
                s["force"] = None
            # done when the entry (main) returns: a RETURN whose arg looks like the final {hp,score,x,y}
            if f.event == Event.RETURN:
                a = _parse_state(f.arg)
                if isinstance(a, dict) and "hp" in a and "player" not in a:
                    s["done"] = True
    return [s["frames"] for s in states]


def run(model_path, tp, n, seed, lora, out, kmin=None, kmax=None, tmin=None, tmax=None):
    rng = random.Random(seed)
    progs = []
    seen = set()
    while len(progs) < n and len(seen) < n * 12:
        K = rng.randint(kmin, kmax) if (kmin and kmax) else None
        T = rng.randint(tmin, tmax) if (tmin and tmax) else None
        src, entry, meta = gen_game_tick(rng, k_enemies=K, t_ticks=T, return_meta=True)
        if src in seen:
            continue
        seen.add(src)
        gts = ground_truth_states(src, meta)
        full = trace_program(src, entry, expand_objects=False)
        if gts and full and len(full) > 5:
            progs.append({"src": src, "meta": meta, "gts": gts, "full_len": len(full)})

    m = CWMvLLM(model_path, tp=tp, max_model_len=16384, lora_path=lora)
    print(f"== CWM loaded {'+LoRA '+lora if lora else '(base)'} | game-tick STEP-OVER (tick abstraction) n={len(progs)} ==", flush=True)

    frames_all = batched_stepover(m, progs, max_frames=60, lora_kwargs=m._gen_kwargs())

    per_tick_accs, final_oks, comp = [], [], []
    abstract_lens = []
    detail = []
    for prog, frames in zip(progs, frames_all):
        gts = prog["gts"]
        # predicted per-tick states = RETURN frames whose arg is a full state (has 'player'/'enemies')
        pred_states = []
        for f in frames:
            if f.event == Event.RETURN:
                a = _parse_state(f.arg)
                if isinstance(a, dict) and "player" in a and "enemies" in a:
                    pred_states.append(a)
        ntick = min(len(gts), len(pred_states))
        tick_ok = [state_eq(pred_states[i], gts[i]) for i in range(ntick)]
        # missing predicted ticks count against accuracy
        denom = len(gts)
        per_tick = sum(tick_ok) / denom if denom else 0.0
        per_tick_accs.append(per_tick)
        final_ok = (ntick == len(gts) and tick_ok and all(tick_ok))
        final_oks.append(bool(final_ok))
        abstract_lens.append(len(frames))
        comp .append(prog["full_len"] / max(1, len(frames)))
        detail.append({"K": prog["meta"]["K"], "T": prog["meta"]["T"],
                       "n_pred_ticks": len(pred_states), "n_true_ticks": len(gts),
                       "per_tick_ok": tick_ok, "abstract_frames": len(frames),
                       "full_frames": prog["full_len"]})

    res = {
        "lora": lora, "n": len(progs), "seed": seed,
        "per_tick_state_acc": round(mean(per_tick_accs), 4),
        "all_ticks_correct_rate": round(mean(final_oks), 4),
        "mean_abstract_frames": round(mean(abstract_lens), 1),
        "mean_full_frames": round(mean(p["full_len"] for p in progs), 1),
        "mean_compression_x": round(mean(comp), 1),
        "detail": detail,
    }
    print(f"  per_tick_state_acc={res['per_tick_state_acc']}  all_ticks_correct={res['all_ticks_correct_rate']}", flush=True)
    print(f"  compression: {res['mean_full_frames']} full frames -> {res['mean_abstract_frames']} abstract ({res['mean_compression_x']}x)", flush=True)
    for d in detail[:10]:
        print(f"    K={d['K']} T={d['T']} ticks_ok={d['per_tick_ok']} ({d['abstract_frames']}f vs {d['full_frames']} full)", flush=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", default="results/gametick_abstract.json")
    ap.add_argument("--kmin", type=int, default=None); ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--tmin", type=int, default=None); ap.add_argument("--tmax", type=int, default=None)
    a = ap.parse_args()
    run(a.model_path, a.tp, a.n, a.seed, a.lora, a.out, a.kmin, a.kmax, a.tmin, a.tmax)
