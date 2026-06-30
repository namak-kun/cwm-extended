"""IDM via FORWARD SEARCH: use the (step-over-SFT'd) CWM FDM as an Inverse Dynamics Model.

We never train an IDM. Given a transition (s_i, s_{i+1}) and the small discrete action set
{U,D,L,R}, we run the FDM forward on EACH candidate action and pick the one(s) whose predicted
next-state matches s_{i+1}:
    IDM(s_i, s_{i+1}) = { a : FDM(s_i, a) == s_{i+1} }

This tests whether an accurate forward model yields a usable inverse model for free. Metrics:
  - action_recovery_acc : true action is among CWM's matched set (aliasing-aware: an action that
                          really produces s_{i+1} also counts as correct)
  - unique_recovery     : CWM matched EXACTLY one action and it's the true one (strict)
  - fdm_ranks_true      : even if no exact match, does FDM(s,a_true) come CLOSEST to s_{i+1}?
                          (component-distance argmin -> the forgiving, ranking-based IDM)
One adapter per process (vLLM LoRA-switch bug).
"""
from __future__ import annotations

import argparse
import json
import random
from statistics import mean

from vllm import TokensPrompt

from models.cwm_trace import (CWMvLLM, Event, FRAME_SEP, EOS, build_prompt, parse_frame)
from game_tick import gen_game_tick, ground_truth_states, gen_one_tick_src, real_step
from run_gametick_abstract import _parse_state, batched_stepover

ACTIONS = ["U", "D", "L", "R"]


def state_distance(pred, truth):
    """#mismatched components across player{4} + each enemy{x,y,alive}. None pred = large."""
    if not isinstance(pred, dict) or "player" not in pred or "enemies" not in pred:
        return 999
    d = 0
    for k in ("x", "y", "hp", "score"):
        if str(pred["player"].get(k)) != str(truth["player"].get(k)):
            d += 1
    pe, te = pred.get("enemies", []), truth["enemies"]
    if not isinstance(pe, list):
        return 999
    if len(pe) != len(te):
        d += abs(len(pe) - len(te)) * 3
    for a, b in zip(pe, te):
        for k in ("x", "y", "alive"):
            if str(a.get(k)) != str(b.get(k)):
                d += 1
    return d


def state_eq(pred, truth):
    return state_distance(pred, truth) == 0


def run(model_path, tp, n, seed, lora, out):
    rng = random.Random(seed)
    # collect transitions (s_i, a_true, s_next) from held-out games
    transitions = []
    seen = set()
    while len(transitions) < n and len(seen) < n * 20:
        src, entry, meta = gen_game_tick(rng, return_meta=True)
        key = src
        if key in seen:
            continue
        seen.add(key)
        states = [meta["init"]] + ground_truth_states(src, meta)
        for i, a in enumerate(meta["actions"]):
            transitions.append({"s": states[i], "a": a, "snext": states[i + 1]})
            if len(transitions) >= n:
                break

    m = CWMvLLM(model_path, tp=tp, max_model_len=8192, lora_path=lora)
    print(f"== CWM loaded {'+LoRA '+lora if lora else '(base)'} | IDM-via-forward-search n={len(transitions)} ==", flush=True)

    # build one item per (transition, candidate action): FDM forward predicts s' = step(s, a)
    items, index = [], []
    for ti, tr in enumerate(transitions):
        for a in ACTIONS:
            osrc, _ = gen_one_tick_src(tr["s"], a)
            items.append({"src": osrc})
            index.append((ti, a))

    frames_all = batched_stepover(m, items, max_frames=24, lora_kwargs=m._gen_kwargs())

    # gather predicted next-state per (transition, action)
    pred = {ti: {} for ti in range(len(transitions))}
    for (ti, a), frames in zip(index, frames_all):
        ps = None
        for f in frames:
            if f.event == Event.RETURN:
                cand = _parse_state(f.arg)
                if isinstance(cand, dict) and "player" in cand and "enemies" in cand:
                    ps = cand   # last full-state return = step()'s result
        pred[ti][a] = ps

    rec_ok, uniq_ok, rank_ok = [], [], []
    detail = []
    for ti, tr in enumerate(transitions):
        snext = tr["snext"]
        a_true = tr["a"]
        # aliasing-aware truth: all actions that REALLY lead to snext
        true_set = [a for a in ACTIONS if real_step(tr["s"], a) == snext]
        # CWM forward-search: actions whose CWM-predicted s' exactly matches snext
        matched = [a for a in ACTIONS if pred[ti].get(a) is not None and state_eq(pred[ti][a], snext)]
        # ranking IDM: action whose CWM prediction is CLOSEST to snext
        dists = {a: state_distance(pred[ti].get(a), snext) for a in ACTIONS}
        best = min(dists, key=dists.get)
        rec_ok.append(any(a in true_set for a in matched) if matched else False)
        uniq_ok.append(len(matched) == 1 and matched[0] in true_set)
        rank_ok.append(best in true_set)
        detail.append({"a_true": a_true, "true_set": true_set, "cwm_matched": matched,
                       "cwm_rank_best": best, "best_dist": dists[best]})

    res = {
        "lora": lora, "n": len(transitions), "seed": seed,
        "action_recovery_acc": round(mean(rec_ok), 4),
        "unique_recovery_acc": round(mean(uniq_ok), 4),
        "fdm_ranks_true_acc": round(mean(rank_ok), 4),
        "detail": detail[:30],
    }
    print(f"  action_recovery_acc (true action in CWM exact-match set) = {res['action_recovery_acc']}", flush=True)
    print(f"  unique_recovery_acc (CWM matched exactly the true action) = {res['unique_recovery_acc']}", flush=True)
    print(f"  fdm_ranks_true_acc  (closest-prediction IDM, forgiving)   = {res['fdm_ranks_true_acc']}", flush=True)
    for d in detail[:12]:
        print(f"    true={d['a_true']} true_set={d['true_set']} cwm_match={d['cwm_matched']} rank_best={d['cwm_rank_best']}(d={d['best_dist']})", flush=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", default="results/idm_search.json")
    a = ap.parse_args()
    run(a.model_path, a.tp, a.n, a.seed, a.lora, a.out)
