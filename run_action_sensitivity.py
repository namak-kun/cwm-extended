#!/usr/bin/env python3
"""Action-sensitivity control (REPORT §32.6) — does the trained FDM actually CONDITION on the action,
or predict next-state from code+state while ~ignoring the action?

For each held-out transition (s_i, a_true, s_next):
  - ground truth: gt[a] = real_step(s_i, a) for a in {U,D,L,R}; action-SEPARABLE iff >1 distinct gt[a].
  - FDM forward: pred[a] = FDM's predicted next state given the one-tick program for (s_i, a).
Metrics per adapter (restricted to action-separable states for the contrast):
  true_acc   = mean[ pred[a_true] == s_next ]                 (maps TRUE action -> TRUE outcome)
  swap_acc   = mean over a'!=a_true [ pred[a'] == s_next ]    (would a WRONG action also 'predict' s_next?)
  true-swap  = the action-conditioning signal. >>0 = conditioned; ~0 = action ignored.
  pred_div   = mean # distinct pred[a] over the 4 actions     (1 = ignores action; up to 4 = sensitive)
  track_acc  = mean over a [ pred[a] == gt[a] ]               (per-action faithfulness)
Compare FDM_0 / FDM_IDM / FDM_oracle; stratify by event type and separability.
"""
import argparse, json, random
from statistics import mean
from collections import defaultdict

from models.cwm_trace import CWMvLLM, Event
from game_tick import gen_game_tick, ground_truth_states, gen_one_tick_src, real_step
from run_gametick_abstract import _parse_state, batched_stepover
from build_flywheel_data import classify_transition

ACTIONS = ["U", "D", "L", "R"]


def canon(st):
    if not isinstance(st, dict):
        return None
    return json.dumps(st, sort_keys=True)


def run(model_path, tp, n, seed, lora, out, kmin=None, kmax=None, tmin=None, tmax=None):
    rng = random.Random(seed)
    transitions, seen = [], set()
    while len(transitions) < n and len(seen) < n * 20:
        K = rng.randint(kmin, kmax) if (kmin and kmax) else None
        T = rng.randint(tmin, tmax) if (tmin and tmax) else None
        src, entry, meta = gen_game_tick(rng, k_enemies=K, t_ticks=T, return_meta=True)
        if src in seen:
            continue
        seen.add(src)
        states = [meta["init"]] + ground_truth_states(src, meta)
        for i, a in enumerate(meta["actions"]):
            transitions.append({"s": states[i], "a": a, "snext": states[i + 1]})
            if len(transitions) >= n:
                break

    # ground-truth per-action outcomes (action separability is model-independent)
    for tr in transitions:
        tr["gt"] = {a: real_step(tr["s"], a) for a in ACTIONS}
        tr["gt_distinct"] = len({canon(tr["gt"][a]) for a in ACTIONS})
        tr["event"] = classify_transition(tr["s"], tr["snext"])

    m = CWMvLLM(model_path, tp=tp, max_model_len=8192, lora_path=lora)
    print(f"== CWM {'+LoRA '+lora if lora else '(base)'} | action-sensitivity n={len(transitions)} ==", flush=True)

    items, index = [], []
    for ti, tr in enumerate(transitions):
        for a in ACTIONS:
            osrc, _ = gen_one_tick_src(tr["s"], a)
            items.append({"src": osrc})
            index.append((ti, a))
    frames_all = batched_stepover(m, items, max_frames=24, lora_kwargs=m._gen_kwargs())

    pred = {ti: {} for ti in range(len(transitions))}
    for (ti, a), frames in zip(index, frames_all):
        ps = None
        for f in frames:
            if f.event == Event.RETURN:
                cand = _parse_state(f.arg)
                if isinstance(cand, dict) and "player" in cand and "enemies" in cand:
                    ps = cand
        pred[ti][a] = ps

    # aggregate (separable states only for the action-conditioning contrast)
    agg = {"all": defaultdict(list)}
    by_event = defaultdict(lambda: defaultdict(list))
    sep_true, sep_swap, sep_div = [], [], []
    insep_true, insep_swap = [], []
    for ti, tr in enumerate(transitions):
        snc = canon(tr["snext"])
        atrue = tr["a"]
        t_ok = 1.0 if canon(pred[ti].get(atrue)) == snc else 0.0
        swap = [1.0 if canon(pred[ti].get(a)) == snc else 0.0 for a in ACTIONS if a != atrue]
        s_ok = mean(swap) if swap else 0.0
        div = len({canon(pred[ti].get(a)) for a in ACTIONS})
        track = mean([1.0 if canon(pred[ti].get(a)) == canon(tr["gt"][a]) else 0.0 for a in ACTIONS])
        agg["all"]["true"].append(t_ok); agg["all"]["swap"].append(s_ok)
        agg["all"]["div"].append(div); agg["all"]["track"].append(track)
        agg["all"]["gtdiv"].append(tr["gt_distinct"])
        for ev in tr["event"]:
            by_event[ev]["true"].append(t_ok); by_event[ev]["swap"].append(s_ok)
            by_event[ev]["div"].append(div)
        if tr["gt_distinct"] > 1:
            sep_true.append(t_ok); sep_swap.append(s_ok); sep_div.append(div)
        else:
            insep_true.append(t_ok); insep_swap.append(s_ok)

    res = {
        "lora": lora, "n": len(transitions), "seed": seed,
        "overall": {
            "true_acc": round(mean(agg["all"]["true"]), 4),
            "swap_acc": round(mean(agg["all"]["swap"]), 4),
            "true_minus_swap": round(mean(agg["all"]["true"]) - mean(agg["all"]["swap"]), 4),
            "pred_diversity": round(mean(agg["all"]["div"]), 3),
            "gt_diversity": round(mean(agg["all"]["gtdiv"]), 3),
            "track_acc": round(mean(agg["all"]["track"]), 4),
        },
        "action_separable": {
            "n": len(sep_true),
            "true_acc": round(mean(sep_true), 4) if sep_true else None,
            "swap_acc": round(mean(sep_swap), 4) if sep_true else None,
            "true_minus_swap": round(mean(sep_true) - mean(sep_swap), 4) if sep_true else None,
            "pred_diversity": round(mean(sep_div), 3) if sep_div else None,
        },
        "action_insensitive": {
            "n": len(insep_true),
            "true_acc": round(mean(insep_true), 4) if insep_true else None,
            "swap_acc": round(mean(insep_swap), 4) if insep_true else None,
        },
        "by_event": {ev: {"n": len(d["true"]), "true_acc": round(mean(d["true"]), 3),
                          "swap_acc": round(mean(d["swap"]), 3),
                          "pred_div": round(mean(d["div"]), 2)} for ev, d in by_event.items()},
    }
    o = res["overall"]; s = res["action_separable"]
    print(f"  OVERALL    true={o['true_acc']} swap={o['swap_acc']} (true-swap={o['true_minus_swap']})  "
          f"pred_div={o['pred_diversity']}/4 gt_div={o['gt_diversity']}/4 track={o['track_acc']}", flush=True)
    print(f"  SEPARABLE  n={s['n']} true={s['true_acc']} swap={s['swap_acc']} (true-swap={s['true_minus_swap']})  "
          f"pred_div={s['pred_diversity']}/4", flush=True)
    print(f"  INSENS     n={res['action_insensitive']['n']} true={res['action_insensitive']['true_acc']} "
          f"swap={res['action_insensitive']['swap_acc']}", flush=True)
    for ev, d in res["by_event"].items():
        print(f"    [{ev:10s}] n={d['n']:3d} true={d['true_acc']} swap={d['swap_acc']} pred_div={d['pred_div']}", flush=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", default="results/action_sensitivity.json")
    ap.add_argument("--kmin", type=int, default=None); ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--tmin", type=int, default=None); ap.add_argument("--tmax", type=int, default=None)
    a = ap.parse_args()
    run(a.model_path, a.tp, a.n, a.seed, a.lora, a.out, a.kmin, a.kmax, a.tmin, a.tmax)
