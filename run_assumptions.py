"""Grounded pre-training assumption tests (gate the whole training program).

A2 — Oracle queryable at arbitrary student-visited states?
   When the model DRIFTS in free rollout, are its predicted states still VALID and
   RUNNABLE by the real update() oracle? If drifted states are unrunnable, DAgger/
   OPSD/RL/re-grounding can't relabel them. We measure valid+runnable rate vs depth.

A3 — Privileged teacher >> blind student? (the OPSD premise)
   Does conditioning on the TRUE current state recover accuracy that drift destroys?
   Compare next-state prediction given TRUE s_t vs given the model's DRIFTED s_hat_t,
   both scored against the true s_{t+1}. A large, growing gap = OPSD has signal.

Small model (Qwen2.5-Coder-1.5B) as the trainable student stand-in.
"""
from __future__ import annotations

import argparse
import json
import time
from statistics import mean

from worlds.gridgen import Game
from models.llm import LLMModel
from eval import gen_trajectory
from metrics import exact_match, field_accuracy, canonical


def schema_ok(state, ref_keys):
    return isinstance(state, dict) and set(state.keys()) == ref_keys


def runnable(game, state, action):
    """Can the real oracle step this (possibly model-predicted) state?"""
    try:
        ns = game.step(state, action)
        return isinstance(ns, dict)
    except Exception:
        return False


def run(model_id, n_games, horizon, out):
    t0 = time.time()
    llm = LLMModel(model_id)
    games = [Game.generate(8000 + i, naming_mode="random") for i in range(n_games)]
    trajs = [gen_trajectory(g, horizon=horizon, init_seed=5, policy_seed=11) for g in games]

    # ---- Free rollout (drifted), tracking A2 validity + recording drifted states ----
    a2_valid = {t: [] for t in range(horizon)}
    a2_runnable = {t: [] for t in range(horizon)}
    drifted = [[s["state"] for s in tr[:1]] for tr in trajs]  # start from true s0
    for gi, (g, tr) in enumerate(zip(games, trajs)):
        ref_keys = set(tr[0]["state"].keys())
        cur = json.loads(json.dumps(tr[0]["state"]))
        for t in range(horizon):
            pred = llm.predict(g.source, [cur], tr[t]["action"])
            sv = schema_ok(pred, ref_keys)
            rn = runnable(g, pred, tr[t]["action"]) if pred is not None else False
            a2_valid[t].append(1.0 if sv else 0.0)
            a2_runnable[t].append(1.0 if rn else 0.0)
            cur = pred if (pred is not None and rn) else cur
            drifted[gi].append(cur)

    # ---- A3 privilege gap: teacher(true s_t) vs student(drifted s_hat_t) ----
    a3_teacher = {t: [] for t in range(horizon)}   # given TRUE current state
    a3_student = {t: [] for t in range(horizon)}   # given DRIFTED current state
    for gi, (g, tr) in enumerate(zip(games, trajs)):
        for t in range(horizon):
            true_s = tr[t]["state"]
            drift_s = drifted[gi][t]
            true_next = tr[t]["next_state"]
            a = tr[t]["action"]
            p_teacher = llm.predict(g.source, [true_s], a)
            p_student = llm.predict(g.source, [drift_s], a)
            a3_teacher[t].append(1.0 if exact_match(p_teacher, true_next) else 0.0)
            a3_student[t].append(1.0 if exact_match(p_student, true_next) else 0.0)

    result = {
        "model": llm.name, "n_games": n_games, "horizon": horizon,
        "A2_valid_schema_by_depth": {t: round(mean(v), 3) for t, v in a2_valid.items()},
        "A2_runnable_by_depth": {t: round(mean(v), 3) for t, v in a2_runnable.items()},
        "A3_teacher_exact_by_depth": {t: round(mean(v), 3) for t, v in a3_teacher.items()},
        "A3_student_exact_by_depth": {t: round(mean(v), 3) for t, v in a3_student.items()},
        "A3_privilege_gap_by_depth": {t: round(mean(a3_teacher[t]) - mean(a3_student[t]), 3)
                                       for t in range(horizon)},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(result, open(out, "w"), indent=2)
    # readable summary
    print("depth | A2 valid | A2 runnable | A3 teacher(true) | A3 student(drift) | gap")
    for t in range(horizon):
        print(f"  {t:3} |  {result['A2_valid_schema_by_depth'][t]:.2f}   |    "
              f"{result['A2_runnable_by_depth'][t]:.2f}     |      "
              f"{result['A3_teacher_exact_by_depth'][t]:.2f}       |       "
              f"{result['A3_student_exact_by_depth'][t]:.2f}        | "
              f"{result['A3_privilege_gap_by_depth'][t]:+.2f}")
    print(f"\nsaved -> {out} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--out", default="results/assumptions_a2a3.json")
    a = ap.parse_args()
    run(a.model, a.games, a.horizon, a.out)
