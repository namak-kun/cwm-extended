"""Rollout drift: free (closed-loop, model eats its own predictions) vs
teacher-forced. Deterministic games. Shows how errors compound and whether
predicted states leave the valid manifold. Batched lockstep across games.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from statistics import mean

from worlds.gridgen import Game
from models.factory import get_model
from metrics import exact_match, field_accuracy, invariant_violations
from eval import gen_trajectory


def rollout(llm, games, trajs, teacher_forced: bool):
    N, H = len(games), len(trajs[0])
    pred = [json.loads(json.dumps(trajs[i][0]["state"])) for i in range(N)]
    curves = [[] for _ in range(N)]
    for t in range(H):
        reqs = []
        for i in range(N):
            hist = [trajs[i][t]["state"]] if teacher_forced else [pred[i]]
            reqs.append({"code": games[i].source, "history": hist,
                         "action": trajs[i][t]["action"], "rng_log": trajs[i][t]["rng_log"]})
        preds = llm.predict_requests(reqs)
        for i, p in enumerate(preds):
            tn = trajs[i][t]["next_state"]
            gw, gh = games[i].ns["GW"], games[i].ns["GH"]
            pf = games[i].spec.fname("pos")
            curves[i].append({
                "t": t + 1, "exact": exact_match(p, tn),
                "field_acc": field_accuracy(p, tn)["field_acc"],
                "viol": bool(invariant_violations(p, gw, gh, pf, set(tn.keys()))),
            })
            if not teacher_forced:
                pred[i] = p if p is not None else pred[i]
    return curves


def agg(curves, H):
    out = {}
    for t in range(1, H + 1):
        col = [c[t - 1] for c in curves]
        out[t] = {"exact": round(mean(x["exact"] for x in col), 3),
                  "field_acc": round(mean(x["field_acc"] for x in col), 3),
                  "viol_rate": round(mean(x["viol"] for x in col), 3)}
    firstv = [next((x["t"] for x in c if x["viol"]), None) for c in curves]
    out["mean_first_violation_t"] = round(
        mean([f for f in firstv if f is not None]), 2) if any(firstv) else None
    out["frac_ever_violating"] = round(mean([f is not None for f in firstv]), 3)
    return out


def run(llm, n_games, horizon, out):
    t0 = time.time()
    modes = ["semantic", "random", "misleading"]
    result = {"model": llm.name, "n_games_per_mode": n_games, "horizon": horizon}
    for m in modes:
        games = [Game.generate(2000 + i, naming_mode=m) for i in range(n_games)]
        trajs = [gen_trajectory(g, horizon=horizon, init_seed=5, policy_seed=11) for g in games]
        print(f"[rollout {m}] free + teacher-forced, {n_games} games x {horizon} steps", flush=True)
        free = agg(rollout(llm, games, trajs, teacher_forced=False), horizon)
        tf = agg(rollout(llm, games, trajs, teacher_forced=True), horizon)
        result[m] = {"free": free, "teacher_forced": tf}
        print(f"  {m}: free field_acc@1={free[1]['field_acc']} @10={free[10]['field_acc']} "
              f"@20={free[horizon]['field_acc']} | first_viol={free['mean_first_violation_t']}")
    result["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out} ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--out", default="results/exp2_rollout.json")
    a = ap.parse_args()
    run(get_model(a.model, a.backend, a.tp), a.games, a.horizon, a.out)
