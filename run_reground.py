"""Drift mitigation: periodic RE-GROUNDING.

The report's headline problem is closed-loop drift. The proposed fix is to run the
real engine every k steps to reset the predicted state ('engine in the loop').
This quantifies the tradeoff: engine-call rate (1/k) vs sustained accuracy.

k=1   -> teacher-forced (engine every step)
k=inf -> pure free rollout (engine never)
"""
from __future__ import annotations

import argparse
import json
import time
from statistics import mean

from worlds.gridgen import Game
from models.factory import get_model
from metrics import exact_match, field_accuracy
from eval import gen_trajectory


def reground(llm, games, trajs, k):
    N, H = len(games), len(trajs[0])
    pred = [json.loads(json.dumps(trajs[i][0]["state"])) for i in range(N)]
    exact = [[] for _ in range(N)]
    fields = [[] for _ in range(N)]
    for t in range(H):
        reqs = [{"code": games[i].source, "history": [pred[i]],
                 "action": trajs[i][t]["action"], "rng_log": trajs[i][t]["rng_log"]}
                for i in range(N)]
        preds = llm.predict_requests(reqs)
        for i, p in enumerate(preds):
            tn = trajs[i][t]["next_state"]
            exact[i].append(exact_match(p, tn))
            fields[i].append(field_accuracy(p, tn)["field_acc"])
            pred[i] = p if p is not None else pred[i]
            if (t + 1) % k == 0:           # RE-GROUND from the real engine
                pred[i] = json.loads(json.dumps(tn))
    flat_e = [x for c in exact for x in c]
    flat_f = [x for c in fields for x in c]
    return {"mean_exact": round(mean(flat_e), 3), "mean_field": round(mean(flat_f), 3),
            "engine_calls_per_step": round(1.0 / k, 3) if k < 9999 else 0.0}


def run(llm, n_games, horizon, ks, out):
    t0 = time.time()
    games = [Game.generate(4000 + i, naming_mode="random") for i in range(n_games)]
    trajs = [gen_trajectory(g, horizon=horizon, init_seed=5, policy_seed=9) for g in games]
    result = {"model": llm.name, "n_games": n_games, "horizon": horizon, "k": {}}
    for k in ks:
        r = reground(llm, games, trajs, k)
        result["k"][k] = r
        label = "teacher-forced" if k == 1 else ("pure-free" if k >= 9999 else f"every {k}")
        print(f"  k={k:>5} ({label:>14}): mean_exact={r['mean_exact']} "
              f"mean_field={r['mean_field']} engine/step={r['engine_calls_per_step']}", flush=True)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--ks", default="1,2,3,5,10,9999")
    ap.add_argument("--out", default="results/exp6_reground_32b.json")
    a = ap.parse_args()
    run(get_model(a.model, a.backend, a.tp), a.games, a.horizon,
        [int(x) for x in a.ks.split(",")], a.out)
