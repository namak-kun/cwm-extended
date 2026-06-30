"""Complexity sweep: does one-step accuracy hold as PROGRAMS get bigger?
Scales code length / #actions / state size via the `complexity` knob (random
naming, code shown). Directly answers 'can we go to bigger programs?'.
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


def run(llm, n_games, horizon, levels, out):
    t0 = time.time()
    result = {"model": llm.name, "n_games": n_games, "horizon": horizon, "levels": {}}
    for c in levels:
        games = [Game.generate(7000 + c * 100 + i, naming_mode="random", complexity=c)
                 for i in range(n_games)]
        reqs, truth = [], []
        for g in games:
            for s in gen_trajectory(g, horizon=horizon, init_seed=42, policy_seed=7):
                reqs.append({"code": g.source, "history": [s["state"]], "action": s["action"]})
                truth.append(s["next_state"])
        preds = llm.predict_requests(reqs)
        result["levels"][c] = {
            "exact": round(mean(exact_match(p, t) for p, t in zip(preds, truth)), 3),
            "field_acc": round(mean(field_accuracy(p, t)["field_acc"] for p, t in zip(preds, truth)), 3),
            "n": len(truth),
            "avg_code_lines": round(mean(g.source.count("\n") for g in games), 1),
            "avg_actions": round(mean(len(g.LEGAL_ACTIONS) for g in games), 1),
            "avg_grid": round(mean(g.ns["GW"] * g.ns["GH"] for g in games), 1),
        }
        lv = result["levels"][c]
        print(f"  complexity={c}: exact={lv['exact']} (code_lines={lv['avg_code_lines']}, "
              f"actions={lv['avg_actions']}, grid={lv['avg_grid']}, n={lv['n']})", flush=True)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--out", default="results/exp5_complexity_7b.json")
    a = ap.parse_args()
    run(get_model(a.model, a.backend, a.tp), a.games, a.horizon,
        [int(x) for x in a.levels.split(",")], a.out)
