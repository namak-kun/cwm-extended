"""Main experiment: one-step accuracy across naming modes, the code-ablation
(prior baseline), and the counterfactual collapse probe. Batched LLM calls.

Produces the core of the duck's target table:
  - generated weird (semantic/random/misleading names), code shown -> code-conditioned ability
  - code OMITTED (prior-only baseline) -> prior leakage
  - Copy/no-change baseline -> trivial floor
  - counterfactual: same state, all actions -> action sensitivity vs collapse
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from statistics import mean

from worlds.gridgen import Game
from models.base import CopyModel
from models.factory import get_model
from metrics import exact_match, field_accuracy, canonical
from eval import gen_trajectory


def run(llm, n_games: int, horizon: int, out: str):
    t0 = time.time()
    modes = ["semantic", "random", "misleading"]
    games = {m: [Game.generate(1000 + i, naming_mode=m) for i in range(n_games)]
             for m in modes}

    # ---- One-step requests (code shown + code omitted) ----
    reqs, meta, truth = [], [], []
    for m in modes:
        for g in games[m]:
            steps = gen_trajectory(g, horizon=horizon, init_seed=42, policy_seed=7)
            for s in steps:
                for code_shown in (True, False):
                    reqs.append({"code": g.source if code_shown else "",
                                 "history": [s["state"]], "action": s["action"]})
                    meta.append((m, code_shown))
                    truth.append(s["next_state"])
    print(f"[one-step] {len(reqs)} predictions...", flush=True)
    preds = llm.predict_requests(reqs)

    agg = defaultdict(lambda: {"exact": [], "field": [], "valid": []})
    copy = CopyModel()
    copy_agg = defaultdict(lambda: {"exact": [], "field": []})
    for (m, cs), p, tr, rq in zip(meta, preds, truth, reqs):
        a = agg[(m, cs)]
        a["exact"].append(exact_match(p, tr))
        fr = field_accuracy(p, tr)
        a["field"].append(fr["field_acc"])
        a["valid"].append(fr["valid"])
        if cs:  # copy baseline once per example
            cp = copy.predict("", rq["history"], rq["action"])
            copy_agg[m]["exact"].append(exact_match(cp, tr))
            copy_agg[m]["field"].append(field_accuracy(cp, tr)["field_acc"])

    onestep = {}
    for (m, cs), d in agg.items():
        onestep[f"{m}|code={'on' if cs else 'OFF'}"] = {
            "exact": round(mean(d["exact"]), 3),
            "field_acc": round(mean(d["field"]), 3),
            "valid": round(mean(d["valid"]), 3),
            "n": len(d["exact"]),
        }
    for m in modes:
        onestep[f"{m}|COPY_baseline"] = {
            "exact": round(mean(copy_agg[m]["exact"]), 3),
            "field_acc": round(mean(copy_agg[m]["field"]), 3),
        }

    # ---- Counterfactual collapse probe (code shown) ----
    cf_reqs, cf_meta, cf_truth = [], [], []
    for m in modes:
        for g in games[m]:
            st = g.initial_state(99)
            for a in g.LEGAL_ACTIONS:
                cf_reqs.append({"code": g.source, "history": [st], "action": a})
                cf_meta.append((m, g, st))
                cf_truth.append(g.step(st, a, None))
    print(f"[counterfactual] {len(cf_reqs)} predictions...", flush=True)
    cf_preds = llm.predict_requests(cf_reqs)

    cf_by_mode = defaultdict(lambda: {"exact": [], "pred_sets": defaultdict(set),
                                      "true_sets": defaultdict(set)})
    for (m, g, st), p, tr in zip(cf_meta, cf_preds, cf_truth):
        gid = id(g)
        cf_by_mode[m]["exact"].append(exact_match(p, tr))
        cf_by_mode[m]["pred_sets"][gid].add(canonical(p) if p is not None else "INVALID")
        cf_by_mode[m]["true_sets"][gid].add(canonical(tr))
    counterfactual = {}
    for m in modes:
        d = cf_by_mode[m]
        ratios = [len(d["pred_sets"][k]) / len(d["true_sets"][k]) for k in d["true_sets"]]
        counterfactual[m] = {
            "action_exact_rate": round(mean(d["exact"]), 3),
            "sensitivity_ratio": round(mean(ratios), 3),
        }

    result = {
        "model": llm.name, "n_games_per_mode": n_games, "horizon": horizon,
        "one_step": onestep, "counterfactual": counterfactual,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nsaved -> {out}  ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", default="results/exp1_onestep.json")
    a = ap.parse_args()
    run(get_model(a.model, a.backend, a.tp), a.games, a.horizon, a.out)
