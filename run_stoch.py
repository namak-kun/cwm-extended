"""Stochasticity: separate epistemic vs aleatoric (per duck).
Conditions on spawn-bearing steps (the genuinely random ones):
  REVEALED: model is given the rng draws -> can it apply stochastic code exactly?
  HIDDEN  : model gets no draws -> deterministic-field accuracy + reachable-set
            coverage (does true outcome appear among k sampled predictions?).
Never punish exact-match on the hidden random field.
"""
from __future__ import annotations

import argparse
import json
import time
from statistics import mean

from worlds.gridgen import Game
from models.factory import get_model
from models.prompting import build_prompt, extract_json
from metrics import (exact_match, field_accuracy, deterministic_field_accuracy,
                     canonical, in_support)
from eval import gen_trajectory


def run(llm, n_games, horizon, k_samples, out):
    t0 = time.time()
    games = [Game.generate(3000 + i, naming_mode="random", stochastic=True)
             for i in range(n_games)]
    # collect spawn-bearing steps (rng_log non-empty)
    items_field = {}
    examples = []  # (game, step)
    for g in games:
        items_field[id(g)] = g.spec.fname("items")
        for s in gen_trajectory(g, horizon=horizon, init_seed=5, policy_seed=13, stochastic=True):
            if s["rng_log"]:
                examples.append((g, s))
    print(f"[stoch] {len(examples)} spawn-bearing steps from {n_games} games", flush=True)
    if not examples:
        print("no stochastic steps; aborting")
        return

    # REVEALED
    rev = llm.predict_requests([
        {"code": g.source, "history": [s["state"]], "action": s["action"], "rng_log": s["rng_log"]}
        for g, s in examples])
    rev_exact = mean(exact_match(p, s["next_state"]) for p, (g, s) in zip(rev, examples))
    rev_field = mean(field_accuracy(p, s["next_state"])["field_acc"] for p, (g, s) in zip(rev, examples))

    # HIDDEN greedy
    hid = llm.predict_requests([
        {"code": g.source, "history": [s["state"]], "action": s["action"]}  # no rng_log
        for g, s in examples])
    hid_exact = mean(exact_match(p, s["next_state"]) for p, (g, s) in zip(hid, examples))
    hid_detacc = mean(
        deterministic_field_accuracy(p, s["next_state"], {items_field[id(g)]})
        for p, (g, s) in zip(hid, examples))

    # HIDDEN sampled -> reachable-set coverage
    users = [build_prompt(g.source, [s["state"]], s["action"], None) for g, s in examples]
    samp = llm.sample_batch(users, k=k_samples, temperature=0.9)
    cov_full, cov_items = [], []
    for (g, s), texts in zip(examples, samp):
        cands = [extract_json(t) for t in texts]
        cands = [c for c in cands if c is not None]
        cov_full.append(in_support(s["next_state"], cands))
        itf = items_field[id(g)]
        true_items = canonical(s["next_state"].get(itf))
        cov_items.append(any(canonical(c.get(itf)) == true_items for c in cands))

    result = {
        "model": llm.name, "n_spawn_steps": len(examples), "k_samples": k_samples,
        "REVEALED": {"exact": round(rev_exact, 3), "field_acc": round(rev_field, 3)},
        "HIDDEN_greedy": {"exact": round(hid_exact, 3),
                          "deterministic_field_acc": round(hid_detacc, 3)},
        "HIDDEN_sampled": {"reachable_set_coverage_full": round(mean(cov_full), 3),
                           "reachable_set_coverage_items": round(mean(cov_items), 3)},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="results/exp3_stoch.json")
    a = ap.parse_args()
    run(get_model(a.model, a.backend, a.tp), a.games, a.horizon, a.k, a.out)
