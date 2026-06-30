"""Evaluator: trajectory generation, one-step, counterfactual, and rollout drift."""
from __future__ import annotations

import json
import random
from statistics import mean

from worlds.gridgen import Game, ScriptedRNG
from metrics import (canonical, exact_match, field_accuracy,
                     deterministic_field_accuracy, in_support, invariant_violations)


def gen_trajectory(game: Game, horizon: int, init_seed: int,
                   policy_seed: int, stochastic: bool = False) -> list[dict]:
    state = game.initial_state(init_seed)
    rng = ScriptedRNG(seed=policy_seed) if stochastic else None
    pol = random.Random(policy_seed + 1)
    steps = []
    for _ in range(horizon):
        action = pol.choice(game.LEGAL_ACTIONS)
        if rng is not None:
            rng.log = []
        nxt = game.step(state, action, rng)
        steps.append({
            "state": state, "action": action,
            "rng_log": list(rng.log) if rng is not None else None,
            "next_state": nxt,
        })
        state = nxt
    return steps


def one_step_eval(model, game: Game, steps: list[dict],
                  code_shown: bool = True, history_len: int = 1) -> dict:
    code = game.source if code_shown else ""
    exact, fa, valid = [], [], []
    for i, s in enumerate(steps):
        hist = [steps[j]["state"] for j in range(max(0, i - history_len + 1), i + 1)]
        pred = model.predict(code, hist, s["action"], s["rng_log"])
        exact.append(exact_match(pred, s["next_state"]))
        r = field_accuracy(pred, s["next_state"])
        fa.append(r["field_acc"])
        valid.append(r["valid"])
    return {
        "n": len(steps),
        "exact_match": mean(exact) if exact else 0.0,
        "field_acc": mean(fa) if fa else 0.0,
        "valid_rate": mean(valid) if valid else 0.0,
    }


def counterfactual_eval(model, game: Game, state: dict) -> dict:
    """Same state, every action. Tests code-mediated input causality vs prior collapse."""
    preds, per_action = {}, {}
    for a in game.LEGAL_ACTIONS:
        true = game.step(state, a, None)
        pred = model.predict(game.source, [state], a, None)
        preds[a] = pred
        per_action[a] = {"exact": exact_match(pred, true)}
    pred_distinct = len({canonical(p) for p in preds.values() if p is not None})
    true_distinct = len({canonical(game.step(state, a, None)) for a in game.LEGAL_ACTIONS})
    return {
        "per_action_exact": {a: r["exact"] for a, r in per_action.items()},
        "action_exact_rate": mean([r["exact"] for r in per_action.values()]),
        "pred_distinct": pred_distinct,
        "true_distinct": true_distinct,
        # collapse score: 1.0 = perfectly mirrors true action-sensitivity, ->0 = collapsed
        "sensitivity_ratio": pred_distinct / true_distinct if true_distinct else 1.0,
    }


def rollout_eval(model, game: Game, steps: list[dict], teacher_forced: bool) -> list[dict]:
    gw, gh = game.ns["GW"], game.ns["GH"]
    posfield = game.spec.fname("pos")
    schema = set(steps[0]["state"].keys())
    pred_state = json.loads(json.dumps(steps[0]["state"]))
    curve = []
    for t, s in enumerate(steps):
        hist = [s["state"]] if teacher_forced else [pred_state]
        pred = model.predict(game.source, hist, s["action"], s["rng_log"])
        curve.append({
            "t": t + 1,
            "exact": exact_match(pred, s["next_state"]),
            "field_acc": field_accuracy(pred, s["next_state"])["field_acc"],
            "violations": invariant_violations(pred, gw, gh, posfield, schema),
        })
        if not teacher_forced:
            pred_state = pred if pred is not None else pred_state
    return curve


def summarize_rollout(curve: list[dict]) -> dict:
    by_t = {c["t"]: c for c in curve}
    return {
        "horizon": len(curve),
        "exact_at": {t: by_t[t]["exact"] for t in (1, 5, 10, 20) if t in by_t},
        "field_acc_at": {t: round(by_t[t]["field_acc"], 3) for t in (1, 5, 10, 20) if t in by_t},
        "first_violation_t": next((c["t"] for c in curve if c["violations"]), None),
        "violation_rate": mean([1.0 if c["violations"] else 0.0 for c in curve]),
    }
