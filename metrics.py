"""Metrics for state-prediction evaluation.

Honest accounting (per duck): separate exact-match, field-level accuracy, invalid
outputs, deterministic-field accuracy under stochasticity, reachable-set coverage,
and manifold/invariant violations.
"""
from __future__ import annotations

import json
from typing import Any


def canonical(state: Any) -> str:
    return json.dumps(state, sort_keys=True)


def exact_match(pred: Any, true: Any) -> bool:
    if pred is None:
        return False
    return canonical(pred) == canonical(true)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict/list into {path: scalar}."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def field_accuracy(pred: Any, true: Any) -> dict:
    """Leaf-level accuracy over the TRUE state's fields.

    Returns fraction of true leaves correctly predicted, plus structure flags.
    """
    if pred is None:
        return {"field_acc": 0.0, "n_true": len(flatten(true)), "valid": False,
                "key_set_match": False}
    ft, fp = flatten(true), flatten(pred)
    if not ft:
        return {"field_acc": 1.0, "n_true": 0, "valid": True, "key_set_match": True}
    correct = sum(1 for k, v in ft.items() if k in fp and fp[k] == v)
    return {
        "field_acc": correct / len(ft),
        "n_true": len(ft),
        "valid": True,
        "key_set_match": set(ft.keys()) == set(fp.keys()),
    }


def deterministic_field_accuracy(pred: Any, true: Any, random_fields: set[str]) -> float:
    """Field accuracy restricted to NON-random fields (for hidden-stochastic eval).

    `random_fields` are top-level field names whose value depends on hidden RNG.
    """
    if pred is None:
        return 0.0
    ft, fp = flatten(true), flatten(pred)
    det = {k: v for k, v in ft.items() if k.split(".")[0] not in random_fields}
    if not det:
        return 1.0
    correct = sum(1 for k, v in det.items() if k in fp and fp[k] == v)
    return correct / len(det)


def in_support(true: Any, candidates: list[Any]) -> bool:
    """Reachable-set coverage: is the true next-state among predicted candidates?"""
    ct = canonical(true)
    return any(canonical(c) == ct for c in candidates if c is not None)


def invariant_violations(state: Any, gw: int, gh: int, pos_field: str,
                         schema_keys: set[str]) -> list[str]:
    """Detect departures from the valid state manifold."""
    v = []
    if state is None:
        return ["unparseable"]
    if not isinstance(state, dict):
        return ["not_a_dict"]
    if set(state.keys()) != schema_keys:
        v.append("key_set_changed")
    p = state.get(pos_field)
    if not (isinstance(p, list) and len(p) == 2 and all(isinstance(x, int) for x in p)):
        v.append("pos_malformed")
    else:
        if not (0 <= p[0] < gw and 0 <= p[1] < gh):
            v.append("pos_out_of_bounds")
    return v
