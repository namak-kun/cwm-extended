"""Game-tick generator + ground truth for the WITHIN-TICK MULTI-ENTITY SALIENCE failure.

This is the game-prediction-relevant failure (REPORT 15): CWM tracks the salient actor (player
x/y) but DROPS consequential side-effects buried in a secondary loop over many entities -- the hp
decrement on enemy contact, the score++/enemy-death on stomp. Frame-accuracy MASKS this (the dropped
var is 1 of hundreds), so the right metric is the OUTCOME (final state checksum) + the specific
side-effect variables (hp/score) per tick.

gen_game_tick(rng) -> (src, entry): a self-contained, deterministic Python game tick loop:
  player {x,y,hp,score} + K enemies {x,y,alive}; per tick:
    1. player moves by scripted action (the SALIENT, easy-to-track part)
    2. STOMP: player landing on an enemy kills it and +score   (side-effect in entity loop)
    3. CHASE: each alive enemy steps toward player              (multi-entity update)
    4. CONTACT: an enemy landing on the player -hp              (side-effect in entity loop)
  returns a checksum = hp*1000 + score*100 + x*10 + y  (encodes ALL consequential state).

Difficulty knobs: K enemies (more = more entities to track within a tick) and T ticks.
"""
from __future__ import annotations

import random


def gen_game_tick(rng: random.Random, k_enemies=None, t_ticks=None, return_meta=False) -> tuple[str, str]:
    K = k_enemies if k_enemies is not None else rng.randint(3, 5)
    T = t_ticks if t_ticks is not None else rng.randint(2, 3)
    W = H = 7
    enemies = [{"x": rng.randint(1, W), "y": rng.randint(1, H)} for _ in range(K)]
    en_lits = ", ".join('{"x": %d, "y": %d, "alive": True}' % (e["x"], e["y"]) for e in enemies)
    actions = [rng.choice(["U", "D", "L", "R"]) for _ in range(T)]
    src = f'''def sign(a):
    return 1 if a > 0 else (-1 if a < 0 else 0)

def step(state, action):
    W, H = {W}, {H}
    p = state["player"]
    if action == "U": p["y"] -= 1
    elif action == "D": p["y"] += 1
    elif action == "L": p["x"] -= 1
    elif action == "R": p["x"] += 1
    if p["x"] < 1: p["x"] = 1
    if p["x"] > W: p["x"] = W
    if p["y"] < 1: p["y"] = 1
    if p["y"] > H: p["y"] = H
    for e in state["enemies"]:
        if e["alive"] and e["x"] == p["x"] and e["y"] == p["y"]:
            e["alive"] = False
            p["score"] += 10
    for e in state["enemies"]:
        if e["alive"]:
            dx = p["x"] - e["x"]
            dy = p["y"] - e["y"]
            if abs(dx) >= abs(dy):
                e["x"] += sign(dx)
            else:
                e["y"] += sign(dy)
            if e["x"] == p["x"] and e["y"] == p["y"]:
                p["hp"] -= 1
    return state

def main():  # << START_OF_TRACE
    state = {{"player": {{"x": 4, "y": 4, "hp": 9, "score": 0}}, "enemies": [{en_lits}]}}
    for a in {actions}:
        state = step(state, a)
    p = state["player"]
    return {{"hp": p["hp"], "score": p["score"], "x": p["x"], "y": p["y"]}}

main()
'''
    if return_meta:
        init = {"player": {"x": 4, "y": 4, "hp": 9, "score": 0},
                "enemies": [{"x": e["x"], "y": e["y"], "alive": True} for e in enemies]}
        return src, "main", {"init": init, "actions": actions, "K": K, "T": T}
    return src, "main"


def ground_truth_states(src: str, meta: dict) -> list[dict]:
    """Replay step() over the actions, returning the FULL state after each tick (per-tick ground truth
    for the step-over abstraction eval). Uses the generated source's own step() for exactness."""
    import copy
    ns: dict = {}
    exec(src.split("def main()")[0], ns)   # defines sign + step only
    state = copy.deepcopy(meta["init"])
    out = []
    for a in meta["actions"]:
        state = ns["step"](state, a)
        out.append(copy.deepcopy(state))
    return out


_STEP_SRC = '''def sign(a):
    return 1 if a > 0 else (-1 if a < 0 else 0)

def step(state, action):
    W, H = 7, 7
    p = state["player"]
    if action == "U": p["y"] -= 1
    elif action == "D": p["y"] += 1
    elif action == "L": p["x"] -= 1
    elif action == "R": p["x"] += 1
    if p["x"] < 1: p["x"] = 1
    if p["x"] > W: p["x"] = W
    if p["y"] < 1: p["y"] = 1
    if p["y"] > H: p["y"] = H
    for e in state["enemies"]:
        if e["alive"] and e["x"] == p["x"] and e["y"] == p["y"]:
            e["alive"] = False
            p["score"] += 10
    for e in state["enemies"]:
        if e["alive"]:
            dx = p["x"] - e["x"]
            dy = p["y"] - e["y"]
            if abs(dx) >= abs(dy):
                e["x"] += sign(dx)
            else:
                e["y"] += sign(dy)
            if e["x"] == p["x"] and e["y"] == p["y"]:
                p["hp"] -= 1
    return state
'''


def gen_one_tick_src(state: dict, action: str) -> tuple[str, str]:
    """A one-tick program: start at `state`, apply a single `action`, return the new full state.
    Used for FDM forward prediction from an arbitrary state (and IDM-via-forward-search)."""
    src = _STEP_SRC + f'''
def main():  # << START_OF_TRACE
    state = {state!r}
    state = step(state, "{action}")
    return state

main()
'''
    return src, "main"


def real_step(state: dict, action: str) -> dict:
    """Ground-truth single transition step(state, action) -> new state (exact)."""
    import copy
    ns: dict = {}
    exec(_STEP_SRC, ns)
    return ns["step"](copy.deepcopy(state), action)


def gen_game_tick_short(rng: random.Random) -> tuple[str, str]:
    """SHORT game-tick variant (2-3 enemies, 2 ticks) -> ~80-110 frames, fits SFT context.
    Same within-tick multi-entity side-effect PATTERN (stomp +score, contact -hp buried in entity
    loops) as gen_game_tick, so it serves as SFT supervision for the salience failure while the
    full-size game_tick (and the held-out Lua arena) stay as the eval."""
    return gen_game_tick(rng, k_enemies=rng.randint(2, 3), t_ticks=2)


def true_return(src: str) -> list | None:
    """Run the generated program and capture main()'s return (ground-truth outcome)."""
    ns: dict = {}
    try:
        exec(src.replace("main()\n", "__out = main()\n", 1) if "main()\n" in src else src, ns)
        return ns.get("__out")
    except Exception:
        return None


if __name__ == "__main__":
    rng = random.Random(0)
    for i in range(3):
        src, entry = gen_game_tick(rng)
        print(f"--- prog {i} truth={true_return(src)} ---")
