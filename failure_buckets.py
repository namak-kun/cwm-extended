"""Failure-mode dataset generators: produce VARIED programs for each bucket where
CWM was observed to fail, with frame-level ground truth via sys.settrace.

Buckets (from the failure catalog §11/§15/§16):
  - recursion: exploding call trees (fib-like, varied depth/branching)
  - oop: encapsulated state mutated through methods
  - multientity: within-tick side-effects across N sub-entities (game ticks)
  - arithmetic: long interacting-accumulator loops
  - easy: simple programs for anti-forgetting replay

Each generator is seeded -> reproducible, diverse corpus. Output integrates with
trace_dataset.serialize_trace to make CWM-format (input_ids, labels) examples.
"""
from __future__ import annotations

import random


def gen_recursion(rng) -> tuple[str, str]:
    kind = rng.choice(["fib", "sumdigits", "power", "ackermann_lite"])
    if kind == "fib":
        n = rng.randint(5, 8)
        src = f'''def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def main():  # << START_OF_TRACE
    return fib({n})

main()
'''
    elif kind == "sumdigits":
        n = rng.randint(100, 9999)
        src = f'''def sd(n):
    if n == 0:
        return 0
    return n % 10 + sd(n // 10)

def main():  # << START_OF_TRACE
    return sd({n})

main()
'''
    elif kind == "power":
        base, exp = rng.randint(2, 4), rng.randint(3, 6)
        src = f'''def power(b, e):
    if e == 0:
        return 1
    return b * power(b, e - 1)

def main():  # << START_OF_TRACE
    return power({base}, {exp})

main()
'''
    else:  # ackermann_lite (small, bounded)
        mm, nn = rng.randint(1, 2), rng.randint(1, 3)
        src = f'''def ack(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return ack(m - 1, 1)
    return ack(m - 1, ack(m, n - 1))

def main():  # << START_OF_TRACE
    return ack({mm}, {nn})

main()
'''
    return src, "main"


def gen_oop(rng) -> tuple[str, str]:
    vals = [rng.randint(1, 9) for _ in range(rng.randint(5, 9))]
    op = rng.choice(["x", "x*x", "x+1"])
    src = f'''class Acc:
    def __init__(self):
        self.total = 0
        self.count = 0
        self.maxv = -999
    def add(self, v):
        self.total += v
        self.count += 1
        if v > self.maxv:
            self.maxv = v
    def report(self):
        return self.total + self.maxv - self.count

def main():  # << START_OF_TRACE
    a = Acc()
    for x in {vals}:
        a.add({op})
    return a.report()

main()
'''
    return src, "main"


def gen_multientity(rng) -> tuple[str, str]:
    n_en = rng.randint(2, 4)
    enemies = [{"x": rng.randint(0, 6), "y": rng.randint(0, 6)} for _ in range(n_en)]
    en_lits = ", ".join('{"x": %d, "y": %d, "alive": True}' % (e["x"], e["y"]) for e in enemies)
    moves = [rng.choice(["R", "L", "U", "D"]) for _ in range(rng.randint(4, 7))]
    src = f'''def step(state, action):
    p = state["player"]
    if action == "R": p["x"] += 1
    elif action == "L": p["x"] -= 1
    elif action == "U": p["y"] -= 1
    elif action == "D": p["y"] += 1
    for e in state["enemies"]:
        if e["alive"]:
            dx = p["x"] - e["x"]
            dy = p["y"] - e["y"]
            if abs(dx) >= abs(dy):
                e["x"] += (1 if dx > 0 else -1 if dx < 0 else 0)
            else:
                e["y"] += (1 if dy > 0 else -1 if dy < 0 else 0)
            if e["x"] == p["x"] and e["y"] == p["y"]:
                p["hp"] -= 1
    return state

def main():  # << START_OF_TRACE
    state = {{"player": {{"x": 3, "y": 3, "hp": 5}}, "enemies": [{en_lits}]}}
    for a in {moves}:
        state = step(state, a)
    p = state["player"]
    return p["hp"] * 100 + p["x"] * 10 + p["y"]

main()
'''
    return src, "main"


def gen_arithmetic(rng) -> tuple[str, str]:
    steps = rng.randint(10, 25)
    m1, m2, m3 = rng.randint(80, 97), rng.randint(80, 97), rng.randint(80, 97)
    src = f'''def main():  # << START_OF_TRACE
    a, b, c = {rng.randint(0,5)}, {rng.randint(0,5)}, {rng.randint(1,9)}
    for i in range({steps}):
        a = (a * 3 + b) % {m1}
        b = (b + c * 2) % {m2}
        c = (c + a) % {m3}
    return a * 10000 + b * 100 + c

main()
'''
    return src, "main"


def gen_easy(rng) -> tuple[str, str]:
    steps = rng.randint(3, 6)
    src = f'''def main():  # << START_OF_TRACE
    s = 0
    for i in range({steps}):
        s += i
    return s

main()
'''
    return src, "main"


def gen_multientity_short(rng) -> tuple[str, str]:
    """SHORT multientity variant (2 enemies, 2-3 moves) -> ~30-40 frames, fits SFT max_len.
    Same dict-mutation + within-tick hp-side-effect PATTERN as gen_multientity (the §22
    forgetting victim), so it serves as anti-forgetting REPLAY for that mode family while
    the full-size FAILURE_PROGRAMS multientity (11k+ tokens) stays HELD-OUT for eval."""
    enemies = [{"x": rng.randint(0, 5), "y": rng.randint(0, 5)} for _ in range(2)]
    en_lits = ", ".join('{"x": %d, "y": %d, "alive": True}' % (e["x"], e["y"]) for e in enemies)
    moves = [rng.choice(["R", "L", "U", "D"]) for _ in range(rng.randint(2, 3))]
    src = f'''def step(state, action):
    p = state["player"]
    if action == "R": p["x"] += 1
    elif action == "L": p["x"] -= 1
    elif action == "U": p["y"] -= 1
    elif action == "D": p["y"] += 1
    for e in state["enemies"]:
        if e["alive"]:
            dx = p["x"] - e["x"]
            dy = p["y"] - e["y"]
            if abs(dx) >= abs(dy):
                e["x"] += (1 if dx > 0 else -1 if dx < 0 else 0)
            else:
                e["y"] += (1 if dy > 0 else -1 if dy < 0 else 0)
            if e["x"] == p["x"] and e["y"] == p["y"]:
                p["hp"] -= 1
    return state

def main():  # << START_OF_TRACE
    state = {{"player": {{"x": 3, "y": 3, "hp": 5}}, "enemies": [{en_lits}]}}
    for a in {moves}:
        state = step(state, a)
    p = state["player"]
    return p["hp"] * 100 + p["x"] * 10 + p["y"]

main()
'''
    return src, "main"


BUCKETS = {
    "recursion": gen_recursion,
    "oop": gen_oop,
    "multientity": gen_multientity,
    "arithmetic": gen_arithmetic,
    "easy": gen_easy,
}

# default failure-weighted mixture (easy is anti-forgetting replay only)
DEFAULT_WEIGHTS = {"recursion": 0.25, "oop": 0.25, "multientity": 0.25,
                   "arithmetic": 0.15, "easy": 0.10}


def generate_corpus(n: int, seed: int = 0, weights: dict | None = None):
    rng = random.Random(seed)
    weights = weights or DEFAULT_WEIGHTS
    names = list(weights)
    probs = [weights[k] for k in names]
    out = []
    for i in range(n):
        bucket = rng.choices(names, probs)[0]
        src, entry = BUCKETS[bucket](rng)
        out.append({"bucket": bucket, "src": src, "entry": entry})
    return out


if __name__ == "__main__":
    import subprocess, sys, tempfile, os
    from collections import Counter
    from gt_trace import trace_program
    corpus = generate_corpus(60, seed=1)
    print("bucket distribution:", Counter(c["bucket"] for c in corpus))
    # validate every program traces and runs
    bad = 0
    flens = []
    for c in corpus:
        gt = trace_program(c["src"], c["entry"])
        if not gt:
            bad += 1
            continue
        flens.append(len(gt))
    import statistics
    print(f"traced OK: {len(corpus)-bad}/{len(corpus)}; frames min/mean/max = "
          f"{min(flens)}/{round(statistics.mean(flens))}/{max(flens)}")
