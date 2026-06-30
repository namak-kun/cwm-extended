"""How far can CWM track state? Free-rollout native trace vs ground truth.

We build parametric programs whose execution trace grows with N, let CWM predict
the WHOLE trace from source (free rollout, native trace tokens), and compare to
the real execution to find the first-divergence depth and accuracy-vs-depth.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

from models.cwm_trace import (CWMvLLM, Event, CALL_SEP, build_prompt,
                              parse_full_trace, resolve_locals)
from gt_trace import trace_program, score_trace


# ---------------- parametric programs (entry marked START_OF_TRACE) ----------
def prog_counter(n: int) -> tuple[str, str]:
    src = f'''def simulate():  # << START_OF_TRACE
    a = 1
    b = 0
    c = 5
    for i in range({n}):
        a = (a * 3 + 1) % 100
        b = (b + a) % 97
        c = (c + b) % 89
    return (a, b, c)

simulate()
'''
    return src, "simulate"


def prog_grid(n: int, seed: int = 0) -> tuple[str, str]:
    rng = random.Random(seed)
    acts = [rng.choice("RLUD") for _ in range(n)]
    src = f'''ACTIONS = {acts!r}

def walk():  # << START_OF_TRACE
    x = 0
    y = 0
    score = 0
    for a in ACTIONS:
        if a == "R":
            x += 1
        elif a == "L":
            x -= 1
        elif a == "U":
            y += 1
        elif a == "D":
            y -= 1
        x = max(0, min(9, x))
        y = max(0, min(9, y))
        if (x + y) % 3 == 0:
            score += 1
    return (x, y, score)

walk()
'''
    return src, "walk"


def prog_list(n: int, seed: int = 0) -> tuple[str, str]:
    rng = random.Random(seed)
    ops = [rng.choice(["push", "pop", "rot"]) for _ in range(n)]
    src = f'''OPS = {ops!r}

def run():  # << START_OF_TRACE
    stack = [1, 2, 3]
    total = 0
    for op in OPS:
        if op == "push":
            stack.append((total + 1) % 10)
        elif op == "pop":
            if stack:
                total += stack.pop()
        elif op == "rot":
            if len(stack) > 1:
                stack = stack[1:] + stack[:1]
    return (stack, total)

run()
'''
    return src, "run"


PROGRAMS = {"counter": prog_counter, "grid": prog_grid, "list": prog_list}


def estimate_trace_tokens(m, gt) -> int:
    """Upper-bound the CWM token length of the true trace (full-locals encoding;
    CWM's diff format is shorter, so this is a safe cap)."""
    total = 0
    for g in gt:
        total += 6  # event + action_sep + frame_sep + slack
        total += len(m.encode(g.source_line))
        if g.event in ("call", "line"):
            total += len(m.encode(json.dumps(g.locals)))
        if g.ret is not None:
            total += len(m.encode(json.dumps(g.ret))) + 1
    return total


def run(model_path, tp, programs, ns, out, dump_dir=None, cap_max=24000):
    t0 = time.time()
    m = CWMvLLM(model_path, tp=tp, max_model_len=40960)
    print("== CWM loaded ==", flush=True)

    # Build all (program, N) jobs, then generate them as ONE batch so vLLM
    # decodes all traces in parallel (tight cap keeps runaway cheap).
    jobs = []
    for pname in programs:
        for n in ns:
            src, entry = PROGRAMS[pname](n)
            gt = trace_program(src, entry)
            cap = min(int(estimate_trace_tokens(m, gt) * 1.05) + 128, cap_max)
            prompt = build_prompt(m, src, [], force_event=Event.CALL)
            jobs.append({"pname": pname, "n": n, "src": src, "gt": gt,
                         "cap": cap, "prompt": prompt})
    print(f"== generating {len(jobs)} traces in one batch ==", flush=True)
    gens = m.gen_full_trace_batch([j["prompt"] for j in jobs], [j["cap"] for j in jobs])

    results = {"model": model_path, "programs": {p: {} for p in programs}}
    for j, gen in zip(jobs, gens):
        gen = [CALL_SEP] + gen
        pred = parse_full_trace(m, gen)
        sc = score_trace(j["gt"], pred, resolve_locals)
        sc.pop("per_frame", None)
        results["programs"][j["pname"]][j["n"]] = sc
        print(f"[{j['pname']} N={j['n']:3}] gt_frames={sc['n_gt']:4} pred={sc['n_pred']:4} "
              f"cap={j['cap']:5} first_div_frame={sc['first_divergence_frame']:4} "
              f"frame_acc={sc['frame_acc']:.3f} ctrl_acc={sc['control_acc']:.3f} "
              f"full={sc['fully_correct']}", flush=True)
        if dump_dir:
            with open(f"{dump_dir}/{j['pname']}_N{j['n']}.json", "w") as f:
                json.dump({"source": j["src"],
                           "gt": [(g.event, g.source_line, g.locals, g.ret) for g in j["gt"]],
                           "pred": [(p.event.name, p.source_line, resolve_locals(p), p.arg) for p in pred]},
                          f, indent=2)
    results["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--programs", default="counter,grid,list")
    ap.add_argument("--ns", default="5,10,20,40,80")
    ap.add_argument("--out", default="results/cwm_track.json")
    ap.add_argument("--dump_dir", default="results/cwm_traces")
    ap.add_argument("--cap_max", type=int, default=24000)
    a = ap.parse_args()
    import os
    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
    run(a.model_path, a.tp, a.programs.split(","),
        [int(x) for x in a.ns.split(",")], a.out, a.dump_dir, a.cap_max)
