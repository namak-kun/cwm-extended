"""Privilege-gap probe on FAILURE CASES (decides OPSD vs SFT vs RL with evidence).

For each program where CWM is suspected to fail, at every depth d:
  TEACHER (privileged): prompt = source + TRUE frames[0:d], predict frame d.
  STUDENT (drifted):    prompt = source + CWM's OWN free-rollout frames[0:d], predict frame d.
Both scored against the TRUE frame d. Diagnostic:
  teacher >> student            -> drift-induced failure -> OPSD/DAgger have signal
  teacher ~ student, both LOW   -> capability hole        -> SFT/RL needed (not distillation)
  teacher ~ student, both HIGH  -> not actually a failure case

Failure modes instantiated in PYTHON (so sys.settrace gives exact frame-level GT):
  - deep recursion (call-tree depth)             [§11 C fib analog]
  - OOP / encapsulated state behind methods      [§11 C++ Acc analog]
  - within-tick multi-entity side-effects        [§15 Lua arena analog]
  - long arithmetic accumulation
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from statistics import mean

from models.cwm_trace import (CWMvLLM, Event, FRAME_SEP, EOS, CALL_SEP,
                              build_prompt, parse_frame, parse_full_trace, resolve_locals)
from gt_trace import trace_program, gt_to_input_frames, score_frame
from run_cwm_track import estimate_trace_tokens


FAILURE_PROGRAMS = {
    # deep recursion: fibonacci, exploding call tree
    "recursion_fib": ('''def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def main():  # << START_OF_TRACE
    return fib(7)

main()
''', "main"),

    # OOP encapsulation: state hidden in object, mutated via methods
    "oop_encapsulation": ('''class Acc:
    def __init__(self):
        self.total = 0
        self.count = 0
    def add(self, v):
        self.total += v
        self.count += 1
    def mean(self):
        return self.total // max(1, self.count)

def main():  # << START_OF_TRACE
    a = Acc()
    for x in [3, 1, 4, 1, 5, 9, 2, 6]:
        a.add(x * x)
    return a.mean()

main()
''', "main"),

    # within-tick multi-entity side effects (the §15 Lua arena failure, in Python)
    "multientity_sideeffect": ('''def step(state, action):
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
    state = {"player": {"x": 3, "y": 3, "hp": 5},
             "enemies": [{"x": 1, "y": 1, "alive": True},
                         {"x": 5, "y": 5, "alive": True},
                         {"x": 2, "y": 4, "alive": True}]}
    for a in ["R", "D", "R", "U", "L", "D"]:
        state = step(state, a)
    return state["player"]["hp"] * 100 + state["player"]["x"] * 10 + state["player"]["y"]

main()
''', "main"),

    # long arithmetic accumulation (interacting accumulators, many steps)
    "long_arithmetic": ('''def main():  # << START_OF_TRACE
    a, b, c = 1, 0, 7
    for i in range(20):
        a = (a * 3 + b) % 97
        b = (b + c * 2) % 89
        c = (c + a) % 83
    return a * 10000 + b * 100 + c

main()
''', "main"),
}


def teacher_probes(m, jobs, stride=1, max_prompt=23000):
    from vllm import TokensPrompt
    probes = []
    for ji, j in enumerate(jobs):
        cf = j["cwm_frames"]
        for d in range(1, len(cf), stride):
            p = build_prompt(m, j["src"], cf[:d], force_event=None)
            if len(p) <= max_prompt:
                probes.append((ji, d, p))
    sp = m.SP(temperature=0.0, max_tokens=512, stop_token_ids=[FRAME_SEP, EOS])
    outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for _, _, p in probes], sp, use_tqdm=True)
    by_job = defaultdict(dict)
    for (ji, d, _), o in zip(probes, outs):
        gen = list(o.outputs[0].token_ids)
        j = jobs[ji]
        pf = parse_frame(m, gen + [FRAME_SEP], forced_event=None, prev=j["cwm_frames"][d-1])
        by_job[ji][d] = score_frame(j["gt"][d], pf, resolve_locals)["frame_ok"] if pf else False
    return by_job


def student_probes(m, jobs):
    """Build each job's drifted prefix from its OWN free rollout, then probe frame d
    from the drifted prefix. Free rollouts are BATCHED (parallel decode)."""
    from vllm import TokensPrompt
    # 1) free rollout each job — batched in one generate() call
    fr_prompts, fr_caps = [], []
    for j in jobs:
        fr_prompts.append(build_prompt(m, j["src"], [], force_event=Event.CALL))
        fr_caps.append(min(int(estimate_trace_tokens(m, j["gt"]) * 1.3) + 256, 12000))
    fr_sps = [m.SP(temperature=0.0, max_tokens=c, stop_token_ids=[EOS]) for c in fr_caps]
    fr_outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for p in fr_prompts],
                             fr_sps, use_tqdm=True)
    fr_frames = [parse_full_trace(m, [CALL_SEP] + list(o.outputs[0].token_ids)) for o in fr_outs]
    # 2) probe frame d from drifted prefix[0:d], scored vs TRUE frame d
    probes = []
    for ji, j in enumerate(jobs):
        df = fr_frames[ji]
        upto = min(len(j["gt"]), len(df))
        for d in range(1, upto):
            p = build_prompt(m, j["src"], df[:d], force_event=None)
            if len(p) <= 23000:
                probes.append((ji, d, p))
    sp = m.SP(temperature=0.0, max_tokens=512, stop_token_ids=[FRAME_SEP, EOS])
    outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for _, _, p in probes], sp, use_tqdm=True)
    by_job = defaultdict(dict)
    for (ji, d, _), o in zip(probes, outs):
        gen = list(o.outputs[0].token_ids)
        j = jobs[ji]
        prev = fr_frames[ji][d-1]
        pf = parse_frame(m, gen + [FRAME_SEP], forced_event=None, prev=prev)
        by_job[ji][d] = score_frame(j["gt"][d], pf, resolve_locals)["frame_ok"] if pf else False
    return by_job, fr_frames


def diagnose(teacher_acc, student_acc):
    if teacher_acc - student_acc > 0.15:
        return "DRIFT-INDUCED -> OPSD/DAgger have signal"
    if teacher_acc < 0.7:
        return "CAPABILITY HOLE -> SFT/RL (not distillation)"
    return "not a failure case (both high)"


def run(model_path, tp, out):
    t0 = time.time()
    m = CWMvLLM(model_path, tp=tp, max_model_len=24576)
    print("== CWM loaded ==", flush=True)

    jobs = []
    for nm, (src, entry) in FAILURE_PROGRAMS.items():
        gt = trace_program(src, entry)
        jobs.append({"name": nm, "src": src, "gt": gt, "cwm_frames": gt_to_input_frames(gt)})
        print(f"  {nm}: {len(gt)} true frames", flush=True)

    print("== teacher probes (privileged) ==", flush=True)
    teacher = teacher_probes(m, jobs)
    print("== student probes (free-rollout drifted prefix) ==", flush=True)
    student, fr_frames = student_probes(m, jobs)

    results = {"model": model_path, "by_program": {}}
    print("\nprogram                 | teacher | student |  gap  | diagnosis")
    for ji, j in enumerate(jobs):
        t_acc = mean(teacher[ji].values()) if teacher[ji] else 0.0
        s_acc = mean(student[ji].values()) if student[ji] else 0.0
        # also: did the free rollout actually drift? (frame match vs true by index)
        nmin = min(len(j["gt"]), len(fr_frames[ji]))
        free_ok = mean(score_frame(j["gt"][i], fr_frames[ji][i], resolve_locals)["frame_ok"]
                       for i in range(nmin)) if nmin else 0.0
        diag = diagnose(t_acc, s_acc)
        results["by_program"][j["name"]] = {
            "true_frames": len(j["gt"]), "pred_frames": len(fr_frames[ji]),
            "teacher_acc": round(t_acc, 3), "student_acc": round(s_acc, 3),
            "privilege_gap": round(t_acc - s_acc, 3),
            "free_rollout_acc": round(free_ok, 3), "diagnosis": diag}
        print(f"  {j['name']:22} |  {t_acc:.2f}   |  {s_acc:.2f}   | {t_acc-s_acc:+.2f} | {diag}")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--out", default="results/cwm_privilege_gap.json")
    a = ap.parse_args()
    run(a.model_path, a.tp, a.out)
