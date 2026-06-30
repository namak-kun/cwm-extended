"""CWM-NATIVE assumption tests (A2, A3) — measured on the REAL model, not a stand-in.

Why native: A3 (privilege gap) and A2 (drift validity) are CWM-SPECIFIC properties.
Testing them on a non-trace-trained stand-in (any Qwen) is an unjustified transfer.
CWM is the model we'd train, so we measure it directly.

A3 — privilege gap (the OPSD premise): at each depth d, compare frame-d prediction
   accuracy given the TRUE prefix (teacher/privileged) vs given CWM's OWN free-rollout
   prefix (student/drifted), both scored against the true frame d. Big, growing gap =>
   OPSD/on-policy distillation from a privileged teacher has real signal.
   (This refines §8's teacher-forced-vs-free result into a matched-depth gap curve.)

A2 — drift validity (oracle-at-arbitrary-states): are CWM's free-rollout frames still
   STRUCTURALLY VALID (parseable locals, correct variable set) as depth grows? If
   drifted frames are malformed, no oracle (DAgger/OPSD/RL/re-ground) can relabel them.

Batched teacher probes (fast, prefix-cached); one free rollout per program (capped).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from statistics import mean

from models.cwm_trace import (CWMvLLM, Event, FRAME_SEP, EOS, CALL_SEP,
                              build_prompt, parse_frame, parse_full_trace, resolve_locals)
from gt_trace import trace_program, gt_to_input_frames, score_frame, score_trace, EVT_MAP
from run_cwm_track import PROGRAMS, estimate_trace_tokens


def teacher_probes(m, jobs, stride=1):
    """Batched: predict frame d from the TRUE prefix[0:d], for all d, all jobs."""
    from vllm import TokensPrompt
    probes = []
    for ji, j in enumerate(jobs):
        cf = j["cwm_frames"]
        for d in range(1, len(cf), stride):
            probes.append((ji, d, build_prompt(m, j["src"], cf[:d], force_event=None)))
    sp = m.SP(temperature=0.0, max_tokens=512, stop_token_ids=[FRAME_SEP, EOS])
    outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for _, _, p in probes],
                          sp, use_tqdm=True)
    by_job = defaultdict(dict)
    for (ji, d, _), o in zip(probes, outs):
        gen = list(o.outputs[0].token_ids)
        j = jobs[ji]
        pf = parse_frame(m, gen + [FRAME_SEP], forced_event=None, prev=j["cwm_frames"][d-1])
        ok = score_frame(j["gt"][d], pf, resolve_locals)["frame_ok"] if pf else False
        by_job[ji][d] = ok
    return by_job


def a2_structural_valid(frame, ref_keys) -> bool:
    """A2: is a free-rollout frame a usable runtime state? parseable locals + right vars."""
    lv = resolve_locals(frame)
    if not isinstance(lv, dict):
        return False
    if any(v == "_PARSE_ERR_" for v in lv.keys()):
        return False
    # for LINE/CALL frames, the variable SET should match the true schema at that scope
    if frame.event in (Event.LINE, Event.CALL):
        return len(lv) > 0 and all(isinstance(k, str) for k in lv)
    return True


def run(model_path, tp, specs, out):
    t0 = time.time()
    m = CWMvLLM(model_path, tp=tp, max_model_len=24576)
    print("== CWM loaded ==", flush=True)

    jobs = []
    for pname, n in specs:
        src, entry = PROGRAMS[pname](n)
        gt = trace_program(src, entry)
        jobs.append({"pname": pname, "n": n, "src": src, "gt": gt,
                     "cwm_frames": gt_to_input_frames(gt)})

    # ---- A3 teacher arm (privileged: true prefix) ----
    print(f"== A3 teacher probes ({sum(len(j['cwm_frames'])-1 for j in jobs)}) ==", flush=True)
    teacher = teacher_probes(m, jobs)

    # ---- student arm: free rollout (drifted prefix) + A2 validity ----
    results = {"model": model_path, "by_program": {}}
    teach_by_depth, stud_by_depth, a2_by_depth = defaultdict(list), defaultdict(list), defaultdict(list)
    for ji, j in enumerate(jobs):
        prompt = build_prompt(m, j["src"], [], force_event=Event.CALL)
        cap = min(int(estimate_trace_tokens(m, j["gt"]) * 1.1) + 128, 12000)
        gen = m.gen_full_trace_tokens(prompt, max_tokens=cap)
        pred = parse_full_trace(m, [CALL_SEP] + gen)
        ref_keys = set(j["gt"][0].locals.keys())

        # bucket by depth: teacher acc, student (free) acc vs true-by-index, A2 validity
        depth_band = lambda d: ("1-10" if d <= 10 else "11-25" if d <= 25 else
                                "26-50" if d <= 50 else "51-100" if d <= 100 else "100+")
        prog = {"n_true": len(j["gt"]), "n_pred": len(pred),
                "teacher_acc": round(mean(teacher[ji].values()), 3) if teacher[ji] else 0.0}
        # student frame acc vs true (by index)
        nmin = min(len(j["gt"]), len(pred))
        stud_ok = [score_frame(j["gt"][i], pred[i], resolve_locals)["frame_ok"] for i in range(nmin)]
        prog["student_free_acc"] = round(mean(stud_ok), 3) if stud_ok else 0.0
        # A2 validity by depth
        a2_ok = [a2_structural_valid(pred[i], ref_keys) for i in range(len(pred))]
        prog["a2_valid_rate"] = round(mean(a2_ok), 3) if a2_ok else 0.0
        prog["first_invalid_depth"] = next((i for i, v in enumerate(a2_ok) if not v), None)
        results["by_program"][f"{j['pname']}_N{j['n']}"] = prog

        for d, ok in teacher[ji].items():
            teach_by_depth[depth_band(d)].append(ok)
        for i in range(nmin):
            stud_by_depth[depth_band(i)].append(stud_ok[i])
        for i in range(len(pred)):
            a2_by_depth[depth_band(i)].append(a2_ok[i])

    bands = ["1-10", "11-25", "26-50", "51-100", "100+"]
    results["A3_privilege_gap_by_band"] = {}
    results["A2_valid_by_band"] = {}
    print("\nband      | A3 teacher(true) | A3 student(drift) | GAP | A2 valid")
    for b in bands:
        if not teach_by_depth[b] and not stud_by_depth[b]:
            continue
        t_acc = mean(teach_by_depth[b]) if teach_by_depth[b] else float("nan")
        s_acc = mean(stud_by_depth[b]) if stud_by_depth[b] else float("nan")
        a2 = mean(a2_by_depth[b]) if a2_by_depth[b] else float("nan")
        results["A3_privilege_gap_by_band"][b] = {
            "teacher": round(t_acc, 3), "student": round(s_acc, 3), "gap": round(t_acc - s_acc, 3)}
        results["A2_valid_by_band"][b] = round(a2, 3)
        print(f"  {b:7} |      {t_acc:.2f}        |       {s_acc:.2f}         | {t_acc-s_acc:+.2f}|  {a2:.2f}")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--specs", default="counter:30,grid:24,list:30")
    ap.add_argument("--out", default="results/cwm_assumptions.json")
    a = ap.parse_args()
    specs = [(s.split(":")[0], int(s.split(":")[1])) for s in a.specs.split(",")]
    run(a.model_path, a.tp, specs, a.out)
