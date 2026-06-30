"""How far can CWM track state? TEACHER-FORCED next-frame prediction at depth.

For each execution depth D, give CWM the TRUE trace prefix (frames 0..D-1) and
ask it to predict frame D. Compare to ground truth. This isolates "given a correct
history of length D, can CWM predict the next state?" and how that decays with D.

All (program, N, depth) probes are independent single-frame generations, batched
in one vLLM call -> fast even on slow tp=4 decode (short outputs, cached prefixes).
Also runs a couple of FREE rollouts (small N, tight cap) for the compounding view.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from statistics import mean

from models.cwm_trace import (CWMvLLM, Event, CALL_SEP, FRAME_SEP,
                              build_prompt, parse_frame, parse_full_trace, resolve_locals)
from gt_trace import (trace_program, gt_to_input_frames, score_frame, score_trace)
from run_cwm_track import PROGRAMS, estimate_trace_tokens


def teacher_forced(m, jobs, stride=1, max_depth=400):
    """Build all depth probes across jobs, batch-generate one frame each."""
    from vllm import TokensPrompt
    probes = []   # (job_idx, depth, prompt_ids)
    for ji, j in enumerate(jobs):
        cf = j["cwm_frames"]
        L = len(cf)
        depths = list(range(1, min(L, max_depth)))
        depths = depths[::stride] if stride > 1 else depths
        for d in depths:
            prompt = build_prompt(m, j["src"], cf[:d], force_event=None)
            probes.append((ji, d, prompt))
    sp = m.SP(temperature=0.0, max_tokens=512, stop_token_ids=[FRAME_SEP, m_eos()])
    outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for _, _, p in probes],
                          sp, use_tqdm=True)
    by_job = defaultdict(list)
    for (ji, d, _), o in zip(probes, outs):
        gen = list(o.outputs[0].token_ids)
        j = jobs[ji]
        pf = parse_frame(m, gen + [FRAME_SEP], forced_event=None,
                         prev=j["cwm_frames"][d - 1])
        if pf is None:
            sc = {"frame_ok": False, "ctrl": False, "vals_ok": False, "vals": (0, 0), "ret_ok": False}
        else:
            sc = score_frame(j["gt"][d], pf, resolve_locals)
        by_job[ji].append((d, sc))
    return by_job


def m_eos():
    from models.cwm_trace import EOS
    return EOS


def free_rollout_batch(m, jobs):
    """One free rollout per job (tight cap), batched."""
    prompts = [build_prompt(m, j["src"], [], force_event=Event.CALL) for j in jobs]
    caps = [min(int(estimate_trace_tokens(m, j["gt"]) * 1.05) + 64, 8000) for j in jobs]
    gens = m.gen_full_trace_batch(prompts, caps)
    out = []
    for j, gen in zip(jobs, gens):
        pred = parse_full_trace(m, [CALL_SEP] + gen)
        sc = score_trace(j["gt"], pred, resolve_locals)
        sc.pop("per_frame", None)
        out.append(sc)
    return out


def run(model_path, tp, specs, free_specs, out):
    t0 = time.time()
    m = CWMvLLM(model_path, tp=tp, max_model_len=24576)
    print("== CWM loaded ==", flush=True)

    jobs = []
    for pname, n in specs:
        src, entry = PROGRAMS[pname](n)
        gt = trace_program(src, entry)
        jobs.append({"pname": pname, "n": n, "src": src, "gt": gt,
                     "cwm_frames": gt_to_input_frames(gt)})
    print(f"== teacher-forced: {sum(len(j['cwm_frames'])-1 for j in jobs)} depth probes ==", flush=True)
    by_job = teacher_forced(m, jobs)

    results = {"model": model_path, "teacher_forced": {}, "free_rollout": {}}
    for ji, j in enumerate(jobs):
        rows = sorted(by_job[ji])
        # accuracy buckets by depth
        depths = [d for d, _ in rows]
        frame_ok = [sc["frame_ok"] for _, sc in rows]
        ctrl_ok = [sc["ctrl"] for _, sc in rows]
        first_fail = next((d for d, sc in rows if not sc["frame_ok"]), None)
        key = f"{j['pname']}_N{j['n']}"
        results["teacher_forced"][key] = {
            "n_frames": len(j["cwm_frames"]),
            "first_value_fail_depth": first_fail,
            "frame_acc_overall": round(mean(frame_ok), 3),
            "ctrl_acc_overall": round(mean(ctrl_ok), 3),
            "frame_acc_by_band": band_acc(rows),
        }
        print(f"[TF {key}] frames={len(j['cwm_frames'])} first_fail_depth={first_fail} "
              f"frame_acc={results['teacher_forced'][key]['frame_acc_overall']:.3f} "
              f"ctrl_acc={results['teacher_forced'][key]['ctrl_acc_overall']:.3f} "
              f"bands={results['teacher_forced'][key]['frame_acc_by_band']}", flush=True)

    # free rollouts
    if free_specs:
        fjobs = []
        for pname, n in free_specs:
            src, entry = PROGRAMS[pname](n)
            gt = trace_program(src, entry)
            fjobs.append({"pname": pname, "n": n, "src": src, "gt": gt})
        print(f"== free rollout: {len(fjobs)} traces ==", flush=True)
        scs = free_rollout_batch(m, fjobs)
        for j, sc in zip(fjobs, scs):
            key = f"{j['pname']}_N{j['n']}"
            results["free_rollout"][key] = sc
            print(f"[FREE {key}] gt={sc['n_gt']} pred={sc['n_pred']} "
                  f"first_div={sc['first_divergence_frame']} frame_acc={sc['frame_acc']:.3f} "
                  f"full={sc['fully_correct']}", flush=True)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out} ({results['elapsed_sec']}s)")


def band_acc(rows):
    """Mean frame accuracy in depth bands."""
    bands = [(1, 10), (11, 25), (26, 50), (51, 100), (101, 200), (201, 9999)]
    out = {}
    for lo, hi in bands:
        sel = [sc["frame_ok"] for d, sc in rows if lo <= d <= hi]
        if sel:
            out[f"{lo}-{hi if hi < 9999 else '+'}"] = round(mean(sel), 3)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--specs", default="counter:40,grid:30,list:40")
    ap.add_argument("--free", default="counter:12,grid:12,list:12")
    ap.add_argument("--out", default="results/cwm_depth.json")
    a = ap.parse_args()
    specs = [(s.split(":")[0], int(s.split(":")[1])) for s in a.specs.split(",")]
    free = [(s.split(":")[0], int(s.split(":")[1])) for s in a.free.split(",")] if a.free else []
    run(a.model_path, a.tp, specs, free, a.out)
