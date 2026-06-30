"""phi-expansion A/B test: does exposing hidden object state in the trace close the
oop privilege gap WITHOUT training?

For each program, build standard-phi and expanded-phi traces, then run the same
teacher(true prefix) + student(free rollout) probes on BOTH. If expanded-phi lifts
the STUDENT accuracy / closes the gap, the failure was observability (A1), fixable
by representation — no training needed for that mode.
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
from run_privilege_gap import FAILURE_PROGRAMS


def teacher_student(m, src, gt, max_prompt=23000):
    """Return (teacher_acc, student_acc, free_acc) for one program+phi."""
    from vllm import TokensPrompt
    cwm_frames = gt_to_input_frames(gt)

    # teacher: predict frame d from TRUE prefix
    t_probes = []
    for d in range(1, len(cwm_frames)):
        p = build_prompt(m, src, cwm_frames[:d], force_event=None)
        if len(p) <= max_prompt:
            t_probes.append((d, p))
    sp = m.SP(temperature=0.0, max_tokens=512, stop_token_ids=[FRAME_SEP, EOS])
    t_outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for _, p in t_probes], sp, use_tqdm=False, **m._gen_kwargs())
    t_ok = []
    for (d, _), o in zip(t_probes, t_outs):
        pf = parse_frame(m, list(o.outputs[0].token_ids) + [FRAME_SEP], forced_event=None, prev=cwm_frames[d-1])
        t_ok.append(score_frame(gt[d], pf, resolve_locals)["frame_ok"] if pf else False)

    # student: free rollout, then probe frame d from DRIFTED prefix
    fp = build_prompt(m, src, [], force_event=Event.CALL)
    cap = min(int(estimate_trace_tokens(m, gt) * 1.3) + 256, 12000)
    fr_out = m.llm.generate([TokensPrompt(prompt_token_ids=fp)],
                            m.SP(temperature=0.0, max_tokens=cap, stop_token_ids=[EOS]),
                            use_tqdm=False, **m._gen_kwargs())
    df = parse_full_trace(m, [CALL_SEP] + list(fr_out[0].outputs[0].token_ids))
    nmin = min(len(gt), len(df))
    free_ok = [score_frame(gt[i], df[i], resolve_locals)["frame_ok"] for i in range(nmin)]
    s_probes = []
    for d in range(1, nmin):
        p = build_prompt(m, src, df[:d], force_event=None)
        if len(p) <= max_prompt:
            s_probes.append((d, p))
    s_outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for _, p in s_probes], sp, use_tqdm=False, **m._gen_kwargs())
    s_ok = []
    for (d, _), o in zip(s_probes, s_outs):
        pf = parse_frame(m, list(o.outputs[0].token_ids) + [FRAME_SEP], forced_event=None, prev=df[d-1])
        s_ok.append(score_frame(gt[d], pf, resolve_locals)["frame_ok"] if pf else False)

    return (round(mean(t_ok), 3) if t_ok else 0.0,
            round(mean(s_ok), 3) if s_ok else 0.0,
            round(mean(free_ok), 3) if free_ok else 0.0,
            len(gt), len(df))


def run(model_path, tp, names, out, lora=None):
    t0 = time.time()
    m = CWMvLLM(model_path, tp=tp, max_model_len=24576, lora_path=lora)
    print(f"== CWM loaded {'+ LoRA ' + lora if lora else '(base)'} ==", flush=True)

    results = {"model": model_path, "by_program": {}}
    print(f"\n{'program':22} {'phi':9} {'teacher':>7} {'student':>7} {'gap':>6} {'free':>6}")
    for nm in names:
        src, entry = FAILURE_PROGRAMS[nm]
        row = {}
        for phi in ("standard", "expanded"):
            gt = trace_program(src, entry, expand_objects=(phi == "expanded"))
            t, s, f, ngt, npred = teacher_student(m, src, gt)
            row[phi] = {"teacher": t, "student": s, "gap": round(t - s, 3),
                        "free_acc": f, "true_frames": ngt, "pred_frames": npred}
            print(f"  {nm:20} {phi:9} {t:>7} {s:>7} {t-s:>+6} {f:>6}", flush=True)
        results["by_program"][nm] = row
        # the headline: did expansion close the gap / lift the student?
        d_student = row["expanded"]["student"] - row["standard"]["student"]
        d_gap = row["standard"]["gap"] - row["expanded"]["gap"]
        print(f"    -> phi-expansion: student {row['standard']['student']}->{row['expanded']['student']} "
              f"(Δ{d_student:+.2f}), gap {row['standard']['gap']}->{row['expanded']['gap']} (closed {d_gap:+.2f})\n", flush=True)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(results, open(out, "w"), indent=2)
    print(f"saved -> {out} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--names", default="oop_encapsulation,multientity_sideeffect")
    ap.add_argument("--out", default="results/cwm_phi_expansion.json")
    ap.add_argument("--lora", default=None)
    a = ap.parse_args()
    run(a.model_path, a.tp, a.names.split(","), a.out, a.lora)
