"""Frozen-teacher VIABILITY PROBE for OPSD-on-CWM (NO training).

Per the rubber-duck design review, before committing training compute we must check
whether a PRIVILEGED teacher context is (a) genuinely better on the capability target
(oop) and (b) DISTRIBUTIONALLY HARMLESS on already-solved modes (multientity) -- because
if the privileged context perturbs solved modes, OPSD's "zero gradient where base is
right" anti-forgetting property breaks and forgetting is reintroduced (the whole point).

For each program (phi-EXPANDED) and each frame depth d, predict gold frame d under 3
teacher contexts and score top-1 frame accuracy vs ground truth:

  (1) tf        : teacher-forcing  [BOS,TCS, src, FS] + gold[:d]            -> frame d
                  (non-privileged; == the validated correct-prefix teacher)
  (2) retrace   : OPSD-literal privilege
                  [BOS,TCS,src,FS, FULL_gold] + [TCS,src,FS] + gold[:d]     -> frame d
                  (teacher has the whole gold trace in-context, then "restarts")
  (3) retrace_x : ablation -- same as (2) but FULL_gold is from a DIFFERENT program
                  (if this helps as much as (2), the signal is OOD-format reaction,
                   not real privileged information)

DECISION GATES (Option 1 / OPSD-literal viable only if ALL hold):
  - capability : acc_retrace(oop) >= acc_tf(oop)            (privilege helps, format not poison)
  - retention  : acc_retrace(multientity) ~= acc_tf(multientity)   (no drop on solved mode)
  - real-signal: acc_retrace(oop) >> acc_retrace_x(oop)     (uses the CORRECT gold, not just length)
If gates fail -> do NOT train Option 1; pivot to the hybrid (gold-prefix capability loss +
same-context base-consistency replay for anti-forgetting).
"""
from __future__ import annotations

import argparse
import json
import time
from statistics import mean

from vllm import TokensPrompt

from models.cwm_trace import (CWMvLLM, Event, FRAME_SEP, EOS, TRACE_CTX_START,
                              build_prompt, frame_to_tokens, parse_frame, resolve_locals)
from gt_trace import trace_program, gt_to_input_frames, score_frame
from failure_buckets import gen_oop, gen_multientity

GENS = {"oop": gen_oop, "multientity": gen_multientity}


def restart_segment(m, src, frames):
    """[TCS, src, FS, frames...] with NO leading BOS (the privileged 're-trace' marker)."""
    return build_prompt(m, src, frames)[1:]  # drop BOS=index0, keep TCS+src+FS+prefix


def full_gold_tokens(m, src, frames):
    """[BOS, TCS, src, FS, all gold frames...]  (no trailing EOS -> avoid 'trace is over')."""
    return build_prompt(m, src, frames)


def probes_for(m, src, cwm_frames, wrong_full_gold, max_prompt=23000, max_depths=25):
    """Build (tag, depth, prompt) probes for all 3 contexts over a depth sample."""
    gold_full = full_gold_tokens(m, src, cwm_frames)
    out = []
    depths = list(range(1, len(cwm_frames)))
    if len(depths) > max_depths:  # subsample evenly for long traces (e.g. multientity ~205)
        step = len(depths) / max_depths
        depths = sorted({depths[min(len(depths) - 1, int(i * step))] for i in range(max_depths)})
    for d in depths:
        tf = build_prompt(m, src, cwm_frames[:d], force_event=None)
        if len(tf) <= max_prompt:
            out.append(("tf", d, tf))
        seg = restart_segment(m, src, cwm_frames[:d])
        rt = gold_full + seg
        if len(rt) <= max_prompt:
            out.append(("retrace", d, rt))
        rx = wrong_full_gold + seg
        if len(rx) <= max_prompt:
            out.append(("retrace_x", d, rx))
    return out


def run(model_path, tp, buckets, n_per, seed, out):
    import random
    rng = random.Random(seed)
    m = CWMvLLM(model_path, tp=tp, max_model_len=24576)
    print("== CWM loaded (frozen viability probe, NO training) ==", flush=True)
    sp = m.SP(temperature=0.0, max_tokens=512, stop_token_ids=[FRAME_SEP, EOS])

    results = {"by_bucket": {}}
    for bucket in buckets:
        gen = GENS[bucket]
        # a pool of programs; one extra serves as the WRONG-gold donor
        progs = []
        seen = set()
        while len(progs) < n_per + 1 and len(seen) < (n_per + 1) * 10:
            src, entry = gen(rng)
            if src in seen:
                continue
            seen.add(src)
            gt = trace_program(src, entry, expand_objects=True)
            if gt:
                progs.append((src, entry, gt))
        donor_src, _, donor_gt = progs[-1]
        donor_full_gold = full_gold_tokens(m, donor_src, gt_to_input_frames(donor_gt))

        acc = {"tf": [], "retrace": [], "retrace_x": []}
        for (src, entry, gt) in progs[:n_per]:
            cwm_frames = gt_to_input_frames(gt)
            pr = probes_for(m, src, cwm_frames, donor_full_gold)
            outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for _, _, p in pr],
                                  sp, use_tqdm=False, **m._gen_kwargs())
            for (tag, d, _), o in zip(pr, outs):
                pf = parse_frame(m, list(o.outputs[0].token_ids) + [FRAME_SEP],
                                 forced_event=None, prev=cwm_frames[d - 1])
                ok = score_frame(gt[d], pf, resolve_locals)["frame_ok"] if pf else False
                acc[tag].append(ok)
        row = {k: round(mean(v), 3) if v else None for k, v in acc.items()}
        row["n_probes"] = {k: len(v) for k, v in acc.items()}
        results["by_bucket"][bucket] = row
        print(f"  [{bucket:13}] tf={row['tf']}  retrace={row['retrace']}  retrace_x={row['retrace_x']}"
              f"  (n={row['n_probes']['tf']})", flush=True)

    # decision gates
    oop = results["by_bucket"].get("oop", {})
    mul = results["by_bucket"].get("multientity", {})
    gates = {}
    if oop.get("tf") is not None and oop.get("retrace") is not None:
        gates["capability_retrace>=tf_oop"] = oop["retrace"] >= oop["tf"]
        gates["real_signal_retrace>>retrace_x_oop"] = (oop["retrace"] - (oop.get("retrace_x") or 0)) >= 0.15
    if mul.get("tf") is not None and mul.get("retrace") is not None:
        gates["retention_retrace~=tf_multientity"] = (mul["tf"] - mul["retrace"]) <= 0.05
    results["gates"] = gates
    verdict = "OPTION-1 VIABLE" if gates and all(gates.values()) else "OPTION-1 NOT VIABLE -> use hybrid"
    results["verdict"] = verdict
    print(f"\n  GATES: {gates}")
    print(f"  VERDICT: {verdict}")

    json.dump(results, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--buckets", default="oop,multientity")
    ap.add_argument("--n_per", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/opsd_viability_probe.json")
    a = ap.parse_args()
    run(a.model_path, a.tp, a.buckets.split(","), a.n_per, a.seed, a.out)
