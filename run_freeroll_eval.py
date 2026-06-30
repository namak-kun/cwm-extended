"""Held-out free-rollout eval for the DAgger A/B (and any CWM adapter).

The metric that matters for "predict how execution evolves under input" is FREE-ROLLOUT frame
accuracy on UNSEEN programs (compounding-drift robustness), not teacher-forced per-frame. This
generates N held-out programs (a seed disjoint from training), free-rolls each (one batched vLLM
call), and reports mean frame accuracy + fully-correct-rollout rate vs phi-expanded ground truth.
"""
from __future__ import annotations

import argparse
import json
import random
from statistics import mean

from vllm import TokensPrompt

from models.cwm_trace import (CWMvLLM, Event, EOS, CALL_SEP,
                              build_prompt, parse_full_trace, resolve_locals)
from gt_trace import trace_program, score_frame
from run_cwm_track import estimate_trace_tokens
from failure_buckets import (gen_oop, gen_multientity_short, gen_arithmetic,
                             gen_recursion, gen_easy)

GENS = {"oop": gen_oop, "multientity_short": gen_multientity_short,
        "arithmetic": gen_arithmetic, "recursion": gen_recursion, "easy": gen_easy}


def run(model_path, tp, bucket, n, seed, expand, lora, out):
    rng = random.Random(seed)
    gen = GENS[bucket]
    progs, seen = [], set()
    while len(progs) < n and len(seen) < n * 10:
        src, entry = gen(rng)
        if src in seen:
            continue
        seen.add(src)
        gt = trace_program(src, entry, expand_objects=expand)
        if gt and len(gt) > 3:
            progs.append((src, entry, gt))

    # NOTE: vLLM (0.23) does NOT switch between LoRA adapters by per-request LoRARequest id within
    # one engine -- it applies whatever adapter the engine was initialized with. lora_request=None
    # DOES correctly disable the adapter (-> base). So we reliably eval exactly TWO conditions per
    # process: base (no adapter) + the single init adapter. Run once per adapter for a clean A/B.
    m = CWMvLLM(model_path, tp=tp, max_model_len=24576, lora_path=(None if lora == "base" else lora))
    print(f"== CWM loaded | held-out free-roll {bucket} n={len(progs)} seed={seed} | adapter={lora} ==", flush=True)

    fps = [build_prompt(m, src, [], force_event=Event.CALL) for (src, _, _) in progs]
    maxcap = min(max(int(estimate_trace_tokens(m, gt) * 1.3) + 256 for (_, _, gt) in progs), 12000)
    sp = m.SP(temperature=0.0, max_tokens=maxcap, stop_token_ids=[EOS])

    conditions = [("base", {})]
    if lora != "base":
        conditions.append((lora, m._gen_kwargs()))  # the init adapter, via the proven native path

    results = {"bucket": bucket, "n": len(progs), "seed": seed, "by_adapter": {}}
    for name, gkw in conditions:
        outs = m.llm.generate([TokensPrompt(prompt_token_ids=fp) for fp in fps], sp,
                              use_tqdm=False, **gkw)
        per_prog, full_ok = [], 0
        for (src, entry, gt), o in zip(progs, outs):
            df = parse_full_trace(m, [CALL_SEP] + list(o.outputs[0].token_ids))
            nmin = min(len(gt), len(df))
            ok = [score_frame(gt[i2], df[i2], resolve_locals)["frame_ok"] for i2 in range(nmin)]
            denom = max(len(gt), len(df))
            acc = sum(ok) / denom if denom else 0.0
            per_prog.append(acc)
            if acc >= 0.999 and len(df) == len(gt):
                full_ok += 1
        row = {
            "free_roll_frame_acc": round(mean(per_prog), 4) if per_prog else None,
            "fully_correct_rollout_rate": round(full_ok / len(progs), 4) if progs else None,
            "per_prog_min": round(min(per_prog), 3) if per_prog else None,
            "per_prog_max": round(max(per_prog), 3) if per_prog else None,
        }
        results["by_adapter"][name] = row
        print(f"  [{name:32}] free_roll_acc={row['free_roll_frame_acc']}  "
              f"fully_correct={row['fully_correct_rollout_rate']}  "
              f"(min {row['per_prog_min']}/max {row['per_prog_max']})", flush=True)

    json.dump(results, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--bucket", default="oop")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=999, help="held-out seed (disjoint from training seed)")
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--lora", default="base", help="single adapter path, or 'base'")
    ap.add_argument("--out", default="results/freeroll_eval.json")
    a = ap.parse_args()
    run(a.model_path, a.tp, a.bucket, a.n, a.seed, a.expand, a.lora, a.out)
