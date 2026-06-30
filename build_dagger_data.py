"""Build DAgger / matched gold-vs-drift training data for CWM (the decisive on-policy test).

Per the synthesis rubber-duck: the real fork for fixing FREE-ROLLOUT DRIFT is the INPUT-STATE
DISTRIBUTION, with gold targets either way:
  - Arm A (gold) : input = GOLD prefix gt[:d],  target = gold frame d   (== per-frame SFT control)
  - Arm B (drift): input = STUDENT-ROLLED prefix df[:d], target = gold frame d   (DAgger recovery)
Both arms: same programs, same depths, same target frames, same token budget. Only the prefix differs.

The target frame d is diff-encoded against the ACTUAL previous frame in the input (gt[d-1] for gold,
df[d-1] for drift) so the CWM '..'=unchanged convention references the visible state -- in drift mode
this makes the target a genuine RECOVERY frame (corrected vars marked changed).

Output: tokenized {input_ids, labels} jsonl (labels=-100 on the prefix, = target on the gold frame),
identical format to build_sft_data.py -> train_lora_cwm.py trains on it directly.

gold mode is CPU-only (no rollout). drift mode needs vLLM (rolls out the student, optionally --lora).
"""
from __future__ import annotations

import argparse
import json
import os
import random

from transformers import AutoTokenizer

from gt_trace import trace_program, gt_to_input_frames, GTFrame
from models.cwm_trace import (Event, Frame, FRAME_SEP, EOS, CALL_SEP,
                              frame_to_tokens, build_prompt, parse_full_trace, resolve_locals,
                              BOS, TRACE_CTX_START, ACTION_SEP)
from failure_buckets import gen_oop, gen_multientity_short, gen_arithmetic, gen_recursion, gen_easy

GENS = {"oop": gen_oop, "multientity_short": gen_multientity_short,
        "arithmetic": gen_arithmetic, "recursion": gen_recursion, "easy": gen_easy}

_EVT_FROM_STR = {"call": Event.CALL, "line": Event.LINE,
                 "return": Event.RETURN, "exception": Event.EXCEPTION}


class TokShim:
    """Minimal model-like shim exposing .encode/.decode for build_prompt/frame_to_tokens (gold mode)."""
    def __init__(self, tok):
        self.tok = tok

    def encode(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


def recovery_diff(gold_gt: GTFrame, prev_full: dict | None, prev_event: str | None) -> dict:
    """gold frame d's locals diff-encoded against the ACTUAL previous frame's FULL locals.
    Mirrors trace_dataset.diff_locals but with an arbitrary (possibly drifted) prev."""
    if prev_full is None or prev_event in ("call", "return"):
        return dict(gold_gt.locals)
    out = {}
    for k, v in gold_gt.locals.items():
        out[k] = ".." if (k in prev_full and prev_full[k] == v) else v
    return out


def target_frame_tokens(m, gold_gt: GTFrame, prev_full: dict | None, prev_event: str | None) -> list[int]:
    """Serialize the gold frame d as a CWM frame, diffed against the visible prev (recovery target)."""
    ev = _EVT_FROM_STR[gold_gt.event]
    lv = recovery_diff(gold_gt, prev_full, prev_event) if ev in (Event.CALL, Event.LINE) else {}
    f = Frame(event=ev, source_line=gold_gt.source_line, local_vars=lv, arg=gold_gt.ret, prev=None)
    return frame_to_tokens(m, f) + [FRAME_SEP]


def make_examples(m, src, gt, prefix_frames, prefix_full_locals, prefix_events, max_depths, max_len):
    """One per-frame example per depth d: [prefix[:d]] -> gold frame d.
    prefix_frames: list of Frame for the input prefix (gold or drifted).
    prefix_full_locals[i], prefix_events[i]: full locals dict + event-str of prefix frame i."""
    gold_frames = gt_to_input_frames(gt)
    n = min(len(gt), len(prefix_frames))
    depths = list(range(1, n))
    if len(depths) > max_depths:
        step = len(depths) / max_depths
        depths = sorted({depths[min(len(depths) - 1, int(i * step))] for i in range(max_depths)})
    out = []
    for d in depths:
        prefix_tok = build_prompt(m, src, prefix_frames[:d])  # [BOS,TCS,src,FS, prefix 0..d-1] FS-terminated
        prev_full = prefix_full_locals[d - 1]
        prev_event = prefix_events[d - 1]
        tgt = target_frame_tokens(m, gt[d], prev_full, prev_event)
        ids = prefix_tok + tgt
        if len(ids) > max_len:
            continue
        labels = [-100] * len(prefix_tok) + tgt
        out.append({"input_ids": ids, "labels": labels, "depth": d, "n_prefix": len(prefix_tok)})
    return out


def gold_prefix_meta(gt):
    """For gold mode: prefix == gold frames; full locals + events straight from GT."""
    return [dict(f.locals) for f in gt], [f.event for f in gt]


def drift_prefix_meta(df_frames):
    """For drift mode: prefix == parsed rollout frames; full locals via resolve_locals."""
    full = [resolve_locals(f) for f in df_frames]
    events = [f.event.name.lower() for f in df_frames]
    return full, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--mode", choices=["gold", "drift"], required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--lora", default=None, help="adapter to roll out as the student (drift mode)")
    ap.add_argument("--n", type=int, default=200, help="number of programs")
    ap.add_argument("--buckets", default="oop:1.0")
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--max_depths", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/dagger_oop.jsonl")
    a = ap.parse_args()

    weights = {}
    for part in a.buckets.split(","):
        k, v = part.split(":")
        weights[k] = float(v)
    names = list(weights); probs = [weights[k] for k in names]
    rng = random.Random(a.seed)

    # programs
    progs, seen = [], set()
    while len(progs) < a.n and len(seen) < a.n * 10:
        b = rng.choices(names, probs)[0]
        src, entry = GENS[b](rng)
        if src in seen:
            continue
        seen.add(src)
        gt = trace_program(src, entry, expand_objects=a.expand)
        if gt and len(gt) > 2:
            progs.append((src, entry, gt))

    if a.mode == "gold":
        m = TokShim(AutoTokenizer.from_pretrained(a.model_path))
        rows = []
        for (src, entry, gt) in progs:
            full, events = gold_prefix_meta(gt)
            rows += make_examples(m, src, gt, gt_to_input_frames(gt), full, events, a.max_depths, a.max_len)
    else:  # drift: roll out the student
        from models.cwm_trace import CWMvLLM
        from vllm import TokensPrompt
        from run_cwm_track import estimate_trace_tokens
        m = CWMvLLM(a.model_path, tp=a.tp, max_model_len=24576, lora_path=a.lora)
        print(f"== CWM loaded {'+LoRA '+a.lora if a.lora else '(base)'} for DAgger rollout ==", flush=True)
        rows = []
        # BATCH all free rollouts into ONE generate call (vLLM continuous batching) -- serial
        # single-sequence decode is comms-bound on this no-NVLink box and ~10x slower.
        fps = [build_prompt(m, src, [], force_event=Event.CALL) for (src, _, _) in progs]
        maxcap = min(max(int(estimate_trace_tokens(m, gt) * 1.3) + 256 for (_, _, gt) in progs), 12000)
        sp = m.SP(temperature=0.0, max_tokens=maxcap, stop_token_ids=[EOS])
        outs = m.llm.generate([TokensPrompt(prompt_token_ids=fp) for fp in fps], sp,
                              use_tqdm=True, **m._gen_kwargs())
        for (src, entry, gt), o in zip(progs, outs):
            df = parse_full_trace(m, [CALL_SEP] + list(o.outputs[0].token_ids))
            if len(df) < 2:
                continue
            full, events = drift_prefix_meta(df)
            rows += make_examples(m, src, gt, df, full, events, a.max_depths, a.max_len)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps({"input_ids": r["input_ids"], "labels": r["labels"]}) + "\n")
    import statistics as st
    lens = [len(r["input_ids"]) for r in rows]
    sup = [sum(1 for x in r["labels"] if x != -100) for r in rows]
    print(f"[{a.mode}] built {len(rows)} per-frame examples from {len(progs)} programs -> {a.out}")
    if rows:
        print(f"  input len min/med/max = {min(lens)}/{int(st.median(lens))}/{max(lens)}")
        print(f"  supervised toks/example med = {int(st.median(sup))}")


if __name__ == "__main__":
    main()
