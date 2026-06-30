"""Build OPSD (on-policy self-distillation) data for CWM trace prediction.

Unlike build_sft_data.py (which emits fully-tokenized {input_ids,labels} SFT targets),
OPSD is ON-POLICY: the student GENERATES the trace at train time. So each row only needs
the raw pieces the self-distill collator turns into student/teacher prompts:

  - source         : the program text (the student's non-privileged context)
  - entry          : trace entry fn
  - ctx_ids        : [BOS, TRACE_CTX_START] + enc(source) + [FRAME_SEP]   (== student prompt)
  - gold_ids       : the gold trace tokens (frames + EOS) -- the PRIVILEGED content the
                     teacher conditions on (phi-EXPANDED, per the section-20 conclusion).
  - n_frames       : number of gold frames (for bookkeeping / curriculum)
  - bucket         : failure-mode tag

This is collator-agnostic: whether privilege is injected as "gold trace in teacher prompt"
(OPSD-literal) or as "correct-prefix teacher forcing", both derive from (ctx_ids, gold_ids).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics

from transformers import AutoTokenizer

from gt_trace import trace_program
from trace_dataset import serialize_trace
from failure_buckets import (
    gen_oop, gen_easy, gen_arithmetic, gen_recursion, gen_multientity,
)

GENS = {
    "oop": gen_oop, "easy": gen_easy, "arithmetic": gen_arithmetic,
    "recursion": gen_recursion, "multientity": gen_multientity,
}


def build(tokenizer, bucket_weights: dict, n: int, seed: int, expand: bool, max_len: int):
    rng = random.Random(seed)
    names = list(bucket_weights)
    probs = [bucket_weights[k] for k in names]
    rows, stats = [], {"total": 0, "kept": 0, "dropped_long": 0, "by_bucket": {}}
    seen = set()
    attempts = 0
    while len(rows) < n and attempts < n * 8:
        attempts += 1
        bucket = rng.choices(names, probs)[0]
        src, entry = GENS[bucket](rng)
        if src in seen:
            continue
        seen.add(src)
        stats["total"] += 1
        gt = trace_program(src, entry, expand_objects=expand)
        if not gt:
            continue
        ser = serialize_trace(tokenizer, src, gt)
        n_ctx = ser["n_ctx"]
        ctx_ids = ser["input_ids"][:n_ctx]
        gold_ids = ser["input_ids"][n_ctx:]
        if len(ctx_ids) + len(gold_ids) > max_len:
            stats["dropped_long"] += 1
            continue
        rows.append({
            "source": src, "entry": entry, "bucket": bucket,
            "ctx_ids": ctx_ids, "gold_ids": gold_ids, "n_frames": len(gt),
        })
        stats["kept"] += 1
        stats["by_bucket"][bucket] = stats["by_bucket"].get(bucket, 0) + 1
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expand", action="store_true",
                    help="phi-expand object state (the section-20 SFT/OPSD target)")
    ap.add_argument("--buckets", default="oop:1.0",
                    help="bucket:weight,...  (OPSD relies on STRUCTURAL anti-forgetting, "
                         "so default is oop-only; add easy/multientity to test replay)")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--out", default="data/opsd_oop_expanded.jsonl")
    a = ap.parse_args()

    weights = {}
    for part in a.buckets.split(","):
        k, v = part.split(":")
        weights[k] = float(v)

    tok = AutoTokenizer.from_pretrained(a.model_path)
    rows, stats = build(tok, weights, a.n, a.seed, a.expand, a.max_len)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ctx_lens = [len(r["ctx_ids"]) for r in rows]
    gold_lens = [len(r["gold_ids"]) for r in rows]
    print(f"built {len(rows)} OPSD rows (expand={a.expand}) -> {a.out}")
    print(f"  stats: {stats}")
    if rows:
        print(f"  ctx  tokens min/mean/max = {min(ctx_lens)}/{round(statistics.mean(ctx_lens))}/{max(ctx_lens)}")
        print(f"  gold tokens min/mean/max = {min(gold_lens)}/{round(statistics.mean(gold_lens))}/{max(gold_lens)}")
        # confirm phi-expansion visible in the gold (teacher's privileged content)
        print("  row0 gold head:", tok.decode(rows[0]["gold_ids"][:80], skip_special_tokens=False)[:280])


if __name__ == "__main__":
    main()
