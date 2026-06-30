"""Build SFT training data for CWM: phi-EXPANDED traces (the §20 conclusion — the
expanded observation format must be TRAINED IN, not swapped at inference).

Produces tokenized {input_ids, labels} examples (labels=-100 on the source context,
= input_ids on the trace) saved to disk for the LoRA trainer. Targets the
encapsulation bucket first (the clean A/B case), with optional easy-replay mix.
"""
from __future__ import annotations

import argparse
import json
import random

from transformers import AutoTokenizer

from gt_trace import trace_program
from trace_dataset import serialize_trace
from failure_buckets import gen_oop, gen_easy, gen_arithmetic, gen_recursion, gen_multientity, gen_multientity_short
from game_tick import gen_game_tick, gen_game_tick_short

GENS = {"oop": gen_oop, "easy": gen_easy, "arithmetic": gen_arithmetic,
        "recursion": gen_recursion, "multientity": gen_multientity,
        "multientity_short": gen_multientity_short, "game_tick": gen_game_tick, "game_tick_short": gen_game_tick_short}


def build(tokenizer, bucket_weights: dict, n: int, seed: int, expand: bool, max_len: int,
          stepover_depth: int | None = None):
    rng = random.Random(seed)
    names = list(bucket_weights)
    probs = [bucket_weights[k] for k in names]
    examples, stats = [], {"total": 0, "kept": 0, "dropped_long": 0, "by_bucket": {}}
    seen = set()
    attempts = 0
    while len(examples) < n and attempts < n * 6:
        attempts += 1
        bucket = rng.choices(names, probs)[0]
        src, entry = GENS[bucket](rng)
        if src in seen:
            continue
        seen.add(src)
        stats["total"] += 1
        # step-over abstraction only applies to game_tick buckets (they have a step() boundary);
        # other buckets stay full-trace even in a mixed corpus.
        so = stepover_depth if bucket.startswith("game_tick") else None
        gt = trace_program(src, entry, expand_objects=expand, stepover_depth=so)
        if not gt:
            continue
        ex = serialize_trace(tokenizer, src, gt)
        if len(ex["input_ids"]) > max_len:
            stats["dropped_long"] += 1
            continue
        ex["bucket"] = bucket
        examples.append(ex)
        stats["kept"] += 1
        stats["by_bucket"][bucket] = stats["by_bucket"].get(bucket, 0) + 1
    return examples, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expand", action="store_true", help="phi-expand object state (the SFT target)")
    ap.add_argument("--buckets", default="oop:0.85,easy:0.15",
                    help="bucket:weight,... (easy = anti-forgetting replay)")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--stepover", type=int, default=None,
                    help="step-over depth for game_tick buckets (1 = tick-level abstraction target)")
    ap.add_argument("--out", default="data/sft_oop_expanded.jsonl")
    a = ap.parse_args()

    weights = {}
    for part in a.buckets.split(","):
        k, v = part.split(":")
        weights[k] = float(v)

    tok = AutoTokenizer.from_pretrained(a.model_path)
    examples, stats = build(tok, weights, a.n, a.seed, a.expand, a.max_len, a.stepover)

    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    import statistics
    lens = [len(e["input_ids"]) for e in examples]
    print(f"built {len(examples)} examples (expand={a.expand}) -> {a.out}")
    print(f"  stats: {stats}")
    print(f"  token lengths: min/mean/max = {min(lens)}/{round(statistics.mean(lens))}/{max(lens)}")
    # show one expanded example's trace head to confirm object state is visible
    if examples:
        ids = examples[0]["input_ids"]
        print("  example 0 decoded tail:", tok.decode(ids[-120:], skip_special_tokens=False)[:300])


if __name__ == "__main__":
    main()
