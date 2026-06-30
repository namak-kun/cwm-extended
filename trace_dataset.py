"""Trace dataset generator: turn the free oracle (Python interpreter) into CWM-format
training data. This is the substrate for SFT / OPSD / RL on CWM.

Each example = (prompt_tokens, target_tokens) where the model learns to predict the
execution trace of a program. Built from real sys.settrace ground truth, serialized
in CWM's native trace-token format. Tests assumption A6: free oracle => cheap data.

This module is training-framework-agnostic: it emits token-id sequences with a loss
mask (only trace tokens contribute to loss, not the source-code context).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from gt_trace import trace_program, GTFrame

# CWM trace token IDs (verified against facebook/cwm tokenizer)
BOS = 128000
EOS = 128001
TRACE_CTX_START = 128107
FRAME_SEP = 128100
ACTION_SEP = 128101
RETURN_SEP = 128102
CALL_SEP = 128103
LINE_SEP = 128104
EXCEPTION_SEP = 128105
ARG_SEP = 128106

EVT_TOK = {"call": CALL_SEP, "line": LINE_SEP, "return": RETURN_SEP, "exception": EXCEPTION_SEP}


def diff_locals(frame: GTFrame, prev: GTFrame | None) -> dict:
    """CWM uses a diff-based locals representation: only changed vars; '..' = unchanged."""
    if prev is None or prev.event in ("call", "return"):
        return {k: v for k, v in frame.locals.items()}
    out = {}
    for k, v in frame.locals.items():
        if k in prev.locals and prev.locals[k] == v:
            out[k] = ".."
        else:
            out[k] = v
    return out


def serialize_trace(tokenizer, source: str, gt: list[GTFrame]) -> dict:
    """Return {input_ids, labels} where labels=-100 on the context (source) tokens
    and = input_ids on the trace tokens (standard causal-LM masking)."""
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    ctx = [BOS, TRACE_CTX_START] + enc(source) + [FRAME_SEP]
    trace_toks: list[int] = []
    prev = None
    for f in gt:
        trace_toks.append(EVT_TOK[f.event])
        if f.event in ("call", "line"):
            trace_toks += enc(json.dumps(diff_locals(f, prev)))
        trace_toks += [ACTION_SEP]
        trace_toks += enc(f.source_line)
        if f.event in ("return", "exception"):
            trace_toks += [ARG_SEP]
            trace_toks += enc(json.dumps(f.ret))
        trace_toks += [FRAME_SEP]
        prev = f
    trace_toks.append(EOS)

    input_ids = ctx + trace_toks
    labels = [-100] * len(ctx) + trace_toks[:]   # learn the trace, not the context
    return {"input_ids": input_ids, "labels": labels, "n_ctx": len(ctx), "n_trace": len(trace_toks)}


# ---- program generators for a training corpus (deterministic, varied) ----
def gen_programs(n: int, seed: int = 0):
    rng = random.Random(seed)
    progs = []
    for i in range(n):
        kind = rng.choice(["accumulate", "transform", "search", "statemachine"])
        if kind == "accumulate":
            ops = rng.choice([("+", 1), ("*", 2), ("+", 3)])
            steps = rng.randint(3, 8)
            src = f'''def f():  # << START_OF_TRACE
    acc = {rng.randint(0,5)}
    for i in range({steps}):
        acc = (acc {ops[0]} {ops[1]}) % {rng.randint(50,100)}
    return acc

f()
'''
            entry = "f"
        elif kind == "transform":
            data = [rng.randint(0, 9) for _ in range(rng.randint(3, 6))]
            src = f'''def f():  # << START_OF_TRACE
    xs = {data}
    out = []
    total = 0
    for x in xs:
        y = x * {rng.randint(2,4)} - {rng.randint(0,3)}
        out.append(y)
        total += y
    return total

f()
'''
            entry = "f"
        elif kind == "search":
            data = [rng.randint(0, 20) for _ in range(rng.randint(4, 7))]
            tgt = rng.choice(data)
            src = f'''def f():  # << START_OF_TRACE
    xs = {data}
    target = {tgt}
    found = -1
    for i in range(len(xs)):
        if xs[i] == target:
            found = i
            break
    return found

f()
'''
            entry = "f"
        else:  # statemachine
            moves = [rng.choice(["U", "D", "L", "R"]) for _ in range(rng.randint(4, 8))]
            src = f'''def f():  # << START_OF_TRACE
    x, y = 0, 0
    for m in {moves}:
        if m == "U": y += 1
        elif m == "D": y -= 1
        elif m == "L": x -= 1
        elif m == "R": x += 1
    return x * 100 + y

f()
'''
            entry = "f"
        progs.append((f"{kind}_{i}", src, entry))
    return progs


def build_dataset(tokenizer, n_programs=200, seed=0, max_len=4096):
    progs = gen_programs(n_programs, seed)
    examples = []
    stats = {"total": 0, "kept": 0, "dropped_long": 0, "dropped_empty": 0, "trace_lens": []}
    for nm, src, entry in progs:
        stats["total"] += 1
        gt = trace_program(src, entry)
        if not gt:
            stats["dropped_empty"] += 1
            continue
        ex = serialize_trace(tokenizer, src, gt)
        if len(ex["input_ids"]) > max_len:
            stats["dropped_long"] += 1
            continue
        ex["name"] = nm
        examples.append(ex)
        stats["kept"] += 1
        stats["trace_lens"].append(ex["n_trace"])
    return examples, stats


if __name__ == "__main__":
    # self-test without the big tokenizer: use the CWM tokenizer if available else a stub
    import sys
    from transformers import AutoTokenizer
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        tok = AutoTokenizer.from_pretrained(path)
        ds, stats = build_dataset(tok, n_programs=50)
        print("dataset stats:", {k: v for k, v in stats.items() if k != "trace_lens"})
        if ds:
            import statistics
            print("trace token lengths: min/mean/max =",
                  min(stats["trace_lens"]), round(statistics.mean(stats["trace_lens"])), max(stats["trace_lens"]))
            print("example 0 name:", ds[0]["name"], "n_ctx:", ds[0]["n_ctx"], "n_trace:", ds[0]["n_trace"])
    else:
        print("pass the CWM model path to test serialization")
