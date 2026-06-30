"""Build step-over SFT data for the cascading-validation gap (REPORT §36).

Trains CWM to one-shot the dispatch effect (the cascade result) on PYTHON cascade data (cascade_py + uidom),
held-out split for in-dist eval. JS cascade (cascade_js) + real vanilla form-validator are kept ENTIRELY for
CROSS-LANGUAGE / real-app transfer eval (never trained) -- the §28 transfer test.

SFT example = serialize_trace of trace_program(prompt_src + bare main(), stepover_depth=1): main's lines +
dispatch CALL/RETURN(truth_state), interior abstracted. (Harvested prompt_src guards main() under
__name__=="__main__", which is false under exec -> append a bare main() so the tracer runs it.)
"""
import argparse, json, random
from transformers import AutoTokenizer
from gt_trace import trace_program
from trace_dataset import serialize_trace


def trace_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def build_examples(tok, rows, max_len, expand=False):
    out, dropped = [], 0
    for r in rows:
        src = r["prompt_src"]
        if "\nmain()\n" not in src and not src.rstrip().endswith("main()"):
            src = src + "\nmain()\n"
        try:
            gt = trace_program(src, r.get("entry", "main"), stepover_depth=1)
            if not gt:
                dropped += 1
                continue
            ex = serialize_trace(tok, r["prompt_src"], gt)   # context = original src (with its driver)
        except Exception:
            dropped += 1
            continue
        if len(ex["input_ids"]) > max_len:
            dropped += 1
            continue
        out.append({"input_ids": ex["input_ids"], "labels": ex["labels"]})
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", nargs="?", default="facebook/cwm")
    ap.add_argument("--train_sources", default="data/uitrans_cascade_py.jsonl,data/uitrans_uidom.jsonl")
    ap.add_argument("--heldout_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--out_train", default="data/sft_cascade_train.jsonl")
    ap.add_argument("--out_heldout", default="data/uitrans_cascade_heldout.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    rng = random.Random(a.seed)
    train_rows, held_rows = [], []
    for path in a.train_sources.split(","):
        rows = trace_rows(path)
        rng.shuffle(rows)
        k = int(len(rows) * (1 - a.heldout_frac))
        train_rows += rows[:k]
        held_rows += rows[k:]
    rng.shuffle(train_rows)

    train_ex, dropped = build_examples(tok, train_rows, a.max_len)
    with open(a.out_train, "w") as f:
        for ex in train_ex:
            f.write(json.dumps(ex) + "\n")
    with open(a.out_heldout, "w") as f:
        for r in held_rows:
            f.write(json.dumps(r) + "\n")
    print(f"train sources: {a.train_sources}")
    print(f"train examples: {len(train_ex)} (dropped {dropped})  -> {a.out_train}")
    print(f"held-out contract rows: {len(held_rows)}  -> {a.out_heldout}")
    print(f"(cross-lang/real eval kept separately: data/uitrans_cascade_js.jsonl, data/uitrans_vanilla.jsonl)")


if __name__ == "__main__":
    main()
