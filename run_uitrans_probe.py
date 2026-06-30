"""Unified real-app UI-transition probe (renderer axis, REPORT §34).

Consumes the data/uitrans_<target>.jsonl CONTRACT produced by the harvesting subagents (TodoMVC, vanilla-JS,
MiniWoB++, Streamlit) and runs ONE CWM step-over probe over all of them (one model load, GPU-serialized):
given a real app's handler + current state + a UI event, force-predict the next state in one shot and score
exact-match + graded field-F1 vs the REAL app's oracle next-state.

This is the decisive distribution-shift test (§33.6): base CWM aces Python-native UI (0.925); does it crumble
on REAL JS / real app code? Per-target breakdown + base-vs-LoRA. One adapter per process (vLLM bug).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean

from vllm import TokensPrompt
from collections import defaultdict

from models.cwm_trace import (CWMvLLM, Event, CALL_SEP, FRAME_SEP, EOS,
                              build_prompt, parse_frame, parse_full_trace)
from vllm import TokensPrompt
from run_gametick_abstract import _norm
from run_uitick_probe import one_event_stepover, field_f1

import subprocess


def _entry_return(frames):
    """Robustly extract the ENTRY function's return = the RETURN frame that brings call-depth back to 0
    (not a helper's return). Falls back to the last RETURN arg if depth never balances (truncated trace)."""
    depth = 0
    last = None
    for f in frames:
        if f.event == Event.CALL:
            depth += 1
        elif f.event == Event.RETURN:
            if f.arg is not None:
                last = f.arg
            depth -= 1
            if depth == 0:
                return f.arg
    return last


def full_trace_preds(m, rows, max_tokens=4096):
    """FULL-TRACE: CWM executes the real handler (incl. helpers); take the entry's return. Robust to
    helper-delegating dispatch (todomvc/vanilla) that one-shot step-over can't capture."""
    prompts = [build_prompt(m, r["prompt_src"], [], force_event=Event.CALL) for r in rows]
    gens = m.gen_full_trace_batch(prompts, [max_tokens] * len(rows))
    raws = []
    for gen in gens:
        try:
            frames = parse_full_trace(m, [CALL_SEP] + gen)
            raws.append(_entry_return(frames))
        except Exception:
            raws.append(None)
    return raws


def depth_stepover(m, items, keep_depth=2, max_frames=48, max_tokens=1024, lora_kwargs=None):
    """Trace the entry + `keep_depth-1` levels (i.e. main + dispatch), but STEP OVER deeper helper/library
    calls (cloneTodos, nextId, copy.deepcopy, checkRequired, anon callbacks) by forcing their RETURN. This
    abstracts both the todomvc-helper one-shot wall and the streamlit deepcopy trace-bloat, uniformly.
    Returns each item's raw entry-RETURN arg string (the predicted next state)."""
    sp = m.SP(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[FRAME_SEP, EOS])
    lk = lora_kwargs or {}
    st = [{"src": it["src"], "frames": [], "force": Event.CALL, "depth": 0,
           "done": False, "raw": None, "last_dict": None} for it in items]
    for _ in range(max_frames):
        active = [s for s in st if not s["done"]]
        if not active:
            break
        prompts = [build_prompt(m, s["src"], s["frames"], force_event=s["force"]) for s in active]
        outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False, **lk)
        for s, o in zip(active, outs):
            f = parse_frame(m, list(o.outputs[0].token_ids) + [FRAME_SEP],
                            forced_event=s["force"], prev=s["frames"][-1] if s["frames"] else None)
            if f is None:
                s["done"] = True
                continue
            s["frames"].append(f)
            if f.event == Event.CALL:
                s["depth"] += 1
            elif f.event == Event.RETURN:
                if f.arg is not None and ("{" in str(f.arg) or "[" in str(f.arg)):
                    s["last_dict"] = f.arg            # remember last structured return as a fallback
                s["depth"] -= 1
                if s["depth"] <= 0:                  # entry (main) returned -> the answer
                    s["raw"] = f.arg
                    s["done"] = True
                    continue
            # decide what to do with the NEXT frame
            if f.event == Event.CALL and s["depth"] > keep_depth:
                s["force"] = Event.RETURN             # step over this deeper helper/library call
            else:
                s["force"] = None
    return [(s["raw"] if s["raw"] is not None else s["last_dict"]) for s in st]


def _last_return_arg(frames):
    """Entry's final state = the last RETURN frame's arg (robust full-trace extraction)."""
    for f in reversed(frames):
        if f.event == Event.RETURN and f.arg is not None:
            return f.arg
    return None


def _node_parse(s):
    """Last-resort: let node eval a JS object literal (unquoted keys / true/false/null) -> JSON -> py."""
    try:
        r = subprocess.run(["node", "-e", "process.stdout.write(JSON.stringify(eval('('+process.argv[1]+')')))", "--", s],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return _norm(json.loads(r.stdout))
    except Exception:
        pass
    return None


def robust_parse(arg):
    """Parse a CWM RETURN arg into a Python object, tolerant of Python-repr AND JS/JSON literals."""
    if arg is None:
        return None
    if isinstance(arg, (dict, list)):
        return _norm(arg)
    s = str(arg).strip()
    import ast
    for attempt in (
        lambda: ast.literal_eval(s),
        lambda: json.loads(s),
        lambda: json.loads(s.replace("'", '"')),
        lambda: ast.literal_eval(s.replace("true", "True").replace("false", "False").replace("null", "None")),
    ):
        try:
            return _norm(attempt())
        except Exception:
            continue
    return _node_parse(s)   # JS object-literal fallback (unquoted keys etc.)


def load_rows(data_args):
    files = []
    for d in data_args:
        files.extend(sorted(glob.glob(d)))
    rows = []
    for f in files:
        if not os.path.exists(f):
            continue
        tgt = os.path.basename(f).replace("uitrans_", "").replace(".jsonl", "")
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            r.setdefault("target", tgt)
            if r.get("prompt_src") and r.get("truth_state") is not None:
                rows.append(r)
    return rows, files


def _flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, f"{p}.{k}"))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(_flat(v, f"{p}[{i}]"))
    else:
        out[p] = o
    return out


def changed_field_acc(pred, truth, before):
    """Of the fields that ACTUALLY change (truth != before), fraction the model gets right. Copy-baseline=0."""
    if not isinstance(pred, (dict, list)):
        return 0.0
    P, T, B = _flat(pred), _flat(_norm(truth)), _flat(_norm(before))
    changed = [k for k in T if B.get(k) != T[k]]
    if not changed:
        return 1.0
    return mean(1.0 if P.get(k) == T[k] else 0.0 for k in changed)


def delta_exact(pred, truth, before):
    """Predicted change-SET == true change-SET (no missed changes, no spurious changes). The strict signal."""
    if not isinstance(pred, (dict, list)):
        return 0.0
    P, T, B = _flat(pred), _flat(_norm(truth)), _flat(_norm(before))
    true_ch = {k: v for k, v in T.items() if B.get(k) != v}
    pred_ch = {k: v for k, v in P.items() if B.get(k) != v}
    return 1.0 if pred_ch == true_ch else 0.0


def run(model_path, tp, data_args, lora, out, maxn=0, mode="stepover"):
    rows, files = load_rows(data_args)
    if maxn and len(rows) > maxn:
        rows = rows[:maxn]
    print(f"== loaded {len(rows)} traceable rows from {len(files)} file(s): {files} ==", flush=True)
    if not rows:
        print("NO ROWS (waiting on harvesters?) — nothing to probe.", flush=True)
        json.dump({"model": model_path, "lora": lora, "n": 0, "files": files}, open(out, "w"), indent=2)
        return

    m = CWMvLLM(model_path, tp=tp, max_model_len=12288, lora_path=lora)
    print(f"== CWM {'+LoRA '+lora if lora else '(BASE)'} | real-app UI-transition probe (mode={mode}) ==", flush=True)
    if mode == "fulltrace":
        # CWM executes the real handler (incl. helpers); robust entry-return extraction. For delegated-handler
        # real apps (todomvc/vanilla) that one-shot step-over can't capture.
        raws = full_trace_preds(m, rows, max_tokens=7500)
    else:
        items = [{"src": r["prompt_src"]} for r in rows]
        preds = one_event_stepover(m, items, max_frames=8, max_tokens=1536,
                                   lora_kwargs=m._gen_kwargs(), return_raw=True)
        raws = [(p if isinstance(p, (dict, list)) else raw) for (p, raw) in preds]

    by_t_exact, by_t_f1 = defaultdict(list), defaultdict(list)
    by_t_chg, by_t_delta, by_t_copyf1 = defaultdict(list), defaultdict(list), defaultdict(list)
    examples = defaultdict(list)
    n_unparsed = 0
    for r, raw in zip(rows, raws):
        t = r["target"]
        truth = _norm(r["truth_state"])
        before = r.get("state_before")
        pnorm = robust_parse(raw)
        if pnorm is None:
            n_unparsed += 1
        ex = (pnorm is not None and pnorm == truth)
        f1 = field_f1(pnorm, truth) if isinstance(pnorm, (dict, list)) else 0.0
        by_t_exact[t].append(1.0 if ex else 0.0)
        by_t_f1[t].append(f1)
        if before is not None:
            by_t_chg[t].append(changed_field_acc(pnorm, truth, before))
            by_t_delta[t].append(delta_exact(pnorm, truth, before))
            by_t_copyf1[t].append(field_f1(_norm(before), truth))   # copy-baseline field_f1
        if len(examples[t]) < 4 and not ex:
            examples[t].append({"action": r.get("action"), "f1": f1,
                                "pred": str(pnorm)[:300], "raw": str(raw)[:200], "truth": str(truth)[:300]})

    all_exact = [v for a in by_t_exact.values() for v in a]
    res = {
        "model": model_path, "lora": lora, "n": len(rows), "files": files,
        "overall_exact": round(mean(all_exact), 4),
        "overall_field_f1": round(mean([v for a in by_t_f1.values() for v in a]), 4),
        "n_unparsed": n_unparsed,
        "by_target": {t: {"exact": round(mean(by_t_exact[t]), 3),
                          "field_f1": round(mean(by_t_f1[t]), 3),
                          "changed_field_acc": round(mean(by_t_chg[t]), 3) if by_t_chg[t] else None,
                          "delta_exact": round(mean(by_t_delta[t]), 3) if by_t_delta[t] else None,
                          "copy_field_f1": round(mean(by_t_copyf1[t]), 3) if by_t_copyf1[t] else None,
                          "n": len(by_t_exact[t])} for t in sorted(by_t_exact)},
        "fail_examples": {t: examples[t] for t in examples},
    }
    print(f"  OVERALL exact={res['overall_exact']}  field_f1={res['overall_field_f1']}  (n={len(rows)}, unparsed={n_unparsed})", flush=True)
    for t in sorted(by_t_exact):
        d = res["by_target"][t]
        print(f"    [{t:10s}] exact={d['exact']:<6} changed_field={d['changed_field_acc']} delta={d['delta_exact']} "
              f"field_f1={d['field_f1']} (copy_f1={d['copy_field_f1']}) n={d['n']}", flush=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--data", default="data/uitrans_*.jsonl",
                    help="comma-separated globs of contract jsonl files")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--maxn", type=int, default=0)
    ap.add_argument("--mode", default="stepover", choices=["stepover","fulltrace"])
    ap.add_argument("--out", default="results/uitrans_probe_base.json")
    a = ap.parse_args()
    run(a.model_path, a.tp, a.data.split(","), a.lora, a.out, a.maxn, a.mode)
