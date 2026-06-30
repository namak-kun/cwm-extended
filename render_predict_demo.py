"""Render CWM's PREDICTED next render-state to pixels (REPORT §35) — the 'frame as generation' capstone.

Given held-out DOM transitions, CWM one-shot-predicts the next DOM; we render predicted vs true through the
SAME browser. pred==truth => identical pixels. Run base and a LoRA to compare visually.
"""
from __future__ import annotations
import argparse, json
from models.cwm_trace import CWMvLLM
from run_uitick_probe import one_event_stepover
from run_uitrans_probe import robust_parse
from run_gametick_abstract import _norm
from dom_render import render_many


def run(model_path, data, lora, n, tag, out_dir, tp=4):
    rows = [json.loads(l) for l in open(data) if l.strip()][:n]
    m = CWMvLLM(model_path, tp=tp, max_model_len=8192, lora_path=lora)
    preds = one_event_stepover(m, [{"src": r["prompt_src"]} for r in rows],
                               max_frames=8, max_tokens=1536, lora_kwargs=m._gen_kwargs(), return_raw=True)
    items, n_ok = [], 0
    for i, (r, (p, raw)) in enumerate(zip(rows, preds)):
        pred = p if isinstance(p, dict) else robust_parse(raw)
        truth = _norm(r["truth_state"])
        ok = (pred == truth)
        n_ok += int(ok)
        app = r.get("app", "app")
        items.append((f"{i}_{app}_before", r["state_before"]))
        items.append((f"{i}_{app}_TRUE", truth))
        items.append((f"{i}_{app}_PRED_{tag}{'_OK' if ok else '_X'}", pred if isinstance(pred, dict) else
                      {"tag": "div", "text": "UNPARSED", "attrs": {}}))
    paths = render_many(items, out_dir)
    print(f"[{tag}] exact {n_ok}/{len(rows)} | rendered {len(paths)} PNGs -> {out_dir}", flush=True)
    return n_ok, len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--data", default="data/uitrans_cascade_heldout.jsonl")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out_dir", default="results/render_predict")
    ap.add_argument("--tp", type=int, default=4)
    a = ap.parse_args()
    run(a.model_path, a.data, a.lora, a.n, a.tag, a.out_dir, a.tp)
