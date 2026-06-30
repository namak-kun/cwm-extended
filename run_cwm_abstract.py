"""Abstraction-level tracing: STEP-OVER (the debugger's `next`) at call boundaries.

Instead of letting CWM descend into library internals (which caused the OOD
failures), we keep the trace at the entry scope: whenever CWM emits a CALL
(descends), we force the very next frame to be that call's RETURN, so CWM
predicts the call's RESULT directly as an opaque operation. This is exactly how
you'd model a game tick: predict update()'s whole effect without tracing inside.

Batched across programs (lockstep) to stay fast on the no-NVLink box.
"""
from __future__ import annotations

import json
import sys
import time

from models.cwm_trace import (CWMvLLM, Event, FRAME_SEP, EOS, CALL_SEP,
                              build_prompt, parse_frame, resolve_locals)
from run_ood import PROGRAMS, true_final, cwm_final_return


def batched_stepover(m, items, max_frames=40, max_tokens=768):
    """items: list of dicts with 'src'. Runs a depth-1 (step-over) rollout for
    each, batched lockstep. Returns predicted frames per item."""
    from vllm import TokensPrompt
    sp = m.SP(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[FRAME_SEP, EOS])

    states = []
    for it in items:
        states.append({"src": it["src"], "frames": [], "force": Event.CALL,
                       "done": False, "depth": 0})

    for _ in range(max_frames):
        active = [s for s in states if not s["done"]]
        if not active:
            break
        prompts = [build_prompt(m, s["src"], s["frames"], force_event=s["force"])
                   for s in active]
        outs = m.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                              sp, use_tqdm=False)
        for s, o in zip(active, outs):
            gen = list(o.outputs[0].token_ids)
            f = parse_frame(m, gen + [FRAME_SEP], forced_event=s["force"],
                            prev=s["frames"][-1] if s["frames"] else None)
            if f is None:
                s["done"] = True
                continue
            s["frames"].append(f)
            # STEP-OVER policy: if CWM descended (CALL below entry), force its return
            if f.event == Event.CALL and len(s["frames"]) > 1:
                s["force"] = Event.RETURN
            elif f.event == Event.RETURN and len(s["frames"]) > 1 and s["frames"][0].event == Event.CALL:
                # a return that pops back to entry scope -> continue normally;
                # if it's the ENTRY's own return, we're done
                s["force"] = None
            else:
                s["force"] = None
            # termination: entry function returned
            if f.event == Event.RETURN and _entry_returned(s["frames"]):
                s["done"] = True
    return [s["frames"] for s in states]


def _entry_returned(frames) -> bool:
    """Heuristic: a RETURN frame whose source line matches the entry function's
    return statement and we're back at top scope."""
    if not frames or frames[-1].event != Event.RETURN:
        return False
    # if the only CALL frame is the entry, any return at this level ends it
    n_calls = sum(1 for f in frames if f.event == Event.CALL)
    n_rets = sum(1 for f in frames if f.event == Event.RETURN)
    return n_rets >= n_calls


def main(model_path, tp=4):
    t0 = time.time()
    m = CWMvLLM(model_path, tp=tp, max_model_len=8192)
    print("== CWM loaded ==\n", flush=True)

    names = ["py_nested", "py_multiproc", "py_threading", "py_numpy_small"]
    items, truths = [], []
    for nm in names:
        lang, src, entry = PROGRAMS[nm]
        items.append({"name": nm, "src": src})
        truths.append(true_final(lang, src))

    all_frames = batched_stepover(m, items, max_frames=24)

    results = {}
    for nm, frames, tv in zip(names, all_frames, truths):
        cv = cwm_final_return(frames)
        ok = (cv == tv) if tv is not None else None
        results[nm] = {"true_final": tv, "cwm_final_abstract": cv, "match": ok,
                       "n_frames": len(frames)}
        print(f"=== {nm} (STEP-OVER) ===  true={tv}  cwm={cv}  MATCH={ok}  frames={len(frames)}")
        for f in frames:
            print(f"    {f.event.name:7} {f.source_line.strip()[:46]:46} "
                  f"{ {k: v for k, v in resolve_locals(f).items()} } arg={f.arg}")
        print()

    json.dump({"model": model_path, "policy": "step-over (depth-1 abstraction)",
               "results": results, "elapsed_sec": round(time.time()-t0, 1)},
              open("results/cwm_abstract.json", "w"), indent=2)
    print(f"saved -> results/cwm_abstract.json")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
