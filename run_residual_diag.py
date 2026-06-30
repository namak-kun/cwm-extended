"""Characterize the oop free-rollout RESIDUAL: where does the gold-SFT model first diverge from
ground truth, and is the same frame wrong regardless of program (structural) vs drift-compounding?"""
import sys, random
from vllm import TokensPrompt
from models.cwm_trace import (CWMvLLM, Event, EOS, CALL_SEP, build_prompt,
                              parse_full_trace, resolve_locals)
from gt_trace import trace_program, score_frame
from failure_buckets import gen_oop
from run_cwm_track import estimate_trace_tokens

CWM = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 else "adapters/cwm_dagger_gold"
m = CWMvLLM(CWM, tp=4, max_model_len=24576, lora_path=adapter)
rng = random.Random(999)
progs = []
seen = set()
while len(progs) < 5:
    src, entry = gen_oop(rng)
    if src in seen:
        continue
    seen.add(src)
    gt = trace_program(src, entry, expand_objects=True)
    if gt and len(gt) > 3:
        progs.append((src, entry, gt))

for pi, (src, entry, gt) in enumerate(progs):
    fp = build_prompt(m, src, [], force_event=Event.CALL)
    cap = min(int(estimate_trace_tokens(m, gt) * 1.3) + 256, 12000)
    o = m.llm.generate([TokensPrompt(prompt_token_ids=fp)],
                       m.SP(temperature=0.0, max_tokens=cap, stop_token_ids=[EOS]),
                       use_tqdm=False, **m._gen_kwargs())
    df = parse_full_trace(m, [CALL_SEP] + list(o[0].outputs[0].token_ids))
    nmin = min(len(gt), len(df))
    print(f"\n=== prog {pi}: len(gt)={len(gt)} len(df)={len(df)} ===")
    wrong = []
    for i in range(nmin):
        r = score_frame(gt[i], df[i], resolve_locals)
        if not r["frame_ok"]:
            wrong.append(i)
    print(f"  wrong frame idxs: {wrong[:12]}{' ...' if len(wrong)>12 else ''}  (total {len(wrong)}/{nmin})")
    # show the FIRST wrong frame: gt vs predicted
    if wrong:
        i = wrong[0]
        print(f"  first-wrong frame {i}:")
        print(f"    GT  : {gt[i].event} | {gt[i].source_line.strip()} | locals={gt[i].locals} | ret={gt[i].ret}")
        pf = df[i]
        print(f"    PRED: {pf.event.name.lower()} | {pf.source_line.strip()} | locals={resolve_locals(pf)} | arg={pf.arg}")
