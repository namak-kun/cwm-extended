"""RL VIABILITY PROBE (gates the whole GRPO effort): at sampling temperature, does CWM produce
arithmetic trace rollouts with EXPLOITABLE reward spread?

GRPO learns from GROUP-RELATIVE advantage: within a group of G rollouts of the SAME program, it
pushes toward the above-average ones. If all G rollouts score identically (no variance), the
advantage is 0 and there is NO learning signal. If best-of-G >> mean, RL can climb toward best-of-G.

For N programs we sample G rollouts at temperature T, score each (frame-accuracy reward in [0,1]),
and report per-group mean / std / max and the population best-of-G vs mean gap. Decision:
  - VIABLE if mean group std is non-trivial AND mean(best-of-G) - mean(mean) is a real gap
    (RL has headroom to climb), and best-of-G clearly beats greedy (temp 0) free-roll.
  - NOT VIABLE if rollouts are near-deterministic at T (no spread) -> GRPO can't learn; use SFT/other.
"""
from __future__ import annotations

import argparse
import json
import random
from statistics import mean, pstdev

from vllm import TokensPrompt

from models.cwm_trace import (CWMvLLM, Event, EOS, CALL_SEP,
                              build_prompt, parse_full_trace, resolve_locals)
from gt_trace import trace_program, score_frame
from run_cwm_track import estimate_trace_tokens
from failure_buckets import gen_oop, gen_multientity_short, gen_arithmetic, gen_recursion, gen_easy

GENS = {"oop": gen_oop, "multientity_short": gen_multientity_short,
        "arithmetic": gen_arithmetic, "recursion": gen_recursion, "easy": gen_easy}


def reward(gt, df):
    nmin = min(len(gt), len(df))
    ok = sum(score_frame(gt[i], df[i], resolve_locals)["frame_ok"] for i in range(nmin))
    denom = max(len(gt), len(df))
    return ok / denom if denom else 0.0


def run(model_path, tp, bucket, n, g, temps, seed, expand, lora, out):
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

    m = CWMvLLM(model_path, tp=tp, max_model_len=24576, lora_path=lora)
    print(f"== CWM loaded {'+LoRA '+lora if lora else '(base)'} | RL viability {bucket} n={len(progs)} G={g} temps={temps} ==", flush=True)

    fps = [build_prompt(m, src, [], force_event=Event.CALL) for (src, _, _) in progs]
    maxcap = min(max(int(estimate_trace_tokens(m, gt) * 1.3) + 256 for (_, _, gt) in progs), 12000)

    # greedy (temp 0) reference reward
    gsp = m.SP(temperature=0.0, max_tokens=maxcap, stop_token_ids=[EOS])
    gouts = m.llm.generate([TokensPrompt(prompt_token_ids=fp) for fp in fps], gsp, use_tqdm=False, **m._gen_kwargs())
    greedy = []
    for (src, entry, gt), o in zip(progs, gouts):
        df = parse_full_trace(m, [CALL_SEP] + list(o.outputs[0].token_ids))
        greedy.append(reward(gt, df))
    greedy_mean = round(mean(greedy), 4)
    print(f"  greedy_reward_mean = {greedy_mean}", flush=True)

    all_res = {"bucket": bucket, "lora": lora, "n": len(progs), "G": g, "greedy_reward_mean": greedy_mean, "by_temp": {}}
    for temp in temps:
        ssp = m.SP(temperature=temp, top_p=0.95, n=g, max_tokens=maxcap, stop_token_ids=[EOS], seed=seed)
        souts = m.llm.generate([TokensPrompt(prompt_token_ids=fp) for fp in fps], ssp, use_tqdm=False, **m._gen_kwargs())
        group_mean, group_std, group_max, group_min = [], [], [], []
        nonzero_adv_groups = groups_with_perfect = total_perfect = 0
        for (src, entry, gt), o in zip(progs, souts):
            rs = []
            for comp in o.outputs:
                df = parse_full_trace(m, [CALL_SEP] + list(comp.token_ids))
                rs.append(reward(gt, df))
            group_mean.append(mean(rs)); group_std.append(pstdev(rs) if len(rs) > 1 else 0.0)
            group_max.append(max(rs)); group_min.append(min(rs))
            if (max(rs) - min(rs)) > 1e-6:
                nonzero_adv_groups += 1
            np_ = sum(1 for r in rs if r >= 0.999)
            total_perfect += np_
            if np_ > 0:
                groups_with_perfect += 1
        row = {
            "sampled_group_mean": round(mean(group_mean), 4),
            "best_of_G_mean": round(mean(group_max), 4),
            "mean_group_std": round(mean(group_std), 4),
            "frac_groups_with_spread": round(nonzero_adv_groups / len(progs), 3),
            "headroom_bestG_minus_greedy": round(mean(group_max) - greedy_mean, 4),
            "frac_groups_with_perfect_rollout": round(groups_with_perfect / len(progs), 3),
            "total_perfect_rollouts": total_perfect, "total_rollouts": len(progs) * g,
        }
        grpo_viable = row["frac_groups_with_spread"] >= 0.5 and row["headroom_bestG_minus_greedy"] >= 0.05
        rest_viable = row["frac_groups_with_perfect_rollout"] >= 0.2
        row["grpo_viable"] = grpo_viable
        row["rest_viable"] = rest_viable
        all_res["by_temp"][str(temp)] = row
        print(f"  T={temp}: best-of-G={row['best_of_G_mean']} spread={row['frac_groups_with_spread']} "
              f"perfect={row['total_perfect_rollouts']}/{row['total_rollouts']} "
              f"GRPO={'Y' if grpo_viable else 'n'} ReST={'Y' if rest_viable else 'n'}", flush=True)

    json.dump(all_res, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--bucket", default="arithmetic")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--temps", default="0.8", help="comma list of temperatures to sweep")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", default="results/rl_viability.json")
    a = ap.parse_args()
    run(a.model_path, a.tp, a.bucket, a.n, a.g, [float(t) for t in a.temps.split(",")],
        a.seed, a.expand, a.lora, a.out)
