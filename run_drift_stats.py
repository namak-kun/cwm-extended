"""Powered drift study (gpt-duck #6): many independent free-roll rollouts, rollout as the unit, bootstrap CIs.

Batches N rollouts in parallel (one batched one-shot prediction per step across all active rollouts) -> fast.
Reports per-step mean accuracy + per-rollout all-correct rate + bootstrap CI over ROLLOUTS (not steps, which
are correlated). Compare base vs LoRA, free-roll vs re-grounded (k).
"""
from __future__ import annotations
import argparse, json, random, copy
from statistics import mean
from models.cwm_trace import CWMvLLM
from run_uitick_probe import one_event_stepover
from run_uitrans_probe import robust_parse
from run_gametick_abstract import _norm
from ui_dom import APPS, APP_NAMES, real_dispatch, gen_one_event_src


def run(model_path, app, n_roll, steps, lora, tag, out, reground_k=0, tp=4, seed0=1000):
    m = CWMvLLM(model_path, tp=tp, max_model_len=8192, lora_path=lora)
    rolls = []
    for r in range(n_roll):
        rng = random.Random(seed0 + r)
        nel = rng.randint(3, 8)
        dom = APPS[app]["init"](rng, nel)
        evs = [APPS[app]["event"](rng, dom, nel) for _ in range(steps)]
        rolls.append({"pred": copy.deepcopy(dom), "gt": copy.deepcopy(dom),
                      "evs": evs, "nel": nel, "ok": []})
    for t in range(steps):
        for ro in rolls:
            ro["gt"] = real_dispatch(app, ro["gt"], ro["evs"][t])
        items = [{"src": gen_one_event_src(app, ro["pred"], ro["evs"][t])[0]} for ro in rolls]
        preds = one_event_stepover(m, items, max_frames=8, max_tokens=1536,
                                   lora_kwargs=m._gen_kwargs(), return_raw=True)
        for ro, (p, raw) in zip(rolls, preds):
            pn = p if isinstance(p, dict) else robust_parse(raw)
            ok = (isinstance(pn, dict) and _norm(pn) == _norm(ro["gt"]))
            ro["ok"].append(bool(ok))
            ro["pred"] = pn if isinstance(pn, dict) else copy.deepcopy(ro["gt"])
            if reground_k and (t + 1) % reground_k == 0:
                ro["pred"] = copy.deepcopy(ro["gt"])
    # stats
    per_step = [mean(ro["ok"][t] for ro in rolls) for t in range(steps)]
    per_roll_all = [all(ro["ok"]) for ro in rolls]
    mean_acc = mean(v for ro in rolls for v in ro["ok"])
    # bootstrap CI over rollouts on mean per-step accuracy
    rng = random.Random(0)
    boot = []
    roll_means = [mean(ro["ok"]) for ro in rolls]
    for _ in range(5000):
        s = [roll_means[rng.randrange(n_roll)] for _ in range(n_roll)]
        boot.append(mean(s))
    boot.sort()
    ci = (round(boot[125], 3), round(boot[4875], 3))
    res = {"tag": tag, "app": app, "lora": lora, "n_roll": n_roll, "steps": steps, "reground_k": reground_k,
           "mean_step_acc": round(mean_acc, 4), "ci95_over_rollouts": ci,
           "per_step_acc": [round(x, 3) for x in per_step],
           "all_steps_correct_rate": round(mean(per_roll_all), 3)}
    print(f"[{tag}] app={app} k={reground_k} mean_step_acc={res['mean_step_acc']} CI95{ci} "
          f"all_correct={res['all_steps_correct_rate']}", flush=True)
    print(f"   per-step: {res['per_step_acc']}", flush=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--app", default="togglelist", choices=APP_NAMES)
    ap.add_argument("--n_roll", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--reground_k", type=int, default=0)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default="results/drift_stats.json")
    ap.add_argument("--tp", type=int, default=4)
    a = ap.parse_args()
    run(a.model_path, a.app, a.n_roll, a.steps, a.lora, a.tag, a.out, a.reground_k, a.tp)
