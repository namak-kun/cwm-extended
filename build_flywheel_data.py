"""FDM<->IDM FLYWHEEL data builder (runs in .venv_vllm: needs vLLM for IDM forward-search + tokenizer).

Self-supervised loop: improve a FORWARD dynamics model (FDM) using ONLY unlabeled state-trajectories
(no action labels, no dynamics oracle for targets), by recovering the actions with the INVERSE model
(IDM = forward-search over the FDM).

Pipeline:
  1. gen N unlabeled trajectories: real game state sequences (s_0..s_T); ACTIONS HIDDEN.
  2. IDM-label each transition (s_i, s_{i+1}) -> recovered action via forward search with --label_fdm.
     (forward-search margin recorded -> used for optional confidence filtering.)
  3. build FDM step-over training traces from (s_i, recovered_a, OBSERVED s_{i+1}):
       - serialize the one-tick step-over trace, but OVERRIDE the step()/main return value to the
         OBSERVED s_{i+1} -> NO dynamics oracle is used for the target (pure observation).
  4. also emit an ORACLE-labeled control (true actions, same trajectories) for upper-bounding.
Reports IDM action-recovery accuracy (the data-quality knob) and writes both jsonls.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from statistics import mean

from transformers import AutoTokenizer
from vllm import TokensPrompt

from models.cwm_trace import Event, CALL_SEP
from gt_trace import trace_program, GTFrame
from trace_dataset import serialize_trace
from game_tick import gen_game_tick, ground_truth_states, gen_one_tick_src, real_step
from run_gametick_abstract import _parse_state
from run_idm_search import state_distance
from models.cwm_trace import CWMvLLM

ACTIONS = ["U", "D", "L", "R"]


def override_returns(gt_frames, s_next_repr):
    """Set every full-state return frame's ret to the OBSERVED next state (observation as target)."""
    out = []
    for f in gt_frames:
        if f.event == "return" and isinstance(f.ret, str) and "player" in f.ret and "enemies" in f.ret:
            out.append(GTFrame(f.event, f.lineno, f.source_line, f.locals, ret=s_next_repr))
        else:
            out.append(f)
    return out


def classify_transition(s, snext):
    """Tag the consequential event(s) in a transition for per-type IDM-accuracy auditing."""
    tags = []
    if snext["player"]["score"] != s["player"]["score"]:
        tags.append("stomp")              # player landed on an enemy (+score, enemy dies)
    if snext["player"]["hp"] != s["player"]["hp"]:
        tags.append("contact")            # an enemy landed on the player (-hp)
    if sum(e["alive"] for e in snext["enemies"]) != sum(e["alive"] for e in s["enemies"]):
        tags.append("death")
    pdx = snext["player"]["x"] - s["player"]["x"]
    pdy = snext["player"]["y"] - s["player"]["y"]
    if pdx == 0 and pdy == 0:
        tags.append("blocked")            # player didn't move (wall clip or aliasing)
    if not tags:
        tags.append("move_only")
    return tags


def idm_label(m, transitions):
    """Forward-search IDM (FAST path): for each transition x action, generate the one-tick step-over
    trace in ONE batched call (the SFT'd FDM emits step-over natively), parse the predicted next state.
    Returns per-transition {a (recovered), margin, rel_margin, best_dist, exact set}."""
    from models.cwm_trace import build_prompt, parse_full_trace
    items, index = [], []
    for ti, tr in enumerate(transitions):
        for a in ACTIONS:
            osrc, _ = gen_one_tick_src(tr["s"], a)
            items.append(osrc)
            index.append((ti, a))
    prompts = [build_prompt(m, src, [], force_event=Event.CALL) for src in items]
    gens = m.gen_full_trace_batch(prompts, [400] * len(prompts))  # one tick is small; 400 tok is ample
    pred = {ti: {} for ti in range(len(transitions))}
    for (ti, a), gen in zip(index, gens):
        frames = parse_full_trace(m, [CALL_SEP] + gen)
        ps = None
        for f in frames:
            if f.event == Event.RETURN:
                cand = _parse_state(f.arg)
                if isinstance(cand, dict) and "player" in cand and "enemies" in cand:
                    ps = cand   # last full-state return = step()'s predicted result
        pred[ti][a] = ps
    labels = []
    for ti, tr in enumerate(transitions):
        dists = sorted((state_distance(pred[ti].get(a), tr["snext"]), a) for a in ACTIONS)
        best_d, best_a = dists[0]
        margin = dists[1][0] - best_d
        rel = margin / max(best_d, 1)
        exact = [a for d, a in dists if d == 0]
        labels.append({"a": best_a, "margin": margin, "rel_margin": round(rel, 3),
                       "best_dist": best_d, "exact": exact})
    return labels


def build_examples(tok, transitions, actions, max_len=4096):
    """Serialize one-tick step-over traces with OBSERVED s_next as target. actions[i] is the action
    label to use (recovered or oracle)."""
    exs = []
    for tr, a in zip(transitions, actions):
        osrc, oentry = gen_one_tick_src(tr["s"], a)
        gt = trace_program(osrc, oentry, stepover_depth=1)
        if not gt:
            continue
        gt = override_returns(gt, repr(tr["snext"]))
        ex = serialize_trace(tok, osrc, gt)
        if len(ex["input_ids"]) <= max_len:
            exs.append(ex)
    return exs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--label_fdm", required=True, help="FDM adapter used as the IDM labeler")
    ap.add_argument("--n_traj", type=int, default=80)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--margin_min", type=float, default=0.0, help="keep only labels with margin >= this")
    ap.add_argument("--out_idm", default="data/flywheel_idm.jsonl")
    ap.add_argument("--out_oracle", default="data/flywheel_oracle.jsonl")
    ap.add_argument("--stats_out", default="results/flywheel_label_stats.json")
    ap.add_argument("--kmin", type=int, default=None); ap.add_argument("--kmax", type=int, default=None)
    ap.add_argument("--tmin", type=int, default=None); ap.add_argument("--tmax", type=int, default=None)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    # unlabeled trajectories: keep the STATES; the actions are hidden (kept only for scoring/oracle)
    transitions = []
    seen = set()
    while len(set(t["gid"] for t in transitions)) < a.n_traj and len(seen) < a.n_traj * 20:
        K = rng.randint(a.kmin, a.kmax) if (a.kmin and a.kmax) else None
        T = rng.randint(a.tmin, a.tmax) if (a.tmin and a.tmax) else None
        src, entry, meta = gen_game_tick(rng, k_enemies=K, t_ticks=T, return_meta=True)
        if src in seen:
            continue
        seen.add(src)
        gid = len(seen)
        states = [meta["init"]] + ground_truth_states(src, meta)
        for i, act in enumerate(meta["actions"]):
            transitions.append({"gid": gid, "s": states[i], "a_true": act, "snext": states[i + 1]})

    tok = AutoTokenizer.from_pretrained(a.model_path)
    m = CWMvLLM(a.model_path, tp=a.tp, max_model_len=8192, lora_path=a.label_fdm)
    print(f"== IDM-labeling {len(transitions)} transitions with FDM {a.label_fdm} ==", flush=True)

    labels = idm_label(m, transitions)

    # aliasing-aware recovery acc (a recovered action counts correct if it REALLY produces s_next)
    def true_set(tr):
        return [x for x in ACTIONS if real_step(tr["s"], x) == tr["snext"]]
    rec = [1.0 if lab["a"] in true_set(tr) else 0.0 for lab, tr in zip(labels, transitions)]
    rec_acc = mean(rec)

    # per-event-type recovery acc (rare events: stomp/contact/death may hide in aggregate)
    by_event = {}
    for lab, tr, ok in zip(labels, transitions, rec):
        for tag in classify_transition(tr["s"], tr["snext"]):
            by_event.setdefault(tag, []).append(ok)
    by_event_acc = {k: {"acc": round(mean(v), 3), "n": len(v)} for k, v in by_event.items()}

    # margin filtering
    kept = [(tr, lab) for tr, lab in zip(transitions, labels) if lab["margin"] >= a.margin_min]
    kept_acc = (mean(1.0 if lab["a"] in true_set(tr) else 0.0 for tr, lab in kept)
                if kept else 0.0)

    idm_actions = [lab["a"] for tr, lab in kept]
    oracle_actions = [tr["a_true"] for tr, lab in kept]
    kept_trans = [tr for tr, lab in kept]

    idm_ex = build_examples(tok, kept_trans, idm_actions)
    oracle_ex = build_examples(tok, kept_trans, oracle_actions)

    os.makedirs("data", exist_ok=True); os.makedirs("results", exist_ok=True)
    with open(a.out_idm, "w") as f:
        for ex in idm_ex:
            f.write(json.dumps({"input_ids": ex["input_ids"], "labels": ex["labels"]}) + "\n")
    with open(a.out_oracle, "w") as f:
        for ex in oracle_ex:
            f.write(json.dumps({"input_ids": ex["input_ids"], "labels": ex["labels"]}) + "\n")

    stats = {
        "label_fdm": a.label_fdm, "n_transitions": len(transitions),
        "idm_action_recovery_acc": round(rec_acc, 4),
        "by_event_type": by_event_acc,
        "n_kept_after_margin": len(kept), "margin_min": a.margin_min,
        "kept_recovery_acc": round(kept_acc, 4),
        "n_idm_examples": len(idm_ex), "n_oracle_examples": len(oracle_ex),
        "mean_margin": round(mean(lab["margin"] for lab in labels), 3),
        "frac_zero_margin": round(mean(1.0 if lab["margin"] == 0 else 0.0 for lab in labels), 3),
    }
    json.dump(stats, open(a.stats_out, "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"saved -> {a.out_idm} ({len(idm_ex)}), {a.out_oracle} ({len(oracle_ex)}), {a.stats_out}")


if __name__ == "__main__":
    main()
