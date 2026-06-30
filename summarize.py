"""Consolidate all results/*.json into one printout for the report."""
import json
import glob
import os


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def tag_of(path, prefix):
    return os.path.basename(path).replace(prefix, "").replace(".json", "")


print("=" * 78)
print("ONE-STEP exact-match (code-ON) | prior baseline (code-OFF) across model sizes")
print("=" * 78)
rows = {}
for p in sorted(glob.glob("results/exp1_onestep_*.json")):
    r = load(p)
    if not r:
        continue
    tag = tag_of(p, "exp1_onestep_")
    rows[tag] = r["one_step"]
hdr = f"{'mode':12}" + "".join(f"{t:>22}" for t in rows)
print(hdr)
for m in ["semantic", "random", "misleading"]:
    line = f"{m:12}"
    for t, os_ in rows.items():
        on = os_[f"{m}|code=on"]["exact"]
        off = os_[f"{m}|code=OFF"]["exact"]
        line += f"{on:.2f} (prior {off:.2f})".rjust(22)
    print(line)

print("\nCounterfactual misleading (action-exact / sensitivity):")
for p in sorted(glob.glob("results/exp1_onestep_*.json")):
    r = load(p)
    if not r:
        continue
    cf = r["counterfactual"]["misleading"]
    print(f"  {tag_of(p, 'exp1_onestep_'):8} action_exact={cf['action_exact_rate']:.2f} "
          f"sensitivity={cf['sensitivity_ratio']:.2f}")

print("\n" + "=" * 78)
print("ROLLOUT drift: free vs teacher-forced exact@20 and field_acc@20")
print("=" * 78)
for p in sorted(glob.glob("results/exp2_rollout_*.json")):
    r = load(p)
    if not r:
        continue
    print(f"[{tag_of(p, 'exp2_rollout_')}]")
    for m in ["semantic", "random", "misleading"]:
        if m not in r:
            continue
        fr, tf = r[m]["free"], r[m]["teacher_forced"]
        H = str(r["horizon"])
        print(f"  {m:11} free: field@20={fr[H]['field_acc']} exact@20={fr[H]['exact']} "
              f"1st_viol={fr['mean_first_violation_t']} | TF: field@20={tf[H]['field_acc']} exact@20={tf[H]['exact']}")

print("\n" + "=" * 78)
print("STOCHASTICITY (revealed vs hidden)")
print("=" * 78)
for p in sorted(glob.glob("results/exp3_stoch_*.json")):
    r = load(p)
    if not r:
        continue
    print(f"[{tag_of(p, 'exp3_stoch_')}] n={r['n_spawn_steps']} | "
          f"REVEALED exact={r['REVEALED']['exact']} | "
          f"HIDDEN greedy exact={r['HIDDEN_greedy']['exact']} det_field={r['HIDDEN_greedy']['deterministic_field_acc']} | "
          f"reachable_items={r['HIDDEN_sampled']['reachable_set_coverage_items']}")

print("\n" + "=" * 78)
print("COMPLEXITY sweep: exact vs program size")
print("=" * 78)
for p in sorted(glob.glob("results/exp5_complexity_*.json")):
    r = load(p)
    if not r:
        continue
    print(f"[{tag_of(p, 'exp5_complexity_')}]")
    for c, lv in r["levels"].items():
        print(f"  c={c}: exact={lv['exact']:.3f}  (code_lines={lv['avg_code_lines']}, "
              f"actions={lv['avg_actions']}, grid={lv['avg_grid']})")

rp = load("results/exp4_render.json")
if rp:
    print("\n" + "=" * 78)
    print("RENDERER sufficiency (mean abs pixel error)")
    print("=" * 78)
    print(" ", rp["mean_abs_pixel_error"])
    print("  interpretation:", rp["interpretation"])

reg = sorted(glob.glob("results/exp6_reground_*.json"))
if reg:
    print("\n" + "=" * 78)
    print("RE-GROUNDING: engine-call rate (1/k) vs sustained mean exact-match")
    print("=" * 78)
    for p in reg:
        r = load(p)
        if not r:
            continue
        print(f"[{tag_of(p, 'exp6_reground_')}]")
        for k, v in r["k"].items():
            print(f"  k={k:>5} (engine/step={v['engine_calls_per_step']}): "
                  f"mean_exact={v['mean_exact']} mean_field={v['mean_field']}")
