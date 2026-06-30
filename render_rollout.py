"""Multi-step DOM-rollout -> rendered frame sequence (the 'video', REPORT §35) + drift curve.

CWM free-rolls the render-state: predict DOM_{i+1} from its OWN predicted DOM_i + event_i (one-shot step-over),
for a sequence of UI events. Render each predicted DOM -> frame_i.png = a generated UI 'video' of the app
responding to input. Compare to the ground-truth rollout (real_dispatch) to get a per-step drift curve. Run
base vs a LoRA to see the SFT's effect on rollout faithfulness. (DRIFT axis: re-grounding would reset to truth
every k steps.)
"""
from __future__ import annotations
import argparse, json, random, copy, os
from models.cwm_trace import CWMvLLM
from run_uitick_probe import one_event_stepover
from run_uitrans_probe import robust_parse
from run_gametick_abstract import _norm
from ui_dom import APPS, APP_NAMES, real_dispatch, gen_one_event_src
from dom_render import render_many


def rollout(model_path, app, n, steps, lora, tag, out_dir, seed=42, tp=4, reground_k=0):
    rng = random.Random(seed)
    m = CWMvLLM(model_path, tp=tp, max_model_len=8192, lora_path=lora)
    nelem = rng.randint(4, 6)
    init = APPS[app]["init"](rng, nelem)
    # fixed event sequence
    events = [APPS[app]["event"](rng, init, nelem) for _ in range(steps)]
    pred = copy.deepcopy(init)
    gt = copy.deepcopy(init)
    items = [(f"step0_init", init)]
    per_step_ok, drift = [], []
    for i, ev in enumerate(events):
        gt = real_dispatch(app, gt, ev)
        # free-roll: predict from the model's OWN previous prediction
        src, _ = gen_one_event_src(app, pred, ev)
        preds = one_event_stepover(m, [{"src": src}], max_frames=8, max_tokens=1536,
                                   lora_kwargs=m._gen_kwargs(), return_raw=True)
        p, raw = preds[0]
        pnext = p if isinstance(p, dict) else robust_parse(raw)
        ok = (isinstance(pnext, dict) and _norm(pnext) == _norm(gt))
        per_step_ok.append(bool(ok))
        if not isinstance(pnext, dict):
            pnext = copy.deepcopy(gt)  # unparsed -> fall back to truth to keep the video going
        pred = pnext
        if reground_k and (i + 1) % reground_k == 0:
            pred = copy.deepcopy(gt)   # DRIFT axis: periodic re-grounding
        items.append((f"step{i+1}_{ev.get('type')}_{ev.get('id','')}_{'OK' if ok else 'X'}", pred))
        items.append((f"step{i+1}_TRUE", gt))
    paths = render_many(items, out_dir)
    # assemble the PREDICTED-frame sequence (init + each step's predicted DOM) into an animated GIF (the 'video')
    try:
        from PIL import Image
        seq = [pth for (nm, _), pth in zip(items, paths) if "_TRUE" not in nm]
        imgs = [Image.open(p).convert("RGB") for p in seq]
        if imgs:
            gif = os.path.join(out_dir, f"rollout_{tag}.gif")
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=900, loop=0)
            print(f"  GIF -> {gif} ({len(imgs)} frames)", flush=True)
    except Exception as e:
        print(f"  (gif skipped: {e})", flush=True)
    print(f"[{tag}] app={app} steps={steps} per-step exact={per_step_ok} "
          f"({sum(per_step_ok)}/{len(per_step_ok)}) | frames -> {out_dir}", flush=True)
    json.dump({"tag": tag, "app": app, "steps": steps, "per_step_ok": per_step_ok,
               "reground_k": reground_k, "events": events},
              open(os.path.join(out_dir, f"rollout_{tag}.json"), "w"), indent=2, default=str)
    return per_step_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--app", default="tabs", choices=APP_NAMES)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out_dir", default="results/render_rollout")
    ap.add_argument("--reground_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tp", type=int, default=4)
    a = ap.parse_args()
    rollout(a.model_path, a.app, 1, a.steps, a.lora, a.tag, a.out_dir, a.seed, a.tp, a.reground_k)
