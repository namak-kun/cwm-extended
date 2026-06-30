"""Renderer-sufficiency probe (no LLM).

Question (the user's pushback): can pixels be produced by a pure render(state) on
the predicted symbolic logic-state, or must the renderer be LEARNED?

Criterion (duck): own-renderer suffices IFF the logged symbolic state is a
sufficient statistic for the pixels. We test this by constructing renderers that
secretly depend on state NOT in the logic-state, and measuring whether
render(logic_state_only) reproduces the true frame.

  R_pure   : pixels are a pure function of logic-state            -> sufficient
  R_anim   : sprite appearance depends on a hidden animation phase (advances each
             frame, independent of position)                      -> insufficient
  R_camera : the whole image is offset by a smoothed camera that depends on the
             HISTORY of positions                                 -> insufficient
  R_trail  : a fading trail depends on the history of positions   -> insufficient

If we AUGMENT the logged state with the hidden fields, sufficiency is restored.
That is the whole point: the bottleneck is state observability, not a missing
neural renderer -- a learned renderer cannot invent a hidden phase/camera from a
single frame either; it would need the same history.
"""
from __future__ import annotations

import json
import numpy as np

from worlds.gridgen import Game
from eval import gen_trajectory

CELL = 16
PHASE_COLORS = [(220, 60, 60), (60, 220, 60), (60, 60, 220), (220, 220, 60)]


def _canvas(gw, gh):
    img = np.zeros((gh * CELL, gw * CELL, 3), dtype=np.uint8)
    img[:] = 30
    return img


def _put(img, gw, gh, cx, cy, rgb, ox=0, oy=0):
    x = (cx + ox) % gw
    y = (cy + oy) % gh
    img[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = rgb


def render(logic, gw, gh, posf, itemsf, *, phase=0, camera=None, trail=()):
    """Full renderer parameterised by (possibly hidden) render-state."""
    img = _canvas(gw, gh)
    ox = oy = 0
    if camera is not None:
        # camera smoothing offsets the whole scene by rounded smoothed pos
        ox = int(round(camera[0])) - logic[posf][0]
        oy = int(round(camera[1])) - logic[posf][1]
    for t, (tx, ty) in enumerate(trail):  # fading trail from history
        fade = int(80 * (t + 1) / (len(trail) + 1))
        _put(img, gw, gh, tx, ty, (fade, fade, fade), ox, oy)
    for it in logic.get(itemsf, []):
        _put(img, gw, gh, it[0], it[1], (200, 180, 40), ox, oy)
    p = logic[posf]
    _put(img, gw, gh, p[0], p[1], PHASE_COLORS[phase % 4], ox, oy)
    return img


def diff(a, b):
    return float(np.mean(np.abs(a.astype(int) - b.astype(int))))


def probe(seed=4321, horizon=24):
    g = Game.generate(seed, naming_mode="random")
    gw, gh = g.ns["GW"], g.ns["GH"]
    posf, itemsf = g.spec.fname("pos"), g.spec.fname("items")
    steps = gen_trajectory(g, horizon=horizon, init_seed=1, policy_seed=2)
    logic_states = [steps[0]["state"]] + [s["next_state"] for s in steps]

    # maintain hidden render-state along the true trajectory
    phase = 0
    camera = list(map(float, logic_states[0][posf]))
    hist = []
    err = {k: [] for k in ["pure", "anim", "camera", "trail",
                           "anim_aug", "camera_aug", "trail_aug"]}
    for ls in logic_states:
        p = ls[posf]
        phase = (phase + 1) % 4
        camera = [0.6 * camera[0] + 0.4 * p[0], 0.6 * camera[1] + 0.4 * p[1]]
        hist.append((p[0], p[1]))
        trail = tuple(hist[-4:-1])

        # TRUE frames (full renderer with hidden state)
        true_pure = render(ls, gw, gh, posf, itemsf)
        true_anim = render(ls, gw, gh, posf, itemsf, phase=phase)
        true_cam = render(ls, gw, gh, posf, itemsf, camera=camera)
        true_trail = render(ls, gw, gh, posf, itemsf, trail=trail)

        # OWN renderer from LOGIC-STATE ONLY (defaults: phase0, camera=pos, no trail)
        own = render(ls, gw, gh, posf, itemsf)
        err["pure"].append(diff(own, true_pure))
        err["anim"].append(diff(own, true_anim))
        err["camera"].append(diff(own, true_cam))
        err["trail"].append(diff(own, true_trail))

        # AUGMENTED: if we LOG the hidden field, own-renderer is exact again
        err["anim_aug"].append(diff(render(ls, gw, gh, posf, itemsf, phase=phase), true_anim))
        err["camera_aug"].append(diff(render(ls, gw, gh, posf, itemsf, camera=camera), true_cam))
        err["trail_aug"].append(diff(render(ls, gw, gh, posf, itemsf, trail=trail), true_trail))

    result = {k: round(float(np.mean(v)), 3) for k, v in err.items()}
    out = {
        "seed": seed, "horizon": horizon,
        "mean_abs_pixel_error": result,
        "interpretation": {
            "pure_sufficient": result["pure"] == 0.0,
            "anim_needs_phase": result["anim"] > 0 and result["anim_aug"] == 0.0,
            "camera_needs_history": result["camera"] > 0 and result["camera_aug"] == 0.0,
            "trail_needs_history": result["trail"] > 0 and result["trail_aug"] == 0.0,
        },
    }
    with open("results/exp4_render.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    probe()
