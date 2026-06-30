"""Procedural 'weird gridworld' generator.

Design principle (from the rubber-duck critique): the emitted source code IS the
ground truth (we exec it to roll out trajectories) AND is the exact context shown
to the model. Because the transition rules are randomly composed per seed, a model
cannot succeed by RECALLING a known game -- it must READ the provided code. This is
the fair test of code-conditioned transition inference.

Knobs that defeat memorization / probe code-mediated causality:
  - naming_mode: 'semantic' | 'random' | 'misleading'
       misleading => action label "LEFT" may move +x (right); field "y" may be the x-axis.
  - boundary: 'clamp' | 'wrap' | 'bounce'
  - action effects are a permuted/sampled composition of primitive ops
  - mode-gated effects (effect depends on a hidden integer 'mode')
  - stochastic spawn with probability p, randomness either revealed or hidden
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Name pools. The KEY anti-prior trick lives here: in 'misleading' mode the
# human-suggestive label is deliberately wired to the opposite/orthogonal effect.
# ---------------------------------------------------------------------------
SEMANTIC_ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
RANDOM_ACTIONS = ["A", "B", "C", "D", "OP0", "OP1", "ACT_3", "Z9", "Q", "K7"]

SEMANTIC_FIELDS = {
    "pos": "pos", "vel": "vel", "score": "score", "mode": "mode",
    "alive": "alive", "items": "items", "target": "target",
}
RANDOM_FIELDS = {
    "pos": "p", "vel": "v", "score": "s", "mode": "m",
    "alive": "ok", "items": "xs", "target": "tg",
}
# misleading: names actively suggest the WRONG thing
MISLEADING_FIELDS = {
    "pos": "color", "vel": "pos", "score": "health", "mode": "score",
    "alive": "won", "items": "walls", "target": "spawn",
}


@dataclass
class GameSpec:
    seed: int
    gw: int
    gh: int
    naming_mode: str
    boundary: str
    names: dict
    actions: list           # labels exposed to the model
    action_ops: dict        # label -> list of op tuples
    spawn_prob: float
    n_modes: int
    has_target: bool
    init_items: int

    def fname(self, role: str) -> str:
        return self.names[role]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_spec(seed: int, *, naming_mode: str | None = None,
                stochastic: bool = False, n_actions: int = 4,
                complexity: int = 1) -> GameSpec:
    rng = random.Random(seed)
    naming_mode = naming_mode or rng.choice(["semantic", "random", "misleading"])
    gw = rng.choice([5, 6, 7, 8]) + 2 * (complexity - 1)
    gh = rng.choice([5, 6, 7, 8]) + 2 * (complexity - 1)
    boundary = rng.choice(["clamp", "wrap", "bounce"])
    n_modes = min(1 + complexity, 4) if complexity > 1 else rng.choice([1, 2, 3])
    # 'complexity' scales code length + state size. Semantic/misleading have only
    # 4 human labels; the random-naming pool supports up to 10 actions.
    if complexity > 1 and naming_mode == "random":
        n_actions = min(3 + complexity, len(RANDOM_ACTIONS))

    if naming_mode == "semantic":
        names = dict(SEMANTIC_FIELDS)
        labels = SEMANTIC_ACTIONS[:n_actions]
    elif naming_mode == "misleading":
        names = dict(MISLEADING_FIELDS)
        labels = SEMANTIC_ACTIONS[:n_actions]   # human labels, wrong wiring
    else:
        names = dict(RANDOM_FIELDS)
        labels = rng.sample(RANDOM_ACTIONS, n_actions)

    # Build the op-set for each action. We sample a primitive composition.
    # Primitive ops (structured; emitted to real python by emit_source):
    #   ("axis", axis, delta)            pos[axis] += delta
    #   ("vel", axis, delta)             vel[axis] += delta
    #   ("apply_vel",)                   pos += vel
    #   ("gate", mode_val, axis, delta)  if mode==mode_val: pos[axis]+=delta
    #   ("cycle_mode",)                  mode=(mode+1)%n_modes
    #   ("score", d)                     score += d
    #   ("score_target", d)              if pos==target: score+=d
    #   ("spawn",)                       stochastic item spawn
    axes = [0, 1]
    action_ops: dict[str, list] = {}
    for i, lbl in enumerate(labels):
        ops: list[tuple] = []
        # Primary movement op. In 'misleading' mode we intentionally scramble the
        # mapping so the semantic label does not match the effect.
        if naming_mode == "misleading":
            axis = rng.choice(axes)
            delta = rng.choice([-1, 1])
            ops.append(("axis", axis, delta))
        else:
            # canonical-ish for semantic; arbitrary for random
            canon = {0: ("axis", 0, -1), 1: ("axis", 0, 1),
                     2: ("axis", 1, -1), 3: ("axis", 1, 1)}
            if naming_mode == "semantic" and i in canon:
                ops.append(canon[i])
            else:
                ops.append(("axis", rng.choice(axes), rng.choice([-1, 1])))
        # Optional secondary effect
        r = rng.random()
        if r < 0.25 and n_modes > 1:
            ops.append(("gate", rng.randrange(n_modes), rng.choice(axes), rng.choice([-1, 1])))
        elif r < 0.45:
            ops.append(("score", rng.choice([1, 2, -1])))
        elif r < 0.60:
            ops.append(("score_target", rng.choice([1, 3])))
        elif r < 0.72 and n_modes > 1:
            ops.append(("cycle_mode",))
        elif r < 0.85:
            ops.append(("vel", rng.choice(axes), rng.choice([-1, 1])))
        if stochastic and rng.random() < 0.5:
            ops.append(("spawn",))
        # complexity adds extra deterministic secondary ops (longer code branches)
        for _ in range(complexity - 1):
            choice = rng.choice(["axis", "vel", "score", "gate"])
            if choice == "axis":
                ops.append(("axis", rng.choice(axes), rng.choice([-1, 1])))
            elif choice == "vel":
                ops.append(("vel", rng.choice(axes), rng.choice([-1, 1])))
            elif choice == "score":
                ops.append(("score", rng.choice([1, 2, -1])))
            elif n_modes > 1:
                ops.append(("gate", rng.randrange(n_modes), rng.choice(axes), rng.choice([-1, 1])))
        action_ops[lbl] = ops

    # Ensure at least one action spawns when stochastic requested
    if stochastic and not any(("spawn",) in v for v in action_ops.values()):
        action_ops[labels[rng.randrange(len(labels))]].append(("spawn",))

    return GameSpec(
        seed=seed, gw=gw, gh=gh, naming_mode=naming_mode, boundary=boundary,
        names=names, actions=labels, action_ops=action_ops,
        spawn_prob=rng.choice([0.3, 0.5, 0.7]) if stochastic else 0.0,
        n_modes=n_modes, has_target=any(
            o[0] == "score_target" for ops in action_ops.values() for o in ops
        ),
        init_items=rng.choice([0, 1, 2]) + (complexity - 1),
    )


# ---------------------------------------------------------------------------
# Source emission. The emitted module is BOTH ground truth and model context.
# ---------------------------------------------------------------------------
def _emit_ops(ops: list, s: GameSpec, ind: str) -> list[str]:
    POS, VEL, SC, MODE, ITEMS, TGT = (
        s.fname("pos"), s.fname("vel"), s.fname("score"),
        s.fname("mode"), s.fname("items"), s.fname("target"),
    )
    lines = []
    for op in ops:
        if op[0] == "axis":
            _, a, d = op
            lines.append(f'{ind}st["{POS}"][{a}] += {d}')
        elif op[0] == "vel":
            _, a, d = op
            lines.append(f'{ind}st["{VEL}"][{a}] += {d}')
        elif op[0] == "apply_vel":
            lines.append(f'{ind}st["{POS}"][0] += st["{VEL}"][0]')
            lines.append(f'{ind}st["{POS}"][1] += st["{VEL}"][1]')
        elif op[0] == "gate":
            _, mv, a, d = op
            lines.append(f'{ind}if st["{MODE}"] == {mv}:')
            lines.append(f'{ind}    st["{POS}"][{a}] += {d}')
        elif op[0] == "cycle_mode":
            lines.append(f'{ind}st["{MODE}"] = (st["{MODE}"] + 1) % {s.n_modes}')
        elif op[0] == "score":
            _, d = op
            lines.append(f'{ind}st["{SC}"] += {d}')
        elif op[0] == "score_target":
            _, d = op
            lines.append(f'{ind}if st["{POS}"] == st["{TGT}"]:')
            lines.append(f'{ind}    st["{SC}"] += {d}')
        elif op[0] == "spawn":
            lines.append(f'{ind}if rng.uniform() < {s.spawn_prob}:')
            lines.append(f'{ind}    st["{ITEMS}"].append([rng.randint(0, {s.gw - 1}), rng.randint(0, {s.gh - 1})])')
    if not ops:
        lines.append(f"{ind}pass")
    return lines


def emit_source(s: GameSpec) -> str:
    POS, VEL, SC, MODE, ALIVE, ITEMS, TGT = (
        s.fname("pos"), s.fname("vel"), s.fname("score"), s.fname("mode"),
        s.fname("alive"), s.fname("items"), s.fname("target"),
    )
    L = []
    L.append(f"# Procedurally generated state machine (seed={s.seed}, naming={s.naming_mode}).")
    L.append(f"# Grid is {s.gw} wide x {s.gh} tall. Boundary policy: {s.boundary}.")
    L.append(f"LEGAL_ACTIONS = {s.actions!r}")
    L.append(f"GW, GH = {s.gw}, {s.gh}")
    L.append("")
    # initial_state
    L.append("def initial_state(seed):")
    L.append("    import random as _r")
    L.append("    g = _r.Random(seed)")
    L.append("    st = {")
    L.append(f'        "{POS}": [g.randrange(GW), g.randrange(GH)],')
    L.append(f'        "{VEL}": [g.choice([-1, 0, 1]), g.choice([-1, 0, 1])],')
    L.append(f'        "{SC}": 0,')
    L.append(f'        "{MODE}": 0,')
    L.append(f'        "{ALIVE}": True,')
    L.append(f'        "{ITEMS}": [[g.randrange(GW), g.randrange(GH)] for _ in range({s.init_items})],')
    if s.has_target:
        L.append(f'        "{TGT}": [g.randrange(GW), g.randrange(GH)],')
    L.append("    }")
    L.append("    return st")
    L.append("")
    # update
    L.append("def update(st, action, rng=None):")
    L.append('    """Apply one input. `rng` supplies random draws when present."""')
    first = True
    for lbl, ops in s.action_ops.items():
        kw = "if" if first else "elif"
        first = False
        L.append(f'    {kw} action == {lbl!r}:')
        L += _emit_ops(ops, s, "        ")
    # boundary
    if s.boundary == "clamp":
        L.append(f'    st["{POS}"][0] = max(0, min(GW - 1, st["{POS}"][0]))')
        L.append(f'    st["{POS}"][1] = max(0, min(GH - 1, st["{POS}"][1]))')
    elif s.boundary == "wrap":
        L.append(f'    st["{POS}"][0] %= GW')
        L.append(f'    st["{POS}"][1] %= GH')
    else:  # bounce
        L.append(f'    if not (0 <= st["{POS}"][0] < GW):')
        L.append(f'        st["{VEL}"][0] = -st["{VEL}"][0]')
        L.append(f'        st["{POS}"][0] = max(0, min(GW - 1, st["{POS}"][0]))')
        L.append(f'    if not (0 <= st["{POS}"][1] < GH):')
        L.append(f'        st["{VEL}"][1] = -st["{VEL}"][1]')
        L.append(f'        st["{POS}"][1] = max(0, min(GH - 1, st["{POS}"][1]))')
    L.append("    return st")
    L.append("")
    # render: pure function of state -> HxWx3 uint8 (own renderer, sufficient by construction)
    L.append("def render(st, cell=16):")
    L.append("    import numpy as _np")
    L.append("    img = _np.zeros((GH * cell, GW * cell, 3), dtype=_np.uint8)")
    L.append("    img[:] = 30")
    L.append("    def put(cx, cy, rgb):")
    L.append("        x = cx % GW; y = cy % GH")
    L.append("        img[y*cell:(y+1)*cell, x*cell:(x+1)*cell] = rgb")
    L.append(f'    for it in st.get("{ITEMS}", []):')
    L.append("        put(it[0], it[1], (200, 180, 40))")
    if s.has_target:
        L.append(f'    tg = st.get("{TGT}")')
        L.append("    if tg: put(tg[0], tg[1], (40, 160, 90))")
    L.append(f'    p = st["{POS}"]; put(p[0], p[1], (220, 60, 60))')
    L.append("    return img")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# RNG that can RECORD (ground-truth rollout) or REPLAY (reveal draws to model)
# ---------------------------------------------------------------------------
class ScriptedRNG:
    def __init__(self, seed: int | None = None, replay: list | None = None):
        self._r = random.Random(seed)
        self._replay = list(replay) if replay is not None else None
        self.log: list[dict] = []   # per-step record of draws (cleared by caller)

    def uniform(self) -> float:
        if self._replay is not None:
            rec = self._replay.pop(0)
            v = rec["v"]
        else:
            v = self._r.random()
        self.log.append({"call": "uniform", "v": round(v, 4)})
        return v

    def randint(self, a: int, b: int) -> int:
        if self._replay is not None:
            rec = self._replay.pop(0)
            v = rec["v"]
        else:
            v = self._r.randint(a, b)
        self.log.append({"call": f"randint({a},{b})", "v": v})
        return v


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, source: str, spec: GameSpec | None = None):
        self.source = source
        self.spec = spec
        ns: dict[str, Any] = {}
        exec(compile(source, "<gen_game>", "exec"), ns)  # trusted: we generated it
        self.ns = ns
        self.LEGAL_ACTIONS = ns["LEGAL_ACTIONS"]
        self._initial = ns["initial_state"]
        self._update = ns["update"]
        self._render = ns["render"]

    def initial_state(self, seed: int = 0) -> dict:
        return self._initial(seed)

    def step(self, state: dict, action: str, rng: ScriptedRNG | None = None) -> dict:
        return self._update(json.loads(json.dumps(state)), action, rng)

    def render(self, state):
        return self._render(state)

    @classmethod
    def generate(cls, seed: int, **kw) -> "Game":
        spec = sample_spec(seed, **kw)
        return cls(emit_source(spec), spec)


if __name__ == "__main__":
    for sd in range(3):
        g = Game.generate(sd, naming_mode="misleading", stochastic=True)
        print("=" * 70)
        print(g.source)
        st = g.initial_state(123)
        print("init:", st)
        rng = ScriptedRNG(seed=7)
        for a in g.LEGAL_ACTIONS[:3]:
            rng.log = []
            st = g.step(st, a, rng)
            print(f"  {a:>6} draws={rng.log} -> {st}")
