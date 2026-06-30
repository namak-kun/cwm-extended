"""Dynamics-model interface + non-LLM baselines.

A DynamicsModel predicts the next symbolic state given:
  - code:    the program source (or "" when omitted, for the prior-only baseline)
  - history: list of recent states (most recent last)
  - action:  the input applied
  - rng_log: revealed random draws for this step (or None when hidden)
"""
from __future__ import annotations

import json
from typing import Any

from worlds.gridgen import Game, ScriptedRNG


class DynamicsModel:
    name = "base"

    def predict(self, code: str, history: list[dict], action: str,
                rng_log: list | None = None) -> dict | None:
        raise NotImplementedError


class CopyModel(DynamicsModel):
    """No-change baseline: predict the state is unchanged. Beating this is the
    floor for 'the model learned anything about the transition'."""
    name = "copy"

    def predict(self, code, history, action, rng_log=None):
        return json.loads(json.dumps(history[-1]))


class OracleModel(DynamicsModel):
    """Perfect predictor (calls the true update). Sanity ceiling = 1.0 everywhere.
    Validates the eval pipeline itself."""
    name = "oracle"

    def __init__(self, game: Game):
        self.game = game

    def predict(self, code, history, action, rng_log=None):
        rng = None
        if rng_log:
            rng = ScriptedRNG(replay=[{"v": d["v"]} for d in rng_log])
        return self.game.step(history[-1], action, rng)


class NoisyOracle(DynamicsModel):
    """Oracle that corrupts one field with probability p -- to verify metrics
    actually move when predictions degrade."""
    name = "noisy_oracle"

    def __init__(self, game: Game, p: float = 0.3, seed: int = 0):
        self.game = game
        self.p = p
        import random
        self.r = random.Random(seed)

    def predict(self, code, history, action, rng_log=None):
        rng = ScriptedRNG(replay=[{"v": d["v"]} for d in rng_log]) if rng_log else None
        st = self.game.step(history[-1], action, rng)
        if self.r.random() < self.p:
            for k, v in st.items():
                if isinstance(v, int) and not isinstance(v, bool):
                    st[k] = v + self.r.choice([-1, 1])
                    break
        return st
