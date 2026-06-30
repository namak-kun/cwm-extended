"""Shared prompting + parsing (no torch dependency, importable from any backend)."""
from __future__ import annotations

import json
from typing import Any

SYS = ("You are a deterministic program interpreter. You execute the given code "
       "exactly as written. You do not explain; you only output the resulting state.")


def build_prompt(code: str, history: list[dict], action: str, rng_log=None) -> str:
    state = history[-1]
    parts = []
    if code:
        parts.append("Here is the program:\n\n```python\n" + code + "\n```\n")
    else:
        parts.append("(The program source is hidden.)\n")
    if len(history) > 1:
        parts.append("Recent previous states (oldest first):\n" +
                     "\n".join(json.dumps(h) for h in history[:-1]) + "\n")
    parts.append("Current state (JSON):\n" + json.dumps(state))
    if rng_log:
        parts.append("Random draws consumed THIS step, in the order the code calls "
                     "rng: " + json.dumps(rng_log))
    parts.append(f"Apply update(state, action) with action = {action!r}.")
    parts.append("Output ONLY the resulting next state as one line of compact JSON.")
    return "\n\n".join(parts)


def extract_json(text: str) -> Any | None:
    """Extract the first balanced {...} object, ignoring code fences/prose."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    return None
