"""Ground-truth execution tracer + scorer for CWM state-tracking evaluation.

We run a program for real under sys.settrace and record frames in CWM's
convention (CALL/LINE/RETURN, full locals rendered as repr-strings). Then we
compare CWM's free-rollout predicted trace against this ground truth, frame by
frame, to answer: HOW FAR can CWM track state before diverging?
"""
from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field


@dataclass
class GTFrame:
    event: str          # 'call' | 'line' | 'return'
    lineno: int
    source_line: str
    locals: dict        # {name: repr(value)}  full locals at this point
    ret: str | None = None


def _expand_value(v, depth=0, max_depth=2):
    """phi-expansion: render objects by their attributes (sufficient statistic)
    instead of the opaque '<Acc object at 0x..>' repr. Bounded depth/size."""
    if depth > max_depth:
        try:
            return repr(v)
        except Exception:
            return "<unrepr>"
    # primitives / containers: keep normal repr-ish but recurse into objects
    if isinstance(v, (int, float, bool, str, type(None))):
        return repr(v)
    if isinstance(v, (list, tuple)):
        inner = [_expand_value(x, depth + 1, max_depth) for x in v[:12]]
        return "[" + ", ".join(inner) + ("]" if len(v) <= 12 else ", ...]")
    if isinstance(v, dict):
        items = list(v.items())[:12]
        return "{" + ", ".join(f"{k!r}: {_expand_value(val, depth+1, max_depth)}"
                               for k, val in items) + "}"
    # objects with __dict__: expand to {attr: value}  (CWM-fix: include EMPTY __dict__
    # too -- a freshly-constructed object at its __init__ call frame has {} and must NOT
    # fall through to repr(), which leaks a non-deterministic '<Cls object at 0x..>' addr)
    d = getattr(v, "__dict__", None)
    if isinstance(d, dict):
        cls = type(v).__name__
        body = ", ".join(f"{k!r}: {_expand_value(val, depth+1, max_depth)}"
                         for k, val in list(d.items())[:16])
        return f"{cls}({{{body}}})"
    try:
        return repr(v)
    except Exception:
        return "<unrepr>"


def trace_program(source: str, entry: str, max_frames: int = 4000,
                  expand_objects: bool = False, stepover_depth: int | None = None) -> list[GTFrame]:
    """Execute `source`, recording the trace of `entry` and everything it calls.

    expand_objects=True applies phi-expansion: object locals are rendered by their
    attributes (a sufficient statistic) rather than the opaque object repr.

    stepover_depth=d: STEP-OVER abstraction -- record LINE events only at call-depth <= d, and
    record CALL/RETURN (incl. the opaque return value) up to depth d+1. So with entry=main at depth 1
    and stepover_depth=1, you get main's lines + each step() call/return (the whole tick effect as one
    opaque transition), skipping step()'s interior. This is the exact ground truth for the tick-level
    abstraction (REPORT 10/29)."""
    ns: dict = {}
    render = (lambda v: _expand_value(v)) if expand_objects else None
    code = compile(source, "<gt>", "exec")
    src_lines = source.splitlines()
    frames: list[GTFrame] = []
    active = {"on": False, "depth": 0}

    def _should_record(event, depth):
        if stepover_depth is None:
            return True
        if event == "line":
            return depth <= stepover_depth
        # call/return: record up to one level below (the stepped-over boundary), not deeper
        return depth <= stepover_depth + 1

    def record(frame, event, arg, depth):
        if len(frames) >= max_frames:
            return tracer
        if not _should_record(event, depth):
            return tracer
        co = frame.f_code
        lineno = frame.f_lineno
        line = src_lines[lineno - 1] if 1 <= lineno <= len(src_lines) else ""
        loc = {}
        for k, v in frame.f_locals.items():
            try:
                loc[k] = render(v) if render else repr(v)
            except Exception:
                loc[k] = "<unrepr>"
        if event == "call":
            frames.append(GTFrame("call", lineno, line, loc))
        elif event == "line":
            frames.append(GTFrame("line", lineno, line, loc))
        elif event == "return":
            try:
                r = render(arg) if render else repr(arg)
            except Exception:
                r = "<unrepr>"
            frames.append(GTFrame("return", lineno, line, loc, ret=r))
        return tracer

    def tracer(frame, event, arg):
        co = frame.f_code
        if not active["on"]:
            if event == "call" and co.co_name == entry:
                active["on"] = True
                active["depth"] = 1
                return record(frame, "call", arg, active["depth"])
            return tracer
        # active
        if event == "call":
            active["depth"] += 1
            return record(frame, "call", arg, active["depth"])
        if event == "line":
            return record(frame, "line", arg, active["depth"])
        if event == "return":
            record(frame, "return", arg, active["depth"])
            active["depth"] -= 1
            if active["depth"] <= 0:
                active["on"] = False
            return tracer
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        exec(code, ns)
    except Exception:
        pass
    finally:
        sys.settrace(old)
    return frames


# -------- value/line normalization for fair comparison --------
def norm_val(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    # CWM stores repr-as-string; GT stores repr. Strip one layer of quotes so
    # "'a'" == 'a', '1' == 1, etc.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s


def norm_line(s: str) -> str:
    return s.strip().rstrip("# << START_OF_TRACE").strip()


EVT_MAP = {"call": "CALL", "line": "LINE", "return": "RETURN"}


def gt_to_input_frames(gt: list[GTFrame]):
    """Convert ground-truth frames into CWM input Frames (full locals, repr
    values) for teacher-forced prompting. Mirrors cwmdbg's frame conventions."""
    from models.cwm_trace import Frame, Event
    evt = {"call": Event.CALL, "line": Event.LINE, "return": Event.RETURN}
    frames = []
    for g in gt:
        f = Frame(event=evt[g.event], source_line=g.source_line,
                  local_vars=dict(g.locals), arg=g.ret,
                  prev=frames[-1] if frames else None)
        frames.append(f)
    return frames


def score_frame(g: GTFrame, p, resolve_fn) -> dict:
    """Score a single predicted frame against one GT frame."""
    pl = resolve_fn(p)
    ev_ok = EVT_MAP.get(g.event, "?") == p.event.name
    line_ok = norm_line(g.source_line) == norm_line(p.source_line)
    nvars = ncorrect = 0
    for k, gv in g.locals.items():
        nvars += 1
        if k in pl and norm_val(pl[k]) == norm_val(gv):
            ncorrect += 1
    vals_ok = (ncorrect == nvars)
    ret_ok = True
    if g.event == "return" and g.ret is not None and p.arg is not None:
        ret_ok = norm_val(p.arg) == norm_val(g.ret)
    return {"ctrl": ev_ok and line_ok, "vals_ok": vals_ok,
            "vals": (ncorrect, nvars), "ret_ok": ret_ok,
            "frame_ok": ev_ok and line_ok and vals_ok and ret_ok}


def score_trace(gt: list[GTFrame], pred_frames, resolve_fn) -> dict:
    """Align predicted frames to GT by index; find first divergence + curves.

    Returns: first_div_frame, n_gt, control_ok_frac, value curves per depth.
    """
    n = min(len(gt), len(pred_frames))
    first_div = None
    control_ok = []
    value_ok = []        # per-frame: all GT vars correctly valued by CWM
    per_frame = []
    for i in range(n):
        g, p = gt[i], pred_frames[i]
        pl = resolve_fn(p)
        # control: same event kind + same source line
        ev_ok = EVT_MAP.get(g.event, "?") == p.event.name
        line_ok = norm_line(g.source_line) == norm_line(p.source_line)
        ctrl = ev_ok and line_ok
        control_ok.append(ctrl)
        # value: every GT local present & equal in CWM resolved locals
        vok, nvars, ncorrect = True, 0, 0
        for k, gv in g.locals.items():
            nvars += 1
            if k in pl and norm_val(pl[k]) == norm_val(gv):
                ncorrect += 1
            else:
                vok = False
        ret_ok = True
        if g.event == "return" and g.ret is not None and p.arg is not None:
            ret_ok = norm_val(p.arg) == norm_val(g.ret)
        frame_ok = ctrl and vok and ret_ok
        value_ok.append(frame_ok)
        per_frame.append({"i": i, "ctrl": ctrl, "vals": (ncorrect, nvars), "ok": frame_ok})
        if first_div is None and not frame_ok:
            first_div = i
    return {
        "n_gt": len(gt),
        "n_pred": len(pred_frames),
        "n_compared": n,
        "first_divergence_frame": first_div if first_div is not None else n,
        "fully_correct": first_div is None and len(pred_frames) >= len(gt),
        "control_acc": round(sum(control_ok) / n, 3) if n else 0.0,
        "frame_acc": round(sum(value_ok) / n, 3) if n else 0.0,
        "per_frame": per_frame,
    }
