"""CWM native trace-prediction client, backed by in-process vLLM.

CWM's real trained capability is execution-trace prediction. Prompt format
(verified token IDs match facebook/cwm tokenizer + PROMPTING_GUIDE):

    <|begin_of_text|><|trace_context_start|>$SOURCE<|frame_sep|>
    then frames:
      <|call_sep|>$LOCALS_JSON<|action_sep|>$SOURCE_LINE<|frame_sep|>
      <|line_sep|>$LOCALS_JSON<|action_sep|>$SOURCE_LINE<|frame_sep|>
      <|return_sep|><|action_sep|>$SOURCE_LINE<|arg_sep|>$VALUE_JSON<|frame_sep|>
      <|exception_sep|><|action_sep|>$SOURCE_LINE<|arg_sep|>$VALUE_JSON<|frame_sep|>
    terminated by <|end_of_text|> when the entry scope is exited.

Locals are diff-based: a frame lists only changed vars (values as JSON strings);
".." means "same value as the previous occurrence in this scope".

This module ports the parsing/resolution logic of the repo's demos/cwmdbg.py but
drives generation through vLLM (no fastgen server needed) and supports both
single-frame stepping (teacher-forced or free) and whole-trace generation.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

# Verified IDs (facebook/cwm tokenizer)
BOS = 128000
EOS = 128001
TRACE_CTX_START = 128107
FRAME_SEP = 128100
ACTION_SEP = 128101
RETURN_SEP = 128102
CALL_SEP = 128103
LINE_SEP = 128104
EXCEPTION_SEP = 128105
ARG_SEP = 128106

EVENT_TOKENS = {CALL_SEP, LINE_SEP, RETURN_SEP, EXCEPTION_SEP}
START_MARKER = "  # << START_OF_TRACE"


class Event(Enum):
    CALL = auto()
    LINE = auto()
    RETURN = auto()
    EXCEPTION = auto()


_EVT2TOK = {Event.CALL: CALL_SEP, Event.LINE: LINE_SEP,
            Event.RETURN: RETURN_SEP, Event.EXCEPTION: EXCEPTION_SEP}
_TOK2EVT = {v: k for k, v in _EVT2TOK.items()}


@dataclass
class Frame:
    event: Event
    source_line: str
    local_vars: dict   # diff-based (values are JSON strings; ".." = unchanged)
    arg: str | None
    prev: "Frame | None" = None


class CWMvLLM:
    def __init__(self, model_path: str, tp: int = 4, max_model_len: int = 32768,
                 temperature: float = 0.0, lora_path: str | None = None):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(model=model_path, tensor_parallel_size=tp, dtype="bfloat16",
                       gpu_memory_utilization=0.92, max_model_len=max_model_len,
                       enforce_eager=False, enable_lora=lora_path is not None,
                       max_lora_rank=32)
        self.SP = SamplingParams
        self.temperature = temperature
        self._lora = None
        if lora_path is not None:
            from vllm.lora.request import LoRARequest
            self._lora = LoRARequest("cwm_adapter", 1, lora_path)

    def _gen_kwargs(self):
        return {"lora_request": self._lora} if self._lora is not None else {}

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    # ---- raw generation: one frame, stop at frame_sep ----
    def _gen_frame_tokens(self, prompt_ids: list[int], max_tokens: int = 2048) -> list[int]:
        from vllm import TokensPrompt
        sp = self.SP(temperature=self.temperature, max_tokens=max_tokens,
                     stop_token_ids=[FRAME_SEP, EOS])
        out = self.llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), sp, use_tqdm=False,
                                **self._gen_kwargs())
        return list(out[0].outputs[0].token_ids)

    def _gen_frames_batch(self, prompts: list[list[int]], max_tokens: int = 2048) -> list[list[int]]:
        from vllm import TokensPrompt
        sp = self.SP(temperature=self.temperature, max_tokens=max_tokens,
                     stop_token_ids=[FRAME_SEP, EOS])
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                 sp, use_tqdm=False, **self._gen_kwargs())
        return [list(o.outputs[0].token_ids) for o in outs]

    # ---- whole-trace generation in one call (true free rollout), stop at EOS ----
    def gen_full_trace_tokens(self, prompt_ids: list[int], max_tokens: int = 16384) -> list[int]:
        from vllm import TokensPrompt
        sp = self.SP(temperature=self.temperature, max_tokens=max_tokens,
                     stop_token_ids=[EOS])
        out = self.llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), sp, use_tqdm=False,
                                **self._gen_kwargs())
        return list(out[0].outputs[0].token_ids)

    def gen_full_trace_batch(self, prompts: list[list[int]], caps: list[int]) -> list[list[int]]:
        """Batched whole-trace generation: vLLM decodes all sequences in parallel
        (much higher GPU utilization than sequential single-sequence calls)."""
        from vllm import TokensPrompt
        sps = [self.SP(temperature=self.temperature, max_tokens=c, stop_token_ids=[EOS])
               for c in caps]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                 sps, use_tqdm=True, **self._gen_kwargs())
        return [list(o.outputs[0].token_ids) for o in outs]


# ---------- frame (de)serialization ----------
def frame_to_tokens(model: CWMvLLM, f: Frame) -> list[int]:
    t = [_EVT2TOK[f.event]]
    if f.event in (Event.CALL, Event.LINE):
        t += model.encode(json.dumps(f.local_vars))
    t += [ACTION_SEP]
    t += model.encode(f.source_line)
    if f.event in (Event.RETURN, Event.EXCEPTION):
        t += [ARG_SEP]
        t += model.encode(json.dumps(f.arg))
    return t


def build_prompt(model: CWMvLLM, source: str, frames: list[Frame],
                 force_event: Event | None = None) -> list[int]:
    t = [BOS, TRACE_CTX_START] + model.encode(source) + [FRAME_SEP]
    for f in frames:
        t += frame_to_tokens(model, f) + [FRAME_SEP]
    if force_event is not None:
        t += [_EVT2TOK[force_event]]
    return t


def parse_frame(model: CWMvLLM, gen_tokens: list[int], forced_event: Event | None,
                prev: Frame | None) -> Frame | None:
    """Parse one generated frame. Returns None if EOS/end-of-trace."""
    rep = deque(gen_tokens)
    # strip trailing stop token if present
    while rep and rep[-1] in (FRAME_SEP,):
        rep.pop()
    if not rep and forced_event is None:
        return None
    event = forced_event
    if event is None:
        head = rep[0] if rep else EOS
        if head in _TOK2EVT:
            event = _TOK2EVT[head]
            rep.popleft()
        elif head == EOS:
            return None
        else:
            # unexpected; treat as line frame content
            event = Event.LINE
    # locals (CALL/LINE only)
    local_tokens: list[int] = []
    if event in (Event.CALL, Event.LINE):
        while rep and rep[0] != ACTION_SEP:
            local_tokens.append(rep.popleft())
    # consume action_sep
    if rep and rep[0] == ACTION_SEP:
        rep.popleft()
    # source line (until arg_sep or end)
    source_tokens: list[int] = []
    while rep and rep[0] not in (ARG_SEP, FRAME_SEP):
        source_tokens.append(rep.popleft())
    arg_tokens: list[int] = []
    if rep and rep[0] == ARG_SEP:
        rep.popleft()
        while rep and rep[0] != FRAME_SEP:
            arg_tokens.append(rep.popleft())

    local_vars = {}
    if local_tokens:
        s = model.decode(local_tokens)
        try:
            local_vars = json.loads(s)
        except json.JSONDecodeError:
            local_vars = {"_PARSE_ERR_": s}
    arg = None
    if arg_tokens:
        s = model.decode(arg_tokens)
        try:
            arg = json.loads(s)
        except json.JSONDecodeError:
            arg = s
    return Frame(event=event, source_line=model.decode(source_tokens).strip("\n"),
                 local_vars=local_vars, arg=arg, prev=prev)


def resolve_locals(frame: Frame) -> dict:
    """Unroll the diff-based representation to full locals at this frame."""
    f = frame
    if f.event in (Event.RETURN, Event.EXCEPTION) and f.prev is not None:
        f = f.prev
    lv = dict(f.local_vars)
    for k, v in list(lv.items()):
        if v != "..":
            continue
        g = f.prev
        while g is not None:
            gv = g.local_vars.get(k, "..")
            if gv != "..":
                lv[k] = gv
                break
            if g.event in (Event.CALL, Event.RETURN):
                break
            g = g.prev
    return lv


def split_frames(tokens: list[int]) -> list[list[int]]:
    """Split a generated token stream into per-frame token lists on FRAME_SEP."""
    frames, cur = [], []
    for t in tokens:
        if t == EOS:
            break
        if t == FRAME_SEP:
            frames.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        frames.append(cur)
    return frames


def parse_full_trace(model: CWMvLLM, gen_tokens: list[int]) -> list[Frame]:
    """Parse a whole generated trace (free rollout) into Frame objects."""
    frames: list[Frame] = []
    for ft in split_frames(gen_tokens):
        if not ft:
            continue
        prev = frames[-1] if frames else None
        f = parse_frame(model, ft + [FRAME_SEP], forced_event=None, prev=prev)
        if f is None:
            break
        frames.append(f)
    return frames
