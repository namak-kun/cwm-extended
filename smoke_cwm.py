"""Smoke test: load CWM, generate a full trace for the repo demo program,
free-rollout (CWM eats its own predicted frames). Validates the format pipeline.
"""
import sys
from models.cwm_trace import (CWMvLLM, Event, build_prompt, parse_frame,
                              resolve_locals, EOS)

CWM_PATH = sys.argv[1] if len(sys.argv) > 1 else "facebook/cwm"

DEMO = '''def count_letters(s, letter):
    n = 0
    for c in s:
        n += int(c == letter)
    return n

def f(c):  # << START_OF_TRACE
    word = "strawberry"
    num = count_letters(word, c)
    return num
'''


def main():
    m = CWMvLLM(CWM_PATH, tp=4, max_model_len=8192)
    print("== CWM loaded ==", flush=True)

    frames = []
    # first frame: force a CALL at the entry
    prompt = build_prompt(m, DEMO, frames, force_event=Event.CALL)
    gen = m._gen_frame_tokens(prompt)
    f = parse_frame(m, gen, forced_event=Event.CALL, prev=None)
    frames.append(f)

    for step in range(40):
        prompt = build_prompt(m, DEMO, frames, force_event=None)
        gen = m._gen_frame_tokens(prompt)
        f = parse_frame(m, gen, forced_event=None, prev=frames[-1])
        if f is None:
            print(f"[step {step}] END OF TRACE (eos)")
            break
        frames.append(f)
        rl = resolve_locals(f)
        print(f"[{step:2}] {f.event.name:9} line={f.source_line!r:45} "
              f"locals={rl} arg={f.arg}")
    print(f"\ntotal frames: {len(frames)}")


if __name__ == "__main__":
    main()
