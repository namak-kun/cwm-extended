"""OOD probe: how does CWM's (Python-trained) trace prediction handle
out-of-distribution programs? C, multiprocessing, threading, numpy/ML.

For each program we get a TRUE final result by really executing it (gcc for C,
exec for Python), let CWM free-roll the native trace, and compare:
  - did CWM predict the correct final RETURN value? (language-agnostic, robust)
  - qualitative: does the predicted trace look coherent / track state?
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

from models.cwm_trace import (CWMvLLM, Event, CALL_SEP, build_prompt,
                              parse_full_trace, resolve_locals)


# ---------------- programs ----------------
C_LOOP = ('c', r'''int main() {  // << START_OF_TRACE
    int a = 1;
    int b = 0;
    for (int i = 0; i < 4; i++) {
        a = a * 2 + 1;
        b = b + a;
    }
    return b;
}
''', "main")

C_ARRAY = ('c', r'''int main() {  // << START_OF_TRACE
    int arr[4] = {5, 2, 8, 1};
    int max = arr[0];
    for (int i = 1; i < 4; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}
''', "main")

PY_MULTIPROC = ('py', '''from multiprocessing import Pool

def sq(x):
    return x * x

def main():  # << START_OF_TRACE
    data = [1, 2, 3, 4]
    with Pool(2) as p:
        results = p.map(sq, data)
    total = sum(results)
    return total

main()
''', "main")

PY_THREADING = ('py', '''import threading

def main():  # << START_OF_TRACE
    counter = [0]
    def inc():
        for _ in range(3):
            counter[0] += 1
    t1 = threading.Thread(target=inc)
    t2 = threading.Thread(target=inc)
    t1.start(); t2.start()
    t1.join(); t2.join()
    return counter[0]

main()
''', "main")

PY_NUMPY_SMALL = ('py', '''import numpy as np

def main():  # << START_OF_TRACE
    a = np.array([1, 2, 3])
    b = a * 2
    c = int(b.sum())
    return c

main()
''', "main")

PY_NUMPY_BIG = ('py', '''import numpy as np

def main():  # << START_OF_TRACE
    m = np.arange(100).reshape(10, 10)
    s = int(m.sum())
    d = int(np.diag(m).sum())
    return s + d

main()
''', "main")

PY_NESTED = ('py', '''def helper(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

def main():  # << START_OF_TRACE
    x = helper(4)
    y = helper(3)
    return x + y

main()
''', "main")

PROGRAMS = {
    "C_loop": C_LOOP, "C_array": C_ARRAY,
    "py_multiproc": PY_MULTIPROC, "py_threading": PY_THREADING,
    "py_numpy_small": PY_NUMPY_SMALL, "py_numpy_big": PY_NUMPY_BIG,
    "py_nested": PY_NESTED,
}


# ---------------- ground-truth execution ----------------
def true_final(lang: str, src: str) -> int | None:
    if lang == "c":
        with tempfile.TemporaryDirectory() as d:
            cf, exe = os.path.join(d, "p.c"), os.path.join(d, "p")
            open(cf, "w").write(src)
            r = subprocess.run(["gcc", "-O0", "-o", exe, cf], capture_output=True)
            if r.returncode != 0:
                return None
            rr = subprocess.run([exe])
            return rr.returncode  # main()'s return (0-255)
    else:
        # Run as a REAL subprocess (robust for multiprocessing / threading).
        body = "\n".join(l for l in src.replace("  # << START_OF_TRACE", "").splitlines()
                         if l.strip() != "main()")
        runner = body + '\n\nif __name__ == "__main__":\n    import sys; sys.stdout.write(str(main()))\n'
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "p.py")
            open(pf, "w").write(runner)
            rr = subprocess.run([sys.executable, pf], capture_output=True, text=True)
            try:
                return int(rr.stdout.strip())
            except (ValueError, TypeError):
                return None


def cwm_final_return(frames):
    """Last RETURN frame's arg = CWM's predicted final return value."""
    for f in reversed(frames):
        if f.event == Event.RETURN and f.arg is not None:
            try:
                return int(str(f.arg).strip().strip('"').strip("'"))
            except (ValueError, TypeError):
                return f.arg
    return None


def main(model_path, tp=4):
    t0 = time.time()
    m = CWMvLLM(model_path, tp=tp, max_model_len=8192)
    print("== CWM loaded ==\n", flush=True)

    names = list(PROGRAMS)
    prompts, caps, truths = [], [], []
    for nm in names:
        lang, src, entry = PROGRAMS[nm]
        truths.append(true_final(lang, src))
        prompts.append(build_prompt(m, src, [], force_event=Event.CALL))
        caps.append(1800)
    gens = m.gen_full_trace_batch(prompts, caps)

    results = {}
    for nm, gen, tv in zip(names, gens, truths):
        lang, src, entry = PROGRAMS[nm]
        pred = parse_full_trace(m, [CALL_SEP] + gen)
        cv = cwm_final_return(pred)
        ok = (cv == tv) if tv is not None else None
        results[nm] = {"lang": lang, "true_final": tv, "cwm_final": cv,
                       "match": ok, "n_pred_frames": len(pred)}
        print(f"=== {nm} ({lang}) ===")
        print(f"  true_final={tv}  cwm_final={cv}  MATCH={ok}  frames={len(pred)}")
        # show a few middle frames for qualitative read
        for f in pred[:6]:
            print(f"    {f.event.name:7} {f.source_line.strip()[:48]:48} {resolve_locals(f)}")
        if len(pred) > 6:
            print(f"    ... (+{len(pred)-6} more frames)")
        print()

    out = "results/cwm_ood.json"
    json.dump({"model": model_path, "results": results,
               "elapsed_sec": round(time.time()-t0, 1)}, open(out, "w"), indent=2)
    print(f"saved -> {out} ({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
