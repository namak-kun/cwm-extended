"""Battery: CWM trace prediction on (A) multiple languages + harder C, and
(B) heavier ML workloads. Metric = does CWM predict the entry function's final
RETURN value? Ground truth = really compiling/running each program.

All programs: an entry function marked `<< START_OF_TRACE` that returns an int,
plus a driver that prints it (so ground truth = stdout). CWM traces the entry
function and we read its predicted RETURN value.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

# ----------------------------------------------------------------------------
# PROGRAMS: (name, lang, source).  Entry fn marked; driver prints result.
# ----------------------------------------------------------------------------
PROGRAMS = []

# ---- Languages -------------------------------------------------------------
PROGRAMS.append(("C_recursion_ptr_struct", "c", r'''#include <stdio.h>
typedef struct { int x, y; } Point;
int fib(int n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}
void shift(Point *p, int dx) { p->x += dx; }
int compute() {  // << START_OF_TRACE
    Point p = {3, 7};
    shift(&p, 5);
    int f = fib(7);
    return p.x + p.y + f;
}
int main() { printf("%d\n", compute()); return 0; }
'''))

PROGRAMS.append(("C_bits_string", "c", r'''#include <stdio.h>
#include <string.h>
int popcount(unsigned int v) {
    int c = 0;
    while (v) { c += v & 1; v >>= 1; }
    return c;
}
int compute() {  // << START_OF_TRACE
    char s[] = "hello";
    int len = strlen(s);
    int bits = popcount(0xB7);   // 10110111 -> 6 set bits
    return len + bits;
}
int main() { printf("%d\n", compute()); return 0; }
'''))

PROGRAMS.append(("Cpp_vector_class", "cpp", r'''#include <iostream>
#include <vector>
class Acc {
    int total = 0;
public:
    void add(int v) { total += v; }
    int get() { return total; }
};
int compute() {  // << START_OF_TRACE
    std::vector<int> v = {2, 4, 6, 8};
    Acc a;
    for (int x : v) a.add(x * x);
    return a.get();
}
int main() { std::cout << compute() << std::endl; }
'''))

PROGRAMS.append(("JS_loops", "js", r'''function compute() {  // << START_OF_TRACE
    let arr = [1, 2, 3, 4, 5];
    let total = 0;
    for (let i = 0; i < arr.length; i++) {
        total += arr[i] * arr[i];
    }
    return total;
}
console.log(compute());
'''))

PROGRAMS.append(("JS_map_reduce", "js", r'''function compute() {  // << START_OF_TRACE
    let arr = [1, 2, 3, 4, 5];
    let doubled = arr.map(x => x * 2);
    let sum = doubled.reduce((a, b) => a + b, 0);
    return sum;
}
console.log(compute());
'''))

PROGRAMS.append(("Rust_iter", "rust", r'''fn compute() -> i32 {  // << START_OF_TRACE
    let v = vec![1, 2, 3, 4];
    let mut total = 0;
    for x in &v {
        total += x * x;
    }
    total
}
fn main() {
    println!("{}", compute());
}
'''))

PROGRAMS.append(("Java_array", "java", r'''public class Prog {
    static int compute() {  // << START_OF_TRACE
        int[] arr = {5, 3, 8, 1};
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) max = arr[i];
        }
        return max * 2;
    }
    public static void main(String[] args) {
        System.out.println(compute());
    }
}
'''))

# ---- ML workloads (Python) -------------------------------------------------
PROGRAMS.append(("ML_matmul_pure", "py", '''def main():  # << START_OF_TRACE
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                C[i][j] += A[i][k] * B[k][j]
    return C[0][0] + C[1][1]

print(main())
'''))

PROGRAMS.append(("ML_nn_forward_pure", "py", '''def main():  # << START_OF_TRACE
    x = [1.0, 2.0]
    W = [[0.5, -0.5], [1.0, 1.0]]
    h = [0.0, 0.0]
    for i in range(2):
        for j in range(2):
            h[i] += W[i][j] * x[j]
        if h[i] < 0:
            h[i] = 0.0
    return int(round((h[0] + h[1]) * 10))

print(main())
'''))

PROGRAMS.append(("ML_grad_descent_pure", "py", '''def main():  # << START_OF_TRACE
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    w = 0.0
    lr = 0.01
    for epoch in range(3):
        for k in range(4):
            pred = w * xs[k]
            err = pred - ys[k]
            w = w - lr * err * xs[k]
    return int(round(w * 1000))

print(main())
'''))

PROGRAMS.append(("ML_numpy_matmul", "py", '''import numpy as np

def main():  # << START_OF_TRACE
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = A @ B
    return int(C[0, 0] + C[1, 1])

print(main())
'''))


# ----------------------------------------------------------------------------
# Ground-truth execution
# ----------------------------------------------------------------------------
def true_output(lang: str, src: str):
    with tempfile.TemporaryDirectory() as d:
        try:
            if lang == "py":
                f = os.path.join(d, "p.py"); open(f, "w").write(src)
                r = subprocess.run([sys.executable, f], capture_output=True, text=True, timeout=60)
            elif lang == "c":
                f, exe = os.path.join(d, "p.c"), os.path.join(d, "p")
                open(f, "w").write(src)
                if subprocess.run(["gcc", "-O0", "-o", exe, f], capture_output=True).returncode: return None
                r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
            elif lang == "cpp":
                f, exe = os.path.join(d, "p.cpp"), os.path.join(d, "p")
                open(f, "w").write(src)
                if subprocess.run(["g++", "-O0", "-std=c++17", "-o", exe, f], capture_output=True).returncode: return None
                r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
            elif lang == "js":
                f = os.path.join(d, "p.js"); open(f, "w").write(src)
                r = subprocess.run(["node", f], capture_output=True, text=True, timeout=60)
            elif lang == "rust":
                f, exe = os.path.join(d, "p.rs"), os.path.join(d, "p")
                open(f, "w").write(src)
                if subprocess.run(["rustc", "-O", "-o", exe, f], capture_output=True).returncode: return None
                r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
            elif lang == "java":
                f = os.path.join(d, "Prog.java"); open(f, "w").write(src)
                r = subprocess.run(["java", f], capture_output=True, text=True, timeout=120)
            else:
                return None
            out = r.stdout.strip().splitlines()
            return int(out[-1]) if out else None
        except (subprocess.TimeoutExpired, ValueError):
            return None


def main_run(model_path, tp=4, only=None, dump=False, lora=None):
    from models.cwm_trace import (CWMvLLM, Event, CALL_SEP, build_prompt, parse_full_trace, resolve_locals)
    from run_ood import cwm_final_return

    progs = [p for p in PROGRAMS if (only is None or p[0] in only)]
    t0 = time.time()
    truths = {nm: true_output(lang, src) for nm, lang, src in progs}
    print("ground truths:", truths, flush=True)

    m = CWMvLLM(model_path, tp=tp, max_model_len=8192, lora_path=lora)
    print(f"== CWM loaded {'+LoRA '+lora if lora else '(base)'} ==", flush=True)

    prompts = [build_prompt(m, src, [], force_event=Event.CALL) for _, _, src in progs]
    caps = [2000] * len(progs)
    gens = m.gen_full_trace_batch(prompts, caps)

    results = {}
    for (nm, lang, src), gen in zip(progs, gens):
        pred = parse_full_trace(m, [CALL_SEP] + gen)
        cv = cwm_final_return(pred)
        tv = truths[nm]
        ok = (cv == tv) if tv is not None else None
        results[nm] = {"lang": lang, "true": tv, "cwm": cv, "match": ok, "frames": len(pred)}
        print(f"[{nm:24} {lang:5}] true={tv}  cwm={cv}  MATCH={ok}  frames={len(pred)}", flush=True)
        if dump:
            print(f"  ---- trace dump ({nm}) ----")
            for f in pred[:50]:
                print(f"    {f.event.name:7} {f.source_line.strip()[:50]:50} {resolve_locals(f)} arg={f.arg}")
            print()

    out = "results/cwm_battery_fail.json" if only else "results/cwm_battery.json"
    if lora:
        out = out.replace(".json", "_lora.json")
    json.dump({"model": model_path, "lora": lora, "results": results, "elapsed_sec": round(time.time()-t0, 1)},
              open(out, "w"), indent=2)
    print(f"\nsaved -> {out} ({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma-separated program names")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None
    main_run(a.model_path, a.tp, only=only, dump=(a.dump or bool(only)), lora=a.lora)
