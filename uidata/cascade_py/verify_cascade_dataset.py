from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "uitrans_cascade_py.jsonl"


def main():
    failures = []
    by_action = Counter()
    max_depts = 0
    max_errors = 0
    with DATA.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            row = json.loads(line)
            proc = subprocess.run(["python3", "-c", row["prompt_src"]], text=True, capture_output=True)
            if proc.returncode != 0:
                failures.append((lineno, "exec", proc.stderr[-500:]))
                continue
            try:
                got = json.loads(proc.stdout)
            except Exception as e:
                failures.append((lineno, "json", repr(e), proc.stdout[-200:]))
                continue
            if got != row["truth_state"]:
                failures.append((lineno, "mismatch", {"got": got, "truth": row["truth_state"]}))
                continue
            by_action[row["action"]["type"]] += 1
            max_depts = max(max_depts, len(row["truth_state"].get("departments", [])))
            max_errors = max(max_errors, len(row["truth_state"].get("errors", {})))
    print(json.dumps({"rows_verified": sum(by_action.values()), "failures": len(failures), "by_action": dict(sorted(by_action.items())), "max_departments": max_depts, "max_error_keys": max_errors}, sort_keys=True))
    if failures:
        print(json.dumps(failures[:3], default=str))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
