from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "uitrans_streamlit.jsonl"


def main():
    kept = 0
    by_app = Counter()
    failures = []
    with DATA.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            row = json.loads(line)
            proc = subprocess.run(["python3", "-c", row["prompt_src"]], text=True, capture_output=True)
            if proc.returncode != 0:
                failures.append((lineno, "exec", proc.stderr[-400:]))
                continue
            try:
                got = json.loads(proc.stdout)
            except Exception as e:
                failures.append((lineno, "json", repr(e)))
                continue
            if got != row["truth_state"]:
                failures.append((lineno, "mismatch", {"got": got, "truth": row["truth_state"]}))
                continue
            kept += 1
            app = row["source_app"].rsplit("/", 1)[-1].replace("_app.py", "")
            by_app[app] += 1
    print(json.dumps({"rows_verified": kept, "failures": len(failures), "by_app": dict(sorted(by_app.items()))}, sort_keys=True))
    if failures:
        print(json.dumps(failures[:3], default=str))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
