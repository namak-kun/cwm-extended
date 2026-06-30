from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "uitrans_cascade_py.jsonl"
APP = HERE / "cascade_app.py"
LOGIC = HERE / "cascade_logic.py"

TEXT_IDX = {
    "org_name": 0, "email": 1, "password": 2, "confirm": 3, "plan": 4,
    "country": 5, "region": 6, "state_code": 7, "vat_id": 8, "coupon": 9,
    "new_dept_name": 10,
}
NUMBER_IDX = {"current_step": 0, "age": 1, "start_day": 2, "end_day": 3, "seats": 4, "monthly_budget": 5}
BOOL_IDX = {"security_addon": 0, "accept_terms": 1, "marketing": 2}


def clean(v: Any) -> Any:
    if is_dataclass(v):
        return clean(asdict(v))
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): clean(v[k]) for k in sorted(v)}
    return v


def canon(at: AppTest) -> dict:
    return clean(at.session_state.filtered_state["model"])


def logic_src() -> str:
    src = LOGIC.read_text(encoding="utf-8")
    src = src.replace("from __future__ import annotations\n\n", "")
    return "import json\n" + src


DISPATCH_SRC = logic_src()


def prompt_src(state: dict, action: dict) -> str:
    return DISPATCH_SRC + "\n" + f'''def main():  # << START_OF_TRACE
    state = {repr(state)}
    state = dispatch(state, {repr(action)})
    return state

if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":")))
'''


def run_prompt(src: str) -> dict:
    proc = subprocess.run(["python3", "-c", src], text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def dyn_counts(at: AppTest) -> int:
    return len(canon(at)["departments"])


def apply_action(at: AppTest, action: dict):
    t = action["type"]
    if t == "set_text":
        at.text_input[TEXT_IDX[action["field"]]].set_value(action["value"]).run()
    elif t == "set_number":
        field = action["field"]
        if field == "new_dept_allocation":
            idx = 6 + dyn_counts(at)
        else:
            idx = NUMBER_IDX[field]
        at.number_input[idx].set_value(action["value"]).run()
    elif t == "set_bool":
        at.checkbox[BOOL_IDX[action["field"]]].set_value(action["value"]).run()
    elif t == "add_department":
        n = dyn_counts(at)
        at.text_input[TEXT_IDX["new_dept_name"]].set_value(action["name"])
        at.number_input[6 + n].set_value(action["allocation"])
        at.button[n].click().run()
    elif t == "set_department_allocation":
        at.number_input[6 + action["index"]].set_value(action["value"]).run()
    elif t == "set_department_approved":
        at.checkbox[3 + action["index"]].set_value(action["value"]).run()
    elif t == "remove_department":
        at.button[action["index"]].click().run()
    else:
        raise ValueError(action)


def add_row(rows: list[dict], at: AppTest, action: dict, drops: list):
    before = canon(at)
    apply_action(at, action)
    truth = canon(at)
    src = prompt_src(before, action)
    try:
        got = run_prompt(src)
    except Exception as e:
        drops.append(("exec", action, repr(e)))
        return
    if got != truth:
        drops.append(("mismatch", action, {"got": got, "truth": truth}))
        return
    rows.append({
        "target": "cascade_py",
        "lang": "python",
        "prompt_src": src,
        "entry": "main",
        "action": action,
        "state_before": before,
        "truth_state": truth,
        "source_app": "uidata/cascade_py/cascade_app.py",
    })


def scripted_actions() -> list[dict]:
    a = [
        {"type":"set_text","field":"org_name","value":"AI"},
        {"type":"set_text","field":"org_name","value":"Acme Research"},
        {"type":"set_text","field":"email","value":"admin@example.com"},
        {"type":"set_text","field":"password","value":"adminpass"},
        {"type":"set_text","field":"password","value":"Stronger42"},
        {"type":"set_text","field":"confirm","value":"Stronger42"},
        {"type":"set_number","field":"age","value":17},
        {"type":"set_number","field":"age","value":30},
        {"type":"set_text","field":"state_code","value":"ZZ"},
        {"type":"set_text","field":"state_code","value":"CA"},
        {"type":"set_number","field":"end_day","value":120},
        {"type":"set_number","field":"end_day","value":60},
        {"type":"set_text","field":"coupon","value":"SAVE10"},
        {"type":"set_number","field":"monthly_budget","value":20},
        {"type":"set_number","field":"monthly_budget","value":120},
        {"type":"set_bool","field":"accept_terms","value":True},
        {"type":"set_number","field":"current_step","value":2},
        {"type":"set_text","field":"plan","value":"team"},
        {"type":"set_number","field":"seats","value":1},
        {"type":"set_number","field":"seats","value":8},
        {"type":"set_number","field":"monthly_budget","value":250},
        {"type":"add_department","name":"Ops","allocation":80},
        {"type":"set_department_approved","index":0,"value":True},
        {"type":"set_department_allocation","index":0,"value":150},
        {"type":"add_department","name":"Security","allocation":200},
        {"type":"set_department_approved","index":1,"value":True},
        {"type":"set_text","field":"country","value":"EU"},
        {"type":"set_text","field":"region","value":"us"},
        {"type":"set_text","field":"region","value":"eu"},
        {"type":"set_text","field":"vat_id","value":"EU-123"},
        {"type":"set_number","field":"current_step","value":4},
        {"type":"set_text","field":"coupon","value":"BETA"},
        {"type":"set_bool","field":"marketing","value":True},
        {"type":"set_text","field":"plan","value":"enterprise"},
        {"type":"set_number","field":"age","value":20},
        {"type":"set_number","field":"age","value":34},
        {"type":"set_number","field":"seats","value":12},
        {"type":"set_number","field":"seats","value":25},
        {"type":"set_bool","field":"security_addon","value":True},
        {"type":"set_text","field":"email","value":"owner@corp.io"},
        {"type":"set_text","field":"password","value":"CorpSecure99"},
        {"type":"set_text","field":"confirm","value":"CorpSecure99"},
        {"type":"set_text","field":"coupon","value":"ENT20"},
        {"type":"set_number","field":"monthly_budget","value":500},
        {"type":"set_number","field":"monthly_budget","value":1800},
        {"type":"set_number","field":"end_day","value":500},
        {"type":"set_number","field":"end_day","value":300},
        {"type":"add_department","name":"Legal","allocation":300},
        {"type":"set_department_approved","index":2,"value":True},
        {"type":"remove_department","index":0},
        {"type":"set_number","field":"current_step","value":5},
    ]
    return a


def random_action(rng: random.Random, at: AppTest) -> dict:
    st = canon(at)
    n = len(st["departments"])
    choices = ["text", "number", "bool", "add"]
    if n:
        choices += ["dept_alloc", "dept_ok", "remove"]
    kind = rng.choice(choices)
    if kind == "text":
        vals = {
            "org_name": ["Zed", "Northwind Labs", "Q", "Deep Cascade Inc"],
            "email": ["bad", "admin@test.com", "lead@northwind.io", "ops@enterprise.dev"],
            "password": ["short", "NoDigitsHere", "ValidPass77", "leadSecure88"],
            "confirm": ["short", "ValidPass77", "leadSecure88", st["fields"]["password"]],
            "plan": ["starter", "team", "enterprise", "trial"],
            "country": ["US", "CA", "EU", "GB", "BR"],
            "region": ["us", "na", "eu", "uk", "apac"],
            "state_code": ["CA", "WA", "ZZ", ""],
            "vat_id": ["", "EU-999", "GB-444", "VAT-777"],
            "coupon": ["", "SAVE10", "ENT20", "BETA", "NOPE"],
        }
        field = rng.choice(list(vals))
        return {"type":"set_text", "field":field, "value":rng.choice(vals[field])}
    if kind == "number":
        vals = {
            "current_step": [1,2,3,4,5], "age": [16,18,20,21,45], "start_day": [1,10,90],
            "end_day": [5,45,120,365,500], "seats": [1,2,5,8,20,40,501],
            "monthly_budget": [40,100,250,750,1500,3000],
        }
        field = rng.choice(list(vals))
        return {"type":"set_number", "field":field, "value":rng.choice(vals[field])}
    if kind == "bool":
        field = rng.choice(["security_addon", "accept_terms", "marketing"])
        return {"type":"set_bool", "field":field, "value":not bool(st["fields"][field])}
    if kind == "add":
        return {"type":"add_department", "name":rng.choice(["Ops", "Risk", "QA", "Finance", "Growth"]), "allocation":rng.choice([0,50,100,250,500])}
    if kind == "dept_alloc":
        return {"type":"set_department_allocation", "index":rng.randrange(n), "value":rng.choice([0,75,100,175,400])}
    if kind == "dept_ok":
        i = rng.randrange(n)
        return {"type":"set_department_approved", "index":i, "value":not bool(st["departments"][i]["approved"])}
    return {"type":"remove_department", "index":rng.randrange(n)}


def main():
    rows: list[dict] = []
    drops: list = []
    at = AppTest.from_file(str(APP)).run()
    for action in scripted_actions():
        add_row(rows, at, action, drops)
    rng = random.Random(20260629)
    while len(rows) < 112:
        action = random_action(rng, at)
        add_row(rows, at, action, drops)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    counts = Counter(r["action"]["type"] for r in rows)
    max_depts = max(len(r["truth_state"]["departments"]) for r in rows)
    print(json.dumps({"rows": len(rows), "drops": len(drops), "by_action": dict(sorted(counts.items())), "max_departments": max_depts, "out": str(OUT)}, sort_keys=True))
    if drops:
        print(json.dumps(drops[:5], default=str))


if __name__ == "__main__":
    main()
