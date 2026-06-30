from __future__ import annotations

import copy

TEXT_FIELDS = [
    "org_name", "email", "password", "confirm", "plan", "country",
    "region", "state_code", "vat_id", "coupon",
]
NUMBER_FIELDS = ["age", "start_day", "end_day", "seats", "monthly_budget"]
BOOL_FIELDS = ["security_addon", "accept_terms", "marketing"]
STEPS = ["account", "plan", "schedule", "budget", "legal"]


def initial_state():
    fields = {
        "org_name": "", "email": "", "password": "", "confirm": "",
        "plan": "starter", "country": "US", "region": "us", "state_code": "",
        "vat_id": "", "coupon": "", "age": 0, "start_day": 1, "end_day": 30,
        "seats": 1, "monthly_budget": 50, "security_addon": False,
        "accept_terms": False, "marketing": False,
    }
    state = {
        "fields": fields,
        "current_step": 1,
        "departments": [],
        "next_dept_id": 1,
        "new_dept_name": "Ops",
        "new_dept_allocation": 0,
        "errors": {},
        "warnings": {},
        "required_fields": [],
        "visible_fields": [],
        "allowed_regions": [],
        "completion_by_step": {},
        "blocked_steps": [],
        "next_step_enabled": False,
        "price": {},
        "error_summary": [],
        "can_submit": False,
    }
    for name, value in fields.items():
        state["input_" + name] = value
    state["input_current_step"] = 1
    state["input_new_dept_name"] = state["new_dept_name"]
    state["input_new_dept_allocation"] = state["new_dept_allocation"]
    return recompute(state)


def _norm_text(x):
    return str(x).strip()


def _email_domain(email):
    return email.rsplit("@", 1)[1].lower() if "@" in email else ""


def _dept_key(dept):
    return str(dept["id"])


def recompute(state):
    f = state["fields"]
    plan = _norm_text(f.get("plan", "")).lower()
    country = _norm_text(f.get("country", "")).upper()
    region = _norm_text(f.get("region", "")).lower()
    email = _norm_text(f.get("email", "")).lower()
    domain = _email_domain(email)
    errors = {}
    warnings = {}

    allowed_regions = {
        "US": ["us", "na"],
        "CA": ["na"],
        "EU": ["eu"],
        "GB": ["eu", "uk"],
    }.get(country, ["us", "eu", "na"])

    required = ["org_name", "email", "password", "confirm", "plan", "country", "region", "age"]
    visible = list(required)
    if country == "US":
        required.append("state_code"); visible.append("state_code")
    if country in ("EU", "GB") or plan == "enterprise":
        required.append("vat_id"); visible.append("vat_id")
    if plan in ("team", "enterprise"):
        required += ["seats", "monthly_budget"]; visible += ["seats", "monthly_budget"]
    if plan == "enterprise":
        required.append("security_addon")
    visible += ["start_day", "end_day", "coupon", "accept_terms", "marketing"]

    if len(_norm_text(f["org_name"])) < 3:
        errors["org_name"] = "min_3_chars"
    if "@" not in email or "." not in domain:
        errors["email"] = "invalid_email"
    elif domain in ("mailinator.com", "test.com", "example.com") and plan == "enterprise":
        errors["email"] = "enterprise_requires_work_domain"
    pwd = str(f["password"])
    if len(pwd) < 10:
        errors["password"] = "min_10_chars"
    elif not any(ch.isdigit() for ch in pwd):
        errors["password"] = "needs_digit"
    elif email and email.split("@", 1)[0] and email.split("@", 1)[0] in pwd.lower():
        errors["password"] = "contains_email_name"
    if str(f["confirm"]) != pwd:
        errors["confirm"] = "mismatch"
    if plan not in ("starter", "team", "enterprise"):
        errors["plan"] = "unknown_plan"
    if country not in ("US", "CA", "EU", "GB"):
        errors["country"] = "unsupported_country"
    if region not in allowed_regions:
        errors["region"] = "not_allowed_for_country"
    if country == "US" and _norm_text(f["state_code"]).upper() not in ("CA", "NY", "WA", "TX", "MA"):
        errors["state_code"] = "required_us_state"
    if (country in ("EU", "GB") or plan == "enterprise") and not _norm_text(f["vat_id"]).upper().startswith(("EU", "GB", "VAT")):
        errors["vat_id"] = "required_tax_id"
    if f["age"] < 18 or (plan == "enterprise" and f["age"] < 21):
        errors["age"] = "too_young"
    if f["start_day"] < 1:
        errors["start_day"] = "must_be_positive"
    duration = f["end_day"] - f["start_day"]
    if duration <= 0:
        errors["end_day"] = "must_be_after_start"
    elif duration > 365 or (plan == "starter" and duration > 90):
        errors["end_day"] = "duration_too_long"
    seats = int(f["seats"])
    if seats < 1:
        errors["seats"] = "at_least_one"
    elif plan == "starter" and seats > 5:
        errors["seats"] = "starter_max_5"
    elif plan == "team" and not (2 <= seats <= 50):
        errors["seats"] = "team_requires_2_to_50"
    elif plan == "enterprise" and not (20 <= seats <= 500):
        errors["seats"] = "enterprise_requires_20_to_500"
    if plan == "enterprise" and not bool(f["security_addon"]):
        errors["security_addon"] = "required_for_enterprise"

    dept_errors = {}
    approved_count = 0
    dept_total = 0
    for dept in state["departments"]:
        dept["name"] = _norm_text(dept.get("name", "")) or "Department"
        dept["allocation"] = int(dept.get("allocation", 0))
        dept["approved"] = bool(dept.get("approved", False))
        dept_total += dept["allocation"]
        if dept["approved"]:
            approved_count += 1
        if dept["allocation"] < 0:
            dept_errors[_dept_key(dept)] = "negative_allocation"
        elif dept["approved"] and dept["allocation"] < 100:
            dept_errors[_dept_key(dept)] = "approved_min_100"
    if dept_errors:
        errors["departments"] = dept_errors
    if plan == "enterprise" and approved_count < 2:
        errors["department_approvals"] = "enterprise_needs_two_approved"
    if plan == "starter" and len(state["departments"]) > 2:
        errors["departments_count"] = "starter_max_two_departments"

    base_price = {"starter": 29, "team": 99, "enterprise": 399}.get(plan, 0)
    seat_price = {"starter": 8, "team": 18, "enterprise": 35}.get(plan, 0)
    security_price = 199 if plan == "enterprise" and f["security_addon"] else (49 if f["security_addon"] else 0)
    subtotal = base_price + seats * seat_price + security_price + dept_total
    coupon = _norm_text(f["coupon"]).upper()
    discount = 0
    if coupon:
        if coupon == "SAVE10" and plan != "enterprise" and "email" not in errors:
            discount = min(50, subtotal // 10)
        elif coupon == "ENT20" and plan == "enterprise" and f["security_addon"] and seats >= 20:
            discount = min(500, subtotal // 5)
        elif coupon == "BETA" and f["marketing"] and duration <= 60:
            discount = 25
        else:
            errors["coupon"] = "not_applicable"
    total_due = max(0, subtotal - discount)
    budget_remaining = int(f["monthly_budget"]) - total_due
    if int(f["monthly_budget"]) < total_due:
        errors["monthly_budget"] = "below_total_due"
    elif budget_remaining < 50 and plan != "starter":
        warnings["monthly_budget"] = "low_margin"

    if not f["accept_terms"]:
        errors["accept_terms"] = "required"

    account_keys = {"org_name", "email", "password", "confirm", "age"}
    plan_keys = {"plan", "country", "region", "state_code", "vat_id", "seats", "security_addon", "department_approvals", "departments_count"}
    schedule_keys = {"start_day", "end_day"}
    budget_keys = {"monthly_budget", "coupon", "departments"}
    legal_keys = {"accept_terms"}
    step_sets = [account_keys, plan_keys, schedule_keys, budget_keys, legal_keys]
    completion = {}
    blocked = []
    for i, (name, keys) in enumerate(zip(STEPS, step_sets), 1):
        ok = not any(k in errors for k in keys)
        completion[name] = ok
        if not ok:
            blocked.append(i)
    current = max(1, min(5, int(state.get("current_step", 1))))
    state["current_step"] = current
    state["input_current_step"] = current
    state["errors"] = errors
    state["warnings"] = warnings
    state["required_fields"] = sorted(set(required))
    state["visible_fields"] = sorted(set(visible))
    state["allowed_regions"] = allowed_regions
    state["completion_by_step"] = completion
    state["blocked_steps"] = blocked
    state["next_step_enabled"] = completion[STEPS[current - 1]] if current <= len(STEPS) else False
    state["price"] = {
        "base_price": base_price,
        "seat_price": seat_price,
        "security_price": security_price,
        "department_total": dept_total,
        "subtotal": subtotal,
        "discount": discount,
        "total_due": total_due,
        "budget_remaining": budget_remaining,
        "approved_departments": approved_count,
    }
    state["error_summary"] = [f"{k}:{errors[k]}" for k in sorted(errors)]
    state["can_submit"] = (len(errors) == 0 and all(completion.values()))
    return state


def dispatch(state, action):
    state = copy.deepcopy(state)
    t = action["type"]
    if t == "set_text":
        name = action["field"]
        value = str(action["value"])
        if name in TEXT_FIELDS:
            state["input_" + name] = value
            state["fields"][name] = value
        elif name == "new_dept_name":
            state["input_new_dept_name"] = value
            state["new_dept_name"] = value
    elif t == "set_number":
        name = action["field"]
        value = int(action["value"])
        if name in NUMBER_FIELDS:
            state["input_" + name] = value
            state["fields"][name] = value
        elif name == "current_step":
            state["input_current_step"] = value
            state["current_step"] = value
        elif name == "new_dept_allocation":
            state["input_new_dept_allocation"] = value
            state["new_dept_allocation"] = value
    elif t == "set_bool":
        name = action["field"]
        value = bool(action["value"])
        if name in BOOL_FIELDS:
            state["input_" + name] = value
            state["fields"][name] = value
    elif t == "add_department":
        name = _norm_text(action.get("name", state.get("new_dept_name", "Department"))) or "Department"
        allocation = int(action.get("allocation", state.get("new_dept_allocation", 0)))
        state["input_new_dept_name"] = name
        state["input_new_dept_allocation"] = allocation
        state["departments"].append({"id": state["next_dept_id"], "name": name, "allocation": allocation, "approved": False})
        state["next_dept_id"] += 1
        state["new_dept_name"] = ""
        state["new_dept_allocation"] = 0
        state["input_new_dept_name"] = ""
        state["input_new_dept_allocation"] = 0
    elif t == "set_department_allocation":
        idx = int(action["index"])
        if 0 <= idx < len(state["departments"]):
            state["departments"][idx]["allocation"] = int(action["value"])
    elif t == "set_department_approved":
        idx = int(action["index"])
        if 0 <= idx < len(state["departments"]):
            state["departments"][idx]["approved"] = bool(action["value"])
    elif t == "remove_department":
        idx = int(action["index"])
        if 0 <= idx < len(state["departments"]):
            state["departments"].pop(idx)
    return recompute(state)
