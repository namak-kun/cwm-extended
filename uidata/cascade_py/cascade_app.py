from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_logic import initial_state, dispatch

if "model" not in st.session_state:
    st.session_state.model = initial_state()


def apply(action):
    st.session_state.model = dispatch(st.session_state.model, action)


def set_text(field):
    apply({"type": "set_text", "field": field, "value": st.session_state["input_" + field]})


def set_number(field):
    apply({"type": "set_number", "field": field, "value": st.session_state["input_" + field]})


def set_bool(field):
    apply({"type": "set_bool", "field": field, "value": st.session_state["input_" + field]})


def _dept_index(dept_id):
    for idx, dept in enumerate(st.session_state.model["departments"]):
        if dept["id"] == dept_id:
            return idx
    return -1


def set_dept_alloc(dept_id):
    idx = _dept_index(dept_id)
    if idx >= 0:
        apply({"type": "set_department_allocation", "index": idx, "value": st.session_state[f"dept_alloc_{dept_id}"]})


def set_dept_approved(dept_id):
    idx = _dept_index(dept_id)
    if idx >= 0:
        apply({"type": "set_department_approved", "index": idx, "value": st.session_state[f"dept_approved_{dept_id}"]})


def add_department_from_inputs():
    apply({
        "type": "add_department",
        "name": st.session_state.input_new_dept_name,
        "allocation": st.session_state.input_new_dept_allocation,
    })


m = st.session_state.model
st.title("Cascading onboarding validator")
st.number_input("Wizard step", min_value=1, max_value=5, key="input_current_step", value=m["input_current_step"], on_change=set_number, args=["current_step"])
for field, label in [
    ("org_name", "Organization"), ("email", "Admin email"), ("password", "Password"),
    ("confirm", "Confirm password"), ("plan", "Plan"), ("country", "Country"),
    ("region", "Data region"), ("state_code", "US state"), ("vat_id", "Tax/VAT id"),
    ("coupon", "Coupon"),
]:
    st.text_input(label, key="input_" + field, value=m["input_" + field], on_change=set_text, args=[field])
for field, label in [
    ("age", "Admin age"), ("start_day", "Start day"), ("end_day", "End day"),
    ("seats", "Seats"), ("monthly_budget", "Monthly budget"),
]:
    st.number_input(label, key="input_" + field, value=m["input_" + field], on_change=set_number, args=[field])
for field, label in [("security_addon", "Security add-on"), ("accept_terms", "Accept terms"), ("marketing", "Marketing opt-in")]:
    st.checkbox(label, key="input_" + field, value=m["input_" + field], on_change=set_bool, args=[field])

st.subheader("Departments")
for i, dept in enumerate(m["departments"]):
    st.write(f"#{dept['id']} {dept['name']}")
    st.number_input(f"Allocation {dept['id']}", key=f"dept_alloc_{dept['id']}", value=dept["allocation"], on_change=set_dept_alloc, args=[dept["id"]])
    st.checkbox(f"Approved {dept['id']}", key=f"dept_approved_{dept['id']}", value=dept["approved"], on_change=set_dept_approved, args=[dept["id"]])
    st.button(f"Remove {dept['id']}", key=f"remove_{i}", on_click=apply, args=[{"type": "remove_department", "index": i}])

st.text_input("New department", key="input_new_dept_name", value=m["input_new_dept_name"], on_change=set_text, args=["new_dept_name"])
st.number_input("New allocation", key="input_new_dept_allocation", value=m["input_new_dept_allocation"], on_change=set_number, args=["new_dept_allocation"])
st.button("Add department", on_click=add_department_from_inputs)

m = dispatch(st.session_state.model, {"type": "noop"})
st.session_state.model = m
st.json({"errors": m["errors"], "price": m["price"], "can_submit": m["can_submit"], "blocked_steps": m["blocked_steps"]})
st.button("Submit", disabled=not m["can_submit"])
