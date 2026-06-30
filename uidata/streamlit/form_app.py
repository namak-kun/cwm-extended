import streamlit as st

if "fields" not in st.session_state:
    st.session_state.fields = {"email": "", "password": "", "confirm": ""}
if "newsletter" not in st.session_state:
    st.session_state.newsletter = False
if "accepted_terms" not in st.session_state:
    st.session_state.accepted_terms = False

def validate():
    f = st.session_state.fields
    errors = {}
    if "@" not in f["email"] or "." not in f["email"]:
        errors["email"] = "invalid"
    if len(f["password"]) < 8:
        errors["password"] = "too_short"
    if f["confirm"] != f["password"]:
        errors["confirm"] = "mismatch"
    if not st.session_state.accepted_terms:
        errors["accepted_terms"] = "required"
    st.session_state.errors = errors
    st.session_state.can_submit = len(errors) == 0

def update_field(name):
    st.session_state.fields[name] = st.session_state[f"input_{name}"]
    validate()

def update_terms():
    st.session_state.accepted_terms = st.session_state.input_terms
    validate()

def update_newsletter():
    st.session_state.newsletter = st.session_state.input_newsletter
    validate()

st.text_input("Email", key="input_email", value=st.session_state.fields["email"], on_change=update_field, args=["email"])
st.text_input("Password", key="input_password", value=st.session_state.fields["password"], on_change=update_field, args=["password"])
st.text_input("Confirm", key="input_confirm", value=st.session_state.fields["confirm"], on_change=update_field, args=["confirm"])
st.checkbox("Accept terms", key="input_terms", value=st.session_state.accepted_terms, on_change=update_terms)
st.checkbox("Newsletter", key="input_newsletter", value=st.session_state.newsletter, on_change=update_newsletter)
validate()
st.button("Submit", disabled=not st.session_state.can_submit)
