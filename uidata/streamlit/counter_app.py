import streamlit as st

if "count" not in st.session_state:
    st.session_state.count = 0
if "step" not in st.session_state:
    st.session_state.step = 1
if "label" not in st.session_state:
    st.session_state.label = "counter"

def recompute():
    st.session_state.is_zero = (st.session_state.count == 0)

def increment():
    st.session_state.count += st.session_state.step
    recompute()

def decrement():
    st.session_state.count -= st.session_state.step
    recompute()

def reset():
    st.session_state.count = 0
    recompute()

def set_step():
    st.session_state.step = st.session_state.step_input
    recompute()

def set_label():
    st.session_state.label = st.session_state.label_input.strip()

st.number_input("Step", min_value=1, max_value=5, key="step_input", value=st.session_state.step, on_change=set_step)
st.text_input("Label", key="label_input", value=st.session_state.label, on_change=set_label)
st.button("Increment", on_click=increment)
st.button("Decrement", on_click=decrement)
st.button("Reset", on_click=reset)
recompute()
st.write(st.session_state.count)
