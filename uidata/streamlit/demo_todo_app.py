# Adapted from streamlit/demo-todo streamlit_app.py (Apache 2.0), with
# deterministic UUIDs for data harvesting.
import streamlit as st
from dataclasses import dataclass, field
import uuid

st.set_page_config(page_title="To-do list", page_icon=":memo:")
state = st.session_state

if "_uid_counter" not in state:
    state._uid_counter = 0

def next_uid():
    state._uid_counter += 1
    return uuid.UUID(int=state._uid_counter)

@dataclass
class Todo:
    text: str
    is_done = False
    uid: uuid.UUID = field(default_factory=next_uid)

if "todos" not in state:
    state.todos = [Todo(text="Buy milk"), Todo(text="Wash dishes"), Todo(text="Write a novel")]

def remove_todo(i):
    state.todos.pop(i)

def add_todo():
    state.todos.append(Todo(text=state.new_item_text))
    state.new_item_text = ""

def check_todo(i, new_value):
    state.todos[i].is_done = new_value

def delete_all_checked():
    state.todos = [t for t in state.todos if not t.is_done]

st.title("To-do list")
with st.form(key="new_item_form", border=False):
    st.text_input("New item", key="new_item_text")
    st.form_submit_button("Add", on_click=add_todo)

if state.todos:
    for i, todo in enumerate(state.todos):
        st.checkbox(todo.text, value=todo.is_done, on_change=check_todo, args=[i, not todo.is_done], key=f"todo-chk-{todo.uid}")
        st.button("Delete", on_click=remove_todo, args=[i], key=f"delete_{i}")
    st.button("Delete all checked", on_click=delete_all_checked)
else:
    st.info("No to-do items.")
