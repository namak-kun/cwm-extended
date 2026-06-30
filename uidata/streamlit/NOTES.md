# Streamlit UI-transition dataset notes

- Streamlit version: 1.58.0, installed only in `wm_probe/.venv_vllm` via `uv pip install --python ... streamlit` because that venv has no `pip` executable/module.
- Apps used: `demo_todo_app.py` (adapted from official `streamlit/demo-todo` with deterministic UUIDs), `counter_app.py`, and `form_app.py`; all are under 150 lines and mutate `st.session_state` through widget callbacks.
- Oracle method: `streamlit.testing.v1.AppTest.from_file(app).run()` starts the app headlessly; widget actions use `button.click().run()`, `text_input.set_value(...).run()`, `checkbox.set_value(...).run()`, and `number_input.set_value(...).run()`; truth is read from `at.session_state.filtered_state` and canonicalized to app state.
- Prompt method: every JSONL row contains a self-contained Python `dispatch(state, action)->state` plus `main()  # << START_OF_TRACE`; dispatch is the extracted callback/session-state update logic from the real Streamlit app.
- Equivalence evidence: `build_streamlit_dataset.py` executes each prompt immediately and keeps only rows where prompt output equals the AppTest truth; `verify_streamlit_dataset.py` independently re-executes all rows with `python3`.
- Kept/dropped: 56 kept, 0 dropped/mismatched in final verification (`counter`: 18, `form`: 18, `todo`: 20). No blocker/fallback rows were needed.
- Coverage: button clicks, number/text inputs, checkboxes, form submit, todo add/toggle/delete/delete-all; todo includes larger states up to 11 items before reductions.
