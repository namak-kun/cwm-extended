# Vanilla real-app UI-transition dataset

Output: `wm_probe/data/uitrans_vanilla.jsonl` with 80 verified JavaScript transitions.

## Apps used

1. `bradtraversy/vanillawebprojects/form-validator/`
   - Local copy: `uidata/vanilla/apps/form-validator/`
   - Source logic cited in rows: `script.js` functions `showError`, `showSuccess`, `checkEmail`, `checkRequired`, `checkLength`, `checkPasswordsMatch`, and the form `submit` listener.
   - State: field values plus each field's `status` (`""`, `"error"`, `"success"`) and current small-message text.
   - Action: `{type:"input", field, value}`; the harness sets the input value and dispatches the real form submit event once so the app's validator logic runs.

2. `bradtraversy/vanillawebprojects/movie-seat-booking/`
   - Local copy: `uidata/vanilla/apps/movie-seat-booking/`
   - Source logic cited in rows: `script.js` functions `setMovieData`, `updateSelectedCount`, `populateUI`, the movie `change` listener, and the seat `click` listener.
   - State: selected non-occupied seat indices, selected movie index, ticket price, count, total, and the localStorage keys written by the app.
   - Actions: `{type:"clickSeat", index}` and `{type:"selectMovie", index}`.

## Truth and equivalence method

`uidata/vanilla/build_dataset.js` loads each real app HTML and `script.js` in jsdom using `NODE_PATH=/home/t-nagupta/CWM_extended/wm_probe/jsdeps/node_modules`. For every candidate transition it applies `state_before` to the real DOM/localStorage, dispatches the real DOM event, reads a canonical next state, then executes the extracted self-contained `prompt_src` in Node's VM and keeps the row only if the JSON output exactly equals the jsdom truth state.

## Verification evidence

Command run:

```bash
cd /home/t-nagupta/CWM_extended/wm_probe && \
NODE_PATH=/home/t-nagupta/CWM_extended/wm_probe/jsdeps/node_modules \
node uidata/vanilla/verify_dataset.js
```

Result: `verified: 80`, with `form-validator: 40` and `movie-seat-booking: 40`. Final kept/dropped: kept 80, dropped 0 in the final generation. Maximum `prompt_src` length is 2769 characters.
