# Cascade JS UI-transition dataset

Output: `wm_probe/data/uitrans_cascade_js.jsonl` with 120 verified JavaScript transitions.

## App and rules used

Local harness app: `uidata/cascade_js/apps/cascade-form/`.

The app extends the same real vanilla-JS validation pattern used by `bradtraversy/vanillawebprojects/form-validator`: `showError`/`showSuccess`-style field state, required checks, email regex, length/format checks, password confirmation, and submit gating. It composes those real-world form-validator rules with common dependent-field rules used by production forms:

- Password strength + confirm-password dependency; editing either field can change both password validity and submit gating.
- Minor/guardian cascade: age `<18` shows/requires guardian email; age `>=18` hides/disables and clears it.
- Personal/business cascade: business accounts show/require company name and VAT ID; switching back to personal hides/disables and clears them.
- Country -> region -> city cascade: country changes recompute valid region options, may clear stale region/city, recomputes city options and errors.
- Terms, newsletter, account/profile/location wizard step status, error summary, error count, active step, and submit button are recomputed on every action.

State includes 14 fields, per-control `status/message/required/visible/disabled`, and derived `canSubmit`, `summary`, `errorCount`, `activeStep`, `stepStatus`, `passwordScore`, `regionOptions`, and `cityOptions`. One event often updates 5-20 state leaves.

## Truth and equivalence method

`build_dataset.js` loads `apps/cascade-form/index.html` and the real `script.js` into jsdom using `NODE_PATH=/home/t-nagupta/CWM_extended/wm_probe/jsdeps/node_modules`. For each candidate transition it:

1. Applies `state_before` to the real DOM.
2. Dispatches the real `input`/`change` event once.
3. Reads a canonical DOM-derived next state.
4. Executes the self-contained `prompt_src` dispatch program in Node during build-time filtering.
5. Keeps the row only when the prompt output exactly equals the jsdom truth state.

Final generation considered 135 candidates, kept 120 verified rows, and dropped 15 prompt/jsdom mismatches. The kept set has 80 `input`, 28 `select`, and 12 `toggle` transitions. Maximum `prompt_src` length is 10,995 characters (under the contract's ~6k-token budget).

## Verification evidence

Command run (verifier executes every `prompt_src` with `node -e` and compares to `truth_state`):

```bash
cd /home/t-nagupta/CWM_extended/wm_probe && \
NODE_PATH=/home/t-nagupta/CWM_extended/wm_probe/jsdeps/node_modules \
node uidata/cascade_js/verify_dataset.js
```

Result: `verified: 120`; field coverage includes username, email, password, password2, age, guardianEmail, accountType, companyName, vatId, country, region, city, newsletter, and terms. Final output file has exactly 120 JSONL rows.
