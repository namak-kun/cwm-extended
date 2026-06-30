# TodoMVC UI-transition dataset

Source: tastejs/todomvc `examples/javascript-es5` (MIT), downloaded under `source/`. The prompt dispatch is adapted from `src/model.js` (create/read/update/remove/count), `src/store.js` (in-memory todos, sequential IDs, save/remove/drop), and `src/controller.js` (addItem, editItemSave, removeItem, removeCompletedItems, toggleComplete, toggleAll, setView/_updateFilterState).

Ground truth: `harvest_todomvc.js` loads the real dist HTML and JS files in jsdom using the repository jsdom install, constructs the real Store/Model/View/Controller, seeds todos through controller methods, applies each action through real controller methods, then reads the model back with `model.read`.

Equivalence check: for every candidate transition, the extracted self-contained dispatch was run and compared with the jsdom real-app state before writing a row. Checked 83 candidate transitions; extracted mismatches dropped: 0.

Self-consistency: every row's `prompt_src` was executed with `node -e` and parsed JSON was compared to `truth_state`. Kept 83; dropped for prompt mismatch: 0; total dropped: 0.

Coverage: {"add":12,"toggleAll":12,"clearCompleted":12,"setFilter":12,"toggle":11,"delete":11,"edit":13}. State sizes include empty, 1-8 item, and larger 10/12/15/20 item lists; some states contain ID holes from prior deletes. Caveat: prompt state intentionally models TodoMVC's observable todo model plus filter, not DOM rendering details or the private Store ID counter; seeding through max id keeps add transitions equivalent.
