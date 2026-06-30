# MiniWoB++ jsdom harvest notes

- Source: Farama-Foundation/miniwob-plusplus `miniwob/html` (MIT). Assets fetched locally under `uidata/miniwob/html/`.
- Runtime: Node + jsdom 29 with local `core.js`, D3 v3, jQuery/jQuery UI where needed; no Chromium/Selenium installed.
- Seeding: after page load and before `core.startEpisodeReal()`, harness calls `Math.seedrandom("miniwob-<task>-<seed>")`.
- Canonical state: JSON snapshot of `#area` only (tag/id/class/direct text/value/checked/sorted attrs/children).
- Fallback: `prompt_src` is intentionally empty. MiniWoB handlers are closures registered through `core.startEpisodeReal()`/D3/jQuery UI; inlining clean self-contained dispatch code for these real browser-plugin handlers was not feasible without reimplementing the runtime. Rows are real jsdom-captured raw transitions and replay-verified.
- Skipped known nondeterministic tasks: click-pie, click-pie-nodelay, terminal, stock-market.

- click-checkboxes: 27 transitions captured; 5/5 deterministic seeds checked; failures 0.
- click-checkboxes-soft: 17 transitions captured; 3/3 deterministic seeds checked; failures 0.
- enter-text: 10 transitions captured; 10/10 deterministic seeds checked; failures 0.
- click-tab: 8 transitions captured; 8/8 deterministic seeds checked; failures 0.
- click-dialog: 6 transitions captured; 6/6 deterministic seeds checked; failures 0.
- click-button: 7 transitions captured; 7/7 deterministic seeds checked; failures 0.
- click-test: 5 transitions captured; 5/5 deterministic seeds checked; failures 0.
- focus-text: 4 transitions captured; 4/4 deterministic seeds checked; failures 0.

Total rows written: 84 to `data/uitrans_miniwob.jsonl`.
Validation: every row was regenerated from the same task+seed and action in a fresh jsdom instance; rows with mismatched `truth_state` were dropped.
