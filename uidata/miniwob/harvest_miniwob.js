#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { JSDOM, VirtualConsole } = require('jsdom');

const ROOT = path.resolve(__dirname, '..', '..');
const HTML_DIR = path.join(__dirname, 'html', 'miniwob');
const OUT = path.join(ROOT, 'data', 'uitrans_miniwob.jsonl');
const NOTES = path.join(__dirname, 'NOTES.md');

const TASKS = [
  { name: 'click-checkboxes', kind: 'checkbox', seeds: range(1, 5) },
  { name: 'click-checkboxes-soft', kind: 'checkbox', seeds: range(11, 13) },
  { name: 'enter-text', kind: 'enter-text', seeds: range(21, 30) },
  { name: 'click-tab', kind: 'click-tab', seeds: range(41, 48) },
  { name: 'click-dialog', kind: 'click-dialog', seeds: range(61, 66) },
  { name: 'click-button', kind: 'click-button', seeds: range(81, 87) },
  { name: 'click-test', kind: 'click-test', seeds: range(101, 105) },
  { name: 'focus-text', kind: 'focus-text', seeds: range(121, 124) },
];

function range(a, b) { const xs = []; for (let i = a; i <= b; i++) xs.push(i); return xs; }
function normText(s) { return (s || '').replace(/\s+/g, ' ').trim(); }
function cssEscape(s) { return String(s).replace(/([ #;?%&,.+*~':"!^$[\]()=>|/@])/g, '\\$1'); }
function stable(obj) {
  if (Array.isArray(obj)) return obj.map(stable);
  if (obj && typeof obj === 'object') {
    const out = {};
    for (const k of Object.keys(obj).sort()) out[k] = stable(obj[k]);
    return out;
  }
  return obj;
}
function canonicalNode(el) {
  const o = { tag: el.tagName.toLowerCase() };
  if (el.id) o.id = el.id;
  const cls = normText(el.className);
  if (cls) o.class = cls.split(/\s+/).sort().join(' ');
  const directText = Array.from(el.childNodes)
    .filter(n => n.nodeType === 3)
    .map(n => n.nodeValue)
    .join(' ');
  const text = normText(directText);
  if (text) o.text = text;
  if ('value' in el && ['input', 'textarea', 'select', 'button'].includes(o.tag)) o.value = el.value || '';
  if ('checked' in el && (el.type === 'checkbox' || el.type === 'radio')) o.checked = !!el.checked;
  const attrs = {};
  for (const a of Array.from(el.attributes || []).sort((a, b) => a.name.localeCompare(b.name))) {
    if (['id', 'class', 'value', 'checked'].includes(a.name)) continue;
    attrs[a.name] = a.value;
  }
  if (Object.keys(attrs).length) o.attrs = attrs;
  const children = Array.from(el.children).map(canonicalNode);
  if (children.length) o.children = children;
  return stable(o);
}
function snapshot(win) {
  const area = win.document.querySelector('#area');
  if (!area) throw new Error('missing #area');
  return canonicalNode(area);
}
function queryText(win) { return normText(win.document.querySelector('#query')?.textContent || ''); }
function selectorFor(el) {
  if (!el) return null;
  if (el.id) return `#${cssEscape(el.id)}`;
  const doc = el.ownerDocument;
  const tag = el.tagName.toLowerCase();
  const sameTag = Array.from(doc.querySelectorAll(tag));
  const ix = sameTag.indexOf(el) + 1;
  return `${tag}:nth-of-type(${ix})`;
}
async function loadTask(taskName, seed) {
  const file = path.join(HTML_DIR, `${taskName}.html`);
  const vc = new VirtualConsole();
  vc.on('jsdomError', () => {});
  vc.on('error', () => {});
  const dom = await JSDOM.fromFile(file, {
    url: pathToFileURL(file).href,
    resources: 'usable',
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    virtualConsole: vc,
  });
  await new Promise((resolve) => {
    if (dom.window.document.readyState === 'complete') resolve();
    else dom.window.addEventListener('load', resolve, { once: true });
  });
  if (!dom.window.core || !dom.window.genProblem || !dom.window.Math.seedrandom) {
    throw new Error('MiniWoB runtime did not initialize');
  }
  // jsdom 29 has no built-in canvas implementation. MiniWoB's reward/click
  // visualizer is outside #area and irrelevant to the harvested UI state.
  dom.window.core.prepareCanvas = () => false;
  dom.window.core.canvasClear = () => {};
  dom.window.core.canvasDrawClick = () => {};
  dom.window.core.canvasDrawElementClick = () => {};
  dom.window.Math.seedrandom(`miniwob-${taskName}-${seed}`);
  if (!dom.window.core.cover_div) dom.window.core.startEpisode();
  dom.window.core.startEpisodeReal();
  await tick(dom.window);
  return dom;
}
function tick(win) { return new Promise(r => win.setTimeout(r, 0)); }
function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('mousedown', { bubbles: true, cancelable: true, view: win }));
  el.dispatchEvent(new win.MouseEvent('mouseup', { bubbles: true, cancelable: true, view: win }));
  el.dispatchEvent(new win.MouseEvent('click', { bubbles: true, cancelable: true, view: win }));
}
async function applyAction(win, action) {
  if (action.type === 'click' || action.type === 'focus') {
    const el = win.document.querySelector(action.target);
    if (!el) throw new Error(`missing target ${action.target}`);
    if (action.type === 'focus') el.focus(); else click(win, el);
  } else if (action.type === 'type') {
    const el = win.document.querySelector(action.target);
    if (!el) throw new Error(`missing target ${action.target}`);
    el.focus();
    el.value = action.text;
    el.dispatchEvent(new win.InputEvent('input', { bubbles: true, inputType: 'insertText', data: action.text }));
    el.dispatchEvent(new win.Event('change', { bubbles: true }));
  } else {
    throw new Error(`unknown action ${action.type}`);
  }
  await tick(win);
}
function candidateActions(win, kind) {
  const doc = win.document;
  if (kind === 'checkbox') {
    return Array.from(doc.querySelectorAll('#area input[type=checkbox]')).map(el => ({ type: 'click', target: selectorFor(el) }));
  }
  if (kind === 'enter-text') {
    const m = queryText(win).match(/Enter "([^"]+)"/);
    const text = m ? m[1] : 'Alice';
    return [{ type: 'type', target: '#tt', text }];
  }
  if (kind === 'click-tab') {
    const q = queryText(win);
    const m = q.match(/Tab #(\d)/);
    const target = m ? `#area a[href="#tabs-${m[1]}"]` : '#area ul a';
    return [{ type: 'click', target }];
  }
  if (kind === 'click-dialog') return [{ type: 'click', target: 'button.ui-dialog-titlebar-close' }];
  if (kind === 'click-button') {
    const m = queryText(win).match(/"([^"]+)"/);
    const buttons = Array.from(doc.querySelectorAll('#area button'));
    const btn = buttons.find(b => normText(b.textContent) === (m && m[1])) || buttons[0];
    return [{ type: 'click', target: selectorFor(btn) }];
  }
  if (kind === 'click-test') return [{ type: 'click', target: '#subbtn' }];
  if (kind === 'focus-text') return [{ type: 'focus', target: '#tt' }];
  return [];
}
function sourceNote(task) {
  return `Farama-Foundation/miniwob-plusplus:miniwob/html/miniwob/${task.name}.html via jsdom; raw fallback DOM transition`;
}
async function capture(task, seed, actionIndex) {
  const dom = await loadTask(task.name, seed);
  const win = dom.window;
  const before = snapshot(win);
  const actions = candidateActions(win, task.kind);
  if (actionIndex >= actions.length) { dom.window.close(); return null; }
  const action = actions[actionIndex];
  await applyAction(win, action);
  const truth = snapshot(win);
  const row = stable({
    target: 'miniwob',
    lang: 'js',
    prompt_src: '',
    entry: 'main',
    action: stable({ task: task.name, seed, ...action }),
    state_before: before,
    truth_state: truth,
    source_app: sourceNote(task),
  });
  dom.window.close();
  return row;
}
async function deterministicBefore(task, seed) {
  const d1 = await loadTask(task.name, seed);
  const s1 = JSON.stringify(snapshot(d1.window));
  d1.window.close();
  const d2 = await loadTask(task.name, seed);
  const s2 = JSON.stringify(snapshot(d2.window));
  d2.window.close();
  return s1 === s2;
}
async function replayMatches(task, seed, actionIndex, expectedTruth) {
  const dom = await loadTask(task.name, seed);
  const actions = candidateActions(dom.window, task.kind);
  await applyAction(dom.window, actions[actionIndex]);
  const got = JSON.stringify(snapshot(dom.window));
  dom.window.close();
  return got === JSON.stringify(expectedTruth);
}
async function main() {
  const rows = [];
  const notes = [];
  notes.push('# MiniWoB++ jsdom harvest notes\n');
  notes.push('- Source: Farama-Foundation/miniwob-plusplus `miniwob/html` (MIT). Assets fetched locally under `uidata/miniwob/html/`.');
  notes.push('- Runtime: Node + jsdom 29 with local `core.js`, D3 v3, jQuery/jQuery UI where needed; no Chromium/Selenium installed.');
  notes.push('- Seeding: after page load and before `core.startEpisodeReal()`, harness calls `Math.seedrandom("miniwob-<task>-<seed>")`.');
  notes.push('- Canonical state: JSON snapshot of `#area` only (tag/id/class/direct text/value/checked/sorted attrs/children).');
  notes.push('- Fallback: `prompt_src` is intentionally empty. MiniWoB handlers are closures registered through `core.startEpisodeReal()`/D3/jQuery UI; inlining clean self-contained dispatch code for these real browser-plugin handlers was not feasible without reimplementing the runtime. Rows are real jsdom-captured raw transitions and replay-verified.');
  notes.push('- Skipped known nondeterministic tasks: click-pie, click-pie-nodelay, terminal, stock-market.\n');
  for (const task of TASKS) {
    let detOk = 0, made = 0, failed = 0;
    for (const seed of task.seeds) {
      try {
        const deterministic = await deterministicBefore(task, seed);
        if (!deterministic) { failed++; notes.push(`- ${task.name} seed ${seed}: skipped; initial DOM was not deterministic.`); continue; }
        detOk++;
        const probe = await loadTask(task.name, seed);
        const nActions = candidateActions(probe.window, task.kind).length;
        probe.window.close();
        for (let i = 0; i < nActions; i++) {
          const row = await capture(task, seed, i);
          if (!row) continue;
          const ok = await replayMatches(task, seed, i, row.truth_state);
          if (!ok) { failed++; notes.push(`- ${task.name} seed ${seed} action ${i}: replay mismatch, dropped.`); continue; }
          rows.push(row); made++;
          if (rows.length >= 100) break;
        }
      } catch (e) {
        failed++;
        notes.push(`- ${task.name} seed ${seed}: jsdom failed: ${e.message}`);
      }
      if (rows.length >= 100) break;
    }
    notes.push(`- ${task.name}: ${made} transitions captured; ${detOk}/${task.seeds.length} deterministic seeds checked; failures ${failed}.`);
    if (rows.length >= 100) break;
  }
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  notes.push(`\nTotal rows written: ${rows.length} to \`${path.relative(ROOT, OUT)}\`.`);
  notes.push('Validation: every row was regenerated from the same task+seed and action in a fresh jsdom instance; rows with mismatched `truth_state` were dropped.');
  fs.writeFileSync(NOTES, notes.join('\n') + '\n');
  console.log(`wrote ${rows.length} rows to ${OUT}`);
}

main().catch(e => { console.error(e.stack || e); process.exit(1); });
