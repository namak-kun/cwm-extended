const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const { JSDOM } = require('jsdom');

const ROOT = path.resolve(__dirname, '../..');
const APP_DIR = path.join(__dirname, 'source', 'dist');
const OUT = path.join(ROOT, 'data', 'uitrans_todomvc.jsonl');
const NOTES = path.join(__dirname, 'NOTES.md');
const TARGET = 'todomvc';

const DISPATCH_SRC = `// TodoMVC ES5 model/controller transition logic, adapted from source/src/model.js,
// source/src/store.js, and source/src/controller.js.
function cloneTodos(todos) {
  var out = [];
  for (var i = 0; i < todos.length; i++) {
    out.push({completed: !!todos[i].completed, id: todos[i].id, title: String(todos[i].title)});
  }
  return out;
}
function nextId(todos) {
  var max = 0;
  for (var i = 0; i < todos.length; i++) if (todos[i].id > max) max = todos[i].id;
  return max + 1;
}
function dispatch(state, action) {
  var next = {filter: state.filter || "all", todos: cloneTodos(state.todos || [])};
  if (action.type === "add") {
    var title = String(action.title || "").trim();
    if (title !== "") next.todos.push({completed: false, id: nextId(next.todos), title: title});
  } else if (action.type === "toggle") {
    for (var i = 0; i < next.todos.length; i++) {
      if (next.todos[i].id === action.id) next.todos[i].completed = !next.todos[i].completed;
    }
  } else if (action.type === "delete") {
    next.todos = next.todos.filter(function (todo) { return todo.id !== action.id; });
  } else if (action.type === "edit") {
    var edited = String(action.title || "").trim();
    if (edited !== "") {
      for (var j = 0; j < next.todos.length; j++) if (next.todos[j].id === action.id) next.todos[j].title = edited;
    } else {
      next.todos = next.todos.filter(function (todo) { return todo.id !== action.id; });
    }
  } else if (action.type === "toggleAll") {
    for (var k = 0; k < next.todos.length; k++) next.todos[k].completed = !!action.completed;
  } else if (action.type === "clearCompleted") {
    next.todos = next.todos.filter(function (todo) { return !todo.completed; });
  } else if (action.type === "setFilter") {
    if (action.filter === "all" || action.filter === "active" || action.filter === "completed") next.filter = action.filter;
  }
  return next;
}
`;

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === 'object') {
    const out = {};
    for (const k of Object.keys(value).sort()) out[k] = stable(value[k]);
    return out;
  }
  return value;
}
function stableString(value) { return JSON.stringify(stable(value)); }
function canonState(state) {
  return stable({
    filter: (state.filter === 'All' || !state.filter) ? 'all' : state.filter,
    todos: (state.todos || []).map(t => stable({ completed: !!t.completed, id: Number(t.id), title: String(t.title) }))
  });
}
function promptSrc(state, action) {
  return DISPATCH_SRC + '\nfunction main(){  // << START_OF_TRACE\n' +
    '  let state = ' + JSON.stringify(canonState(state)) + ';\n' +
    '  state = dispatch(state, ' + JSON.stringify(stable(action)) + ');\n' +
    '  return state;\n}\nconsole.log(JSON.stringify(main()));\n';
}
function extractedDispatch(state, action) {
  const fn = new Function(DISPATCH_SRC + '; return dispatch;')();
  return canonState(fn(JSON.parse(JSON.stringify(canonState(state))), JSON.parse(JSON.stringify(action))));
}

function loadRealApp() {
  const html = fs.readFileSync(path.join(APP_DIR, 'index.html'), 'utf8');
  const dom = new JSDOM(html, { runScripts: 'outside-only', url: 'http://localhost/#/' });
  const w = dom.window;
  for (const f of ['base.js', 'helpers.js', 'store.js', 'model.js', 'template.js', 'view.js', 'controller.js']) {
    w.eval(fs.readFileSync(path.join(APP_DIR, f), 'utf8'));
  }
  const app = {};
  app.storage = new w.app.Store('dataset-' + Math.random().toString(36).slice(2));
  app.model = new w.app.Model(app.storage);
  app.template = new w.app.Template();
  app.view = new w.app.View(app.template);
  app.controller = new w.app.Controller(app.model, app.view);
  app.controller.setView('#/');
  return { dom, app };
}
function readTodos(app) {
  let todos = [];
  app.model.read(data => { todos = data.map(t => ({ completed: !!t.completed, id: Number(t.id), title: String(t.title) })); });
  const route = app.controller._activeRoute;
  return canonState({ filter: route === 'All' ? 'all' : route, todos });
}
function routeFor(filter) { return filter === 'all' ? '#/' : '#/' + filter; }
function seedRealApp(app, state) {
  const byId = new Map(canonState(state).todos.map(t => [t.id, t]));
  const ids = Array.from(byId.keys());
  const maxId = ids.length ? Math.max(...ids) : 0;
  for (let id = 1; id <= maxId; id++) {
    const todo = byId.get(id);
    app.controller.addItem(todo ? todo.title : '__deleted_' + id);
    if (todo && todo.completed) app.controller.toggleComplete(id, true, true);
  }
  for (let id = 1; id <= maxId; id++) if (!byId.has(id)) app.controller.removeItem(id);
  app.controller.setView(routeFor(canonState(state).filter));
}
function applyRealAction(app, action) {
  if (action.type === 'add') app.controller.addItem(action.title);
  else if (action.type === 'toggle') {
    const before = readTodos(app).todos.find(t => t.id === action.id);
    if (before) app.controller.toggleComplete(action.id, !before.completed);
  } else if (action.type === 'delete') app.controller.removeItem(action.id);
  else if (action.type === 'edit') {
    const before = readTodos(app).todos.find(t => t.id === action.id);
    if (before) app.controller.editItem(action.id);
    app.controller.editItemSave(action.id, action.title);
  }
  else if (action.type === 'toggleAll') app.controller.toggleAll(!!action.completed);
  else if (action.type === 'clearCompleted') app.controller.removeCompletedItems();
  else if (action.type === 'setFilter') app.controller.setView(routeFor(action.filter));
}
function realTransition(state, action) {
  const { dom, app } = loadRealApp();
  seedRealApp(app, state);
  applyRealAction(app, action);
  const out = readTodos(app);
  dom.window.close();
  return out;
}
function verifyPrompt(row) {
  const r = spawnSync(process.execPath, ['-e', row.prompt_src], { encoding: 'utf8', timeout: 15000 });
  if (r.status !== 0) return { ok: false, err: (r.stderr || r.stdout).slice(0, 200) };
  try { return { ok: stableString(JSON.parse(r.stdout.trim())) === stableString(row.truth_state) }; }
  catch (e) { return { ok: false, err: e.message }; }
}

const words = ['milk','eggs','bread','tea','rice','soap','code','tests','docs','coffee','laundry','rent','plants','inbox','backup','walk','call','write','read','ship'];
function makeState(maxId, filter, pattern, holes) {
  const holeSet = new Set(holes || []);
  const todos = [];
  for (let id = 1; id <= maxId; id++) {
    if (holeSet.has(id)) continue;
    let completed = pattern === 'allDone' ? true : pattern === 'noneDone' ? false : (id % 2 === 0);
    todos.push({ completed, id, title: words[(id - 1) % words.length] + '-' + id });
  }
  return canonState({ filter, todos });
}
function actionsFor(state, idx) {
  const ids = state.todos.map(t => t.id);
  const first = ids[0], last = ids[ids.length - 1], mid = ids[Math.floor(ids.length / 2)];
  const actions = [
    { type: 'add', title: idx % 3 === 0 ? '  new task ' + idx + '  ' : 'task ' + idx },
    { type: 'toggleAll', completed: idx % 2 === 0 },
    { type: 'clearCompleted' },
    { type: 'setFilter', filter: ['all','active','completed'][idx % 3] }
  ];
  if (ids.length) {
    actions.push({ type: 'toggle', id: mid });
    actions.push({ type: 'delete', id: first });
    actions.push({ type: 'edit', id: last, title: idx % 4 === 0 ? '  renamed ' + idx + '  ' : 'renamed-' + idx });
    if (idx % 5 === 0) actions.push({ type: 'edit', id: mid, title: '   ' });
  }
  return actions;
}
function buildCandidates() {
  const specs = [
    [0,'all','noneDone',[]], [1,'all','noneDone',[]], [2,'active','mixed',[]], [3,'completed','mixed',[]],
    [4,'all','allDone',[2]], [5,'active','noneDone',[]], [6,'completed','mixed',[3]], [8,'all','mixed',[]],
    [10,'active','mixed',[]], [12,'completed','noneDone',[2,7]], [15,'all','mixed',[]], [20,'active','mixed',[5,11,17]]
  ];
  const candidates = [];
  specs.forEach((spec, i) => {
    const state = makeState(spec[0], spec[1], spec[2], spec[3]);
    actionsFor(state, i).forEach(action => candidates.push({ state, action }));
  });
  return candidates.slice(0, 84);
}

function main() {
  const candidates = buildCandidates();
  const rows = [];
  let extractedMismatch = 0, promptMismatch = 0;
  const coverage = {};
  for (const c of candidates) {
    const before = canonState(c.state);
    const action = stable(c.action);
    const truth = realTransition(before, action);
    const extracted = extractedDispatch(before, action);
    if (stableString(truth) !== stableString(extracted)) { extractedMismatch++; continue; }
    const row = stable({
      action,
      entry: 'main',
      lang: 'js',
      prompt_src: promptSrc(before, action),
      source_app: 'tastejs/todomvc examples/javascript-es5 src/model.js, src/store.js, src/controller.js; truth via jsdom-loaded dist app',
      state_before: before,
      target: TARGET,
      truth_state: truth
    });
    const v = verifyPrompt(row);
    if (!v.ok) { promptMismatch++; continue; }
    coverage[action.type] = (coverage[action.type] || 0) + 1;
    rows.push(row);
  }
  fs.writeFileSync(OUT, rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  const equivalenceSamples = Math.min(candidates.length, rows.length + extractedMismatch);
  const notes = `# TodoMVC UI-transition dataset\n\n` +
    `Source: tastejs/todomvc \`examples/javascript-es5\` (MIT), downloaded under \`source/\`. The prompt dispatch is adapted from \`src/model.js\` (create/read/update/remove/count), \`src/store.js\` (in-memory todos, sequential IDs, save/remove/drop), and \`src/controller.js\` (addItem, editItemSave, removeItem, removeCompletedItems, toggleComplete, toggleAll, setView/_updateFilterState).\n\n` +
    `Ground truth: \`harvest_todomvc.js\` loads the real dist HTML and JS files in jsdom using the repository jsdom install, constructs the real Store/Model/View/Controller, seeds todos through controller methods, applies each action through real controller methods, then reads the model back with \`model.read\`.\n\n` +
    `Equivalence check: for every candidate transition, the extracted self-contained dispatch was run and compared with the jsdom real-app state before writing a row. Checked ${equivalenceSamples} candidate transitions; extracted mismatches dropped: ${extractedMismatch}.\n\n` +
    `Self-consistency: every row's \`prompt_src\` was executed with \`node -e\` and parsed JSON was compared to \`truth_state\`. Kept ${rows.length}; dropped for prompt mismatch: ${promptMismatch}; total dropped: ${extractedMismatch + promptMismatch}.\n\n` +
    `Coverage: ${JSON.stringify(coverage)}. State sizes include empty, 1-8 item, and larger 10/12/15/20 item lists; some states contain ID holes from prior deletes. Caveat: prompt state intentionally models TodoMVC's observable todo model plus filter, not DOM rendering details or the private Store ID counter; seeding through max id keeps add transitions equivalent.\n`;
  fs.writeFileSync(NOTES, notes);
  console.log(JSON.stringify({ out: OUT, rows: rows.length, candidates: candidates.length, extractedMismatch, promptMismatch, coverage }));
}
main();
