const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { JSDOM } = require('jsdom');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'data', 'uitrans_cascade_js.jsonl');
const APP_DIR = path.join(__dirname, 'apps', 'cascade-form');
const HTML = fs.readFileSync(path.join(APP_DIR, 'index.html'), 'utf8');
const SCRIPT = fs.readFileSync(path.join(APP_DIR, 'script.js'), 'utf8');
const DISPATCH_SRC = '// ---- cascading form validator logic derived from bradtraversy/vanillawebprojects/form-validator plus dependent-field rules ----\n' + SCRIPT.split('function stateFromDom()')[0];
const SOURCE = 'uidata/cascade_js/apps/cascade-form/script.js:dispatch/validateAll/handleEvent (Brad Traversy form-validator style + dependent business/minor/address/step gating rules)';

function canonical(x) {
  if (Array.isArray(x)) return x.map(canonical);
  if (x && typeof x === 'object') {
    const o = {};
    for (const k of Object.keys(x).sort()) o[k] = canonical(x[k]);
    return o;
  }
  return x;
}
function clone(x) { return JSON.parse(JSON.stringify(x)); }
function same(a, b) { return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b)); }
function loadDom() {
  const dom = new JSDOM(HTML, { runScripts: 'outside-only', url: 'http://localhost/' });
  dom.window.eval(SCRIPT);
  return dom;
}
function makeState(fields) {
  const dom = loadDom();
  const st = dom.window.CWMCascade.defaultState();
  Object.assign(st.fields, fields);
  return canonical(dom.window.CWMCascade.validateAll(st));
}
function runReal(state, action) {
  const dom = loadDom();
  const doc = dom.window.document;
  dom.window.CWMCascade.applyStateToDom(clone(state));
  const el = doc.getElementById(action.field);
  if (el.type === 'checkbox') el.checked = !!action.value; else el.value = action.value;
  const ev = action.type === 'input' ? 'input' : 'change';
  el.dispatchEvent(new dom.window.Event(ev, { bubbles: true, cancelable: true }));
  return canonical(dom.window.CWMCascade.stateFromDom());
}
function promptSrc(state, action) {
  return DISPATCH_SRC + `function main(){  // << START_OF_TRACE\n  let state = ${JSON.stringify(canonical(state))};\n  state = dispatch(state, ${JSON.stringify(canonical(action))});\n  return state;\n}\nconsole.log(JSON.stringify(main()));\n`;
}
function runPrompt(src) {
  const lines = [];
  vm.runInNewContext(src, { console: { log: (s) => lines.push(String(s)) } }, { timeout: 1000 });
  return JSON.parse(lines[lines.length - 1]);
}

const seeds = [
  {},
  {username:'Al', email:'bad', password:'abc', password2:'xyz', age:'12', country:'USA', region:'CA', city:'Los Angeles', terms:false},
  {username:'Nora', email:'nora@example.com', password:'Secret12', password2:'Secret12', age:'22', country:'USA', region:'CA', city:'San Diego', terms:true},
  {username:'BusinessOne', email:'ops@biz.com', password:'Strong9A', password2:'Strong9A', age:'35', accountType:'business', companyName:'Acme', vatId:'US123456', country:'Canada', region:'ON', city:'Toronto', terms:true},
  {username:'teenUser', email:'teen@example.com', password:'TeenPass1', password2:'TeenPass1', age:'16', guardianEmail:'parent@example.com', country:'USA', region:'NY', city:'Buffalo', terms:true},
  {username:'space user', email:'space@example.com', password:'NoNumberA', password2:'NoNumberA', age:'19', country:'Other', terms:true},
  {username:'LongLongLongLong', email:'wrong@site', password:'lowercase1', password2:'lowercase1', age:'121', accountType:'business', companyName:'A', vatId:'123', country:'USA', region:'BC', city:'Toronto', terms:false},
  {username:'Mia', email:'mia@site.org', password:'MiaPass9', password2:'Mismatch9', age:'17', guardianEmail:'badguardian', accountType:'business', companyName:'Mia LLC', vatId:'CA12345678', country:'Canada', region:'BC', city:'Victoria', newsletter:true, terms:false},
  {username:'Omar', email:'omar@example.com', password:'OmarPass7!', password2:'OmarPass7!', age:'13', guardianEmail:'parent@home.org', accountType:'personal', country:'USA', region:'TX', city:'Dallas', newsletter:true, terms:true},
  {username:'Eva', email:'eva@example.com', password:'EvaPass8', password2:'EvaPass8', age:'18', guardianEmail:'old@hidden.com', accountType:'business', companyName:'E Corp', vatId:'GB123456789', country:'Canada', region:'ON', city:'Ottawa', terms:true},
  {username:'Kai', email:'kai@example.com', password:'KaiPass1', password2:'KaiPass1', age:'44', accountType:'business', companyName:'', vatId:'DE123456', country:'USA', region:'NY', city:'New York', terms:true},
  {username:'Lia', email:'lia@example.com', password:'LiaPass2', password2:'LiaPass2', age:'15', guardianEmail:'', accountType:'personal', country:'Canada', region:'BC', city:'Vancouver', terms:true},
  {username:'Max', email:'max@example.com', password:'MaxPass3', password2:'MaxPass3', age:'25', accountType:'personal', country:'', region:'', city:'', terms:false},
  {username:'Quinn', email:'quinn@example.com', password:'QuinnPass4', password2:'QuinnPass4', age:'33', accountType:'business', companyName:'Q Shop', vatId:'FR1234567', country:'USA', region:'TX', city:'Austin', terms:true},
  {username:'Rae', email:'rae@sample.net', password:'RaePass5', password2:'RaePass5', age:'14', guardianEmail:'rae.parent@sample.net', accountType:'business', companyName:'Rae Studio', vatId:'US1234567890', country:'Other', terms:true}
].map(makeState);

const actions = [
  {type:'input', field:'username', value:''}, {type:'input', field:'username', value:'Zo'}, {type:'input', field:'username', value:'Zoe'}, {type:'input', field:'username', value:'bad name'},
  {type:'input', field:'email', value:'not-email'}, {type:'input', field:'email', value:'user@example.com'},
  {type:'input', field:'password', value:'short'}, {type:'input', field:'password', value:'StrongPass9'}, {type:'input', field:'password2', value:'StrongPass9'}, {type:'input', field:'password2', value:'mismatch'},
  {type:'input', field:'age', value:'12'}, {type:'input', field:'age', value:'16'}, {type:'input', field:'age', value:'18'}, {type:'input', field:'age', value:'121'},
  {type:'input', field:'guardianEmail', value:'parent@example.com'}, {type:'input', field:'guardianEmail', value:'bad-parent'},
  {type:'select', field:'accountType', value:'business'}, {type:'select', field:'accountType', value:'personal'},
  {type:'input', field:'companyName', value:'A'}, {type:'input', field:'companyName', value:'Acme Inc'}, {type:'input', field:'vatId', value:'US123456'}, {type:'input', field:'vatId', value:'badvat'},
  {type:'select', field:'country', value:'USA'}, {type:'select', field:'country', value:'Canada'}, {type:'select', field:'country', value:'Other'}, {type:'select', field:'country', value:''},
  {type:'select', field:'region', value:'CA'}, {type:'select', field:'region', value:'NY'}, {type:'select', field:'region', value:'BC'}, {type:'select', field:'region', value:'ON'},
  {type:'select', field:'city', value:'Los Angeles'}, {type:'select', field:'city', value:'Toronto'}, {type:'select', field:'city', value:'Austin'},
  {type:'toggle', field:'newsletter', value:true}, {type:'toggle', field:'newsletter', value:false}, {type:'toggle', field:'terms', value:true}, {type:'toggle', field:'terms', value:false}
];

const rows = [];
const seen = new Set();
let candidates = 0, promptMismatches = 0, realErrors = 0;
outer: for (const seed of seeds) {
  for (const action of actions) {
    candidates++;
    const before = canonical(clone(seed));
    try {
      const truth = runReal(before, action);
      const prompt_src = promptSrc(before, action);
      const got = runPrompt(prompt_src);
      const key = JSON.stringify([before, action, truth]);
      if (!same(got, truth)) { promptMismatches++; continue; }
      if (seen.has(key)) continue;
      seen.add(key);
      rows.push(canonical({ target:'cascade_js', lang:'js', prompt_src, entry:'main', action:canonical(action), state_before:before, truth_state:truth, source_app:SOURCE }));
      if (rows.length >= 120) break outer;
    } catch (e) {
      realErrors++;
    }
  }
}
if (rows.length < 80) throw new Error(`only generated ${rows.length} verified rows`);
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, rows.map(r => JSON.stringify(r)).join('\n') + '\n');
const maxPromptChars = Math.max(...rows.map(r => r.prompt_src.length));
console.log(JSON.stringify({ output: OUT, candidates, kept: rows.length, promptMismatches, realErrors, maxPromptChars }, null, 2));
