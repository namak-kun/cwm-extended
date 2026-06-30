const fs = require('fs');
const path = require('path');
const vm = require('vm');
function canonical(x) {
  if (Array.isArray(x)) return x.map(canonical);
  if (x && typeof x === 'object') {
    const o = {};
    for (const k of Object.keys(x).sort()) o[k] = canonical(x[k]);
    return o;
  }
  return x;
}
function same(a, b) { return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b)); }
function runPrompt(src) {
  const lines = [];
  vm.runInNewContext(src, { console: { log: (s) => lines.push(String(s)) } }, { timeout: 1000 });
  return JSON.parse(lines[lines.length - 1]);
}
const file = path.resolve(__dirname, '../../data/uitrans_vanilla.jsonl');
const rows = fs.readFileSync(file, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
let ok = 0;
const counts = {};
for (let i = 0; i < rows.length; i++) {
  const row = rows[i];
  if (row.target !== 'vanilla' || row.lang !== 'js' || row.entry !== 'main') throw new Error(`schema mismatch at row ${i}`);
  if (!row.prompt_src.includes('// << START_OF_TRACE')) throw new Error(`missing trace marker at row ${i}`);
  const got = runPrompt(row.prompt_src);
  if (!same(got, row.truth_state)) {
    throw new Error(`truth mismatch at row ${i}\n got=${JSON.stringify(canonical(got))}\n exp=${JSON.stringify(canonical(row.truth_state))}`);
  }
  const app = row.source_app.includes('form-validator') ? 'form-validator' : 'movie-seat-booking';
  counts[app] = (counts[app] || 0) + 1;
  ok++;
}
console.log(JSON.stringify({ verified: ok, counts }, null, 2));
