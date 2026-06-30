const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === 'object') {
    const out = {};
    for (const k of Object.keys(value).sort()) out[k] = stable(value[k]);
    return out;
  }
  return value;
}
function same(a, b) { return JSON.stringify(stable(a)) === JSON.stringify(stable(b)); }

const file = process.argv[2] || path.resolve(__dirname, '../../data/uitrans_todomvc.jsonl');
const lines = fs.readFileSync(file, 'utf8').trim().split(/\n/).filter(Boolean);
let ok = 0, bad = 0;
const coverage = {};
for (let i = 0; i < lines.length; i++) {
  const row = JSON.parse(lines[i]);
  const r = spawnSync(process.execPath, ['-e', row.prompt_src], { encoding: 'utf8', timeout: 15000 });
  let pass = false;
  if (r.status === 0) {
    try { pass = same(JSON.parse(r.stdout.trim()), row.truth_state); } catch (_) {}
  }
  if (pass) ok++; else { bad++; console.error('mismatch row', i + 1, r.stderr || r.stdout); }
  coverage[row.action.type] = (coverage[row.action.type] || 0) + 1;
}
console.log(JSON.stringify({ file, rows: lines.length, ok, bad, coverage }));
process.exit(bad ? 1 : 0);
