const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
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
  const out = execFileSync(process.execPath, ['-e', src], { encoding: 'utf8', timeout: 2000, maxBuffer: 1024 * 1024 });
  const lines = out.trim().split(/\n/).filter(Boolean);
  return JSON.parse(lines[lines.length - 1]);
}
const file = path.resolve(__dirname, '../../data/uitrans_cascade_js.jsonl');
const rows = fs.readFileSync(file, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
let ok = 0;
const actionCounts = {}, fieldCounts = {};
let maxPromptChars = 0;
for (let i = 0; i < rows.length; i++) {
  const row = rows[i];
  if (row.target !== 'cascade_js' || row.lang !== 'js' || row.entry !== 'main') throw new Error(`schema mismatch at row ${i}`);
  if (!row.prompt_src.includes('function dispatch(state, action)') || !row.prompt_src.includes('// << START_OF_TRACE')) throw new Error(`prompt_src missing required content at row ${i}`);
  const got = runPrompt(row.prompt_src);
  if (!same(got, row.truth_state)) throw new Error(`truth mismatch at row ${i}\n got=${JSON.stringify(canonical(got))}\n exp=${JSON.stringify(canonical(row.truth_state))}`);
  if (row.prompt_src.length > 36000) throw new Error(`prompt too large at row ${i}: ${row.prompt_src.length}`);
  actionCounts[row.action.type] = (actionCounts[row.action.type] || 0) + 1;
  fieldCounts[row.action.field] = (fieldCounts[row.action.field] || 0) + 1;
  maxPromptChars = Math.max(maxPromptChars, row.prompt_src.length);
  ok++;
}
console.log(JSON.stringify({ verified: ok, actionCounts, fieldCounts, maxPromptChars }, null, 2));
