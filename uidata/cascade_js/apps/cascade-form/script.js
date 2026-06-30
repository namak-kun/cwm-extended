const FIELD_IDS = ['accountType','age','city','companyName','country','email','guardianEmail','newsletter','password','password2','region','terms','username','vatId'];
const SELECT_DATA = {
  USA: { regions: ['CA','NY','TX'], cities: { CA: ['Los Angeles','San Diego'], NY: ['New York','Buffalo'], TX: ['Austin','Dallas'] } },
  Canada: { regions: ['BC','ON'], cities: { BC: ['Vancouver','Victoria'], ON: ['Toronto','Ottawa'] } },
  Other: { regions: [], cities: {} }
};
function canonical(x) {
  if (Array.isArray(x)) return x.map(canonical);
  if (x && typeof x === 'object') {
    const o = {};
    for (const k of Object.keys(x).sort()) o[k] = canonical(x[k]);
    return o;
  }
  return x;
}
function label(id) {
  const names = {password2:'Confirm password',guardianEmail:'Guardian email',accountType:'Account type',companyName:'Company name',vatId:'VAT ID'};
  return names[id] || id.charAt(0).toUpperCase() + id.slice(1);
}
function blankControl() { return { disabled: false, message: 'Error message', required: false, status: '', visible: true }; }
function defaultState() {
  const fields = {}, controls = {};
  FIELD_IDS.forEach(id => { fields[id] = (id === 'newsletter' || id === 'terms') ? false : ''; controls[id] = blankControl(); });
  fields.accountType = 'personal';
  return validateAll({ fields, controls, derived: {} });
}
function setError(state, id, message) { state.controls[id].status = 'error'; state.controls[id].message = message; }
function setSuccess(state, id) { state.controls[id].status = 'success'; state.controls[id].message = 'Error message'; }
function hideDisable(state, id) { state.controls[id].visible = false; state.controls[id].disabled = true; state.controls[id].required = false; state.controls[id].status = ''; state.controls[id].message = 'Error message'; }
function requireVisible(state, id) { state.controls[id].visible = true; state.controls[id].disabled = false; state.controls[id].required = true; }
function optionalVisible(state, id) { state.controls[id].visible = true; state.controls[id].disabled = false; state.controls[id].required = false; }
function emailOk(v) { return /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/.test(String(v).trim()); }
function passwordScore(pw) { let s = 0; if (pw.length >= 8) s++; if (/[A-Z]/.test(pw)) s++; if (/\d/.test(pw)) s++; if (/[^A-Za-z0-9]/.test(pw)) s++; return s; }
function fieldValid(state, id) { const c = state.controls[id]; return !c.visible || c.disabled || c.status === 'success' || (!c.required && c.status === ''); }
function validateRequiredText(state, id) { if (String(state.fields[id]).trim() === '') setError(state, id, label(id) + ' is required'); else setSuccess(state, id); }
function validateAll(state) {
  FIELD_IDS.forEach(id => { const old = state.controls[id] || {}; state.controls[id] = Object.assign(blankControl(), old); state.controls[id].status = ''; state.controls[id].message = 'Error message'; state.controls[id].required = false; state.controls[id].visible = true; state.controls[id].disabled = false; });
  ['username','email','password','password2','age','accountType','country','terms'].forEach(id => requireVisible(state, id));
  optionalVisible(state, 'newsletter'); optionalVisible(state, 'region'); optionalVisible(state, 'city');
  if (state.fields.accountType === 'business') { requireVisible(state, 'companyName'); requireVisible(state, 'vatId'); } else { hideDisable(state, 'companyName'); hideDisable(state, 'vatId'); }
  const ageNum = Number(state.fields.age);
  const under18 = String(state.fields.age).trim() !== '' && !Number.isNaN(ageNum) && ageNum < 18;
  if (under18) requireVisible(state, 'guardianEmail'); else hideDisable(state, 'guardianEmail');
  const country = state.fields.country;
  const regionOptions = country && SELECT_DATA[country] ? SELECT_DATA[country].regions.slice() : [];
  const cityOptions = country && state.fields.region && SELECT_DATA[country] ? (SELECT_DATA[country].cities[state.fields.region] || []).slice() : [];
  if (country === 'Other') { hideDisable(state, 'region'); hideDisable(state, 'city'); }
  else { requireVisible(state, 'region'); requireVisible(state, 'city'); }
  ['username','email','password','password2','age','accountType','country'].forEach(id => validateRequiredText(state, id));
  if (state.fields.username && (state.fields.username.length < 3 || state.fields.username.length > 15 || /\s/.test(state.fields.username))) setError(state, 'username', 'Username must be 3-15 chars with no spaces');
  if (state.fields.email && !emailOk(state.fields.email)) setError(state, 'email', 'Email is not valid');
  const score = passwordScore(String(state.fields.password));
  if (state.fields.password && score < 3) setError(state, 'password', 'Password needs 8 chars, uppercase, and number');
  if (state.fields.password2 && state.fields.password !== state.fields.password2) setError(state, 'password2', 'Passwords do not match');
  if (String(state.fields.age).trim() !== '' && (Number.isNaN(ageNum) || ageNum < 13 || ageNum > 120)) setError(state, 'age', 'Age must be between 13 and 120');
  if (state.controls.guardianEmail.visible) { validateRequiredText(state, 'guardianEmail'); if (state.fields.guardianEmail && !emailOk(state.fields.guardianEmail)) setError(state, 'guardianEmail', 'Guardian email is not valid'); }
  if (state.controls.companyName.visible) { validateRequiredText(state, 'companyName'); if (state.fields.companyName && state.fields.companyName.length < 2) setError(state, 'companyName', 'Company name must be at least 2 characters'); }
  if (state.controls.vatId.visible) { validateRequiredText(state, 'vatId'); if (state.fields.vatId && !/^[A-Z]{2}\d{6,10}$/.test(state.fields.vatId)) setError(state, 'vatId', 'VAT ID must start with 2 letters and 6-10 digits'); }
  if (state.controls.region.visible) { validateRequiredText(state, 'region'); if (state.fields.region && !regionOptions.includes(state.fields.region)) setError(state, 'region', 'Region is not available for country'); }
  if (state.controls.city.visible) { validateRequiredText(state, 'city'); if (state.fields.city && !cityOptions.includes(state.fields.city)) setError(state, 'city', 'City is not available for region'); }
  if (state.fields.terms !== true) setError(state, 'terms', 'Terms must be accepted'); else setSuccess(state, 'terms');
  const summary = FIELD_IDS.filter(id => state.controls[id].visible && state.controls[id].status === 'error').map(id => id + ': ' + state.controls[id].message);
  const steps = { account: ['username','email','password','password2'], profile: ['age','guardianEmail','accountType','companyName','vatId'], location: ['country','region','city','terms'] };
  const stepStatus = {};
  Object.keys(steps).forEach(k => { stepStatus[k] = steps[k].every(id => fieldValid(state, id)) ? 'complete' : 'blocked'; });
  state.derived = { activeStep: stepStatus.account === 'blocked' ? 'account' : (stepStatus.profile === 'blocked' ? 'profile' : (stepStatus.location === 'blocked' ? 'location' : 'done')), canSubmit: summary.length === 0, cityOptions, errorCount: summary.length, passwordScore: score, regionOptions, stepStatus, summary };
  return canonical(state);
}
function dispatch(state, action) {
  if (action.type === 'input' || action.type === 'select') state.fields[action.field] = action.value;
  if (action.type === 'toggle') state.fields[action.field] = !!action.value;
  if (action.field === 'country') { const opts = action.value && SELECT_DATA[action.value] ? SELECT_DATA[action.value].regions : []; if (!opts.includes(state.fields.region)) { state.fields.region = ''; state.fields.city = ''; } }
  if (action.field === 'region') { const c = state.fields.country; const opts = c && SELECT_DATA[c] && SELECT_DATA[c].cities[action.value] ? SELECT_DATA[c].cities[action.value] : []; if (!opts.includes(state.fields.city)) state.fields.city = ''; }
  if (action.field === 'accountType' && action.value !== 'business') { state.fields.companyName = ''; state.fields.vatId = ''; }
  if (action.field === 'age') { const n = Number(action.value); if (String(action.value).trim() === '' || Number.isNaN(n) || n >= 18) state.fields.guardianEmail = ''; }
  return validateAll(state);
}
function stateFromDom() {
  const fields = {}, controls = {};
  FIELD_IDS.forEach(id => { const el = document.getElementById(id); fields[id] = el.type === 'checkbox' ? el.checked : el.value; const fc = document.querySelector(`[data-field="${id}"]`); const small = fc.querySelector('small'); const cls = fc.className.split(/\s+/); controls[id] = { disabled: el.disabled, message: small.innerText || small.textContent || 'Error message', required: el.required, status: cls.includes('error') ? 'error' : (cls.includes('success') ? 'success' : ''), visible: fc.style.display !== 'none' }; });
  return validateAll({ fields, controls, derived: {} });
}
function applyStateToDom(state) {
  render(validateAll(JSON.parse(JSON.stringify(state))));
}
function render(state) {
  const region = document.getElementById('region'), city = document.getElementById('city');
  region.innerHTML = '<option value="">Choose region</option>' + state.derived.regionOptions.map(x => `<option value="${x}">${x}</option>`).join('');
  city.innerHTML = '<option value="">Choose city</option>' + state.derived.cityOptions.map(x => `<option value="${x}">${x}</option>`).join('');
  FIELD_IDS.forEach(id => { const el = document.getElementById(id), c = state.controls[id], fc = document.querySelector(`[data-field="${id}"]`); if (el.type === 'checkbox') el.checked = !!state.fields[id]; else el.value = state.fields[id]; el.disabled = c.disabled; el.required = c.required; fc.style.display = c.visible ? '' : 'none'; fc.className = 'form-control' + (c.status ? ' ' + c.status : ''); fc.querySelector('small').innerText = c.message; });
  document.getElementById('submitBtn').disabled = !state.derived.canSubmit;
  document.getElementById('errorSummary').innerHTML = state.derived.summary.map(x => `<li>${x}</li>`).join('');
}
function handleEvent(e) { const el = e.target; if (!FIELD_IDS.includes(el.id)) return; const state = stateFromDom(); const type = el.type === 'checkbox' ? 'toggle' : (el.tagName === 'SELECT' ? 'select' : 'input'); render(dispatch(state, { type, field: el.id, value: el.type === 'checkbox' ? el.checked : el.value })); }
document.getElementById('profileForm').addEventListener('input', handleEvent);
document.getElementById('profileForm').addEventListener('change', handleEvent);
document.getElementById('profileForm').addEventListener('submit', e => { e.preventDefault(); render(validateAll(stateFromDom())); });
render(defaultState());
window.CWMCascade = { dispatch, validateAll, defaultState, stateFromDom, applyStateToDom, canonical };
