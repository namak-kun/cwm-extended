const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { JSDOM } = require('jsdom');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'data', 'uitrans_vanilla.jsonl');
const FORM_DIR = path.join(__dirname, 'apps', 'form-validator');
const MOVIE_DIR = path.join(__dirname, 'apps', 'movie-seat-booking');
const FORM_HTML = fs.readFileSync(path.join(FORM_DIR, 'index.html'), 'utf8');
const FORM_SCRIPT = fs.readFileSync(path.join(FORM_DIR, 'script.js'), 'utf8');
const MOVIE_HTML = fs.readFileSync(path.join(MOVIE_DIR, 'index.html'), 'utf8');
const MOVIE_SCRIPT = fs.readFileSync(path.join(MOVIE_DIR, 'script.js'), 'utf8');

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

const FORM_FIELDS = ['username', 'email', 'password', 'password2'];
function emptyFormState() {
  const controls = {}, fields = {};
  for (const id of FORM_FIELDS) {
    fields[id] = '';
    controls[id] = { message: 'Error message', status: '' };
  }
  return canonical({ controls, fields });
}
function formStateFromDom(dom) {
  const doc = dom.window.document;
  const controls = {}, fields = {};
  for (const id of FORM_FIELDS) {
    const input = doc.getElementById(id);
    const fc = input.parentElement;
    const small = fc.querySelector('small');
    const parts = fc.className.split(/\s+/).filter(Boolean);
    fields[id] = input.value;
    controls[id] = {
      message: small.innerText !== undefined ? small.innerText : small.textContent,
      status: parts.includes('error') ? 'error' : (parts.includes('success') ? 'success' : '')
    };
  }
  return canonical({ controls, fields });
}
function applyFormState(dom, state) {
  const doc = dom.window.document;
  for (const id of FORM_FIELDS) {
    const input = doc.getElementById(id);
    const fc = input.parentElement;
    const small = fc.querySelector('small');
    input.value = state.fields[id];
    fc.className = state.controls[id].status ? `form-control ${state.controls[id].status}` : 'form-control';
    small.innerText = state.controls[id].message;
  }
}
function runRealForm(state, action) {
  const dom = new JSDOM(FORM_HTML, { runScripts: 'outside-only', url: 'http://localhost/' });
  dom.window.eval(FORM_SCRIPT);
  applyFormState(dom, state);
  dom.window.document.getElementById(action.field).value = action.value;
  dom.window.document.getElementById('form').dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
  return formStateFromDom(dom);
}

const FORM_DISPATCH_SRC = `// ---- real handler logic from bradtraversy/vanillawebprojects/form-validator/script.js ----
const FORM_FIELDS = ['username', 'email', 'password', 'password2'];
function canonical(x) {
  if (Array.isArray(x)) return x.map(canonical);
  if (x && typeof x === 'object') {
    const o = {};
    for (const k of Object.keys(x).sort()) o[k] = canonical(x[k]);
    return o;
  }
  return x;
}
function getFieldName(input) {
  return input.id.charAt(0).toUpperCase() + input.id.slice(1);
}
function showError(state, id, message) {
  state.controls[id].status = 'error';
  state.controls[id].message = message;
}
function showSuccess(state, id) {
  state.controls[id].status = 'success';
}
function checkEmail(state, id) {
  const value = state.fields[id];
  const re = /^(([^<>()\\[\\]\\\\.,;:\\s@"]+(\\.[^<>()\\[\\]\\\\.,;:\\s@"]+)*)|(".+"))@((\\[[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\])|(([a-zA-Z\\-0-9]+\\.)+[a-zA-Z]{2,}))$/;
  if (re.test(value.trim())) showSuccess(state, id);
  else showError(state, id, 'Email is not valid');
}
function checkRequired(state, inputArr) {
  let isRequired = false;
  inputArr.forEach(function(id) {
    if (state.fields[id].trim() === '') {
      showError(state, id, getFieldName({id}) + ' is required');
      isRequired = true;
    } else {
      showSuccess(state, id);
    }
  });
  return isRequired;
}
function checkLength(state, id, min, max) {
  if (state.fields[id].length < min) {
    showError(state, id, getFieldName({id}) + ' must be at least ' + min + ' characters');
  } else if (state.fields[id].length > max) {
    showError(state, id, getFieldName({id}) + ' must be less than ' + max + ' characters');
  } else {
    showSuccess(state, id);
  }
}
function checkPasswordsMatch(state, input1, input2) {
  if (state.fields[input1] !== state.fields[input2]) {
    showError(state, input2, 'Passwords do not match');
  }
}
function dispatch(state, action) {
  state.fields[action.field] = action.value;
  if (checkRequired(state, FORM_FIELDS)) {
    checkLength(state, 'username', 3, 15);
    checkLength(state, 'password', 6, 25);
    checkEmail(state, 'email');
    checkPasswordsMatch(state, 'password', 'password2');
  }
  return canonical(state);
}
`;

function formPrompt(state, action) {
  return FORM_DISPATCH_SRC + `function main(){  // << START_OF_TRACE\n  let state = ${JSON.stringify(canonical(state))};\n  state = dispatch(state, ${JSON.stringify(canonical(action))});\n  return state;\n}\nconsole.log(JSON.stringify(main()));\n`;
}

const MOVIES = [10, 12, 8, 9];
const N_SEATS = 39;
function movieStorage(selectedSeats, movieIndex) {
  return {
    selectedMovieIndex: String(movieIndex),
    selectedMoviePrice: String(MOVIES[movieIndex]),
    selectedSeats: JSON.stringify(selectedSeats)
  };
}
function movieState(selectedSeats = [], movieIndex = 0) {
  const uniq = [...new Set(selectedSeats)].filter(i => i >= 0 && i < N_SEATS).sort((a, b) => a - b);
  return canonical({
    count: uniq.length,
    movieIndex,
    selectedSeats: uniq,
    storage: movieStorage(uniq, movieIndex),
    ticketPrice: MOVIES[movieIndex],
    total: uniq.length * MOVIES[movieIndex]
  });
}
function movieStateFromDom(dom) {
  const doc = dom.window.document;
  const seats = [...doc.querySelectorAll('.row .seat:not(.occupied)')];
  const selectedSeats = [...doc.querySelectorAll('.row .seat.selected')].map(seat => seats.indexOf(seat)).filter(i => i >= 0).sort((a, b) => a - b);
  const movie = doc.getElementById('movie');
  return canonical({
    count: Number(doc.getElementById('count').innerText ?? doc.getElementById('count').textContent),
    movieIndex: movie.selectedIndex,
    selectedSeats,
    storage: {
      selectedMovieIndex: dom.window.localStorage.getItem('selectedMovieIndex'),
      selectedMoviePrice: dom.window.localStorage.getItem('selectedMoviePrice'),
      selectedSeats: dom.window.localStorage.getItem('selectedSeats')
    },
    ticketPrice: Number(movie.value),
    total: Number(doc.getElementById('total').innerText ?? doc.getElementById('total').textContent)
  });
}
function loadMovieDom(state) {
  const dom = new JSDOM(MOVIE_HTML, { runScripts: 'outside-only', url: 'http://localhost/' });
  dom.window.localStorage.clear();
  dom.window.localStorage.setItem('selectedSeats', JSON.stringify(state.selectedSeats));
  dom.window.localStorage.setItem('selectedMovieIndex', String(state.movieIndex));
  dom.window.localStorage.setItem('selectedMoviePrice', String(state.ticketPrice));
  dom.window.eval(MOVIE_SCRIPT);
  return dom;
}
function runRealMovie(state, action) {
  const dom = loadMovieDom(state);
  const doc = dom.window.document;
  if (action.type === 'clickSeat') {
    const seats = [...doc.querySelectorAll('.row .seat:not(.occupied)')];
    seats[action.index].dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }));
  } else if (action.type === 'selectMovie') {
    const movie = doc.getElementById('movie');
    movie.selectedIndex = action.index;
    movie.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
  }
  return movieStateFromDom(dom);
}

const MOVIE_DISPATCH_SRC = `// ---- real handler logic from bradtraversy/vanillawebprojects/movie-seat-booking/script.js ----
const MOVIES = [10, 12, 8, 9];
function canonical(x) {
  if (Array.isArray(x)) return x.map(canonical);
  if (x && typeof x === 'object') {
    const o = {};
    for (const k of Object.keys(x).sort()) o[k] = canonical(x[k]);
    return o;
  }
  return x;
}
function setMovieData(state, movieIndex, moviePrice) {
  state.storage.selectedMovieIndex = String(movieIndex);
  state.storage.selectedMoviePrice = String(moviePrice);
}
function updateSelectedCount(state) {
  state.selectedSeats = [...new Set(state.selectedSeats)].sort((a, b) => a - b);
  state.storage.selectedSeats = JSON.stringify(state.selectedSeats);
  state.count = state.selectedSeats.length;
  state.total = state.count * state.ticketPrice;
  setMovieData(state, state.movieIndex, String(state.ticketPrice));
}
function dispatch(state, action) {
  if (action.type === 'selectMovie') {
    state.movieIndex = action.index;
    state.ticketPrice = MOVIES[action.index];
    setMovieData(state, action.index, String(MOVIES[action.index]));
    updateSelectedCount(state);
  } else if (action.type === 'clickSeat') {
    const i = state.selectedSeats.indexOf(action.index);
    if (i >= 0) state.selectedSeats.splice(i, 1);
    else state.selectedSeats.push(action.index);
    updateSelectedCount(state);
  }
  return canonical(state);
}
`;
function moviePrompt(state, action) {
  return MOVIE_DISPATCH_SRC + `function main(){  // << START_OF_TRACE\n  let state = ${JSON.stringify(canonical(state))};\n  state = dispatch(state, ${JSON.stringify(canonical(action))});\n  return state;\n}\nconsole.log(JSON.stringify(main()));\n`;
}

function runPrompt(src) {
  const lines = [];
  const sandbox = { console: { log: (s) => lines.push(String(s)) } };
  vm.runInNewContext(src, sandbox, { timeout: 1000 });
  return JSON.parse(lines[lines.length - 1]);
}

function makeRows() {
  const rows = [];
  const formSeeds = [
    emptyFormState(),
    canonical({fields:{username:'Al',email:'bad',password:'123',password2:'456'},controls:{username:{status:'error',message:'Username must be at least 3 characters'},email:{status:'error',message:'Email is not valid'},password:{status:'error',message:'Password must be at least 6 characters'},password2:{status:'error',message:'Passwords do not match'}}}),
    canonical({fields:{username:'Alice',email:'alice@example.com',password:'secret1',password2:'secret1'},controls:{username:{status:'success',message:'Error message'},email:{status:'success',message:'Error message'},password:{status:'success',message:'Error message'},password2:{status:'success',message:'Error message'}}}),
    canonical({fields:{username:'',email:'user@site.org',password:'abcdef',password2:'abcdef'},controls:{username:{status:'error',message:'Username is required'},email:{status:'success',message:'Error message'},password:{status:'success',message:'Error message'},password2:{status:'success',message:'Error message'}}}),
    canonical({fields:{username:'VeryVeryLongUserName',email:'x@y.z',password:'longenoughpassword1234567890',password2:'nope'},controls:{username:{status:'success',message:'Error message'},email:{status:'error',message:'Email is not valid'},password:{status:'success',message:'Error message'},password2:{status:'error',message:'Passwords do not match'}}}),
    canonical({fields:{username:'Bob',email:'bob@site.co',password:'',password2:''},controls:{username:{status:'success',message:'Error message'},email:{status:'success',message:'Error message'},password:{status:'error',message:'Password is required'},password2:{status:'error',message:'Password2 is required'}}}),
    canonical({fields:{username:'Zo',email:'zo@example.com',password:'abcdef',password2:'abcdeg'},controls:{username:{status:'error',message:'Username must be at least 3 characters'},email:{status:'success',message:'Error message'},password:{status:'success',message:'Error message'},password2:{status:'error',message:'Passwords do not match'}}}),
    canonical({fields:{username:'Carla',email:'carla@example',password:'12345',password2:'12345'},controls:{username:{status:'success',message:'Error message'},email:{status:'success',message:'Error message'},password:{status:'error',message:'Password must be at least 6 characters'},password2:{status:'success',message:'Error message'}}}),
    canonical({fields:{username:'Dina',email:'dina@example.com',password:'abcdef',password2:'abcdef'},controls:{username:{status:'success',message:'Username is required'},email:{status:'success',message:'Email is not valid'},password:{status:'success',message:'Password must be at least 6 characters'},password2:{status:'success',message:'Passwords do not match'}}})
  ];
  const formActions = [
    {type:'input', field:'username', value:''},
    {type:'input', field:'username', value:'Nora'},
    {type:'input', field:'email', value:'not-email'},
    {type:'input', field:'email', value:'nora@example.com'},
    {type:'input', field:'password', value:'12345'},
    {type:'input', field:'password', value:'abcdef'},
    {type:'input', field:'password2', value:'abcdef'},
    {type:'input', field:'password2', value:'mismatch'}
  ];
  let formCount = 0;
  outerForm: for (const seed of formSeeds) {
    for (const action of formActions) {
      if (formCount >= 40) break outerForm;
      const before = canonical(clone(seed));
      const truth = runRealForm(before, action);
      const prompt_src = formPrompt(before, action);
      const got = runPrompt(prompt_src);
      if (same(got, truth)) {
        rows.push({ target:'vanilla', lang:'js', prompt_src, entry:'main', action:canonical(action), state_before:before, truth_state:truth, source_app:'uidata/vanilla/apps/form-validator/script.js:showError/showSuccess/checkEmail/checkRequired/checkLength/checkPasswordsMatch/form submit listener' });
        formCount++;
      }
    }
  }

  const movieSeeds = [
    movieState([], 0),
    movieState([0], 1),
    movieState([0, 1, 5], 2),
    movieState([10, 20, 38], 3),
    movieState([2, 3, 4, 5, 6], 0),
    movieState([0, 2, 4, 6, 8, 10, 12, 14], 1),
    movieState([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 2),
    movieState([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 3)
  ];
  const movieActions = [
    {type:'clickSeat', index:0},
    {type:'clickSeat', index:5},
    {type:'clickSeat', index:20},
    {type:'clickSeat', index:38},
    {type:'selectMovie', index:0},
    {type:'selectMovie', index:1},
    {type:'selectMovie', index:2},
    {type:'selectMovie', index:3}
  ];
  let movieCount = 0;
  outerMovie: for (const seed of movieSeeds) {
    for (const action of movieActions) {
      if (movieCount >= 40) break outerMovie;
      const before = canonical(clone(seed));
      const truth = runRealMovie(before, action);
      const prompt_src = moviePrompt(before, action);
      const got = runPrompt(prompt_src);
      if (same(got, truth)) {
        rows.push({ target:'vanilla', lang:'js', prompt_src, entry:'main', action:canonical(action), state_before:before, truth_state:truth, source_app:'uidata/vanilla/apps/movie-seat-booking/script.js:setMovieData/updateSelectedCount/populateUI/change listener/click listener' });
        movieCount++;
      }
    }
  }
  return rows;
}

const rows = makeRows();
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, rows.map(r => JSON.stringify(canonical(r))).join('\n') + '\n');
const counts = rows.reduce((a, r) => { const app = r.source_app.includes('form-validator') ? 'form-validator' : 'movie-seat-booking'; a[app] = (a[app] || 0) + 1; return a; }, {});
console.log(JSON.stringify({ output: OUT, rows: rows.length, counts }, null, 2));
