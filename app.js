import * as detector from './detector.js';
import { show, isRed, SUITS, RANKS, JOKER } from './cards.js';
import * as blackjack from './games/blackjack.js';
import * as poker from './games/poker.js';
import * as zolik from './games/zolik.js';

const MODEL_CACHE = 'karty-models-v1'; // same name in sw.js

// `other` names the cards that are not yours; null means every card is yours.
const GAMES = {
  blackjack: { advise: blackjack.advise, other: 'krupiér', hint: 'Ťukni na kartu krupiéra, nech viem, ktorá je jeho.' },
  poker: { advise: poker.advise, other: 'stôl', hint: 'Ťukni na karty na stole (flop, turn, river).' },
  zolik: { advise: zolik.advise, other: null, twoDecks: true, hint: 'Žolíky nevidím, pridaj ich cez + Karta.' },
};

const FRAME_SIDE = 1280;   // longest side of the picture that gets analysed
const MIN_TICK = 80;       // ms between looks, keeps the phone cool
const SHOW_HITS = 2;       // a card must be seen this many times before it counts
const FORGET = 1500;       // ms out of sight before a card is forgotten
const STALE = 700;         // ms out of sight before a card stops counting
const ADVICE_EVERY = 350;  // ms between advice updates

const $ = (id) => document.getElementById(id);
const video = $('cam'), frame = $('frame'), overlay = $('overlay');
const sheet = $('sheet'), scrim = $('scrim'), shutter = $('shutter');
const fctx = frame.getContext('2d'), g = overlay.getContext('2d');
const labelFont = getComputedStyle(document.documentElement).getPropertyValue('--label-font').trim();

const settings = {
  game: 'blackjack', opponents: 3, hitSoft17: false, openPoints: 42, opened: false, threshold: 0.45,
  ...JSON.parse(localStorage.getItem('karty-settings') || '{}'),
};
const saveSettings = () => localStorage.setItem('karty-settings', JSON.stringify(settings));

let mode = 'live';          // 'live' | 'photo'
let frozen = false;
let stream = null, booted = false, failures = 0;
let tracks = [], nextId = 1, lastLook = 0;
const roles = new Map();    // card code -> 'other' | 'ignore'; anything else is yours
let manual = [];            // cards added by hand: { code, role }
let advice = null, adviceKey = '', adviceAt = 0;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false && kid !== '') el.append(kid);
  return el;
}

let toastTimer;
function toast(text) {
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2000);
}

// ---------- start-up ----------

async function boot() {
  $('bootActions').hidden = true;
  showBoot('Pripravujem sa', $('bootNote').textContent);
  try {
    await Promise.all([
      detector.load().catch((err) => { err.stage = 'model'; throw err; }),
      startCamera().catch((err) => { err.stage = 'camera'; throw err; }),
    ]);
  } catch (err) {
    console.error(err);
    if (err.stage === 'camera') return cameraProblem(err);
    return showBoot('Model na rozpoznávanie sa nepodarilo stiahnuť.', 'Skontroluj internet a skús znova.', true);
  }
  booted = true;
  $('boot').hidden = true;
  keepModelsOffline().catch(() => {});
  run();
}

// Copies every library file this launch used into the cache the service worker
// reads from, so the next launch works with no signal.
async function keepModelsOffline() {
  if (!('caches' in window)) return;
  const cache = await caches.open(MODEL_CACHE);
  const used = performance.getEntriesByType('resource').map((e) => e.name)
    .filter((u) => u.startsWith('https://cdn.jsdelivr.net/') || u.endsWith(detector.MODEL_URL));
  for (const url of new Set(used)) {
    if (!(await cache.match(url))) await cache.add(url).catch(() => {});
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('insecure'), { name: 'InsecureError' });
  stopCamera();
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  stream.getVideoTracks()[0].addEventListener('ended', () => {
    stream = null;
    if (!document.hidden && mode === 'live') startCamera().catch(cameraProblem);
  });
  video.srcObject = stream;
  await video.play();
  navigator.wakeLock?.request('screen').catch(() => {});
}

function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
}

function cameraProblem(err) {
  const why = {
    NotAllowedError: 'Prístup ku kamere je vypnutý. Povoľ kameru pre túto stránku a skús znova.',
    NotFoundError: 'Nenašiel som žiadnu kameru.',
    NotReadableError: 'Kameru používa iná aplikácia. Zavri ju a skús znova.',
    InsecureError: 'Kamera funguje len cez https.',
  }[err?.name] ?? 'Kamera sa nespustila.';
  showBoot(why, 'Môžeš použiť aj fotku z galérie.', true);
}

function showBoot(status, note = '', actions = false) {
  $('bootStatus').textContent = status;
  $('bootNote').textContent = note;
  $('bootNote').hidden = !note;
  $('bootActions').hidden = !actions;
  $('boot').hidden = false;
}

// ---------- looking ----------

function grab(source, w, hgt) {
  const k = Math.min(1, FRAME_SIDE / Math.max(w, hgt));
  const fw = Math.round(w * k), fh = Math.round(hgt * k);
  if (frame.width !== fw || frame.height !== fh) { frame.width = fw; frame.height = fh; }
  fctx.drawImage(source, 0, 0, fw, fh);
}

const centre = (b) => [b.x + b.w / 2, b.y + b.h / 2];

// Follows cards from one look to the next so a card missed for a moment does
// not flicker, and a one-off misreading never counts.
function follow(cards, now) {
  const near = 0.12 * Math.max(frame.width, frame.height);
  const free = new Set(tracks);
  for (const c of cards) {
    const [x, y] = centre(c.box);
    let best = null, bestD = Infinity;
    for (const t of free) {
      if (t.code !== c.code) continue;
      const [tx, ty] = centre(t.box);
      const d = Math.hypot(tx - x, ty - y);
      if (d < bestD) { bestD = d; best = t; }
    }
    if (best && bestD < near + Math.max(best.box.w, best.box.h)) {
      free.delete(best);
      Object.assign(best, { box: c.box, corners: c.corners, score: c.score, seen: now, hits: best.hits + 1 });
    } else {
      tracks.push({ id: nextId++, code: c.code, box: c.box, corners: c.corners, score: c.score, seen: now, first: now, hits: 1 });
    }
  }
  tracks = tracks.filter((t) => now - t.seen < FORGET);
  lastLook = now;
}

const counts = (t) => (mode === 'photo' || t.hits >= SHOW_HITS) && lastLook - t.seen < STALE;

async function look() {
  grab(video, video.videoWidth, video.videoHeight);
  const found = await detector.corners(frame, frame.width, frame.height, settings.threshold);
  follow(detector.groupCards(found), performance.now());
}

async function run() {
  for (;;) {
    await new Promise(requestAnimationFrame);
    if (mode !== 'live' || frozen || !stream || video.readyState < 2) continue;
    const started = performance.now();
    try {
      await look();
      failures = 0;
    } catch (err) {
      console.error(err);
      if (++failures === 5) showBoot('Rozpoznávanie prestalo fungovať.', 'Zavri aplikáciu a otvor ju znova.');
    }
    refresh();
    const spare = MIN_TICK - (performance.now() - started);
    if (spare > 0) await sleep(spare);
  }
}

// ---------- whose cards ----------

// Žolík has no dealer or table, so a card given to them earlier counts as yours.
function roleOf(code) {
  const role = roles.get(code) ?? 'me';
  return role === 'other' && !GAMES[settings.game].other ? 'me' : role;
}
const roleName = (role) => (role === 'other' ? GAMES[settings.game].other : role === 'ignore' ? 'ignorujem' : 'moja');

function hand() {
  const game = GAMES[settings.game];
  const mine = [], table = [];
  const seen = tracks.filter(counts).sort((a, b) => a.first - b.first);
  for (const t of seen) {
    const role = roleOf(t.code);
    if (role === 'ignore') continue;
    (role === 'other' && game.other ? table : mine).push(t.code);
  }
  for (const m of manual) (m.role === 'other' && game.other ? table : mine).push(m.code);
  if (game.twoDecks) return { mine, table };
  // One deck: the same card twice is a misreading.
  const t = [...new Set(table)];
  return { mine: [...new Set(mine)].filter((c) => !t.includes(c)), table: t };
}

function nextRole(code) {
  const order = GAMES[settings.game].other ? ['me', 'other', 'ignore'] : ['me', 'ignore'];
  const role = order[(order.indexOf(roleOf(code)) + 1) % order.length];
  if (role === 'me') roles.delete(code);
  else roles.set(code, role);
  toast(`${show(code)}: ${roleName(role)}`);
  refresh(true);
}

// ---------- advice ----------

function refresh(now = false) {
  const state = hand();
  const key = JSON.stringify([settings, state]);
  const t = performance.now();
  if (key !== adviceKey && (now || t - adviceAt > ADVICE_EVERY)) {
    adviceKey = key;
    adviceAt = t;
    advice = GAMES[settings.game].advise(state, settings);
    renderAdvice();
  }
  draw();
}

function renderAdvice() {
  const game = GAMES[settings.game];
  $('action').textContent = advice.action;
  $('action').className = `action ${advice.tone}`;
  const lines = [...advice.lines];
  if (advice.tone === 'wait' && booted && !tracks.some(counts)) lines.push(game.hint);
  $('lines').replaceChildren(...lines.map((l) => h('li', {}, l)));

  const extras = [];
  if (settings.game === 'zolik') {
    extras.push(h('button', {
      type: 'button',
      class: `toggle${settings.opened ? ' on' : ''}`,
      onclick: () => { settings.opened = !settings.opened; saveSettings(); refresh(true); },
    }, settings.opened ? '✓ Už som vyložil' : 'Ešte som nevyložil'));
  }
  for (const m of manual) {
    extras.push(h('button', {
      type: 'button',
      class: `card-chip${isRed(m.code) ? ' red' : ''}${m.role === 'other' ? ' other' : ''}`,
      'aria-label': `Odobrať ${show(m.code)}`,
      onclick: () => { manual = manual.filter((x) => x !== m); refresh(true); },
    }, `${show(m.code)} ✕`));
  }
  $('extras').replaceChildren(...extras);
}

// ---------- drawing ----------

function roundRect(x, y, w, hgt, r) {
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, w, hgt, r);
  else g.rect(x, y, w, hgt);
}

// For žolík: which cards on screen sit in a meld and which one to throw away.
function zolikMarks(seen) {
  const marks = new Map();
  if (settings.game !== 'zolik' || !advice) return marks;
  const inMeld = (advice.melds ?? []).flatMap((m) => m.cards);
  let discardShown = false;
  for (const t of seen) {
    const i = inMeld.indexOf(t.code);
    if (i >= 0) { inMeld.splice(i, 1); marks.set(t, 'meld'); }
  }
  for (const t of seen) {
    if (!marks.has(t) && !discardShown && t.code === advice.discard) { marks.set(t, 'discard'); discardShown = true; }
  }
  return marks;
}

const COLORS = { me: '#ffd21f', other: '#4fd1ff', ignore: 'rgba(255,255,255,.55)', meld: '#3ee07a', discard: '#ff4d3d' };

function draw() {
  const W = overlay.clientWidth, H = overlay.clientHeight, dpr = window.devicePixelRatio || 1;
  if (overlay.width !== Math.round(W * dpr) || overlay.height !== Math.round(H * dpr)) {
    overlay.width = Math.round(W * dpr);
    overlay.height = Math.round(H * dpr);
  }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const fw = frame.width, fh = frame.height;
  if (!fw || !fh) return;

  const s = mode === 'photo' ? Math.min(W / fw, H / fh) : Math.max(W / fw, H / fh);
  const dx = (W - fw * s) / 2, dy = (H - fh * s) / 2;
  const toScreen = (b) => ({ x: b.x * s + dx, y: b.y * s + dy, w: b.w * s, h: b.h * s });

  const seen = tracks.filter(counts).sort((a, b) => a.first - b.first);
  const marks = zolikMarks(seen);
  const placed = [];
  const clash = (x, y, w, hgt) => placed.some((p) => x < p.x + p.w && p.x < x + w && y < p.y + p.h && p.y < y + hgt);

  for (const t of seen) {
    const role = roleOf(t.code);
    const mark = marks.get(t);
    const color = role === 'ignore' ? COLORS.ignore : COLORS[mark ?? role];

    // A thin frame around each corner index the model read.
    g.lineWidth = 2;
    g.setLineDash(role === 'ignore' ? [5, 5] : []);
    g.strokeStyle = color;
    for (const c of t.corners) {
      const b = toScreen(c.box);
      roundRect(b.x - 3, b.y - 3, b.w + 6, b.h + 6, 5);
      g.stroke();
    }
    g.setLineDash([]);

    // The label, stuck above the top-most corner.
    const top = t.corners.reduce((a, c) => (c.box.y < a.box.y ? c : a)).box;
    const b = toScreen(top);
    let text = show(t.code);
    if (role !== 'me') text += ` · ${roleName(role)}`;
    if (mark === 'discard') text += ' · ZAHOĎ';
    g.font = `800 17px ${labelFont}`;
    const sub = `${Math.round(t.score * 100)} %`;
    const tw = g.measureText(text).width + 18;
    g.font = `700 11px ${labelFont}`;
    const sw = g.measureText(sub).width + 10;
    const lw = tw + sw, lh = 28;
    const lx = Math.min(Math.max(6, b.x + b.w / 2 - lw / 2), W - lw - 6);
    // Labels of cards lying close together would cover each other: move up, then down.
    let ly = Math.max(6, b.y - lh - 8);
    for (let step = 1, start = ly; clash(lx, ly, lw, lh) && step < 12; step++) {
      ly = Math.max(6, start + (step % 2 ? -1 : 1) * Math.ceil(step / 2) * (lh + 4));
    }
    placed.push({ x: lx, y: ly, w: lw, h: lh });

    g.save();
    g.shadowColor = 'rgba(0,0,0,.45)';
    g.shadowBlur = 8;
    g.shadowOffsetY = 2;
    roundRect(lx, ly, lw, lh, 6);
    g.fillStyle = role === 'ignore' ? 'rgba(0,0,0,.6)' : color;
    g.fill();
    g.restore();
    g.textBaseline = 'middle';
    g.font = `800 17px ${labelFont}`;
    const dark = role === 'ignore' ? '#fff' : '#16161a';
    g.fillStyle = role !== 'ignore' && isRed(t.code) && !mark ? '#c9291d' : dark;
    g.fillText(text, lx + 9, ly + lh / 2 + 1);
    g.font = `700 11px ${labelFont}`;
    g.fillStyle = role === 'ignore' ? 'rgba(255,255,255,.8)' : 'rgba(22,22,26,.65)';
    g.fillText(sub, lx + tw, ly + lh / 2 + 1);

    t.hit = { x: Math.min(lx, b.x) - 8, y: ly - 8, r: Math.max(lx + lw, b.x + b.w) + 8, b: b.y + b.h + 12 };
    for (const c of t.corners) {
      const cb = toScreen(c.box);
      t.hit.x = Math.min(t.hit.x, cb.x - 10); t.hit.r = Math.max(t.hit.r, cb.x + cb.w + 10);
      t.hit.y = Math.min(t.hit.y, cb.y - 10); t.hit.b = Math.max(t.hit.b, cb.y + cb.h + 10);
    }
  }
}

function hitTest(px, py) {
  const under = tracks.filter((t) => counts(t) && t.hit && px >= t.hit.x && px <= t.hit.r && py >= t.hit.y && py <= t.hit.b);
  return under.sort((a, b) => (a.hit.r - a.hit.x) * (a.hit.b - a.hit.y) - (b.hit.r - b.hit.x) * (b.hit.b - b.hit.y))[0];
}

// ---------- live, frozen and photo views ----------

function setView() {
  const still = mode === 'photo' || frozen;
  frame.hidden = !still;
  frame.classList.toggle('contain', mode === 'photo');
  video.hidden = mode === 'photo';
  shutter.classList.toggle('still', still);
  shutter.setAttribute('aria-label', still ? 'Pokračovať' : 'Zastaviť obraz');
}

function freeze() {
  frozen = true;
  setView();
  refresh(true);
}

async function goLive() {
  frozen = false;
  if (mode === 'photo') {
    mode = 'live';
    tracks = [];
    setView();
    if (!stream) await startCamera().catch(cameraProblem);
  }
  setView();
}

async function usePhoto(file) {
  if (!file) return;
  try {
    const img = await createImageBitmap(file, { imageOrientation: 'from-image' });
    if (!booted) {
      showBoot('Pripravujem sa', '');
      await detector.load();
      booted = true;
      $('boot').hidden = true;
    }
    mode = 'photo';
    frozen = false;
    setView();
    grab(img, img.width, img.height);
    tracks = [];
    const found = await detector.corners(frame, frame.width, frame.height, settings.threshold);
    follow(detector.groupCards(found), performance.now());
    refresh(true);
    if (!tracks.length) toast('Na fotke som nenašiel žiadne karty.');
  } catch (err) {
    console.error(err);
    toast('Túto fotku sa nepodarilo otvoriť.');
  }
}

function newRound() {
  roles.clear();
  manual = [];
  tracks = [];
  if (settings.game === 'zolik') { settings.opened = false; saveSettings(); }
  refresh(true);
  toast('Nové kolo');
}

function setGame(game) {
  settings.game = game;
  saveSettings();
  for (const b of $('games').children) b.setAttribute('aria-pressed', b.dataset.game === game);
  refresh(true);
}

// ---------- sheets ----------

function openSheet(...content) {
  sheet.replaceChildren(...content.flat(Infinity).filter(Boolean));
  sheet.scrollTop = 0;
  sheet.inert = false;
  sheet.classList.add('open');
  scrim.classList.add('open');
}

function closeSheet() {
  sheet.inert = true;
  sheet.classList.remove('open');
  scrim.classList.remove('open');
}

const head = (title) => h('div', { class: 'sheet-head' }, h('h2', {}, title), h('button', { type: 'button', class: 'done', onclick: closeSheet }, 'Hotovo'));

function stepper(label, key, min, max, after = () => {}) {
  const value = h('output', { class: 'step-value' }, String(settings[key]));
  const set = (d) => {
    settings[key] = Math.min(max, Math.max(min, settings[key] + d));
    value.textContent = String(settings[key]);
    saveSettings();
    after();
    refresh(true);
  };
  return h('div', { class: 'step' },
    h('span', {}, label),
    h('div', { class: 'step-ctrl' },
      h('button', { type: 'button', 'aria-label': 'Menej', onclick: () => set(-1) }, '−'),
      value,
      h('button', { type: 'button', 'aria-label': 'Viac', onclick: () => set(1) }, '+')));
}

function check(label, key) {
  return h('label', { class: 'check' },
    h('input', {
      type: 'checkbox',
      checked: !!settings[key],
      onchange: (e) => { settings[key] = e.target.checked; saveSettings(); refresh(true); },
    }),
    h('span', {}, label));
}

function openSettings() {
  const range = h('input', {
    type: 'range', class: 'range', min: '0.25', max: '0.75', step: '0.05', value: String(settings.threshold),
    oninput: (e) => { settings.threshold = Number(e.target.value); saveSettings(); },
  });
  openSheet(
    head('Nastavenia'),
    h('h3', {}, 'Poker (Texas Hold\'em)'),
    stepper('Počet súperov', 'opponents', 1, 9),
    h('h3', {}, 'Blackjack'),
    check('Krupiér berie aj na mäkkých 17', 'hitSoft17'),
    h('h3', {}, 'Žolík'),
    stepper('Body na prvé vyloženie', 'openPoints', 0, 100),
    check('Už som vyložil', 'opened'),
    h('p', { class: 'fine' }, '2 balíčky + 4 žolíky. Postupka aspoň 3 karty jednej farby, skupina 3–4 rovnaké karty rôznych farieb. J, Q, K a A = 10 bodov (A v A-2-3 = 1). V každej kombinácii aspoň 2 skutočné karty.'),
    h('h3', {}, 'Rozpoznávanie'),
    h('label', { class: 'row' }, h('span', {}, 'Ako istý si musí byť, kým kartu ukáže'), range,
      h('div', { class: 'range-ends' }, h('span', {}, 'ukáže viac, aj omyly'), h('span', {}, 'len isté'))),
    h('p', { class: 'fine' }, `Model číta roh karty (číslo a znak), preto drž karty tak, aby boli rohy vidieť. Funguje s bežnými kartami s označením A, K, Q, J. Žolíky nerozozná, pridaj ich cez + Karta. Beží na ${detector.backendName() === 'webgpu' ? 'grafickom čipe' : 'procesore'} telefónu, nič sa neodosiela.`),
    h('button', { type: 'button', class: 'btn outline', onclick: () => { closeSheet(); $('photoInput').click(); } }, 'Rozpoznať fotku z galérie'),
  );
}

function openPicker() {
  const game = GAMES[settings.game];
  let role = 'me';
  const roleRow = game.other
    ? h('div', { class: 'seg' },
      ...['me', 'other'].map((r) => h('button', {
        type: 'button',
        'aria-pressed': String(r === role),
        onclick: (e) => {
          role = r;
          for (const b of e.currentTarget.parentElement.children) b.setAttribute('aria-pressed', String(b === e.currentTarget));
        },
      }, r === 'me' ? 'Moja karta' : game.other[0].toUpperCase() + game.other.slice(1))))
    : null;
  const add = (code) => {
    manual.push({ code, role });
    toast(`Pridaná ${show(code)}`);
    refresh(true);
  };
  const grid = SUITS.map((s) => h('div', { class: 'picker' },
    RANKS.map((r) => h('button', {
      type: 'button',
      class: isRed(r + s) ? 'red' : '',
      onclick: () => add(r + s),
    }, show(r + s)))));
  openSheet(
    head('Pridať kartu'),
    h('p', { class: 'fine' }, 'Pre karty, ktoré kamera nevidí alebo prečítala zle. Ťuknutím na kartu dole na obrazovke ju zase odoberieš.'),
    roleRow,
    grid,
    settings.game === 'zolik' ? h('button', { type: 'button', class: 'btn', onclick: () => add(JOKER) }, '+ Žolík') : null,
  );
}

// ---------- wiring ----------

overlay.addEventListener('click', (e) => {
  if (!booted) return;
  const t = hitTest(e.clientX, e.clientY);
  if (t) nextRole(t.code);
});
shutter.addEventListener('click', () => (mode === 'photo' || frozen ? goLive() : freeze()));
$('roundBtn').addEventListener('click', newRound);
$('addBtn').addEventListener('click', openPicker);
$('settingsBtn').addEventListener('click', openSettings);
for (const b of $('games').children) b.addEventListener('click', () => setGame(b.dataset.game));
scrim.addEventListener('click', closeSheet);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });
$('photoInput').addEventListener('change', (e) => { usePhoto(e.target.files[0]); e.target.value = ''; });
$('bootPhotoBtn').addEventListener('click', () => $('photoInput').click());
$('retryBtn').addEventListener('click', boot);
window.addEventListener('resize', () => draw());
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopCamera();
  else if (booted && mode === 'live') startCamera().catch(cameraProblem);
});

if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});

setGame(settings.game);
setView();
boot();
