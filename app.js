import * as db from './db.js';

const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const FACE = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15';
const GCS = 'https://storage.googleapis.com/mediapipe-models';
const MODEL_CACHE = 'nametag-models-v1'; // same name in sw.js

const FRAME_SIDE = 1280;         // longest side of the frame the models look at
const FACE_CROP = 416;           // longest side of the crop searched for a face
const MIN_FACE = 32;             // narrower faces are too blurry to tell apart
const MAX_PEOPLE = 4;
const MAX_THINGS = 5;
const MIN_TICK = 120;            // ms between looks, keeps the phone cool
const STEADY = 3;                // looks in a row before a sighting is logged
const SIGHTING_GAP = 2 * 60e3;   // out of view this long starts a new sighting
const PLACE_RADIUS = 150;        // metres
const MAX_EXAMPLES = 24;

const $ = (id) => document.getElementById(id);
const video = $('cam'), frame = $('frame'), overlay = $('overlay');
const sheet = $('sheet'), scrim = $('scrim'), shutter = $('shutter');
const fctx = frame.getContext('2d'), g = overlay.getContext('2d');
const crop = document.createElement('canvas'), cctx = crop.getContext('2d');
const labelFont = getComputedStyle(document.documentElement).getPropertyValue('--label-font').trim();

let detector, embedder, faceapi, cropFaces, frameFaces;
let labels = [], places = [], strictness = 0.5;
let mode = 'live';               // 'live' | 'photo'
let frozen = null;               // null | 'user' | 'tap'
let stream = null, booted = false;
let targets = [], ticking = Promise.resolve(), failures = 0;
let pos = null, placeOverride = '';
const live = new Map();          // label id -> { streak, sighting, saved }

// ---------- small helpers ----------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const uid = () => crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
const area = (b) => b.w * b.h;

function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') el.className = v;
    else if (k === 'value') el.value = v ?? '';
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false && kid !== '') el.append(kid);
  return el;
}

function iou(a, b) {
  const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
  const r = Math.min(a.x + a.w, b.x + b.w), bt = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, r - x) * Math.max(0, bt - y);
  return inter / (area(a) + area(b) - inter || 1);
}

function clip(b, w, hgt) {
  const x = Math.max(0, b.x), y = Math.max(0, b.y);
  return { x, y, w: Math.min(w, b.x + b.w) - x, h: Math.min(hgt, b.y + b.h) - y };
}

function when(t, midSentence = false) {
  const d = new Date(t), now = new Date();
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (t >= midnight) return `${midSentence ? 'today' : 'Today'} ${time}`;
  if (t >= midnight - 864e5) return `${midSentence ? 'yesterday' : 'Yesterday'} ${time}`;
  const year = d.getFullYear() === now.getFullYear() ? undefined : 'numeric';
  return `${d.toLocaleDateString([], { day: 'numeric', month: 'short', year })} ${time}`;
}

function listNames(ids) {
  const names = ids.map((id) => labels.find((l) => l.id === id)?.name).filter(Boolean);
  if (names.length < 2) return names[0] ?? '';
  return `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`;
}

let toastTimer;
function toast(text) {
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2400);
}

const pending = new Map();
let saveTimer;
function saveSoon(key, write) {
  pending.set(key, write);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flushSaves, 500);
}
function flushSaves() {
  for (const write of pending.values()) write();
  pending.clear();
}

// ---------- models ----------

async function loadModels() {
  const [mp, fa] = await Promise.all([
    import(`${MP}/vision_bundle.mjs`),
    import(`${FACE}/dist/face-api.esm.js`),
  ]);
  faceapi = fa;
  const files = await mp.FilesetResolver.forVisionTasks(`${MP}/wasm`);

  const things = (async () => {
    detector = await mp.ObjectDetector.createFromOptions(files, {
      baseOptions: { modelAssetPath: `${GCS}/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`, delegate: 'CPU' },
      runningMode: 'VIDEO',
      scoreThreshold: 0.4,
      maxResults: 10,
    });
    embedder = await mp.ImageEmbedder.createFromOptions(files, {
      baseOptions: { modelAssetPath: `${GCS}/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite`, delegate: 'CPU' },
      runningMode: 'IMAGE',
      l2Normalize: true,
    });
  })();

  const faces = (async () => {
    if (!(await fa.tf.setBackend('webgl').catch(() => false))) await fa.tf.setBackend('cpu');
    await fa.tf.ready();
    await Promise.all([
      fa.nets.tinyFaceDetector.loadFromUri(`${FACE}/model`),
      fa.nets.faceLandmark68Net.loadFromUri(`${FACE}/model`),
      fa.nets.faceRecognitionNet.loadFromUri(`${FACE}/model`),
    ]);
    cropFaces = new fa.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.45 });
    frameFaces = new fa.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.45 });
  })();

  await Promise.all([things, faces]);
}

// Copies every model file this launch used into the cache the service worker
// reads from, so the next launch works with no signal.
async function keepModelsOffline() {
  if (!('caches' in window)) return;
  const cache = await caches.open(MODEL_CACHE);
  const used = performance.getEntriesByType('resource').map((e) => e.name)
    .filter((u) => u.startsWith('https://cdn.jsdelivr.net/') || u.startsWith('https://storage.googleapis.com/'));
  for (const url of new Set(used)) {
    if (!(await cache.match(url))) await cache.add(url).catch(() => {});
  }
}

// ---------- camera ----------

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('insecure'), { name: 'InsecureError' });
  stopCamera();
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
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
    NotAllowedError: 'Camera access is off. Allow the camera for Nametag, then try again.',
    NotFoundError: 'No camera found on this device.',
    NotReadableError: 'Another app is using the camera. Close it, then try again.',
    InsecureError: 'The camera only works when the page is opened over https.',
  }[err?.name] ?? 'The camera did not start.';
  showBoot(why, 'A photo from your library works too.', true);
}

function showBoot(status, note = '', actions = false) {
  $('bootStatus').textContent = status;
  $('bootNote').textContent = note;
  $('bootNote').hidden = !note;
  $('bootActions').hidden = !actions;
  $('boot').hidden = false;
}

// ---------- looking at a frame ----------

function grab(source) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  const k = Math.min(1, FRAME_SIDE / Math.max(sw, sh));
  const w = Math.round(sw * k), hgt = Math.round(sh * k);
  if (frame.width !== w || frame.height !== hgt) { frame.width = w; frame.height = hgt; }
  fctx.drawImage(source, 0, 0, w, hgt);
}

// The head sits at the top of a person box, so only that part is searched.
// Searching a crop instead of the whole frame is what finds faces across a room.
async function faceIn(box) {
  const pad = box.w * 0.1;
  const x = Math.max(0, box.x - pad), y = Math.max(0, box.y - pad);
  const w = Math.min(frame.width - x, box.w + pad * 2);
  const hgt = Math.min(frame.height - y, box.h + pad, Math.max(box.w * 1.3, box.h * 0.5));
  const k = Math.min(1, FACE_CROP / Math.max(w, hgt));
  crop.width = Math.max(1, Math.round(w * k));
  crop.height = Math.max(1, Math.round(hgt * k));
  cctx.drawImage(frame, x, y, w, hgt, 0, 0, crop.width, crop.height);
  const found = await faceapi.detectSingleFace(crop, cropFaces).withFaceLandmarks().withFaceDescriptor();
  if (!found || found.detection.box.width < MIN_FACE) return null;
  const b = found.detection.box;
  return { box: { x: x + b.x / k, y: y + b.y / k, w: b.width / k, h: b.height / k }, descriptor: found.descriptor };
}

function embed(box) {
  const fw = frame.width, fh = frame.height;
  const res = embedder.embed(frame, {
    regionOfInterest: { left: box.x / fw, top: box.y / fh, right: (box.x + box.w) / fw, bottom: (box.y + box.h) / fh },
  });
  return Float32Array.from(res.embeddings[0].floatEmbedding);
}

function distance(a, b, kind) {
  let s = 0;
  if (kind === 'person') {
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return Math.sqrt(s);
  }
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return 1 - s;
}

function closest(label, sample) {
  let best = Infinity;
  for (const s of label.samples) best = Math.min(best, distance(sample, s, label.kind));
  return best;
}

function recognise(t) {
  t.label = null;
  let best = null;
  for (const label of labels) {
    if (label.kind !== t.kind) continue;
    const d = closest(label, t.sample);
    if (!best || d < best.d) best = { label, d };
  }
  if (!best) return;
  // Faces: 0.6 is the usual cut-off for this model, lower is stricter.
  let limit = t.kind === 'person' ? 0.6 - 0.2 * strictness : 0.4 - 0.3 * strictness;
  if (t.kind === 'thing' && best.label.type !== t.type) limit -= 0.08;
  if (best.d <= limit) { t.label = best.label; t.d = best.d; }
}

async function analyse(source) {
  if (source) grab(source);
  const fw = frame.width, fh = frame.height;
  const out = [];
  for (const d of detector.detectForVideo(frame, performance.now()).detections) {
    const c = d.categories[0], b = d.boundingBox;
    const box = clip({ x: b.originX, y: b.originY, w: b.width, h: b.height }, fw, fh);
    if (box.w < 12 || box.h < 12) continue;
    out.push({ kind: c.categoryName === 'person' ? 'person' : 'thing', type: c.categoryName, box, face: null, sample: null, label: null });
  }

  const people = out.filter((t) => t.kind === 'person').sort((a, b) => area(b.box) - area(a.box)).slice(0, MAX_PEOPLE);
  if (people.length) {
    for (const p of people) {
      const face = await faceIn(p.box);
      // Two people standing close can both frame the same face; the first box keeps it.
      if (face && !people.some((o) => o.face && iou(o.face.box, face.box) > 0.5)) p.face = face;
    }
  } else {
    const found = await faceapi.detectAllFaces(frame, frameFaces).withFaceLandmarks().withFaceDescriptors();
    for (const f of found.slice(0, MAX_PEOPLE)) {
      const b = f.detection.box;
      if (b.width < MIN_FACE) continue;
      const face = { box: { x: b.x, y: b.y, w: b.width, h: b.height }, descriptor: f.descriptor };
      const m = b.width * 0.25;
      out.push({ kind: 'person', type: 'person', box: clip({ x: b.x - m, y: b.y - m, w: b.width + m * 2, h: b.height + m * 2 }, fw, fh), face, sample: null, label: null });
    }
  }

  const knowsThings = labels.some((l) => l.kind === 'thing');
  let embedded = 0;
  for (const t of out) {
    if (t.kind === 'person') t.sample = t.face?.descriptor ?? null;
    else if (knowsThings && embedded++ < MAX_THINGS) t.sample = embed(t.box);
    if (t.sample) recognise(t);
  }
  // One name per frame: the closer match keeps it.
  for (const t of out) {
    if (t.label && out.some((o) => o !== t && o.label === t.label && o.d < t.d)) t.label = null;
  }

  targets = out;
  if (mode === 'live') logSightings();
}

async function run() {
  for (;;) {
    await new Promise(requestAnimationFrame);
    if (mode !== 'live' || frozen || !stream || video.readyState < 2) continue;
    const started = performance.now();
    ticking = analyse(video).then(() => { failures = 0; draw(); }, tickFailed);
    await ticking;
    const spare = MIN_TICK - (performance.now() - started);
    if (spare > 0) await sleep(spare);
  }
}

function tickFailed(err) {
  console.error(err);
  if (++failures === 5) showBoot('Recognition stopped working.', 'Close Nametag and open it again.');
}

// ---------- drawing ----------

function roundRect(x, y, w, hgt, r) {
  g.beginPath();
  if (g.roundRect) g.roundRect(x, y, w, hgt, r);
  else g.rect(x, y, w, hgt);
}

function tapeText(t) {
  if (t.label) return t.label.name;
  if (t.kind === 'thing') return t.type;
  return t.face ? 'who?' : 'person';
}

function draw() {
  const W = overlay.clientWidth, H = overlay.clientHeight, dpr = window.devicePixelRatio || 1;
  if (overlay.width !== Math.round(W * dpr) || overlay.height !== Math.round(H * dpr)) {
    overlay.width = Math.round(W * dpr);
    overlay.height = Math.round(H * dpr);
  }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  updateHint();
  const fw = frame.width, fh = frame.height;
  if (!fw || !fh) return;

  const s = mode === 'photo' ? Math.min(W / fw, H / fh) : Math.max(W / fw, H / fh);
  const dx = (W - fw * s) / 2, dy = (H - fh * s) / 2;
  const top = $('top').getBoundingClientRect().bottom + 8;
  const bottom = $('bottom').getBoundingClientRect().top + 24;

  for (const t of [...targets].sort((a, b) => area(b.box) - area(a.box))) {
    const x = Math.max(6, t.box.x * s + dx), y = Math.max(6, t.box.y * s + dy);
    const r = Math.min(W - 6, (t.box.x + t.box.w) * s + dx), b = Math.min(H - 6, (t.box.y + t.box.h) * s + dy);
    if (r - x < 16 || b - y < 16) { t.hit = t.tape = null; continue; }
    const known = !!t.label, person = t.kind === 'person';

    roundRect(x, y, r - x, b - y, 12);
    g.setLineDash([]);
    g.lineWidth = 5;
    g.strokeStyle = 'rgba(0,0,0,.28)';
    g.stroke();
    g.setLineDash(known ? [] : [7, 7]);
    g.lineWidth = 2.5;
    g.strokeStyle = known ? (person ? '#ffd21f' : '#f4f4ef') : 'rgba(255,255,255,.8)';
    g.stroke();
    g.setLineDash([]);

    // The tape label, stuck slightly crooked across the box's top edge.
    const text = tapeText(t);
    g.font = `800 16px ${labelFont}`;
    const tw = Math.min(g.measureText(text).width + 20, W - 16), th = 30;
    const tx = Math.min(Math.max(8, x + 8), W - tw - 8);
    const ty = Math.min(Math.max(top, y - th / 2), bottom - th);
    let tilt = 0;
    for (const ch of text) tilt = (tilt * 31 + ch.charCodeAt(0)) % 5;
    g.save();
    g.translate(tx, ty + th / 2);
    g.rotate((tilt - 2) * 0.012);
    g.shadowColor = 'rgba(0,0,0,.4)';
    g.shadowBlur = 8;
    g.shadowOffsetY = 2;
    roundRect(0, -th / 2, tw, th, 5);
    g.fillStyle = known ? (person ? '#ffd21f' : '#f4f4ef') : 'rgba(0,0,0,.58)';
    g.fill();
    g.shadowColor = 'transparent';
    if (!known) { g.lineWidth = 1; g.strokeStyle = 'rgba(255,255,255,.55)'; g.stroke(); }
    g.fillStyle = known ? '#16161a' : '#fff';
    g.textBaseline = 'middle';
    g.fillText(text, 10, 1, tw - 20);
    g.restore();

    t.hit = { x, y, r, b };
    t.tape = { x: tx, y: ty, r: tx + tw, b: ty + th };
  }
}

function updateHint() {
  const show = booted && !labels.length && !sheet.classList.contains('open') && $('boot').hidden;
  $('hint').hidden = !show;
  if (show) $('hint').textContent = targets.length ? 'Tap anything to give it a name' : 'Point at a person or a thing';
}

// A tap on a tape label picks that label; otherwise the smallest box under the finger.
function hitTest(px, py) {
  const under = (r, pad) => r && px >= r.x - pad && px <= r.r + pad && py >= r.y - pad && py <= r.b + pad;
  const smallest = (list) => list.sort((a, b) => area(a.box) - area(b.box))[0] ?? null;
  return smallest(targets.filter((t) => under(t.tape, 6))) ?? smallest(targets.filter((t) => under(t.hit, 10)));
}

// ---------- live, frozen and photo views ----------

function setView() {
  const still = mode === 'photo' || !!frozen;
  frame.hidden = !still;
  frame.classList.toggle('contain', mode === 'photo');
  shutter.classList.toggle('still', still);
  shutter.setAttribute('aria-label', still ? 'Back to the live camera' : 'Freeze the picture');
}

// Stops on the frame the models last looked at, so boxes hold still for tapping.
async function freeze(by, tapped = null) {
  frozen = by;
  await ticking;
  setView();
  draw();
  if (!tapped) return null;
  const again = targets.filter((t) => t.kind === tapped.kind).sort((a, b) => iou(b.box, tapped.box) - iou(a.box, tapped.box))[0];
  return again && iou(again.box, tapped.box) > 0.3 ? again : null;
}

async function goLive() {
  mode = 'live';
  frozen = null;
  setView();
  if (!stream) {
    targets = [];
    draw();
    try { await startCamera(); $('boot').hidden = true; } catch (err) { cameraProblem(err); }
  }
}

async function usePhoto(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  try { await img.decode(); } catch { toast('That photo would not open'); URL.revokeObjectURL(url); return; }
  mode = 'photo';
  frozen = null;
  await ticking;
  stopCamera();
  ticking = analyse(img).catch(tickFailed);
  await ticking;
  URL.revokeObjectURL(url);
  $('boot').hidden = true;
  setView();
  draw();
  if (!targets.length) toast('Nothing recognisable in this photo');
}

// ---------- sightings ----------

function track(id) {
  if (!live.has(id)) live.set(id, { streak: 0, sighting: null, saved: 0 });
  return live.get(id);
}

function logSightings() {
  const seen = new Set(targets.filter((t) => t.label).map((t) => t.label));
  for (const [id, s] of live) if (![...seen].some((l) => l.id === id)) s.streak = 0;
  for (const label of seen) track(label.id).streak++;
  // A name has to hold for a few looks in a row, so one wrong guess logs nothing.
  const steady = [...seen].filter((l) => track(l.id).streak >= STEADY);
  for (const label of steady) touchSighting(label, steady.filter((o) => o !== label).map((o) => o.id));
}

function touchSighting(label, withIds = []) {
  const now = Date.now(), s = track(label.id), here = placeName();
  let sg = s.sighting, dirty = false;
  if (!sg || now - sg.tEnd > SIGHTING_GAP) {
    sg = s.sighting = { id: uid(), labelId: label.id, t: now, tEnd: now, place: here, lat: pos?.lat ?? null, lon: pos?.lon ?? null, withIds: [], note: '' };
    label.seenCount = (label.seenCount || 0) + 1;
    dirty = true;
  }
  for (const id of withIds) if (!sg.withIds.includes(id)) { sg.withIds.push(id); dirty = true; }
  if (!sg.place && here) { sg.place = here; dirty = true; }
  if (sg.lat == null && pos) { sg.lat = pos.lat; sg.lon = pos.lon; dirty = true; }
  sg.tEnd = now;
  if (dirty || now - s.saved > 15e3) {
    s.saved = now;
    label.lastSeen = now;
    label.lastPlace = sg.place;
    db.put('sightings', sg).catch(console.error);
    db.put('labels', label).catch(console.error);
  }
  return sg;
}

async function dropSighting(label, sg) {
  const s = track(label.id);
  s.sighting = null;
  s.streak = 0;
  await db.del('sightings', sg.id);
  const rest = (await db.sightingsFor(label.id)).sort((a, b) => b.tEnd - a.tEnd);
  label.seenCount = rest.length;
  label.lastSeen = rest[0]?.tEnd ?? null;
  label.lastPlace = rest[0]?.place ?? '';
  await db.put('labels', label);
}

// ---------- places ----------

function metresBetween(a, b) {
  const rad = Math.PI / 180, R = 6371000;
  const dLat = (b.lat - a.lat) * rad, dLon = (b.lon - a.lon) * rad;
  const x = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(x));
}

function nearestPlace() {
  if (!pos) return null;
  let best = null;
  for (const p of places) {
    const m = metresBetween(pos, p);
    if (m <= PLACE_RADIUS && (!best || m < best.m)) best = { p, m };
  }
  return best?.p ?? null;
}

const placeName = () => placeOverride || nearestPlace()?.name || '';

function refreshPlace() {
  const name = placeName();
  $('placeText').textContent = name || 'Name this place';
  $('placeBtn').classList.toggle('named', !!name);
}

function watchPlace() {
  navigator.geolocation?.watchPosition(
    (p) => { pos = { lat: p.coords.latitude, lon: p.coords.longitude }; refreshPlace(); },
    () => {},
    { enableHighAccuracy: true, maximumAge: 30e3 },
  );
}

// ---------- sheets ----------

function openSheet(...content) {
  flushSaves();
  sheet.replaceChildren(...content.flat(Infinity).filter(Boolean));
  sheet.scrollTop = 0;
  sheet.inert = false;
  sheet.classList.add('open');
  scrim.classList.add('open');
  updateHint();
}

function closeSheet() {
  if (!sheet.classList.contains('open')) return;
  document.activeElement?.blur?.();
  flushSaves();
  sheet.inert = true;
  sheet.classList.remove('open');
  scrim.classList.remove('open');
  $('savedCount').textContent = labels.length;
  if (frozen === 'tap') { frozen = null; setView(); }
  draw();
}

const doneBtn = () => h('button', { class: 'done', type: 'button', onclick: closeSheet }, 'Done');
const head = (title) => h('div', { class: 'sheet-head' }, h('h2', {}, title), doneBtn());
const field = (name, control) => h('label', { class: 'row' }, h('span', {}, name), control);
const chip = (label, onclick) => h('button', { class: `chip ${label.kind}`, type: 'button', onclick }, label.name);

function thumbOf(t) {
  const b = t.face?.box ?? t.box;
  const side = Math.min(frame.width, frame.height, Math.max(b.w, b.h) * (t.face ? 1.7 : 1.1));
  const x = Math.min(frame.width - side, Math.max(0, b.x + b.w / 2 - side / 2));
  const y = Math.min(frame.height - side, Math.max(0, b.y + b.h / 2 - side / 2));
  const c = document.createElement('canvas');
  c.width = c.height = 160;
  c.getContext('2d').drawImage(frame, x, y, side, side, 0, 0, 160, 160);
  return c.toDataURL('image/jpeg', 0.82);
}

// Something the camera found that has no name yet.
function openName(target) {
  const person = target.kind === 'person';
  if (person && !target.sample) {
    openSheet(head('Who is this?'), h('p', { class: 'body' }, 'No face is visible in this shot. Get their face in view, then tap them again.'));
    return;
  }
  if (!person) target.sample ??= embed(target.box);

  const input = h('input', {
    class: `tape-input ${person ? 'yellow' : 'white'}`,
    placeholder: person ? 'Their name' : `Name this ${target.type}`,
    'aria-label': 'Name', autocomplete: 'off', autocapitalize: 'words', enterkeyhint: 'done',
  });
  const save = (e) => {
    e.preventDefault();
    const name = input.value.trim();
    if (name) nameTarget(target, name); else input.focus();
  };
  const known = labels.filter((l) => l.kind === target.kind)
    .sort((a, b) => closest(a, target.sample) - closest(b, target.sample)).slice(0, 4);

  openSheet(h('form', { onsubmit: save },
    h('div', { class: 'who' }, h('img', { class: 'thumb', src: thumbOf(target), alt: '' }), input, doneBtn()),
    h('button', { class: 'btn', type: 'submit' }, 'Save name'),
    known.length > 0 && [
      h('h3', {}, person ? 'Or someone you already saved' : 'Or something you already saved'),
      h('div', { class: 'chips' }, known.map((l) => chip(l, () => addExample(l, target)))),
    ],
  ));
}

async function nameTarget(target, name) {
  const same = labels.find((l) => l.kind === target.kind && l.name.toLowerCase() === name.toLowerCase());
  if (same) return addExample(same, target);
  const label = {
    id: uid(), kind: target.kind, type: target.type, name, about: '',
    samples: [target.sample], thumb: thumbOf(target),
    created: Date.now(), seenCount: 0, lastSeen: null, lastPlace: '',
  };
  labels.push(label);
  target.label = label;
  target.taught = true;
  await db.put('labels', label);
  toast(`Saved ${name}`);
  openLabel(label, target);
}

async function addExample(label, target) {
  label.samples.push(target.sample);
  if (label.samples.length > MAX_EXAMPLES) label.samples.splice(1, 1);
  target.label = label;
  target.taught = true;
  await db.put('labels', label);
  toast(`Added another example of ${label.name}`);
  openLabel(label, target);
}

function seenLine(label) {
  if (!label.lastSeen) return 'Not seen by the live camera yet';
  const times = label.seenCount === 1 ? 'once' : `${label.seenCount} times`;
  return `Seen ${times}, last ${when(label.lastSeen, true)}${label.lastPlace ? ` at ${label.lastPlace}` : ''}`;
}

function sightingRow(sg) {
  const who = listNames(sg.withIds);
  return h('li', { class: 'seen' },
    h('p', { class: 'seen-when' }, `${when(sg.t)}${sg.place ? ` at ${sg.place}` : ''}`),
    who && h('p', { class: 'seen-more' }, `With ${who}`),
    sg.note && h('p', { class: 'seen-more' }, sg.note),
    sg.lat != null && h('a', { class: 'link', href: `https://maps.apple.com/?q=${sg.lat},${sg.lon}`, target: '_blank', rel: 'noopener' }, 'Show on map'),
  );
}

// A saved name: what it is, where it was last seen, and who with.
async function openLabel(label, target = null) {
  const person = label.kind === 'person';
  const others = targets.filter((t) => t.label && t.label !== label).map((t) => t.label.id);
  const now = target && mode === 'live' ? touchSighting(label, others) : null;
  const past = (await db.sightingsFor(label.id)).filter((s) => s.id !== now?.id).sort((a, b) => b.t - a.t);
  const saveLabel = () => saveSoon(label.id, () => db.put('labels', label));

  const summary = h('p', { class: 'summary' }, seenLine(label));
  const name = h('input', { class: `tape-input ${person ? 'yellow' : 'white'}`, value: label.name, 'aria-label': 'Name', autocomplete: 'off', autocapitalize: 'words', enterkeyhint: 'done' });
  name.addEventListener('input', () => {
    if (!name.value.trim()) return;
    label.name = name.value.trim();
    saveLabel();
  });

  const about = h('textarea', { class: 'field', rows: 3, value: label.about, placeholder: person ? 'Who they are, how you know them' : 'Whose it is, where it belongs' });
  about.addEventListener('input', () => { label.about = about.value; saveLabel(); });

  let nowSection = null;
  if (now) {
    const saveNow = () => saveSoon(now.id, () => { db.put('sightings', now); db.put('labels', label); });
    const where = h('input', { class: 'field', value: now.place, placeholder: 'School, home, the gym', autocomplete: 'off' });
    where.addEventListener('input', () => {
      now.place = label.lastPlace = where.value.trim();
      summary.textContent = seenLine(label);
      saveNow();
    });
    const note = h('input', { class: 'field', value: now.note, placeholder: person ? 'What they were doing, who else was there' : 'Where exactly, who had it', autocomplete: 'off' });
    note.addEventListener('input', () => { now.note = note.value.trim(); saveNow(); });
    const company = now.withIds.map((id) => labels.find((l) => l.id === id)).filter(Boolean);
    nowSection = [
      h('h3', {}, 'Seen just now'),
      field('Where', where),
      company.length > 0 && h('div', { class: 'row' }, h('span', {}, 'With'), h('div', { class: 'chips' }, company.map((l) => chip(l, () => openLabel(l))))),
      field('Note', note),
    ];
  }

  const remove = h('button', { class: 'btn danger', type: 'button' }, `Remove ${label.name}`);
  remove.addEventListener('click', async () => {
    if (!remove.dataset.sure) {
      remove.dataset.sure = '1';
      remove.textContent = `Tap again to remove ${label.name} and every sighting`;
      return;
    }
    labels = labels.filter((l) => l !== label);
    live.delete(label.id);
    for (const t of targets) if (t.label === label) t.label = null;
    await db.del('labels', label.id);
    await db.deleteSightingsFor(label.id);
    toast(`Removed ${label.name}`);
    closeSheet();
  });

  const examples = label.samples.length === 1 ? '1 example' : `${label.samples.length} examples`;
  openSheet(
    h('div', { class: 'who' }, h('img', { class: 'thumb', src: label.thumb, alt: '' }), name, doneBtn()),
    summary,
    nowSection,
    h('h3', {}, person ? 'About them' : 'About it'),
    about,
    h('h3', {}, now ? 'Seen before' : 'Seen'),
    past.length
      ? h('ul', { class: 'seen-list' }, past.slice(0, 40).map(sightingRow))
      : h('p', { class: 'fine' }, now ? 'This is the first time.' : 'Sightings are added whenever the live camera recognises this name.'),
    target?.sample && !target.taught && h('button', { class: 'btn outline', type: 'button', onclick: () => addExample(label, target) }, 'Add this view as another example'),
    target && !target.taught && h('button', { class: 'btn outline', type: 'button', onclick: async () => {
      if (now) await dropSighting(label, now);
      target.label = null;
      openName(target);
    } }, `This isn't ${label.name}`),
    h('p', { class: 'fine' }, `${examples} saved. More angles and lighting make ${label.name} easier to recognise.`),
    remove,
  );
}

function openSaved() {
  const list = h('ul', { class: 'saved' });
  const empty = h('p', { class: 'body' });
  const search = h('input', { class: 'field', type: 'search', placeholder: 'Find a name', 'aria-label': 'Find a name', autocomplete: 'off' });
  const fill = () => {
    const q = search.value.trim().toLowerCase();
    const rows = labels
      .filter((l) => !q || l.name.toLowerCase().includes(q) || (l.about || '').toLowerCase().includes(q))
      .sort((a, b) => (b.lastSeen ?? b.created) - (a.lastSeen ?? a.created));
    list.replaceChildren(...rows.map((l) => h('li', {},
      h('button', { class: 'saved-row', type: 'button', onclick: () => openLabel(l) },
        h('img', { class: 'thumb', src: l.thumb, alt: '' }),
        h('span', { class: 'saved-text' },
          h('span', { class: `chip ${l.kind}` }, l.name),
          h('span', { class: 'seen-more' }, l.lastSeen ? `${when(l.lastSeen)}${l.lastPlace ? ` at ${l.lastPlace}` : ''}` : 'Not seen yet'))))));
    empty.hidden = rows.length > 0;
    empty.textContent = labels.length ? 'No saved name matches that.' : 'Nothing saved yet. Tap a person or a thing in the camera view and give it a name.';
  };
  search.addEventListener('input', fill);
  fill();
  openSheet(head('Saved'), labels.length > 5 && search, empty, list);
}

function openPlace() {
  const near = nearestPlace();
  const input = h('input', { class: 'tape-input white', value: placeName(), placeholder: 'School, home, the gym', 'aria-label': 'Place name', autocomplete: 'off', autocapitalize: 'words', enterkeyhint: 'done' });
  const save = async (e) => {
    e.preventDefault();
    const name = input.value.trim();
    if (!name) { input.focus(); return; }
    if (pos) {
      const p = near ?? { id: uid(), lat: pos.lat, lon: pos.lon };
      p.name = name;
      if (!near) places.push(p);
      await db.put('places', p);
      placeOverride = '';
    } else placeOverride = name;
    refreshPlace();
    closeSheet();
    toast(`Saved ${name}`);
  };
  openSheet(h('form', { onsubmit: save },
    head('Where are you?'),
    input,
    h('p', { class: 'fine' }, pos
      ? 'Saved with your location. It fills in by itself whenever you are back within about 150 metres.'
      : 'Location is off, so this name is kept only until you close Nametag. Allow location to have places fill in by themselves.'),
    h('button', { class: 'btn', type: 'submit' }, 'Save place'),
    near && h('button', { class: 'btn danger', type: 'button', onclick: async () => {
      places = places.filter((p) => p !== near);
      await db.del('places', near.id);
      refreshPlace();
      closeSheet();
      toast(`Removed ${near.name}`);
    } }, `Remove ${near.name}`),
  ));
}

function openSettings() {
  const range = h('input', { class: 'range', type: 'range', min: 0, max: 1, step: 0.05, value: strictness, 'aria-label': 'How sure it has to be' });
  range.addEventListener('input', () => {
    strictness = Number(range.value);
    saveSoon('strictness', () => db.put('settings', strictness, 'strictness'));
  });

  const erase = h('button', { class: 'btn danger', type: 'button' }, 'Erase everything');
  erase.addEventListener('click', async () => {
    if (!erase.dataset.sure) {
      erase.dataset.sure = '1';
      erase.textContent = 'Tap again to erase every name, sighting and place';
      return;
    }
    await Promise.all(['labels', 'sightings', 'places'].map(db.clear));
    labels = [];
    places = [];
    live.clear();
    for (const t of targets) t.label = null;
    refreshPlace();
    closeSheet();
    toast('Erased everything');
  });

  openSheet(
    head('Settings'),
    h('h3', {}, 'How sure it has to be'),
    range,
    h('div', { class: 'range-ends' }, h('span', {}, 'Guesses more'), h('span', {}, 'Stays quiet unless sure')),
    h('p', { class: 'fine' }, 'Move it right if wrong names show up. Move it left if it keeps asking who someone is.'),
    h('h3', {}, 'Backup'),
    h('p', { class: 'fine' }, 'Names, examples and sightings are stored only on this phone. Nothing is uploaded. A backup file keeps them safe or moves them to another phone.'),
    h('button', { class: 'btn outline', type: 'button', onclick: saveBackup }, 'Save a backup file'),
    h('button', { class: 'btn outline', type: 'button', onclick: () => $('backupInput').click() }, 'Restore a backup file'),
    erase,
  );
}

// ---------- backup ----------

async function saveBackup() {
  const data = {
    app: 'nametag', version: 1, saved: Date.now(), strictness, places,
    labels: labels.map((l) => ({ ...l, samples: l.samples.map((s) => Array.from(s)) })),
    sightings: await db.getAll('sightings'),
  };
  const file = new File([JSON.stringify(data)], `nametag-backup-${new Date().toISOString().slice(0, 10)}.json`, { type: 'application/json' });
  if (navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file] }); return; } catch (err) { if (err.name === 'AbortError') return; }
  }
  const a = h('a', { href: URL.createObjectURL(file), download: file.name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10e3);
}

async function restoreBackup(file) {
  let data;
  try { data = JSON.parse(await file.text()); } catch { data = null; }
  if (data?.app !== 'nametag' || !Array.isArray(data.labels)) { toast('That file is not a Nametag backup'); return; }
  for (const l of data.labels) await db.put('labels', { ...l, samples: l.samples.map((s) => Float32Array.from(s)) });
  for (const s of data.sightings ?? []) await db.put('sightings', s);
  for (const p of data.places ?? []) await db.put('places', p);
  [labels, places] = await Promise.all([db.getAll('labels'), db.getAll('places')]);
  refreshPlace();
  closeSheet();
  toast(`Restored ${labels.length === 1 ? '1 name' : `${labels.length} names`}`);
}

// ---------- wiring ----------

overlay.addEventListener('click', async (e) => {
  let target = hitTest(e.clientX, e.clientY);
  if (!target) return;
  if (mode === 'live' && !frozen) {
    target = await freeze('tap', target);
    if (!target) { frozen = null; setView(); toast('Lost it. Tap again.'); return; }
  }
  if (target.label) openLabel(target.label, target); else openName(target);
});

shutter.addEventListener('click', () => (mode === 'photo' || frozen ? goLive() : freeze('user')));
$('savedBtn').addEventListener('click', openSaved);
$('placeBtn').addEventListener('click', openPlace);
$('settingsBtn').addEventListener('click', openSettings);
$('photoBtn').addEventListener('click', () => $('photoInput').click());
$('bootPhotoBtn').addEventListener('click', () => $('photoInput').click());
$('retryBtn').addEventListener('click', () => (detector && faceapi ? goLive() : location.reload()));
$('photoInput').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) usePhoto(f); });
$('backupInput').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) restoreBackup(f); });
scrim.addEventListener('click', closeSheet);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });
window.addEventListener('resize', draw);
window.addEventListener('pagehide', flushSaves);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { flushSaves(); stopCamera(); }
  else if (booted && mode === 'live') startCamera().then(() => { $('boot').hidden = true; }, cameraProblem);
});

async function boot() {
  navigator.serviceWorker?.register('sw.js').catch(() => {});
  navigator.storage?.persist?.().catch(() => {});
  try {
    [labels, places, strictness = 0.5] = await Promise.all([db.getAll('labels'), db.getAll('places'), db.get('settings', 'strictness')]);
  } catch {
    showBoot('This browser is blocking storage.', 'Nametag needs it to remember names. Turn off private browsing and open it again.');
    return;
  }
  $('savedCount').textContent = labels.length;
  refreshPlace();

  const camera = startCamera().then(() => null, (err) => err);
  try {
    await loadModels();
  } catch (err) {
    console.error(err);
    showBoot('The recognition models did not download.', 'Check your connection, then try again.', true);
    $('bootPhotoBtn').hidden = true;
    return;
  }
  booted = true;
  const cameraErr = await camera;
  if (cameraErr) cameraProblem(cameraErr); else $('boot').hidden = true;
  watchPlace();
  keepModelsOffline();
  run();
}

boot();
