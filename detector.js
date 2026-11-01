// Finds playing cards in a picture.
//
// The model (YOLOv8n, see models/README.md) does not look for whole cards: it
// finds the small rank-and-suit index printed in a card's corners, which stays
// visible even when cards overlap in a fanned hand. A card usually shows two
// corners, so detections of the same card are grouped afterwards.

export const ORT_VERSION = '1.31.1';
const ORT = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const MODEL_URL = 'models/cards.onnx';
const SIZE = 640;            // the model's input is SIZE x SIZE
const ANCHORS = 8400;        // candidate boxes the model scores
const OVERLAP = 0.5;         // boxes overlapping more than this are the same corner
const CARD_REACH = 8;        // two corners of one card are at most ~8 index heights apart

export const LABELS = [
  '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S',
  '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S',
  '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S',
  'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS',
  'QC', 'QD', 'QH', 'QS',
];

let ort, session, backend = '';
const input = document.createElement('canvas');
input.width = input.height = SIZE;
const ictx = input.getContext('2d', { willReadFrequently: true });
const pixels = new Float32Array(3 * SIZE * SIZE);

export const backendName = () => backend;

// Safe to call more than once: later calls wait for the first one.
let loading = null;
export function load() {
  loading ??= start().catch((err) => { loading = null; throw err; });
  return loading;
}

async function start() {
  ort = await import(`${ORT}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT;
  // Threads need a cross-origin isolated page, which GitHub Pages cannot provide.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const model = await (await fetch(MODEL_URL)).arrayBuffer();

  // The graphics chip is much faster; fall back to the processor if it is not available.
  const tries = 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];
  for (const ep of tries) {
    try {
      session = await ort.InferenceSession.create(model, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      await session.run({ images: new ort.Tensor('float32', pixels, [1, 3, SIZE, SIZE]) }); // warm up
      backend = ep;
      return;
    } catch (err) {
      console.warn(`${ep} did not work`, err);
      session = null;
    }
  }
  throw new Error('model');
}

// Copies `source` (video, image or canvas, `w` x `h`) into the model's square
// input, scaled to fit with grey bars, and returns the scale used.
function prepare(source, w, h) {
  const k = SIZE / Math.max(w, h);
  ictx.fillStyle = 'rgb(114,114,114)';
  ictx.fillRect(0, 0, SIZE, SIZE);
  ictx.drawImage(source, 0, 0, w, h, 0, 0, Math.round(w * k), Math.round(h * k));
  const rgba = ictx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  for (let i = 0, p = 0; p < plane; i += 4, p++) {
    pixels[p] = rgba[i] / 255;
    pixels[p + plane] = rgba[i + 1] / 255;
    pixels[p + 2 * plane] = rgba[i + 2] / 255;
  }
  return k;
}

function iou(a, b) {
  const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
  const r = Math.min(a.x + a.w, b.x + b.w), bt = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, r - x) * Math.max(0, bt - y);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}

// Returns the card corners found: [{ code, score, box: {x, y, w, h} }] in source pixels.
export async function corners(source, w, h, threshold = 0.45) {
  const k = prepare(source, w, h);
  const out = await session.run({ images: new ort.Tensor('float32', pixels, [1, 3, SIZE, SIZE]) });
  const data = out.output0.data; // [1, 4 + classes, ANCHORS], one row per value
  const classes = LABELS.length;

  const found = [];
  for (let i = 0; i < ANCHORS; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < classes; c++) {
      const s = data[(4 + c) * ANCHORS + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < threshold) continue;
    const cx = data[i] / k, cy = data[ANCHORS + i] / k;
    const bw = data[2 * ANCHORS + i] / k, bh = data[3 * ANCHORS + i] / k;
    found.push({ code: LABELS[cls], score: best, box: { x: cx - bw / 2, y: cy - bh / 2, w: bw, h: bh } });
  }

  // Keep the strongest of overlapping boxes, whatever card they think it is.
  found.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of found) if (!kept.some((o) => iou(o.box, f.box) > OVERLAP)) kept.push(f);
  return kept;
}

// Groups corners that belong to one physical card: same card name and close
// enough to be two corners of it. Returns [{ code, score, box, corners }].
export function groupCards(found) {
  const groups = [];
  for (const f of found) {
    const cx = f.box.x + f.box.w / 2, cy = f.box.y + f.box.h / 2;
    const g = groups.find((gr) => gr.code === f.code && gr.corners.some((o) => {
      const reach = CARD_REACH * Math.max(o.box.h, f.box.h, o.box.w, f.box.w);
      return Math.hypot(o.box.x + o.box.w / 2 - cx, o.box.y + o.box.h / 2 - cy) < reach;
    }));
    if (g) g.corners.push(f);
    else groups.push({ code: f.code, corners: [f] });
  }
  return groups.map((g) => {
    const x = Math.min(...g.corners.map((c) => c.box.x));
    const y = Math.min(...g.corners.map((c) => c.box.y));
    const r = Math.max(...g.corners.map((c) => c.box.x + c.box.w));
    const b = Math.max(...g.corners.map((c) => c.box.y + c.box.h));
    return { code: g.code, score: Math.max(...g.corners.map((c) => c.score)), box: { x, y, w: r - x, h: b - y }, corners: g.corners };
  });
}
