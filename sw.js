const SHELL_CACHE = 'karty-shell-v1';
const MODEL_CACHE = 'karty-models-v1'; // same name in app.js
const MODEL_HOSTS = ['cdn.jsdelivr.net'];
const SHELL = [
  './', 'index.html', 'styles.css', 'app.js', 'detector.js', 'cards.js',
  'games/blackjack.js', 'games/poker.js', 'games/zolik.js', 'manifest.webmanifest',
  'icons/icon-180.png', 'icons/icon-192.png', 'icons/icon-512.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL_CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

// Drop caches from older versions (including the old Nametag app).
self.addEventListener('activate', (e) => e.waitUntil((async () => {
  for (const key of await caches.keys()) if (key !== SHELL_CACHE && key !== MODEL_CACHE) await caches.delete(key);
  await self.clients.claim();
})()));

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  if (MODEL_HOSTS.includes(url.hostname) || url.pathname.endsWith('.onnx')) e.respondWith(cacheFirst(e.request));
  else if (url.origin === self.location.origin) e.respondWith(networkFirst(e.request));
});

// Libraries are pinned to a version and the model file is large, so a cached
// copy is used as is. A new model needs a new file name or MODEL_CACHE name.
async function cacheFirst(request) {
  const cache = await caches.open(MODEL_CACHE);
  const hit = await cache.match(request.url);
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok) cache.put(request.url, res.clone());
  return res;
}

// The app's own files: fresh when online, cached copy when not.
async function networkFirst(request) {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const res = await fetch(request);
    if (res.ok) cache.put(request, res.clone());
    return res;
  } catch {
    return (await cache.match(request, { ignoreSearch: true })) ?? Response.error();
  }
}
