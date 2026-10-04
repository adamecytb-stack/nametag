const SHELL_CACHE = 'nametag-shell-v1';
const MODEL_CACHE = 'nametag-models-v1'; // same name in app.js
const MODEL_HOSTS = ['cdn.jsdelivr.net', 'storage.googleapis.com'];
const SHELL = [
  './', 'index.html', 'styles.css', 'app.js', 'db.js', 'manifest.webmanifest',
  'icons/icon-180.png', 'icons/icon-192.png', 'icons/icon-512.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL_CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  if (MODEL_HOSTS.includes(url.hostname)) e.respondWith(cacheFirst(e.request));
  else if (url.origin === self.location.origin) e.respondWith(networkFirst(e.request));
});

// Model files are pinned to a version, so a cached copy never goes stale.
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
