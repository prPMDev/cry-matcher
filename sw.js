// Cry Matcher service worker: keeps the app files on the phone so the page works offline.
// Your clips are not here. They are in the page's own storage (IndexedDB) on this phone.
const CACHE = "cry-matcher-v5";
const FILES = ["./", "index.html", "engine.js", "tf.min.js", "yamnet-weights.js", "manifest.webmanifest",
  "icon-192.png", "icon-512.png", "icon-maskable-512.png"];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET" || new URL(e.request.url).origin !== location.origin) return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request).then((res) => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
    return res;
  })));
});
