// Everything the app remembers lives in IndexedDB on this device.
const NAME = 'nametag';
const VERSION = 1;
let opening;

function open() {
  opening ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(NAME, VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      d.createObjectStore('labels', { keyPath: 'id' });
      d.createObjectStore('sightings', { keyPath: 'id' }).createIndex('labelId', 'labelId');
      d.createObjectStore('places', { keyPath: 'id' });
      d.createObjectStore('settings');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return opening;
}

async function run(store, mode, work) {
  const d = await open();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(store, mode);
    const req = work(tx.objectStore(store));
    tx.oncomplete = () => resolve(req?.result);
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}

export const getAll = (store) => run(store, 'readonly', (s) => s.getAll());
export const get = (store, key) => run(store, 'readonly', (s) => s.get(key));
export const put = (store, value, key) => run(store, 'readwrite', (s) => s.put(value, key));
export const del = (store, key) => run(store, 'readwrite', (s) => s.delete(key));
export const clear = (store) => run(store, 'readwrite', (s) => s.clear());

export const sightingsFor = (labelId) =>
  run('sightings', 'readonly', (s) => s.index('labelId').getAll(labelId));

export const deleteSightingsFor = (labelId) =>
  run('sightings', 'readwrite', (s) => {
    const req = s.index('labelId').openCursor(IDBKeyRange.only(labelId));
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) { cursor.delete(); cursor.continue(); }
    };
  });
