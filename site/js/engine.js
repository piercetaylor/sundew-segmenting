// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// The model for index.html: in a module worker (js/worker.js) when the browser
// can decode photos there, otherwise in the page. Same classify.js either way.
//
//   const engine = await createEngine('wasm', (got, total) => ...);
//   engine.labels, engine.info, engine.inWorker
//   const { logits, timings } = await engine.classify(file);

import { classifyBlob } from '../classify.js';
import { loadModel } from './loader.js';

const HELLO_TIMEOUT_MS = 20000;

const rebuildError = ({ name, message }) => Object.assign(new Error(message), { name });

// Resolves with a started worker, or null if module workers or a 2D OffscreenCanvas are missing.
function startWorker() {
  if (typeof Worker === 'undefined') return Promise.resolve(null);
  let w;
  try {
    w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  } catch (e) {
    console.warn('no module worker, running in the page:', e);
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    const done = (ok, why) => {
      clearTimeout(timer);
      w.onmessage = w.onerror = null;
      if (ok) return resolve(w);
      console.warn('worker not usable, running in the page:', why);
      w.terminate();
      resolve(null);
    };
    // Engines without module workers load worker.js as a classic script and fail on `import`.
    const timer = setTimeout(() => done(false, 'no answer'), HELLO_TIMEOUT_MS);
    w.onerror = (e) => { e.preventDefault?.(); done(false, e.message || 'error event'); };
    w.onmessage = ({ data }) => { if (data.type === 'hello') done(data.ok, data.reason); };
  });
}

async function workerEngine(w, backend, onProgress) {
  const pending = new Map();
  let nextId = 0;
  let dead = null;
  const ready = new Promise((resolve, reject) => {
    w.onmessage = ({ data }) => {
      if (data.type === 'progress') onProgress(data.got, data.total);
      else if (data.type === 'ready') resolve(data);
      else if (data.type === 'result' || (data.type === 'error' && data.id != null)) {
        const p = pending.get(data.id);
        pending.delete(data.id);
        if (data.type === 'result') p?.resolve({ logits: data.logits, timings: data.timings });
        else p?.reject(rebuildError(data));
      } else if (data.type === 'error') reject(rebuildError(data));
    };
    // A worker that dies (e.g. the browser runs out of memory) fails whatever is waiting.
    w.onerror = (e) => {
      e.preventDefault?.();
      dead = new Error(`the model stopped running${e.message ? ': ' + e.message : ''}`);
      reject(dead);
      for (const p of pending.values()) p.reject(dead);
      pending.clear();
    };
  });
  w.postMessage({ type: 'init', backend });
  const { labels, info } = await ready;
  return {
    labels, info, inWorker: true,
    classify(file) {
      if (dead) return Promise.reject(dead);
      const id = nextId++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        w.postMessage({ type: 'classify', id, file });
      });
    },
  };
}

async function pageEngine(backend, onProgress) {
  const clf = await loadModel(backend, onProgress);
  return {
    labels: clf.labels, info: clf.info, inWorker: false,
    async classify(file) {
      const r = await classifyBlob(clf, file, { method: 'pil' });
      return { logits: r.logits, timings: r.timings };
    },
  };
}

/** Load the model; `?worker=0` (read by index.html) forces the in-page path. */
export async function createEngine(backend, onProgress, { worker = true } = {}) {
  const w = worker ? await startWorker() : null;
  return w ? workerEngine(w, backend, onProgress) : pageEngine(backend, onProgress);
}
