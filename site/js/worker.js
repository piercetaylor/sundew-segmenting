// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// Module worker that decodes, resizes and classifies photos, so the page stays
// responsive while the model runs (about 0.5-3 s per photo on one thread).
// Started by js/engine.js; the same code runs in the page if workers can't be used.
//
// Messages in:  {type: 'init', backend} | {type: 'classify', id, file}
// Messages out: {type: 'hello', ok, reason?} | {type: 'progress', got, total}
//               | {type: 'ready', labels, info} | {type: 'result', id, logits, timings}
//               | {type: 'error', id?, name, message}

import { classifyBlob } from '../classify.js';
import { loadModel } from './loader.js';

let clf = null;

// The pil path reads pixels through a 2D OffscreenCanvas (Safari 16.4+, Firefox 105+, Chrome 69+).
function canDecodeHere() {
  if (typeof createImageBitmap !== 'function') return 'no createImageBitmap';
  if (typeof OffscreenCanvas === 'undefined') return 'no OffscreenCanvas';
  try { if (!new OffscreenCanvas(1, 1).getContext('2d')) return 'no 2d OffscreenCanvas'; } catch (e) { return String(e); }
  return null;
}

const errorMessage = (e, id) => ({ type: 'error', id, name: e?.name ?? 'Error', message: e?.message ?? String(e) });

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    try {
      clf = await loadModel(data.backend, (got, total) => self.postMessage({ type: 'progress', got, total }));
      self.postMessage({ type: 'ready', labels: clf.labels, info: clf.info });
    } catch (e) {
      console.error(e);
      self.postMessage(errorMessage(e));
    }
  } else if (data.type === 'classify') {
    try {
      const r = await classifyBlob(clf, data.file, { method: 'pil' });
      const logits = Float32Array.from(r.logits);
      self.postMessage({ type: 'result', id: data.id, logits, timings: r.timings }, [logits.buffer]);
    } catch (e) {
      console.error(e);
      self.postMessage(errorMessage(e, data.id));
    }
  }
};

const reason = canDecodeHere();
self.postMessage({ type: 'hello', ok: !reason, reason });
