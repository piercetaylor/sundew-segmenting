// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// Load the species model for the site: release.json check, Cache API, download
// with progress, onnxruntime-web session. Runs in the page or in js/worker.js;
// it uses only APIs that exist in both (fetch, caches, Blob, crypto.subtle).
//
// URLs are resolved against this file, so they are the same from either context
// (and the Cache API key is the same as before the worker existed).

import { loadOrt, createClassifier, ORT_VERSION, MODEL_SHA256 } from '../classify.js';
import { MODEL_SHA256 as DECISION_SHA256 } from './decision.js';

// Copied from release/species-v1.0.0/ at build time (.github/workflows/pages.yml).
const MODEL_DIR = new URL('../model/v1.0.0/', import.meta.url);
export const MODEL_URL = new URL('model-int8.onnx', MODEL_DIR).href;
const LABELS_URL = new URL('labels.json', MODEL_DIR).href;
const RELEASE_URL = new URL('release.json', MODEL_DIR).href;
export const CACHE_NAME = 'sundew-model-v1.0.0';
// onnxruntime-web 1.30.0 is self-hosted (copied from the npm package by pages.yml). The optional,
// unverified ?backend=webgpu build is not self-hosted and still loads from jsDelivr (ORT_CDN).
const ORT_DIR = new URL('../ort/1.30.0/', import.meta.url).href;

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

// The Cache API needs a secure context; on plain http (a LAN IP) it is missing, so every step may fail.
async function openCache() {
  try {
    const cache = await caches.open(CACHE_NAME);
    for (const k of await caches.keys()) if (k.startsWith('sundew-model-') && k !== CACHE_NAME) await caches.delete(k);
    return cache;
  } catch { return null; }
}

async function download(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body) return r.blob();
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total);
  }
  return new Blob(chunks, { type: 'application/octet-stream' });
}

// Model bytes from the Cache API if present, else the network. classify.js gets an object URL
// and still checks the sha256 (when crypto.subtle exists), so a bad cached copy is caught.
async function loadClassifier(ort, backend, onProgress) {
  const cache = await openCache();
  let cached = null;
  try { cached = cache && await cache.match(MODEL_URL); } catch { cached = null; }
  for (const fromCache of cached ? [true, false] : [false]) {
    const blob = fromCache ? await cached.blob() : await download(MODEL_URL, onProgress);
    const url = URL.createObjectURL(blob);
    try {
      const c = await createClassifier({ ort, modelUrl: url, labelsUrl: LABELS_URL, backend });
      if (!fromCache && cache) {
        try { await cache.put(MODEL_URL, new Response(blob, { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) { console.warn('model not cached:', e); }
      }
      c.fromCache = fromCache;
      return c;
    } catch (e) {
      // Only a failed sha256 check means the cached copy is bad; other errors (offline, labels) are rethrown.
      if (!fromCache || !/sha256/.test(e.message)) throw e;
      console.warn('cached model rejected, downloading again:', e);
      try { await cache.delete(MODEL_URL); } catch {}
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

/**
 * Check release.json against this code, load onnxruntime-web and the model.
 * onProgress(got, total) is called while the model downloads.
 * Returns the classifier from classify.js plus { fromCache, info } (info is plain data for the status line).
 */
export async function loadModel(backend, onProgress = () => {}) {
  const release = await fetchJson(RELEASE_URL);
  const sha = release.files?.['model-int8.onnx']?.sha256;
  if (sha !== MODEL_SHA256 || sha !== DECISION_SHA256) throw new Error('release.json does not match this page');
  const ort = await loadOrt(backend, backend === 'wasm' ? { cdn: ORT_DIR } : {});
  const clf = await loadClassifier(ort, backend, onProgress);
  clf.info = {
    version: release.version, ort: ORT_VERSION, backend, threads: ort.env.wasm.numThreads,
    loadMs: clf.timings.total_ms, fromCache: clf.fromCache,
  };
  return clf;
}
