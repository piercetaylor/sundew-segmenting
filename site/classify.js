// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// Sundew species model in the browser: onnxruntime-web + the export preprocessing.
//
// ES module, no build step. Serve it with pilresize.js, and model-int8.onnx +
// labels.json from the same origin (see docs/reports/species-release.md).
//
// Model contract (release/species-v1.0.0/release.json):
//   input  "pixels"  float32 [1,3,224,224], RGB, NCHW, values pixel/255 (normalisation is in the graph)
//   output "logits"  float32 [1,110], labels.json order
//   probabilities = softmax(logits / 0.74)
// Training/export preprocessing (PIL): short side -> 255 bicubic, centre crop 224.

import { preprocessPIL, cropToTensor, shortSideSize } from './pilresize.js';

export const ORT_VERSION = '1.30.0';
export const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const TEMPERATURE = 0.74;
export const MODEL_SHA256 = 'd7ee150eaf5d1c7ddae00a2a94e66ddcfd48b55166b940f4dfa35f0e3cc8af16';
export const SIZE = 224;
export const RESIZE = 255;
export const MAX_CANVAS_PIXELS = 16e6;
export const PRESHRINK = 1152;

/**
 * Load onnxruntime-web from the CDN. 'wasm' pulls the smaller WASM-only build;
 * 'webgpu' pulls the WebGPU build (which also contains WASM, used as fallback
 * for ops WebGPU does not implement, e.g. the int8 DynamicQuantizeLinear /
 * MatMulInteger nodes of this model).
 */
export async function loadOrt(backend = 'wasm', { cdn = ORT_CDN, numThreads } = {}) {
  const file = backend === 'webgpu' ? 'ort.webgpu.min.mjs' : 'ort.wasm.min.mjs';
  const ort = await import(/* webpackIgnore: true */ cdn + file);
  ort.env.wasm.wasmPaths = cdn;
  // Multi-threaded WASM needs SharedArrayBuffer, i.e. a cross-origin-isolated
  // page (COOP: same-origin + COEP: require-corp). Otherwise ORT uses 1 thread.
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  ort.env.wasm.numThreads = numThreads ?? (isolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);
  return ort;
}

async function sha256Hex(buf) {
  if (!(globalThis.crypto && crypto.subtle)) return null; // crypto.subtle needs a secure context (https or localhost)
  const d = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Fetch the model (with optional progress callback), check its sha256, create the session. */
export async function createClassifier({ ort, modelUrl, labelsUrl, backend = 'wasm', onProgress, checkSha = true }) {
  const t0 = performance.now();
  const [modelBuf, labels] = await Promise.all([fetchBytes(modelUrl, onProgress), fetch(labelsUrl).then((r) => { if (!r.ok) throw new Error(`${labelsUrl}: HTTP ${r.status}`); return r.json(); })]);
  const tFetch = performance.now();
  let sha = null;
  if (checkSha) {
    sha = await sha256Hex(modelBuf);
    if (sha && sha !== MODEL_SHA256) throw new Error(`model sha256 ${sha} does not match release.json`);
  }
  const executionProviders = backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  const session = await ort.InferenceSession.create(new Uint8Array(modelBuf), {
    executionProviders,
    graphOptimizationLevel: 'all',
  });
  const tSession = performance.now();
  return {
    ort, session, labels, backend,
    sha256: sha,
    timings: { fetch_ms: tFetch - t0, session_ms: tSession - tFetch, total_ms: tSession - t0 },
  };
}

async function fetchBytes(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!onProgress || !r.body) return r.arrayBuffer();
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
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

/**
 * Decode an image Blob/File. EXIF orientation is applied ('from-image'), as a
 * user expects; colour-space conversion is turned off to stay close to PIL,
 * which ignores embedded ICC profiles.
 */
export async function decodeBitmap(blob, { orientation = 'from-image' } = {}) {
  try {
    return await createImageBitmap(blob, {
      imageOrientation: orientation,
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    });
  } catch (e) {
    // Older engines reject unknown option values; fall back to the defaults.
    console.warn('createImageBitmap options rejected, using defaults:', e);
    return createImageBitmap(blob);
  }
}

/** Full-resolution RGBA pixels of a bitmap. */
export function bitmapPixels(bitmap) {
  const c = makeCanvas(bitmap.width, bitmap.height);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
}

/**
 * Preprocess a decoded bitmap into the model input.
 *   method 'pil'        : JS port of Pillow bicubic (matches training for the same decoded pixels)
 *   method 'pil-2step'  : JS Pillow LANCZOS to short side 576, then 'pil' (mimics the training cache)
 *   method 'canvas'     : browser drawImage with imageSmoothingQuality 'high' (fast, NOT bicubic-identical;
 *                         every browser uses its own downscaler)
 */
export function preprocess(bitmap, method = 'pil') {
  let w = bitmap.width, h = bitmap.height;
  if (method !== 'canvas' && w * h > MAX_CANVAS_PIXELS) {
    // Phone photos (12-50 MP) exceed iOS Safari's canvas limit (~16.7 MP) and cost
    // ~200 MB of RGBA. Let the browser shrink to short side 1152 (2 x 576) first;
    // iNaturalist originals (<= 2048 px) never take this branch.
    const [nw, nh] = shortSideSize(w, h, PRESHRINK);
    const c = makeCanvas(nw, nh);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, nw, nh);
    const rgba = ctx.getImageData(0, 0, nw, nh).data;
    return preprocessPIL(rgba, nw, nh, 4, { twoStep: method === 'pil-2step', resize: RESIZE, crop: SIZE });
  }
  if (method === 'canvas') {
    const [nw, nh] = shortSideSize(w, h, RESIZE);
    const c = makeCanvas(nw, nh);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, nw, nh);
    return cropToTensor(ctx.getImageData(0, 0, nw, nh).data, nw, nh, 4, SIZE);
  }
  const rgba = bitmapPixels(bitmap);
  return preprocessPIL(rgba, w, h, 4, { twoStep: method === 'pil-2step', resize: RESIZE, crop: SIZE });
}

/** Run the model on a preprocessed tensor. Returns raw logits (Float32Array, 110). */
export async function runLogits(clf, tensor) {
  const x = new clf.ort.Tensor('float32', tensor, [1, 3, SIZE, SIZE]);
  const out = await clf.session.run({ pixels: x });
  return out.logits.data;
}

export function softmaxT(logits, t = TEMPERATURE) {
  let m = -Infinity;
  for (const v of logits) m = Math.max(m, v / t);
  const e = Array.from(logits, (v) => Math.exp(v / t - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function topK(logits, labels, k = 5) {
  const p = softmaxT(logits);
  const idx = Array.from(p.keys()).sort((a, b) => logits[b] - logits[a] || a - b).slice(0, k);
  return idx.map((i) => ({ index: i, label: labels[i], prob: p[i] }));
}

/** One call: Blob/File -> top-5. */
export async function classifyBlob(clf, blob, { method = 'pil', k = 5 } = {}) {
  const t0 = performance.now();
  const bitmap = await decodeBitmap(blob);
  let tensor;
  try {
    ({ tensor } = preprocess(bitmap, method));
  } finally {
    bitmap.close?.();
  }
  const t1 = performance.now();
  const logits = await runLogits(clf, tensor);
  const t2 = performance.now();
  return { top: topK(logits, clf.labels, k), logits, timings: { preprocess_ms: t1 - t0, inference_ms: t2 - t1 } };
}
