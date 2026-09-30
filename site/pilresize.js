// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
// Port of Pillow's resampler; Pillow's licence (MIT-CMU) is in THIRD_PARTY_NOTICES.md.
//
// Pillow-compatible image resampling in plain JS (no DOM), so the browser can
// reproduce the training preprocessing pixel for pixel on the same decoded RGB.
//
// Port of Pillow's src/libImaging/Resample.c (8-bit path): separable
// convolution, horizontal pass then vertical pass, antialiased (the filter
// support is stretched by the downscale factor), coefficients normalised and
// converted to 22-bit fixed point, each pass rounded and clipped to uint8.
// This is what torchvision's Resize(..., BICUBIC) calls on a PIL image, and
// what make_full_cache.py's LANCZOS pre-shrink used.
//
// Checked to give the same bytes as Pillow 12.3 on validation photos.

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 1 << PRECISION_BITS;
const HALF = 1 << (PRECISION_BITS - 1);
const MAX = ONE * 255;

function sinc(x) {
  if (x === 0.0) return 1.0;
  x *= Math.PI;
  return Math.sin(x) / x;
}

export const FILTERS = {
  bicubic: {
    support: 2.0,
    fn(x) {
      const a = -0.5;
      if (x < 0.0) x = -x;
      if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
      if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
      return 0.0;
    },
  },
  lanczos: {
    support: 3.0,
    fn(x) {
      if (-3.0 <= x && x < 3.0) return sinc(x) * sinc(x / 3);
      return 0.0;
    },
  },
};

function precomputeCoeffs(inSize, outSize, filter) {
  const scale = inSize / outSize;
  const filterscale = scale < 1.0 ? 1.0 : scale;
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const tmp = new Float64Array(ksize);
  const ss = 1.0 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0.0;
    for (let x = 0; x < xmax; x++) {
      const w = filter.fn((x + xmin - center + 0.5) * ss);
      tmp[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0.0 ? tmp[x] / ww : tmp[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, kk, bounds };
}

function clip8(v) {
  if (v >= MAX) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

/**
 * Resize an interleaved 8-bit image the way Pillow's Image.resize does.
 * src: Uint8Array/Uint8ClampedArray, `channels` values per pixel (3 = RGB,
 * 4 = RGBA from canvas getImageData; alpha is dropped). Returns RGB
 * (3 channels) Uint8Array of dw x dh.
 */
export function resizePIL(src, sw, sh, channels, dw, dh, filterName = 'bicubic') {
  const filter = FILTERS[filterName];
  if (!filter) throw new Error(`unknown filter ${filterName}`);
  // Normalise to packed RGB first.
  let cur = src;
  if (channels !== 3) {
    cur = new Uint8Array(sw * sh * 3);
    for (let i = 0, j = 0; i < sw * sh; i++, j += channels) {
      cur[i * 3] = src[j]; cur[i * 3 + 1] = src[j + 1]; cur[i * 3 + 2] = src[j + 2];
    }
  }
  let w = sw, h = sh;
  const needH = dw !== sw;
  const needV = dh !== sh;
  let vert = null, yFirst = 0, yLast = sh;
  if (needV) {
    vert = precomputeCoeffs(sh, dh, filter);
    // Only the rows the vertical pass reads need the horizontal pass (as Pillow does).
    yFirst = vert.bounds[0];
    yLast = vert.bounds[(dh - 1) * 2] + vert.bounds[(dh - 1) * 2 + 1];
  }
  if (needH) {
    const { ksize, kk, bounds } = precomputeCoeffs(sw, dw, filter);
    const rows = yLast - yFirst;
    const out = new Uint8Array(dw * rows * 3);
    for (let y = 0; y < rows; y++) {
      const rowIn = (y + yFirst) * sw * 3;
      const rowOut = y * dw * 3;
      for (let xx = 0; xx < dw; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], k0 = xx * ksize;
        let s0 = HALF, s1 = HALF, s2 = HALF;
        for (let x = 0; x < xmax; x++) {
          const k = kk[k0 + x], p = rowIn + (xmin + x) * 3;
          s0 += cur[p] * k; s1 += cur[p + 1] * k; s2 += cur[p + 2] * k;
        }
        const o = rowOut + xx * 3;
        out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
      }
    }
    cur = out; w = dw; h = rows;
  } else if (needV && (yFirst !== 0 || yLast !== sh)) {
    cur = cur.subarray(yFirst * sw * 3, yLast * sw * 3);
    h = yLast - yFirst;
  }
  if (needV) {
    const { ksize, kk, bounds } = vert;
    const out = new Uint8Array(w * dh * 3);
    for (let yy = 0; yy < dh; yy++) {
      const ymin = bounds[yy * 2] - yFirst, ymax = bounds[yy * 2 + 1], k0 = yy * ksize;
      const rowOut = yy * w * 3;
      for (let xx = 0; xx < w; xx++) {
        let s0 = HALF, s1 = HALF, s2 = HALF;
        for (let y = 0; y < ymax; y++) {
          const k = kk[k0 + y], p = ((ymin + y) * w + xx) * 3;
          s0 += cur[p] * k; s1 += cur[p + 1] * k; s2 += cur[p + 2] * k;
        }
        const o = rowOut + xx * 3;
        out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
      }
    }
    cur = out; h = dh;
  }
  if (cur === src) cur = Uint8Array.from(src); // no-op resize of RGB input: return a copy
  return cur;
}

/** Python 3 round() (half to even), as torchvision's CenterCrop and make_full_cache.py use. */
export function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** torchvision Resize(size) on the short side: returns [newW, newH]. */
export function shortSideSize(w, h, size) {
  if (w <= h) return [size, Math.trunc((size * h) / w)];
  return [Math.trunc((size * w) / h), size];
}

/**
 * The export preprocessing on decoded RGB(A):
 *   optional pre-shrink to short side 576 with LANCZOS (the training cache),
 *   Resize(255, bicubic), CenterCrop(224), /255, NCHW float32.
 * Returns { tensor: Float32Array(3*224*224), crop: Uint8Array RGB 224x224 }.
 */
export function preprocessPIL(src, w, h, channels, { twoStep = false, resize = 255, crop = 224 } = {}) {
  let img = src, cw = w, ch = h, cc = channels;
  if (twoStep) {
    const sc = 576 / Math.min(w, h);
    const tw = pyRound(w * sc), th = pyRound(h * sc);
    img = resizePIL(img, cw, ch, cc, tw, th, 'lanczos');
    cw = tw; ch = th; cc = 3;
  }
  const [nw, nh] = shortSideSize(cw, ch, resize);
  const r = resizePIL(img, cw, ch, cc, nw, nh, 'bicubic');
  return cropToTensor(r, nw, nh, 3, crop);
}

/** CenterCrop(crop) as torchvision, then ToTensor (RGB / 255, NCHW). */
export function cropToTensor(rgb, w, h, channels, crop = 224) {
  const top = pyRound((h - crop) / 2), left = pyRound((w - crop) / 2);
  const plane = crop * crop;
  const tensor = new Float32Array(3 * plane);
  const out = new Uint8Array(plane * 3);
  for (let y = 0; y < crop; y++) {
    for (let x = 0; x < crop; x++) {
      const p = ((y + top) * w + (x + left)) * channels;
      const i = y * crop + x;
      for (let c = 0; c < 3; c++) {
        const v = rgb[p + c];
        tensor[c * plane + i] = v / 255;
        out[i * 3 + c] = v;
      }
    }
  }
  return { tensor, crop: out };
}
