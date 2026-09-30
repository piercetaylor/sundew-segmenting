# Species model in the browser: parity and speed

*Measured 2026-09-29 on the cluster; no phone results yet.*

The web page ([`site/`](../../site/)) runs `release/species-v1.0.0/model-int8.onnx`
with onnxruntime-web 1.30.0 (WebAssembly). It reproduces the training
preprocessing in JavaScript: a port of Pillow's bicubic resize to a 255 px short
side, then a centre crop to 224. For identical decoded pixels it produces the
same bytes as Python on all 990 crops tested. Unlike training, it applies EXIF
orientation.

## Sample

The sample is 330 validation photos, 3 per species for all 110 species (seed
20260929); the test split was not read. The photos go through two paths: `s576`
(the training cache, short side 576) and `original` (iNaturalist originals, at
most 2048 px). "Agreement" means the browser and Python gave the same top-1
species. The reference is Python ONNX Runtime 1.30.0 on an Intel Xeon Gold 6252.

| Run | s576 agreement | original agreement | Balanced accuracy on sample (s576 / original) |
| --- | ---: | ---: | --- |
| Python ORT int8, Intel (reference) | 1 | 1 | 0.767 / 0.767 |
| Python ORT int8, AMD EPYC, same pixels | 0.979 | n/a | 0.767 / n/a |
| Node.js onnxruntime-web WASM, same pixels | 0.982 | 0.991 | 0.758 / 0.758 |
| Headless Chrome 154, WASM, JS resize, browser JPEG decode | 0.982 | 0.991 | 0.758 / 0.758 |
| Headless Chrome 154, WASM, the browser's own canvas resize (110 photos) | 0.955 | 0.955 | 0.727 / 0.727 |

The remaining differences come from the int8 kernels, not the preprocessing.
Headless Chrome's logits equal Node's to within 5e-6. On all 3,959 validation
photos, with identical input tensors, WASM agreed with Intel ORT on 98.9% of
photos and with AMD ORT on 98.7%. Intel and AMD agreed with each other on
99.1%. Balanced accuracy was 0.760 in WASM, against 0.764 (Intel), 0.763 (AMD)
and 0.763 (fp32). The canvas resize costs accuracy and saves only 20-120 ms, so
the site uses the JS resize.

## Speed (one Xeon core, WebAssembly, one thread)

- Model alone: 525 ms (median).
- Whole photo (decode, resize, model): 0.56 s for s576 and 0.69 s for
  originals. Native ONNX Runtime takes 118 ms, so WASM is about 4.5x slower.
- Page ready 1.0-1.3 s after opening from localhost; 0.6-0.8 s when the model
  comes from the Cache API.
- Two WASM threads (needs COOP/COEP headers, which GitHub Pages does not send):
  model 285 ms.

## Site checks (headless Chrome 154)

- **Large photos:** 30 originals upscaled to 20 MP JPEGs take the pre-shrink
  branch without error. Top-1 matched the direct path on 27 of 30; every
  disagreement was a near tie (top probability 0.13-0.49, all below the "not
  sure" threshold).
- **EXIF orientation:** 30 originals saved with EXIF Orientation 6 decode
  upright. Top-1 matches Python on the `exif_transpose`d image for all 30.
- **Hosting:** the self-hosted runtime makes no third-party requests, and a
  file the browser can't decode gets a "use a JPEG" message.
- **Web Worker:** the model loads and runs in a module worker
  (`site/js/worker.js`), so the page keeps responding while a photo is
  identified. On a 20 MP photo, the longest main-thread gap was 107 ms with the
  worker and 923 ms without it (`?worker=0`). Both paths gave the same top-5,
  and warm timings per photo are the same. Browsers without module workers or
  a 2D `OffscreenCanvas` (Safari before 16.4) run the model in the page as
  before. Over plain http on a LAN IP (no Cache API, no `crypto.subtle`), the
  worker still loads the model and names the photo.
- **Before every deploy:** `pages.yml` loads the built site in Chrome, runs a
  synthetic photo through it and runs `site/test/privacy-check.mjs`: a result
  with five names, and only same-origin GETs made before the photo was chosen
  (worker requests included).

## Out-of-list photos

On the 20 photos of other things in [the out-of-list check](species-out-of-list-check.md),
the browser and Python ONNX Runtime differed by up to 10 points in top-1
probability (a mug: 0.695 in Python, 0.595 in the browser, 0.757 in fp32). The
browser's decoded pixels and 224 px input match PIL's exactly on all 20 photos,
including two Adobe RGB JPEGs. The whole difference comes from the int8 kernels:
the logits differ by at most 0.29, with a mean absolute difference of 0.039.
That is the same spread as on the 330 validation photos (at most 0.43, mean
0.040), and neither int8 build is closer to fp32. So a photo near the 0.65
threshold can fall on either side depending on the device. Like training, the
page reads JPEG colour values without colour management
(`colorSpaceConversion: 'none'` = PIL `convert("RGB")`), which also holds for
iPhone Display P3 photos, except in Safari, which is untested.

## Not yet measured

- Phones (iPhone Safari, Android Chrome) and desktop Firefox and Safari.
- Real HEIC photos.
- WebGPU.
