# Sundew species model, v1.0.0

Names the species of a sundew (*Drosera*) from a photo: one of 110 species. It
is a DINOv2-S image classifier distilled from a DINOv2-L teacher, exported to
int8 ONNX. It is part of
[sundew-segmenting](https://github.com/piercetaylor/sundew-segmenting); how it
was built is in [docs/species-classifier.md](../../docs/species-classifier.md).

| File | What it is |
| --- | --- |
| `model-int8.onnx` | the model, 22 MB, opset 17 |
| `labels.json` | the 110 species names, in output order |
| `release.json` | input spec, temperature, checksums, metrics |
| `LICENSE` | CC BY-NC 4.0, the licence for these weights |
| `ATTRIBUTION.md` | credit for the 23,244 photos it was trained on |

## Using it

Input `pixels`: float32, 1 x 3 x 224 x 224, RGB, values in [0, 1] (pixel / 255).
The ImageNet normalisation is inside the model, so don't apply it yourself.

1. Resize the whole photo so its short side is 255 px (bicubic), then take the
   centre 224 x 224 px. Use the whole photo; do not crop to the plant first.
2. Run the model. Output `logits`: 1 x 110, in `labels.json` order.
3. Probabilities are `softmax(logits / 0.74)`. The temperature was fitted on
   validation, because the raw model is under-confident.

It runs in about 120 ms on one CPU thread with ONNX Runtime (Python). In a
browser (onnxruntime-web 1.30.0, WebAssembly, one thread, headless Chrome on one
server core) the model takes about 525 ms, and a whole photo (decode, resize,
model) 0.56 s at 576 px or 0.69 s for an iNaturalist original (up to 2048 px).
On 330 validation photos the browser gave the same top-1 species as Python for
99.1% of originals ([details](../../docs/reports/species-browser.md)). The web page is in [`site/`](../../site/).

## How well it works

Balanced accuracy is the mean of per-species accuracy, so every species counts
equally.

| | Balanced accuracy | Top-5 |
| --- | ---: | ---: |
| Validation (3,959 photos) | 0.764 | 0.957 |
| Held-out test (2,601 photos, scored once) | 0.775 [0.752, 0.794] | 0.962 |

On test, species with under 40 training photos score 0.652, against 0.837 for
the 42 commonest.

## Limits

- **It always names one of the 110 species.** It has no "not a sundew" or
  "unknown" answer, so a photo of another plant still gets a sundew name. Show
  the top few answers with their probabilities, and treat a low top probability
  as "not sure" (see the rule below).
- **Rare species are the weak spot** (see above).
- **Preprocessing differs slightly from training.** Training photos were first
  shrunk to a 576 px short side (LANCZOS), and EXIF orientation was ignored.
  The web page reproduces the bicubic resize exactly, but rotates by EXIF.
- **int8 results vary slightly by CPU.** Two server CPUs agreed on 99.1% of
  top-1 answers.
- It is an identification aid, not an authority. Identifications that matter,
  for conservation or law, need an expert.

## Recommended decision rule

*Added 2026-09-29, changed 2026-09-30 after an open-set test; the weights and
`release.json` are unchanged.*

From the logits, with T = 0.74: `p` is the top-1 of `softmax(logits / T)`,
and the energy is `T * logsumexp(logits / T)` (how strongly the photo matches
any of the 110 species). Compare unrounded values.

1. Always show the top 5 with whole-number percentages (`p` and the rest).
2. Name the species only when `p >= 0.65` **and** energy `>= 5.955`. On
   validation that answers 71.6% of photos, 91.5% of them right (80.0% right
   without a rule).
3. Otherwise say "not sure". If the summed probability of one section's
   species (`data/species-110-sections.json`) is at least 0.7, name the
   section: that covers 58.8% of the "not sure" photos, 93.6% of them right.

The energy condition was added because `p >= 0.65` alone gave a confident
sundew name to about 1 in 5 photos of other plants (research-grade
iNaturalist photos of other carnivorous genera and other plants); with both
conditions it is about 1 in 10, and about 1 in 3 for *Drosera* species outside
the 110 ([open-set test](../../docs/reports/species-open-set.md),
[rule change](../../docs/reports/species-open-set-fallback.md)). Species with
under 40 training photos get fewer answers (66.2%, 82.2% of them right). Full
tables: [docs/reports/species-abstain.md](../../docs/reports/species-abstain.md).
Reference code: `src/sundew_segmentation/species_decision.py` and
`site/js/decision.js`.

## Licence and credit

The weights are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (`LICENSE`):
share and adapt them with credit, not for commercial use. Most training photos
are CC BY-NC, and iNaturalist's terms prohibit using its data for commercial AI
training.

- **Training photos:** from iNaturalist contributors, credited in `ATTRIBUTION.md`.
- **Base model:** DINOv2 (Meta AI) via timm, Apache-2.0; its licence text is in
  the repository's [LICENSE](../../LICENSE).
- **Code:** Apache-2.0, repository root.

Cite the repository with the commit used (`CITATION.cff`).
