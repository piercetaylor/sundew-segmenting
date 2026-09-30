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

It runs in about 120 ms on one CPU thread with ONNX Runtime (Python). Browser
speed and accuracy with onnxruntime-web have not been measured yet.

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
- **Preprocessing differences are not measured.** Training photos were first
  shrunk to a 576 px short side (LANCZOS), and EXIF orientation was ignored.
  Browsers rotate by EXIF and resize differently.
- **int8 results vary slightly by CPU.** Two server CPUs agreed on 99.1% of
  top-1 answers.
- It is an identification aid, not an authority. Identifications that matter,
  for conservation or law, need an expert.

## Recommended decision rule

*Added 2026-09-29; the weights and `release.json` are unchanged.*

Let `p` be the top-1 probability from `softmax(logits / 0.74)`. Compare the
unrounded value.

1. Always show the top 5 with whole-number percentages.
2. Name the species only when `p >= 0.65`. On validation that answers 77.8% of
   photos, and 90.1% of those answers are right (80.0% without the rule).
3. Otherwise say "not sure". If the summed probability of one section's
   species (`data/species-110-sections.json`) is at least 0.7, name the
   section: that covers about half of the "not sure" photos, 95.5% of them
   right.

The thresholds were picked on validation, which holds only the 110 species, so
they say nothing about photos of other plants. Species with under 40 training
photos get fewer and less accurate answers (68.5% answered, 81.9% right). Full
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
