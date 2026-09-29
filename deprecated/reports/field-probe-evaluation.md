# Step 1: the first field number

The `combined` checkpoint that ran the 50-image scrape probe
(`models/recipe/combined/seed-17`, curated validation IoU 0.6838) was scored
against the 47 masks that came back from annotation, split 30/17 by
`docs/field-probe-return.md`. The curated test split was not touched.

Reproduce with `sbatch scripts/hellbender_field_probe_eval.slurm` (CPU only —
`evaluate_checkpoint.py` never moves the model to CUDA; ~2 minutes on `general`).

## The gap

| | Images | Mean IoU | Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curated validation (reference) | 19 | **0.683** | — | — | — |
| `field-eval` (held out) | 17 | **0.499** | 0.634 | 0.680 | 0.690 |
| `field-train` (unseen at time of measurement) | 30 | **0.444** | 0.581 | 0.630 | 0.655 |

**−0.184 IoU.** For scale, the entire SegFormer-vs-U-Net architecture effect was
+0.022 and the entire `combined` recipe win was +0.051. Distribution shift is
roughly four times the largest lever this project has previously found, and it
runs the other way.

It is not a tail effect. On `field-eval` only 4 of 17 images reach curated-like
quality (IoU >= 0.65); 6 of 17 fall below 0.50. Across both splits, 25 of 47 are
below 0.50 and 9 are hard failures below 0.30.

The `field-train` number is descriptive and expires the moment those 30 images
are trained on. It is recorded here because it will never be measurable again.

## Percent cover, which is the actual deliverable

The project's research question is projected area, so relative area error
matters more than IoU. Comparing predicted foreground fraction against the
annotated fraction, at the default 0.5 threshold:

| | Signed area error | Mean abs | Median abs |
| --- | ---: | ---: | ---: |
| Curated validation (from `data-sufficiency.md`) | +14.0% | 21.7% | 17.9% |
| `field-eval` | **+33.9%** | 65.1% | **42.1%** |
| `field-train` | +78.9% | 113.6% | 38.9% |

Read the median column; the means are inflated by images where the true plant is
tiny, which makes relative error explode. Median relative area error roughly
**doubles** on field images, and the known over-prediction bias more than
doubles.

This also puts a caveat on the threshold-0.70 calibration proposed in
`data-sufficiency.md`, which removed the +14% bias almost exactly. That
threshold was fitted on 19 curated validation images against a +14% bias. The
field bias is +34%, so 0.70 should not be assumed to transfer. There is now a
held-out field set on which that can be checked properly.

## What the failures look like

Sorted worst-first in `data/reports/field-probe-evaluation/*-overlays.jpg`. The
sub-0.50 images do not share one bias; they split into two opposite ones.

| Mode | n | Signature |
| --- | ---: | --- |
| Over-inclusive | 7 | low precision, high recall — mask spreads onto surrounding grass, dead leaves and litter |
| Under-segmenting | 8 | high precision, low recall — `inat_472599992` is P=1.00 R=0.08, `inat_330483307` is P=1.00 R=0.18 |
| Both / collapsed | 10 | mask and truth substantially disagree in both directions |

The scrape probe anticipated the over-inclusive mode. The under-segmenting mode
is new, and it is the larger group. Because the two modes cancel in aggregate,
neither is visible in the mean precision and recall, which are both ~0.68.

Worth noting against the original plan: `inat_686282548`, pinned to `field-eval`
by the handoff doc as the named tape-measure failure to be fixed, scores 0.509 —
unremarkable. The catastrophic tape-measure image is `inat_592910855`, a
different *D. capillaris* frame that also landed in `field-eval`. The pinning
still does its job, via a different image than intended.

## A failed measurement, recorded on purpose

Reading the overlay sheets suggested the model was systematically missing red
and purple anthocyanin-rich tissue — that it had learned "green means sundew".
This was tested by comparing, within each image, the redness `(R-G)/(R+G)` of
ground-truth sundew pixels the model **found** against ground-truth sundew
pixels it **missed**. Both populations are annotated sundew, so unlike the YCbCr
skin detector discarded in `scrape-probe-50.md`, the comparison needs no colour
model of "sundew" and cannot be fooled by sand or leaf litter.

**The hypothesis is wrong, and wrong in the opposite direction.** Missed tissue
is *less* red than found tissue: mean difference −0.056 (SD 0.078, SE 0.012,
t = −4.78 on 43 df), with found tissue redder in 33 of 44 images. Pigment is not
the mechanism.

The error came from misreading the visualisation: `evaluate_checkpoint.py`
paints false negatives magenta `(255, 0, 180)`, so every missed region looks
pink whatever colour the plant is. Anyone reading those overlay sheets should
know that before drawing conclusions from them.

## No automated flag catches the bad cases

Per-image IoU against every cue available without ground truth (n=47):

| Cue | r | t(45) |
| --- | ---: | ---: |
| mean confidence | +0.444 | 3.32 |
| largest-component fraction | +0.388 | 2.82 |
| n connected components | −0.358 | −2.57 |
| predicted foreground fraction | +0.332 | 2.36 |
| ground-truth foreground fraction | +0.053 | 0.36 |
| image megapixels | −0.011 | −0.07 |

Confidence correlates with quality on average, which is a weaker claim than it
sounds. As a screen it still fails, because it fails hardest exactly where it
matters:

| Screening rule | Flagged | Caught | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| confidence < 0.85 | 8 | 7 | 0.88 | 0.28 |
| confidence < 0.90 | 28 | 18 | 0.64 | 0.72 |
| n_components >= 5 | 11 | 10 | 0.91 | 0.40 |
| largest/total < 0.40 | 3 | 3 | 1.00 | 0.12 |

The five worst images defeat all of them:

| Image | IoU | Confidence | Components | Largest/total |
| --- | ---: | ---: | ---: | ---: |
| `inat_592910855` | **0.000** | **0.957** | 1 | 1.00 |
| `inat_460367205` | 0.035 | 0.799 | 7 | 0.50 |
| `inat_371821067` | 0.039 | 0.859 | 4 | 0.61 |
| `inat_453365067` | 0.047 | 0.934 | 2 | 0.80 |
| `inat_472599992` | 0.081 | 0.919 | 2 | 0.57 |

The worst image in the batch — IoU exactly 0.000, no overlap with truth at all —
produced a single clean compact component at the **highest confidence of all 50
scraped images**. By every signal available without a ground-truth mask it is
indistinguishable from a perfect prediction.

This confirms recommendation 1 of `scrape-probe-50.md` and strengthens
recommendation 3: eye review is not a precaution that better statistics will
eventually replace. On this evidence there is no automated substitute for it.

## What this means for step 2

The retrain described in `docs/field-probe-return.md` is still the right next
move, and its stated caveat now has a number attached: 30 field images added to
144 curated is a 20% addition against a 0.184 gap. "Not enough data yet" is a
likely outcome and would itself justify the next annotation round.

Two adjustments the measurement argues for:

- **Score area error, not only IoU, when comparing the two runs.** Area is the
  deliverable and it degraded proportionally worse than IoU.
- **Re-fit the decision threshold on `field-eval` after the retrain** rather
  than reusing 0.70 from curated validation.
