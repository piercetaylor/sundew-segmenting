# Do 30 field masks close the field gap?

**In short.** Yes, partly. Adding 30 masks of uncurated field photos raised IoU
on 17 held-out field images from 0.552 to 0.613 (+0.061, 5 of 5 seeds,
significant) and roughly halved area error. That closes 58% of the gap between
curated and field photos. A separate flag change, turning growth-form
balancing off, was worth another +0.052. One failure mode, bright artificial
objects, is not fixed.

## Setup

Two arms, five seeds each (17, 101, 202, 303, 404), identical flags, differing
only in the dataset root:

- `frozen`: the 144 curated train images;
- `field`: those plus 30 field-train masks (174).

Both use the `combined` recipe (1024 px, 80 epochs, lr 3e-4, bce-tversky;
[segmentation-recipe.md](segmentation-recipe.md)) with
`--no-balanced-growth-forms`. Both are scored on the 17 held-out `field-eval`
images. The curated test split was not touched. Seeds are paired because
[baseline-seed-sweep.md](baseline-seed-sweep.md) showed a single run cannot
settle a comparison on this dataset.

Reproduce with `sbatch scripts/hellbender_field_compare.slurm` (10 GPU jobs,
4-15 minutes each; indices 0-4 are `frozen`, 5-9 are `field`). Score with
`python scripts/score_field_compare.py`.

## Result: the gap roughly halves

| Arm | 17 | 101 | 202 | 303 | 404 | Mean | SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen` (144 images) | 0.5676 | 0.5164 | 0.5469 | 0.5540 | 0.5725 | 0.5515 | 0.0221 |
| `field` (174 images) | 0.6086 | 0.6315 | 0.6166 | 0.6055 | 0.6022 | **0.6129** | 0.0117 |

**Field wins on 5 of 5 seeds: mean paired difference +0.0614 IoU, 95% CI
[+0.020, +0.103], t = 4.10 on 4 df against a critical 2.776, significant at
p = 0.05.** The arms do not overlap: the worst `field` run (0.6022) beats the
best `frozen` run (0.5725).

This is the largest single improvement recorded for the segmenter (architecture
gave +0.022, the whole `combined` recipe +0.051), and the first to come from
annotation rather than configuration.

The field gap is each arm's curated validation IoU minus its `field-eval` IoU:

| | Curated validation | `field-eval` | Gap |
| --- | ---: | ---: | ---: |
| Original `combined` | 0.6838 | 0.4994 | 0.1844 |
| `frozen` arm | 0.6808 | 0.5515 | 0.1293 |
| `field` arm | 0.6906 | 0.6129 | **0.0778** |

58% of the gap is closed; the rest is not.

## How sure is this

The first run used three seeds and did **not** reach significance (t = 3.49
against a critical 4.303 on 2 df). The power estimate had assumed the seed SD
of 0.0176 from `baseline-seed-sweep.md`, but the SD of the *difference* was
about twice that. Seeds 303 and 404 were then added to both arms.

At five seeds both reasonable pairings agree:

| Pairing | n | Mean delta | t | crit (p=.05) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| By seed (image-averaged) | 5 | +0.0614 | 4.10 | 2.776 | [+0.020, +0.103] |
| By image (seed-averaged) | 17 | +0.0614 | 2.89 | 2.120 | [+0.016, +0.106] |

The effect shrank from +0.0753 at three seeds to +0.0614 at five: the two new
seeds gave the two smallest deltas (+0.0515, +0.0297). That is regression
toward the mean. The interval is wide (+0.02 to +0.10), so read this as a
direction and an order of magnitude, not a precise number. What makes it
credible is the separation: 5 of 5 seeds, 14 of 17 images, and no overlap
between the arms.

## Part of the gain is a flag, not the data

The original `combined` seed-17 checkpoint was trained **with** growth-form
balancing and scores 0.4994 on `field-eval`. The `frozen` arm differs from it
only by `--no-balanced-growth-forms`.

| Change | field-eval IoU | Gain |
| --- | ---: | ---: |
| Original `combined` (balanced, 1 seed) | 0.4994 | — |
| + turn off growth-form balancing | 0.5515 | +0.052 |
| + add 30 field masks | 0.6129 | +0.061 |

Turning balancing off is worth **+0.052 on its own**, as much as the field
masks. Without a matched rerun, all +0.114 would have been credited to the
masks. The 0.4994 row is a single seed and the least certain; the other two
are five-seed means.

## What got fixed, and what did not

Per-image IoU at threshold 0.5, averaged over five seeds, worst first:

| Image | `frozen` | `field` | Change |
| --- | ---: | ---: | ---: |
| `inat_592910855` (tape measure) | 0.061 | 0.030 | **−0.031** |
| `inat_436079974` | 0.287 | 0.324 | +0.037 |
| `inat_511350090` | 0.356 | 0.377 | +0.021 |
| `inat_686282548` | 0.455 | 0.515 | +0.060 |
| `inat_472599992` | 0.482 | **0.824** | **+0.342** |
| `inat_158581358` | 0.587 | 0.765 | +0.178 |
| `inat_230880856` | 0.639 | 0.737 | +0.098 |
| `inat_632125737` | 0.580 | 0.671 | +0.091 |

14 of 17 images improved; the three that got worse lost 0.031 or less.

**Under-segmentation is largely fixed.** `inat_472599992` had precision 1.00
and recall 0.08 with the original checkpoint; it gains +0.342. The other large
gains are the same failure mode.

**The tape measure is not.** `inat_592910855` (IoU 0.000 at 0.957 confidence
with the original checkpoint) does not improve. The field masks taught the
model about ordinary clutter, not that a bright yellow tape measure is not a
plant: the batch has essentially one such example, and it is in evaluation.

## Area error, which is the deliverable

Median absolute relative area error over 3 seeds x 17 images:

| Threshold | `frozen` | `field` | `field` IoU |
| ---: | ---: | ---: | ---: |
| 0.4 | 32.9% | 21.7% | 0.6135 |
| 0.5 | 32.4% | 18.4% | **0.6129** |
| 0.6 | 30.9% | **16.1%** | 0.6097 |
| 0.7 | 31.1% | 17.8% | 0.6028 |
| 0.8 | 32.8% | 21.6% | 0.5877 |

Area error roughly halves, from about 32% to 16-18%. At threshold 0.6 the field
model's median (16.1%) is close to the 17.9% measured on *curated* validation
([segmentation-recipe.md](segmentation-recipe.md)), so field photos are no
longer much worse than curated ones on area.

The `frozen` column is flat across thresholds; `field` has a clear optimum. A
threshold can tune a model whose errors are calibration errors, not one whose
errors are not.

**Decision: use threshold 0.6**, not the 0.70 suggested by curated validation.
It costs 0.003 IoU against the 0.5 peak and gains 2.3 points of area error.

### The remaining bias is the remaining failures

Mean signed area error stays high (+33.5% `field`, +39.5% `frozen` at 0.5),
unlike the median. It is driven by three images, the three the retrain did not
fix:

| Image | Relative area error | Status |
| --- | ---: | --- |
| `inat_592910855` | ~+276% | the tape measure |
| `inat_436079974` | ~+213% | barely moved |
| `inat_686282548` | ~+77% | improved but still weak |

This is not a calibration property to tune away; it is a handful of hard
failures.

## What this changes

The curated data curve ([segmentation-recipe.md](segmentation-recipe.md))
said more training images were not the lever. That holds for curated images
only: 30 *field* images gained +0.061 IoU and halved area error, where the last
36 curated images gained −0.007. The difference is the distribution, not the
volume. Annotation helps again when the images come from the uncurated photos
the model will actually see.

Next steps, in order of expected return:

1. **Hard negatives.** Bright artificial objects are the one failure mode the
   retrain did not touch, and the batch had one training example of it
   (recommendation 2 of
   [`deprecated/reports/scrape-probe-50.md`](../../deprecated/reports/scrape-probe-50.md)).
2. **A larger field evaluation set.** 17 images give about ±0.10 on the mean,
   which is why the estimate moved between three and five seeds. More
   evaluation images would tighten it more cheaply than more seeds.
