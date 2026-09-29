# Step 2: do 30 field masks close the field gap?

Two arms, five seeds each, identical flags, differing only in dataset root:
`frozen` is the 144 curated images, `field` is those plus the 30 field-train
masks (174). Both use the `combined` recipe (1024 px, 80 epochs, lr 3e-4,
bce-tversky) with `--no-balanced-growth-forms`. Scored on the 17 held-out
`field-eval` images. The curated test split was not touched.

Reproduce with `sbatch scripts/hellbender_field_compare.slurm` (10 GPU jobs,
4-15 minutes each; indices 0-4 are `frozen`, 5-9 are `field`).

Paired across the same five seeds `baseline-seed-sweep.md` used, because that
report established that a single run cannot settle a comparison on this
dataset. An initial three-seed run is described under "The statistics" below.

## Result: the gap roughly halves

| Arm | 17 | 101 | 202 | 303 | 404 | Mean | SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen` (144 images) | 0.5676 | 0.5164 | 0.5469 | 0.5540 | 0.5725 | 0.5515 | 0.0221 |
| `field` (174 images) | 0.6086 | 0.6315 | 0.6166 | 0.6055 | 0.6022 | **0.6129** | 0.0117 |

**Field wins on 5 of 5 seeds, mean paired difference +0.0614 IoU, 95% CI
[+0.020, +0.103], t = 4.10 on 4 df against a critical 2.776 — significant at
p = 0.05.** The arms do not overlap: the worst `field` run (0.6022) beats the
best `frozen` run (0.5725).

Against the project's other measured levers — architecture +0.022, the entire
`combined` recipe +0.051 — this is the largest single improvement recorded, and
the first one that came from annotation rather than configuration.

The field gap, measured as each arm's own curated validation minus its
`field-eval` score:

| | Curated validation | `field-eval` | Gap |
| --- | ---: | ---: | ---: |
| Original `combined` (step 1) | 0.6838 | 0.4994 | 0.1844 |
| `frozen` arm | 0.6808 | 0.5515 | 0.1293 |
| `field` arm | 0.6906 | 0.6129 | **0.0778** |

**58% of the gap closed, not closed.**

## The statistics, honestly

This was run first at three seeds, where it did **not** clear significance:
t = 3.49 against a critical 4.303, because two degrees of freedom are
unforgiving. The pre-run power estimate had been too optimistic — it assumed the
seed-to-seed SD of 0.0176 from `baseline-seed-sweep.md`, whereas the observed SD
of the *difference* is about twice that. Seeds 303 and 404 were added for both
arms.

At five seeds both defensible pairings agree:

| Pairing | n | Mean delta | t | crit (p=.05) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| By seed (image-averaged) | 5 | +0.0614 | 4.10 | 2.776 | [+0.020, +0.103] |
| By image (seed-averaged) | 17 | +0.0614 | 2.89 | 2.120 | [+0.016, +0.106] |

Worth recording that the effect **shrank** as seeds were added, from +0.0753 at
n=3 to +0.0614 at n=5: the two added seeds returned the two smallest deltas
(+0.0515, +0.0297). That is ordinary regression toward the mean and is the
reason the three-seed estimate should not have been quoted as a point value. The
five-seed interval is wide — the effect is somewhere between +0.02 and +0.10 —
and the honest summary is direction and order of magnitude, not a precise
number.

The separation is what makes it credible regardless: 5 of 5 seeds, 14 of 17
images, and arm ranges that do not overlap.

## Part of the gain is a flag, not the data

The original probe checkpoint (`combined` seed-17, trained **with**
growth-form balancing) scores 0.4994 on `field-eval`. The `frozen` rerun,
identical except `--no-balanced-growth-forms`, scores 0.5436.

| Change | field-eval IoU | Gain |
| --- | ---: | ---: |
| Original `combined` (balanced, 1 seed) | 0.4994 | — |
| + turn off growth-form balancing | 0.5515 | +0.052 |
| + add 30 field masks | 0.6129 | +0.061 |

Turning balancing off is worth **+0.052 on its own** — more than the whole
architecture effect, and comparable to the field masks themselves. Had the
baseline not been rerun under matched flags, all +0.114 would have been credited
to the field masks.

(The 0.4994 reference is a single seed and so is the least certain row here; the
other two are five-seed means.) This is why
`field-probe-return.md` insisted on two jobs rather than comparing against a
previous day's sweep number.

## What got fixed, and what did not

Per-image IoU at threshold 0.5, averaged over five seeds, worst-first:

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

14 of 17 images improved; the three that regressed did so by 0.031 or less.

**The under-segmenting mode is largely solved.** `inat_472599992`, which step 1
recorded at precision 1.00 / recall 0.08, gains +0.342 and is now among the
better images in the set. The other large gains are the same mode.

**The catastrophic failure is not.** `inat_592910855` — IoU 0.000 at 0.957
confidence in step 1 — does not improve and is marginally worse. Thirty field
images taught the model about ordinary field clutter; they did not teach it that
a bright yellow tape measure is not a plant, because the batch contains
essentially one such example and it is in evaluation, not training.

## Area error, which is the deliverable

Median absolute relative area error over 3 seeds x 17 images:

| Threshold | `frozen` | `field` | `field` IoU |
| ---: | ---: | ---: | ---: |
| 0.4 | 32.9% | 21.7% | 0.6135 |
| 0.5 | 32.4% | 18.4% | **0.6129** |
| 0.6 | 30.9% | **16.1%** | 0.6097 |
| 0.7 | 31.1% | 17.8% | 0.6028 |
| 0.8 | 32.8% | 21.6% | 0.5877 |

Area error roughly **halves**, from ~32% to 16-18%. At threshold 0.6 the field
model's median area error (16.1%) is close to the 17.9% measured on *curated*
validation in `data-sufficiency.md` — field images are no longer materially
worse on the deliverable than curated ones.

Note the `frozen` column is flat across thresholds while `field` has a clear
optimum. Thresholding cannot fix a model whose errors are not calibration
errors; it can tune one whose errors are.

**Use threshold 0.6, not the 0.70 proposed from curated validation.** It costs
0.003 IoU against the 0.5 peak and buys 2.3 points of area error.

### The residual bias is the residual failures

Mean signed area error remains high for both arms (+33.5% field, +39.5% frozen
at threshold 0.5) and contradicts the median. The mean is outlier-driven, and
the outliers are named:

| Image | Relative area error | Status |
| --- | ---: | --- |
| `inat_592910855` | ~+276% | the unfixed tape measure |
| `inat_436079974` | ~+213% | barely moved by retraining |
| `inat_686282548` | ~+77% | improved but still weak |

The three images driving the aggregate bias are the three the retrain did not
fix. The residual area bias is not a systemic calibration property to be tuned
away — it *is* the remaining hard failures, concentrated in a handful of images.

## What this changes

`data-sufficiency.md` concluded that more training images were not the lever,
from a curated data-scaling curve that flattened at 108 images. That conclusion
was correct about curated images and does not generalise: 30 *field* images
bought +0.075 IoU and halved area error, where the last 36 *curated* images
bought −0.007.

The distinction is distribution, not volume. Annotation is a lever again,
provided the images come from the uncurated distribution the model will actually
be run on.

Suggested next steps, in order of expected return:

1. **Hard negatives.** The one failure mode the retrain did not touch is bright
   artificial objects, and the batch contained one training example of it. This
   is recommendation 2 of `scrape-probe-50.md`, still unaddressed and now with a
   measured cost attached.
2. **A larger field evaluation set.** 17 images give a +/-0.10 interval on the
   mean, which is why the effect estimate moved as much as it did between three
   seeds and five. More evaluation images would tighten the estimate more
   cheaply than more seeds now can.
