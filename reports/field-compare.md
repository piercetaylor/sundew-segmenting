# Step 2: do 30 field masks close the field gap?

Two arms, three seeds each, identical flags, differing only in dataset root:
`frozen` is the 144 curated images, `field` is those plus the 30 field-train
masks (174). Both use the `combined` recipe (1024 px, 80 epochs, lr 3e-4,
bce-tversky) with `--no-balanced-growth-forms`. Scored on the 17 held-out
`field-eval` images. The curated test split was not touched.

Reproduce with `sbatch scripts/hellbender_field_compare.slurm` (6 GPU jobs,
4-15 minutes each).

Paired across three seeds rather than run once, because
`baseline-seed-sweep.md` established that a single run cannot settle a
comparison on this dataset.

## Result: the gap roughly halves

| Arm | seed 17 | seed 101 | seed 202 | Mean | SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen` (144 images) | 0.5676 | 0.5164 | 0.5469 | 0.5436 | 0.0258 |
| `field` (174 images) | 0.6086 | 0.6315 | 0.6166 | **0.6189** | 0.0116 |

**Field wins on 3 of 3 seeds, mean paired difference +0.0753 IoU.** The arms do
not overlap: the worst `field` run (0.6086) beats the best `frozen` run
(0.5676).

Against the project's other measured levers — architecture +0.022, the entire
`combined` recipe +0.051 — this is the largest single improvement recorded, and
the first one that came from annotation rather than configuration.

The field gap, measured as each arm's own curated validation minus its
`field-eval` score, narrows from 0.184 (step 1) to **0.072**. Roughly 60% closed,
not closed.

## The statistics, honestly

Two defensible pairings, and they disagree about significance:

| Pairing | n | Mean delta | t | crit (p=.05) | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| By seed (image-averaged) | 3 | +0.0753 | 3.49 | 4.303 | **not significant** |
| By image (seed-averaged) | 17 | +0.0753 | 3.17 | 2.120 | significant, 95% CI [+0.025, +0.126] |

The seed-level test is the one this project's methodology established, and at
n=3 it does not clear p=0.05 — two degrees of freedom demand t > 4.3. The
pre-run power estimate was too optimistic: it assumed the seed-to-seed SD of
0.0176 from `baseline-seed-sweep.md`, but the observed SD of the *difference*
is 0.0374, twice that.

What carries the result is not either p-value but the separation: 3 of 3 seeds,
13 of 17 images, and non-overlapping arm ranges. **Two more seeds per arm would
settle it properly** and cost about 35 GPU-minutes.

## Part of the gain is a flag, not the data

The original probe checkpoint (`combined` seed-17, trained **with**
growth-form balancing) scores 0.4994 on `field-eval`. The `frozen` rerun,
identical except `--no-balanced-growth-forms`, scores 0.5436.

| Change | field-eval IoU | Gain |
| --- | ---: | ---: |
| Original `combined` (balanced) | 0.4994 | — |
| + turn off growth-form balancing | 0.5436 | +0.044 |
| + add 30 field masks | 0.6189 | +0.075 |

Turning balancing off is worth **+0.044 on its own** — more than the whole
architecture effect. Had the baseline not been rerun under matched flags, all
+0.120 would have been credited to the field masks. This is why
`field-probe-return.md` insisted on two jobs rather than comparing against a
previous day's sweep number.

## What got fixed, and what did not

Per-image IoU at threshold 0.5, averaged over three seeds, worst-first:

| Image | `frozen` | `field` | Change |
| --- | ---: | ---: | ---: |
| `inat_592910855` (tape measure) | 0.047 | 0.032 | **−0.015** |
| `inat_436079974` | 0.281 | 0.281 | −0.000 |
| `inat_511350090` | 0.372 | 0.389 | +0.017 |
| `inat_686282548` | 0.398 | 0.497 | +0.100 |
| `inat_472599992` | 0.474 | **0.845** | **+0.371** |
| `inat_158581358` | 0.559 | 0.760 | +0.201 |
| `inat_632125737` | 0.570 | 0.699 | +0.129 |
| `inat_230880856` | 0.626 | 0.758 | +0.132 |

13 of 17 images improved; the four that regressed did so by less than 0.035.

**The under-segmenting mode is largely solved.** `inat_472599992`, which step 1
recorded at precision 1.00 / recall 0.08, gains +0.371 and is now among the
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
| 0.4 | 32.0% | 19.7% | 0.6181 |
| 0.5 | 32.4% | 17.5% | **0.6189** |
| 0.6 | 31.8% | **14.5%** | 0.6172 |
| 0.7 | 31.1% | 15.1% | 0.6116 |
| 0.8 | 32.8% | 17.6% | 0.5984 |

Area error roughly **halves**, from ~32% to 14.5-17.5%. At threshold 0.6 the
field model's median area error (14.5%) is better than the 17.9% measured on
*curated* validation in `data-sufficiency.md`.

Note the `frozen` column is flat across thresholds while `field` has a clear
optimum. Thresholding cannot fix a model whose errors are not calibration
errors; it can tune one whose errors are.

**Use threshold 0.6, not the 0.70 proposed from curated validation.** It costs
0.0017 IoU against the 0.5 peak and buys 3 points of area error.

### The residual bias is the residual failures

Mean signed area error looks bad for the field arm (+40.8% at 0.5, against
+35.1% frozen), which contradicts the median (+13.9%). The mean is outlier-driven:

| Image | Relative area error | Status |
| --- | ---: | --- |
| `inat_592910855` | +275.8% | the unfixed tape measure |
| `inat_436079974` | +213.4% | unchanged by retraining |
| `inat_686282548` | +77.2% | improved but still weak |

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

1. **Two more seeds per arm** (~35 GPU-min) to settle significance properly.
2. **Hard negatives.** The one failure mode the retrain did not touch is bright
   artificial objects, and the batch contained one training example of it. This
   is recommendation 2 of `scrape-probe-50.md`, still unaddressed and now with a
   measured cost attached.
3. **A larger field evaluation set.** 17 images give a +/-0.10 interval on the
   mean; that is what forced the underpowered seed comparison above.
