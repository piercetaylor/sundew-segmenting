# Step 2: does the segmentation crop earn its place?

Ten species, 1,753 scraped images, observer-grouped split, ResNet-18 from
ImageNet at 224 px. Two arms differing only in which image file is read, paired
across five seeds. Then a third arm, added after the first result, to rule out a
preprocessing confound.

Reproduce with `sbatch scripts/hellbender_species_compare.slurm` (10 jobs) and
`sbatch scripts/hellbender_species_control.slurm` (5 jobs), 1-6 minutes each.

## Answer: yes, and it is cropping rather than an artefact

Best validation balanced accuracy, five seeds:

| Arm | 17 | 101 | 202 | 303 | 404 | Mean | SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` (original frames) | 0.7381 | 0.7065 | 0.7414 | 0.7370 | 0.7242 | 0.7295 | 0.0144 |
| `full768` (uncropped, 768 px) | 0.7125 | 0.7187 | 0.7318 | 0.7472 | 0.7376 | 0.7296 | 0.0140 |
| `crop` (predicted box) | 0.7472 | 0.7450 | 0.7607 | 0.7534 | 0.7649 | **0.7542** | 0.0085 |

| Comparison | Isolates | Mean | t (4 df, crit 2.776) | 95% CI | Wins |
| --- | --- | ---: | ---: | --- | ---: |
| `crop` - `full` | cropping + resampling | +0.0248 | 3.95 | [+0.007, +0.042] | 5/5 |
| `full768` - `full` | **resampling alone** | +0.0001 | 0.02 | [-0.021, +0.021] | 3/5 |
| `crop` - `full768` | **cropping alone** | +0.0246 | **5.10** | [+0.011, +0.038] | 5/5 |

**Cropping is the entire effect.** Resampling contributes +0.0001, which is as
close to nothing as this design can measure. The isolated cropping effect
(+0.0246, t = 5.10) is slightly *cleaner* than the original comparison, because
removing the resampling difference removed noise rather than signal.

### Why the control was run

The crop arm's advantage did not grow as the predicted box got tighter
(r = -0.025, t = -0.50, n=391) — it was +0.031 even on the third of images where
the box covers 95% of the frame and the two arms see nearly identical pixels.
That is not the profile a cropping benefit should have, and it had an obvious
alternative explanation: `render_crop_review.py` saves crops as 768 px LANCZOS
thumbnails while full frames keep up to 1600 px, so the crop arm received a
two-stage downsample before the 224 px resize.

The suspicion was wrong. It is recorded because the reasoning was sound and the
result was not predictable from it, and because a +0.025 effect resting on an
unexamined preprocessing difference would have been a bad thing to build on.

## The mechanism is not magnification, and is not yet established

| Box size tercile | Median box | `full` | `crop` | Delta | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| tightest third | 38.2% | 0.675 | 0.705 | +0.029 | 130 |
| middle third | 69.4% | 0.751 | 0.758 | +0.008 | 130 |
| loosest third | 95.2% | 0.769 | 0.800 | +0.031 | 131 |

The benefit is flat across box size. Whatever the crop supplies, it is not
zoom — a box covering 95% of the frame delivers as much as one covering 38%.

The leading hypothesis is **centring, not magnification**. Both arms are
evaluated with `Resize(255)` then `CenterCrop(224)`, which discards the frame's
edges. A plant that sits off-centre in the original photograph can be partly cut
away; the predicted box is centred on the plant by construction, so even a box
that removes almost no area changes what survives the centre crop.

**This is a hypothesis, not a result.** Testing it means comparing against full
frames resized without a centre crop, and doing that honestly requires matching
the training augmentation too, which changes more than one thing at once. It is
written down as the next cheap experiment rather than guessed at.

If it holds, the practical consequence is large: the value of segmentation to
the species pipeline is **localisation, not boundary precision**. Tighter masks
would buy nothing; a correctly placed box is the whole contribution, and
`reports/field-probe-evaluation.md` already measured that at 95.4% crop
coverage.

## Per class

| Species | `full` | `crop` | Delta |
| --- | ---: | ---: | ---: |
| *D. brevifolia* | 0.605 | 0.721 | **+0.116** |
| *D. anglica* | 0.563 | 0.647 | +0.084 |
| *D. rotundifolia* | 0.652 | 0.705 | +0.052 |
| *D. capillaris* | 0.574 | 0.621 | +0.046 |
| *D. glanduligera* | 0.911 | 0.947 | +0.037 |
| *D. intermedia* | 0.795 | 0.826 | +0.032 |
| *D. aberrans* | 0.838 | 0.856 | +0.019 |
| *D. arcturi* | 0.905 | 0.889 | -0.016 |
| *D. spatulata* | 0.573 | 0.514 | **-0.059** |
| *D. auriculata* | 0.878 | 0.816 | **-0.063** |

The aggregate hides real disagreement: the crop helps seven classes and hurts
three, and *D. auriculata* and *D. spatulata* lose about as much as
*D. anglica* gains. Per-class counts are 32-51 validation images, so each row
carries roughly +/-0.13 at 95%, and none of these individually clears noise.
The pattern is a prompt for inspection, not a finding.

## Design notes

- **238 photo IDs excluded** because the segmentation model trained on them.
  The crop on a training image is unrealistically good, and that advantage is
  precisely what the experiment measures, so leaving them in would have decided
  the result quietly.
- **Observer-grouped split**, 690 train observers against 40 validation, zero
  shared, matching the segmentation splits. A first greedy assignment overshot
  to 25.8% validation and starved *D. glanduligera* to 84 training images; the
  assignment now sends each observer to whichever side leaves per-class counts
  closest to target, giving 22.3% with every class between 20% and 28%.
- **The training script refuses to run** if any image is missing from one arm,
  rather than silently comparing different image sets.
- **Wall-clock is not a result.** The crop arm ran 3.4x faster (1.4 vs 4.8
  minutes), which is JPEG decoding of 768 px versus 1600 px files. The
  `full768` arm ran in 1.5 minutes and scored the same as `full`, which
  confirms the speed and the accuracy have nothing to do with each other.

## What this means for the scrape

The crop is justified, so step 1's failure rate stays on the critical path:
a crop that misses the plant now costs something measurable. The decision rule
in `docs/crop-readiness-plan.md` stands unchanged.

Note the ceiling this sets. At 75% balanced accuracy over ten well-represented
species, a 10-20k scrape is not going to produce a reliable 129-way classifier.
That is a reasonable baseline for a from-scratch ResNet-18 on ~1,400 training
images and would improve with more data and a stronger backbone, but the
species goal should be scoped against 75% on ten classes rather than against an
assumption.
