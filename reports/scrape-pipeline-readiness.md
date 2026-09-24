# Is segmentation ready to drive a scraped species dataset?

The plan: use the segmentation model to localise sundews in images scraped from
iNaturalist or Carnivorous Plant Photo Finder, then train a species classifier
on the result. Scraped observations carry species labels, which removes the
labelling bottleneck described in `data-sufficiency.md`.

Three experiments. The headline: **for a crop-then-classify pipeline the
segmentation is already sufficient, and recipe work beats more masks.**

## 1. The model finds plants it has never seen

First and only use of the 28-image test split, SegFormer-B0 at threshold 0.5.
Six test species never appear in training, which makes this a direct proxy for
scraped images of unlabelled species.

| | Species seen in train (n=22) | Species never in train (n=6) |
| --- | ---: | ---: |
| Mask IoU | 0.6312 | **0.5197** |
| Bounding-box IoU | 0.7912 | 0.7492 |
| Crop coverage | 99.6% | **99.6%** |

Crop coverage is the fraction of true sundew tissue falling inside the
predicted bounding box padded by 10% — that is, what a crop-then-classify
pipeline actually consumes.

Mask IoU falls by 0.11 on unseen species. Bounding-box IoU barely moves, and
crop coverage does not move at all. Across all 28 test images:

- crop coverage >= 90%: **28 / 28**
- crop coverage >= 95%: 27 / 28
- total failures (IoU < 0.3): 1 / 28

The model is worse at *tracing* novel morphology and essentially unaffected at
*finding* it. Boundary precision is what degrades on unseen species, and
boundary precision is not what a species classifier needs.

Caveat: six unseen images is a thin basis. This is a strong hint, not a
guarantee, and the test split is now spent.

## 2. Recipe work beats more masks

Six variants, two seeds each, full 144-image train split, fixed 19-image
validation (`scripts/hellbender_recipe_sweep.slurm`). All use flags that
already existed.

| Variant | Mean val IoU | vs baseline | Epochs | Time |
| --- | ---: | ---: | ---: | ---: |
| baseline (768, 30ep, lr 1e-3, tversky) | 0.6323 | — | 14 | 87 s |
| res1024 | 0.6452 | +0.0129 | 15 | 159 s |
| long (80ep, patience 15) | 0.6450 | +0.0127 | 34 | 212 s |
| dice loss | 0.6460 | +0.0137 | 18 | 115 s |
| lowlr-long (lr 3e-4, 80ep) | 0.6627 | +0.0304 | 40 | 242 s |
| **combined (1024, 80ep, lr 3e-4)** | **0.6830** | **+0.0507** | 33 | 348 s |

`combined` gains **+0.051 IoU** for six minutes of GPU and zero annotation. For
comparison, the data-scaling curve's entire last step (108 -> 144 images) was
-0.007, and the SegFormer-vs-U-Net architecture gap was +0.022.

The two `combined` seeds landed at 0.6838 and 0.6823 — a spread of 0.0015
against a baseline seed SD of 0.0223. Individually each single-factor change
sits inside seed noise; only the combination clears it.

So the flat tail of the data-scaling curve was a **recipe ceiling, not a data
ceiling**. The earlier conclusion stands but sharpens: more training images
were not the lever, and now there is a measured lever that is.

Caveat: two seeds, and `combined` has not been evaluated on the test split
because that would be tuning on test. Its 0.683 is a validation number and is
not comparable to the 0.607 test number in section 1.

## 3. What this means for 250-500 more labelled masks

For the **scraping pipeline**, more masks are close to worthless. Crop coverage
is 99.6% and cannot go much higher; the localisation the pipeline needs is
already solved, including on species the model has never seen.

For **projected area and percent cover** — the project's stated research
question — more masks are also not the lever: the curve saturated at 108
images. Recipe work is, and so is threshold calibration (see
`data-sufficiency.md`, which removes a systematic +14% area bias).

Where more labelled images *would* pay:

- **Validation and test.** Section 1 rests on 6 unseen images; section 2 on 19
  validation images with a 95% interval of about +/-0.046. Both conclusions are
  measurement-limited, not model-limited.
- **`linear_or_forked` morphology.** Both unseen-species failures (*D. binata*
  0.277, *D. murfetii* 0.321) are linear/forked, the weakest growth form
  throughout. Targeted masks here would attack a real, identified weakness
  rather than adding more rosettes to a distribution already 72% rosette.

250 indiscriminate new masks would mostly be rosettes and would mostly not
help. 250 masks weighted toward held-out evaluation and toward filiform
morphology would.
