# Why the segmentation recipe is what it is

**In short.** The segmenter is SegFormer-B0 trained with the `combined` recipe:
1024 px, up to 80 epochs (patience 15), learning rate 3e-4, BCE + Tversky loss.
On the 19 validation images it scored 0.683, against 0.632 for the 768 px,
30-epoch default. Once the recipe was fixed, more curated training images no
longer helped, but field images did ([field-compare.md](field-compare.md)).

This report merges two earlier reports from 2026-09-19, kept unchanged in
[`deprecated/reports/data-sufficiency.md`](../../deprecated/reports/data-sufficiency.md)
and
[`deprecated/reports/scrape-pipeline-readiness.md`](../../deprecated/reports/scrape-pipeline-readiness.md).
All runs use the frozen v0.3.0 dataset (144 train, 19 validation, 28 test).

## The recipe

| Setting | Default in `scripts/train_baseline.py` | `combined` |
| --- | --- | --- |
| `--image-size` | 768 | **1024** |
| `--epochs` / `--patience` | 30 / 5 | **80 / 15** |
| `--learning-rate` | 1e-3 | **3e-4** |
| `--loss` | `bce-tversky` | `bce-tversky` |

The Tversky term weights false positives 0.3 and false negatives 0.7. That buys
recall and makes the model over-segment slightly (see "Threshold" below).

The shipped segmenter (`models/field-compare/field/seed-101`) uses these flags
plus `--no-balanced-growth-forms`, trained on the 144 curated images and 30
field masks, and runs at threshold 0.6. The reasons for those last three
choices are in [field-compare.md](field-compare.md).

## More curated training images stopped helping

SegFormer-B0 at the default recipe, trained on 25/50/75/100% of the 144 train
images, three seeds each (17, 101, 202), all scored on the same 19 validation
images. Job 17828458, `scripts/hellbender_data_curve.slurm`.

| Train images | Fraction | Mean val IoU | SD | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 0.25 | 0.5428 | 0.0045 | |
| 72 | 0.50 | 0.6191 | 0.0119 | +0.0763 |
| 108 | 0.75 | 0.6482 | 0.0116 | +0.0291 |
| 144 | 1.00 | 0.6408 | 0.0223 | **-0.0074** |

Four times the data (36 to 144) gained +0.098 IoU, but the last 36 images
gained nothing. A plateau like this can mean the recipe has run out of room
rather than the data, so the next step was to test the recipe.

## Recipe changes beat more images

Six variants, two seeds each (17, 101), full 144-image train split, same 19
validation images. Every variant uses flags that already existed. Job
17828645, `scripts/hellbender_recipe_sweep.slurm`.

| Variant | Mean val IoU | vs baseline | Epochs | Time |
| --- | ---: | ---: | ---: | ---: |
| baseline (768, 30ep, lr 1e-3, tversky) | 0.6323 | — | 14 | 87 s |
| res1024 | 0.6452 | +0.0129 | 15 | 159 s |
| long (80ep, patience 15) | 0.6450 | +0.0127 | 34 | 212 s |
| dice loss | 0.6460 | +0.0137 | 18 | 115 s |
| lowlr-long (lr 3e-4, 80ep) | 0.6627 | +0.0304 | 40 | 242 s |
| **combined (1024, 80ep, lr 3e-4)** | **0.6830** | **+0.0507** | 33 | 348 s |

`combined` gains **+0.051 IoU** for six minutes of GPU time and no new
annotation. For comparison, the last 36 training images gave -0.007 and the
SegFormer-vs-U-Net choice gave +0.022
([baseline-seed-sweep.md](baseline-seed-sweep.md)).

The two `combined` seeds scored 0.6838 and 0.6823, 0.0015 apart, against a
baseline seed SD of 0.0223. Each single change sits inside seed noise; only the
combination clears it. The flat end of the data curve was a recipe ceiling,
not a data ceiling.

Caveats: two seeds only, and 0.683 is a validation number. `combined` was never
scored on the test split, because that would be tuning on test.

## The model finds species it has never seen

The 28-image test split was used once, on the default-recipe SegFormer-B0 at
threshold 0.5 (overall test IoU 0.607). Six test species never appear in
training, which stands in for scraped photos of new species.

| | Species seen in train (n=22) | Species never in train (n=6) |
| --- | ---: | ---: |
| Mask IoU | 0.6312 | **0.5197** |
| Bounding-box IoU | 0.7912 | 0.7492 |
| Crop coverage | 99.6% | **99.6%** |

Crop coverage is the share of true sundew pixels inside the predicted box
padded by 10%, which is what a crop-then-classify pipeline uses. Across all 28
test images, 28 had coverage of at least 90%, 27 at least 95%, and 1 was a
total failure (IoU < 0.3).

Mask IoU drops by 0.11 on unseen species, but the box barely moves: the model
traces new shapes less precisely but still finds them. Both unseen-species
failures (*D. binata* 0.277, *D. murfetii* 0.321) are linear or forked, the
weakest growth form throughout.

Caveat: six unseen images is a thin basis, and the test split is now spent.
The crop was later measured properly on 493 uncurated photos
([crop-review-result.md](crop-review-result.md)).

## Threshold: area and IoU want different points

The deliverable is plant area, so area error matters more than IoU. At the
default threshold 0.5, SegFormer-B0 on validation has mean absolute relative
area error 21.7% (median 17.9%), a systematic +14.0% over-prediction, and
R^2 0.84 between true and predicted area. The over-prediction comes from the
recall-weighted Tversky loss.

| Threshold | IoU | Mean abs area err | Signed area err |
| ---: | ---: | ---: | ---: |
| 0.40 | **0.6453** | 26.2% | +21.5% |
| 0.50 (default) | 0.6442 | 21.7% | +14.0% |
| 0.60 | 0.6376 | 18.9% | +6.9% |
| 0.70 | 0.6225 | **17.3%** | **-0.7%** |
| 0.80 | 0.5912 | 18.1% | -9.8% |

On these 19 images 0.70 removes the bias and cuts area error by a fifth, for
0.022 IoU. The threshold was fitted on the same images it was scored on, so the
gain is optimistic. The field-trained model later chose **0.6** on held-out
field images, and that is the threshold in use
([field-compare.md](field-compare.md)).

## The evaluation sets are too small

Per-image validation IoU has SD 0.1025. At n=19 the standard error is 0.0235,
so the 95% interval on a headline number is about ±0.046.

| To reliably detect a gap of | You need about |
| ---: | ---: |
| 0.05 IoU | 65 validation images |
| 0.03 IoU | 180 validation images |
| 0.02 IoU | 404 validation images |

The SegFormer-vs-U-Net gap (0.0215) is invisible on one 19-image split; only
pairing across seeds showed it. Any comparison here should be paired across
seeds, and new annotation is worth more in evaluation sets than in train.

## What this meant for annotation and species ID

- **Indiscriminate new masks would not help.** They would mostly be rosettes
  (the train split is already 72% rosette), and the curated curve was flat.
  Masks aimed at held-out evaluation, at linear or forked plants, and at the
  uncurated photos the model will actually see were worth making. The 30 field
  masks were the result, and they gained +0.061 IoU
  ([field-compare.md](field-compare.md)).
- **Species ID needed labels, not better masks.** The 191 masks span 51
  species, median 1 image per species; only 9 species have at least 5 training
  images, and 10 species sit only in validation or test. A classifier could not
  be trained on this set. Scraping labelled iNaturalist photos and cropping
  them with this segmenter removed that limit
  ([dataset-card.md](dataset-card.md#species-corpus)).

## Reproduce

```bash
sbatch scripts/hellbender_data_curve.slurm    # 4 fractions x 3 seeds, writes models/curve/
sbatch scripts/hellbender_recipe_sweep.slurm  # 6 variants x 2 seeds, writes models/recipe/
```

The `combined` checkpoints are `models/recipe/combined/seed-{17,101}/segformer-b0-best.pt`.
