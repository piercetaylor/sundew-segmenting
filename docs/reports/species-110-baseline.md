# 110 species: the ResNet-18 baseline, full frame vs crop

**In short.** A small ResNet-18 reaches 0.546 validation balanced accuracy on
cropped plants and 0.525 on full frames (chance is 0.009). The crop helps by
+0.021, 5 of 5 seeds, about as much as it did at ten species (+0.025). On the
split with a held-out test set, the anchor that later models are compared
against is **0.516 crop / 0.499 full**.

The ten-species comparison (`deprecated/reports/species-crop-comparison.md`),
rerun unchanged on the 110-species corpus: ResNet-18 from ImageNet at 224 px,
same recipe, same five seeds, same split by observer (the person who took the
photo, so no photographer is in both training and validation).

Reproduce with `sbatch --array=0-4 scripts/hellbender_species_110.slurm`, then
`--array=5-9` once `scripts/hellbender_species_110_crops.slurm` has written every
crop. Full-frame tasks take 45-56 min on an A100, crop tasks 10-12 min (the
gap is JPEG decoding of ~2048 px originals against 768 px crops).

## Result: the crop still wins

Validation balanced accuracy (every species weighted equally) at the best
epoch, five seeds:

| Arm | 17 | 101 | 202 | 303 | 404 | Mean | SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.5313 | 0.5261 | 0.5248 | 0.5220 | 0.5189 | 0.5246 | 0.0047 |
| `crop` | 0.5498 | 0.5533 | 0.5421 | 0.5407 | 0.5439 | **0.5460** | 0.0054 |

| Comparison (paired over seeds) | Mean | t (4 df, crit 2.776) | 95% CI | Wins |
| --- | ---: | ---: | --- | ---: |
| `crop` - `full`, best-epoch balanced acc | +0.0213 | 10.74 | [+0.016, +0.027] | 5/5 |
| `crop` - `full`, last-epoch balanced acc | +0.0221 | 6.96 | [+0.013, +0.031] | 5/5 |
| `crop` - `full`, plain accuracy | +0.0134 | 7.59 | [+0.009, +0.018] | 5/5 |

- The worst crop seed (0.5407) beats the best full-frame seed (0.5313).
- Picking the best epoch on validation does not inflate the result: the
  last epoch gives the same answer (best and last differ by 0.000-0.008).
- Plain accuracy is higher (0.58 crop) and the gap smaller (+0.013). The crop
  helps the rare species more, and balanced accuracy counts them more.

## Reading 0.55

The prior in `deprecated/docs/crop-readiness-plan.md` — far below the 0.754
at ten species — held. Three reasons the number is low:

- **The classes are unbalanced.** Training images per species run from 29
  (*D. monticola*) to 240 (*D. rotundifolia*), median 118; validation from 9
  (*D. collinsiae*) to 76 (*D. pygmaea*), median 34. A 9-image class moves
  balanced accuracy in steps of 11 points, so per-class results are noisy.
- **Validation is concentrated**: 3,959 images from 308 observers (training:
  13,719 from 5,745). Intervals must resample observers, not images.
- **ResNet-18 underfits.** Final training loss is 0.71 in every run against
  a floor of about 0.43 (label smoothing 0.05, 110 classes), and several runs
  peaked on the last epoch. More capacity is needed before more regularisation.

This run did not log top-k, per-class results or confusions, and could not say
whether the backbone is the ceiling. That is the next experiment
([plan](../species-classifier-plan.md), [screen](species-backbone-screen.md)).

Checkpoints: `models/species-110/{full,crop}/seed-*/classifier-best.pt`.
Logs: `logs/sundew-species-110-{17903910,17910113}_*.out`.

## Rerun on `split-110-test`: the anchor for the fine-tunes

Same recipe and seeds, on the split with a held-out test set carved from
training (`scripts/carve_test_split.py`; validation unchanged, training
13,719 -> 11,118). Job 17928590:
`sbatch --export=ALL,SPLIT_NAME=split-110-test,OUT_ROOT=models/species-110-test/resnet18 scripts/hellbender_species_110.slurm`.

| Arm | 17 | 101 | 202 | 303 | 404 | Mean | SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.4997 | 0.4913 | 0.4999 | 0.5022 | 0.4992 | 0.4985 | 0.0042 |
| `crop` | 0.5225 | 0.5169 | 0.5050 | 0.5108 | 0.5229 | **0.5156** | 0.0077 |

| Comparison (paired over seeds) | Mean | t (4 df) | 95% CI | Wins |
| --- | ---: | ---: | --- | ---: |
| `crop` - `full`, best epoch | +0.0172 | 4.01 | [+0.005, +0.029] | 5/5 |
| `crop` - `full`, last epoch | +0.0190 | 5.74 | [+0.010, +0.028] | 5/5 |

19% less training data cost 0.026-0.030 on both arms; the crop effect held
(+0.017). **0.516 crop / 0.499 full are the numbers the fine-tunes are
compared against**, not 0.546 / 0.525. These models never saw the test split;
their test scores are in [species-test.md](species-test.md).
