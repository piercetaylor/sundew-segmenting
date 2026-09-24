# 110 species: the ResNet-18 baseline, full frame vs crop

The ten-species comparison in `reports/species-crop-comparison.md`, rerun
unchanged on the scaled corpus. Only the class set and the data volume moved:
same ResNet-18 from ImageNet at 224 px, same recipe, same five seeds, same
observer-grouped split logic.

Reproduce with `sbatch --array=0-4 scripts/hellbender_species_110.slurm`, then
`--array=5-9` once `scripts/hellbender_species_110_crops.slurm` has written every
crop. Full-frame tasks take 45-56 min on an A100, crop tasks 10-12 min; the gap
is JPEG decoding of ~2048 px originals against 768 px crops, not the model.

## Answer: the crop still wins, by the same margin as at ten species

Validation balanced accuracy at the best epoch, five seeds:

| Arm | 17 | 101 | 202 | 303 | 404 | Mean | SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.5313 | 0.5261 | 0.5248 | 0.5220 | 0.5189 | 0.5246 | 0.0047 |
| `crop` | 0.5498 | 0.5533 | 0.5421 | 0.5407 | 0.5439 | **0.5460** | 0.0054 |

| Comparison (paired over seeds) | Mean | t (4 df, crit 2.776) | 95% CI | Wins |
| --- | ---: | ---: | --- | ---: |
| `crop` - `full`, best-epoch balanced acc | +0.0213 | 10.74 | [+0.016, +0.027] | 5/5 |
| `crop` - `full`, last-epoch balanced acc | +0.0221 | 6.96 | [+0.013, +0.031] | 5/5 |
| `crop` - `full`, plain accuracy | +0.0134 | 7.59 | [+0.009, +0.018] | 5/5 |

- **The crop effect carried over.** +0.021 here against +0.025 at ten species.
  The worst crop seed (0.5407) is above the best full-frame seed (0.5313).
- **Best-epoch selection is not inflating it.** Model selection is on the
  validation split, but last-epoch numbers give the same answer; best and last
  differ by 0.000-0.008 per run.
- **Plain accuracy is higher (0.58 crop) and the gap smaller (+0.013).** Balanced
  accuracy weights every species equally, so the thin classes — where the crop
  helps more — count for more.

## Reading 0.55

The prior written into `docs/crop-readiness-plan.md` before this run — far
below the 0.754 at ten species — held. For scale: chance is 1/110 = 0.9%. The
numbers are low because the problem is harder, and because the model is small:

- **The classes are not balanced.** Training images per species run from 29
  (*D. monticola*) to 240 (*D. rotundifolia*), median 118; validation from 9
  (*D. collinsiae*) to 76 (*D. pygmaea*), median 34. Class-weighted loss
  compensates in training, but a 9-image validation class moves balanced
  accuracy in steps of 11 points, so per-class results will be noisy.
- **Validation is concentrated.** 3,959 validation images come from 308
  observers (training: 13,719 from 5,745). Any interval on these numbers has
  to resample observers, not images.
- **ResNet-18 is underfitting.** Final training loss is 0.71 in every run
  against a floor of roughly 0.43 for label smoothing 0.05 at 110 classes, and
  several runs peaked on the final epoch. More capacity or a longer schedule
  is indicated before more regularisation.

## What the run does not tell us

- **How close the misses are.** Only exact-species accuracy was logged; no
  top-k, no per-class, no confusions. A 110-species error inside one
  section (the pygmy sundews of section *Bryastrum*, for instance) is a
  different failure from a cross-continent one.
- **Whether the backbone is the ceiling.** That is the next experiment:
  `docs/species-classifier-plan.md`.

Checkpoints: `models/species-110/{full,crop}/seed-*/classifier-best.pt`.
Logs: `logs/sundew-species-110-{17903910,17910113}_*.out`.

## Rerun on `split-110-test`: the anchor for the fine-tunes

Same recipe, same seeds, on the split with a held-out test set carved from
training (`scripts/carve_test_split.py`; validation unchanged, training 13,719 ->
11,118). Run with `sbatch --export=ALL,SPLIT_NAME=split-110-test,OUT_ROOT=models/species-110-test/resnet18
scripts/hellbender_species_110.slurm`, job 17928590.

| Arm | 17 | 101 | 202 | 303 | 404 | Mean | SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.4997 | 0.4913 | 0.4999 | 0.5022 | 0.4992 | 0.4985 | 0.0042 |
| `crop` | 0.5225 | 0.5169 | 0.5050 | 0.5108 | 0.5229 | **0.5156** | 0.0077 |

| Comparison (paired over seeds) | Mean | t (4 df) | 95% CI | Wins |
| --- | ---: | ---: | --- | ---: |
| `crop` - `full`, best epoch | +0.0172 | 4.01 | [+0.005, +0.029] | 5/5 |
| `crop` - `full`, last epoch | +0.0190 | 5.74 | [+0.010, +0.028] | 5/5 |

19% less training data cost 0.026-0.030 on both arms, and the crop effect
held (+0.017 against +0.021). **0.516 crop / 0.499 full are the numbers the
fine-tuned backbones are compared against**, not 0.546 / 0.525. These models
have not seen the test split and can be scored on it at the end alongside
the fine-tunes.
