# Held-out test: read once

**In short.** The shipped model (int8, seed 17) scores **0.775** balanced
accuracy on the held-out test split, top-5 0.962; the teacher ensemble scores
0.842. Distillation holds: +0.043 over plain fine-tuning (CE-100c), 5 of 5
seeds. Species with under 40 training photos remain the weak spot (0.652).

The test split (2,601 images, 1,095 observers, 110 species) was scored once,
by job 18082949 on 2026-09-29, under the protocol in the
[plan](../species-classifier-plan.md): the 2026-09-26 test protocol, its
2026-09-27 amendment, the shipped-student protocol of 2026-09-29 and the
scoring details recorded before the run. The same job first recomputed
validation, and every reproduction check passed (the shipped int8 model
within 1e-3, all stored predictions bit for bit). Generated tables:
[test-results.md](species-test/test-results.md) and `.json`. Test logits are
in `models/species-110/test-scoring/` (not in Git).

Reproduce: `sbatch --export=ALL,SPLIT=test scripts/hellbender_species_test.slurm`.
It refuses to run while `docs/reports/species-test/test-results.json` exists.

## Result

Balanced accuracy, observer-grouped bootstrap, 2,000 resamples. Students at
the last epoch; the teacher and the anchor at the best epoch.

| Item | Validation | Test [95% CI] | Top-5 (test) | ECE after T (test) |
| --- | ---: | --- | ---: | ---: |
| **Shipped: KDw-100c+T seed 17, int8 ONNX, T 0.74** | 0.764 | **0.775 [0.752, 0.794]** | **0.962** | 0.031 |
| KDw-100c+T, seed mean | 0.768 | 0.783 [0.763, 0.802] | 0.967 | 0.026 |
| KDw-100c+T, 5-seed ensemble (not shipped) | 0.780 | 0.796 [0.774, 0.816] | 0.970 | 0.028 |
| CE-100c, seed mean | 0.719 | 0.740 [0.721, 0.758] | 0.934 | 0.043 |
| DINOv2-L teacher, 5-seed ensemble | 0.834 | 0.842 [0.822, 0.862] | 0.974 | 0.025 |
| ResNet-18 anchor, full, seed mean | 0.498 | 0.522 [0.503, 0.540] | 0.828 | 0.054 |
| ResNet-18 anchor, crop, seed mean | 0.516 | 0.548 [0.529, 0.567] | 0.834 | 0.046 |

- **Above the release floor** (0.75, top-5 0.95). The floor was judged on
  validation, where it was met; the test is reported, not a gate.
- **Distillation holds.** KDw-100c+T minus CE-100c: +0.043 [+0.033, +0.053],
  5/5 seeds (validation +0.049). Positive in every bin: under 40 +0.063, 40-79
  +0.021, 80-129 +0.030, 130 and over +0.052.
- **Test is slightly easier than validation for every model**, by +0.008
  (teacher) to +0.032 (anchor, crop). No student, teacher or ensemble falls
  outside its validation interval; the anchor seed means do, as do four
  single anchor seeds and one CE-100c seed. A shift that lifts every model
  points to the split, not to over-fitting on validation: the test split was
  carved from training by observer and has 1,095 observers against
  validation's 308.
- **The gap to the teacher is about the same**: 0.059 for the student seed
  mean on test, 0.066 on validation.

## Where the shipped model is weak

Per-species accuracy by training count, test:

| Training images | Species | Shipped | Teacher ensemble |
| --- | ---: | ---: | ---: |
| under 40 | 23 | 0.652 [0.562, 0.727] | 0.777 |
| 40-79 | 28 | 0.744 [0.689, 0.788] | 0.828 |
| 80-129 | 17 | 0.839 [0.800, 0.872] | 0.879 |
| 130 and over | 42 | 0.837 [0.819, 0.855] | 0.873 |

The thin species are 0.125 below the teacher, against 0.036 on the commonest.
Seed 17 in PyTorch scores 0.683 on this bin (seed mean 0.677), so int8 costs
the thin species 0.031 on test. Overall int8 costs 0.008 on test (0.775
against 0.784 for seed 17 in PyTorch), against nothing on validation, where
the 0.01 rule was applied. These 23 species have few test photos each, and
the interval is wide.

## Calibration

With its validation temperature (0.74), the shipped model's test ECE is
0.031 (0.100 before). Every model is under-confident before temperature on
both splits, and one temperature fitted on validation carries to test.

## Caveats

- **The shipped int8 numbers are CPU-specific.** Dynamic int8 kernels differ
  between CPU families: the model was scored on Intel (AVX512-VNNI), exported
  on AMD. On validation the two agree on 0.991 of top-1 predictions. A
  browser's WebAssembly backend is a third implementation, not measured.
- **Browser preprocessing is not measured**: training read a 576 px LANCZOS
  cache and ignored EXIF orientation ([species-release.md](species-release.md)).
- **Open set is not tested.** Every test photo is one of the 110 species. The
  model names one of them for any photo, including other Drosera and other
  genera.
- Per-bin intervals are not corrected for four comparisons.
