# Distillation KD-25: no net gain; the teacher shifts accuracy toward common species

The first arm of the distillation run pre-registered in
`docs/species-classifier-plan.md`, "Distillation into DINOv2-S (written
2026-09-27, before any run)". The DINOv2-L `full` 5-seed ensemble (best-epoch
checkpoints) teaches `dinov2-s`, 25 epochs, KL to the mean softmax, tau 1, no
label term, no class weights. Everything else, including the batches and
starting weights, matches the bake-off's label-trained `dinov2-s` (CE-25),
seed for seed. The generated table is `species-distill/kd25-results.md`.

Reproduce with
`sbatch --array=0-4 --export=ALL,MODEL=dinov2-s,KD=1 scripts/hellbender_species_finetune.slurm`,
then
`python scripts/summarize_species_finetune.py --models dinov2-s-kd dinov2-s --arms full --title "Distillation KD-25" --report reports/species-distill.md --out-md reports/species-distill/kd25-results.md --out-json reports/species-distill/kd25-summary.json`.

The run:

- **Job**: 18012985, five A100 tasks of 50-52 min (4.3 A100-h), 2026-09-27.
- **Epochs**: 117-122 s, inside the 110-180 s estimate. The five teacher
  forwards hid under JPEG decoding.
- **Teacher check**: the rebuilt ensemble scored 0.8341 on full validation,
  inside the required 0.834 +- 0.002.
- **Smoke test**: passed beforehand (job 18012907).

## Answer

`full` arm, 224 px, `split-110-test`, validation balanced accuracy, seed mean
at the **last epoch** (the pre-registered metric), observer-grouped bootstrap.

| Arm | Last epoch [95% CI] | Plain accuracy | Single-model ECE | 5-seed ensemble | Top-5 (ensemble) |
| --- | --- | ---: | ---: | --- | ---: |
| CE-25 (`dinov2-s`) | 0.712 [0.691, 0.736] | 0.741 | 0.079 | 0.735 [0.712, 0.761] | 0.946 |
| KD-25 (`dinov2-s-kd`) | 0.711 [0.688, 0.736] | 0.750 | **0.030** | 0.732 [0.708, 0.759] | 0.950 |

**KD-25 - CE-25: -0.001 [-0.008, +0.006]**, 0 of 5 paired seeds won (best
epoch: -0.002 [-0.008, +0.004]). The ensembles differ by -0.003 [-0.013, +0.005].

Fidelity to the teacher: top-1 agreement on validation is 0.80. The mean KL
from the teacher is 0.46 on validation, against a final training KD loss of
0.25-0.26.

## Decision, applied as written

| Rule | Fires? |
| --- | --- |
| KD-25 - CE-25: CI excludes 0, positive | **no**: -0.001 [-0.008, +0.006] |
| KD-100 - CE-100 | not run yet |
| Both differences < +0.01 | half known: KD-25 is below +0.01; the rule needs KD-100 too |
| A run diverges (seed SD > 0.02) or fails | no: all 25 epochs, SD 0.004 |

The teacher does not raise balanced accuracy at a matched 25-epoch schedule.

## Against the prior

| Prior | Outcome |
| --- | --- |
| KD-25 0.735 (0.72-0.75) | **missed**: 0.711, below the range |
| A single student 5-9 points below the teacher | missed: 11 points below (0.824) |

Fable's estimate (0.74) missed the same way.

## Why: the gain and the loss cancel across species

This is the per-class diagnostic the pre-registration fixed. Per-species
validation accuracy is averaged over seeds at the last epoch, and binned by
training count. Intervals are the observer bootstrap of the paired
difference, 1,000 resamples.

| Training images | Species | Teacher | CE-25 | KD-25 | KD - CE [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| under 40 | 23 | 0.758 | 0.639 | 0.601 | **-0.039 [-0.066, -0.022]** |
| 40-79 | 28 | 0.792 | 0.660 | 0.660 | +0.000 [-0.015, +0.016] |
| 80-129 | 17 | 0.854 | 0.750 | 0.746 | -0.003 [-0.014, +0.007] |
| 130 and over | 42 | 0.868 | 0.771 | 0.791 | **+0.020 [+0.015, +0.024]** |

**KD moved accuracy from the rare species to the common ones.** The teacher
itself does well on thin species (0.758), so the loss is not inherited from
it. It comes from the unweighted KD term: each image counts once, so the
common species dominate the gradient, while the CE baseline re-weights toward
rare ones. The design assumed the student would inherit the balanced prior
by matching a class-weighted teacher. On the thin species it did not.
Consistent with that:

- plain accuracy rose (+0.009);
- the share of predictions going to the 42 commonest species rose from 0.596
  to 0.612, against a true share of 0.599.

The pre-registration named this outcome and its follow-up: **if KD loses to
CE on species with under 40 training images, a class-weighted KD term is the
follow-up.**

A second reading, which the first does not exclude: on its own training
images the teacher's targets are close to one-hot (its training loss ends at
0.72 against a label-smoothing floor of 0.43). The student fits them on
training images (KL 0.26) far better than on unseen ones (0.46). That is the
case Stanton et al. 2021 (arXiv:2106.05945) describe: fidelity is limited by
the transfer data. More or different images, or a longer schedule, are the
remedies (Beyer et al. 2022, arXiv:2106.05237).

## What KD did buy

- **Calibration**: single-model ECE 0.030 against 0.079, and mean confidence
  0.72 against 0.66 for accuracy 0.75 against 0.74. The pre-registration
  expected the student to inherit the ensemble's under-confidence. It did
  not: an unweighted, unsmoothed KL target leaves it better calibrated than
  the class-weighted, smoothed CE student. For an abstain threshold this
  matters, though a temperature fitted on validation can recover much of it
  for CE too.
- **Plain accuracy** +0.009: better for what users mostly photograph, the
  common species, at the cost of the rare ones.

Neither is an input to the rule.

## Caveats

- **One schedule.** 25 epochs is short for distillation (Beyer et al. 2022);
  KD-100 is not yet run.
- **The teacher checkpoints were chosen on validation** (best epoch), a
  negligible leak recorded in the plan.
- **The per-class bins were fixed in advance** (the teacher analysis's
  bins); the interval for each bin is not corrected for four comparisons.
  The two extreme bins are far from 0 either way.

## Next

Recorded as an amendment in the plan, made after seeing this result:

1. **A fast data pipeline first.** Every run is bound by decoding the
   original JPEGs. A cache of full frames with the short side at 576 px is
   built, and CE-25 is rerun on it as an equivalence check.
2. **Class-weighted KD** (the pre-registered follow-up), and the
   **100-epoch** CE and KD arms, on the fast pipeline.
3. **A transfer set** of photos the teacher never trained on, as its own
   pre-registered arm.

# Class-weighted KD, 25 epochs, on the cache

Pre-registered in `docs/species-classifier-plan.md`, "Class-weighted KD
(written 2026-09-27, before any run)". Everything runs on the adopted 576 px
cache:

- CE-25c (`dinov2-s-c576`, job 18014847);
- KD-25c (`dinov2-s-kd-c576`, job 18015109);
- KDw-25c (`dinov2-s-kdw-c576`, job 18015110): the KL of each image weighted
  by its label's class weight, normalised as the weighted CE is.

Five seeds each, 25 epochs, paired. KD tasks took 41-42 min (93 s per
epoch). Two KD-25c tasks were preempted and resumed. Generated table:
`species-distill/kdw25-results.md`.

Reproduce with
`sbatch --array=0-4 --export=ALL,MODEL=dinov2-s,KD=1,CACHE=1 scripts/hellbender_species_finetune.slurm`,
the same with `KD_WEIGHT=label`, then
`python scripts/summarize_species_finetune.py --models dinov2-s-kdw-c576 dinov2-s-kd-c576 dinov2-s-c576 --arms full --title "Class-weighted KD, 25 epochs, cache" --report reports/species-distill.md --out-md reports/species-distill/kdw25-results.md --out-json reports/species-distill/kdw25-summary.json`.

## Answer

Validation balanced accuracy, seed mean at the last epoch, observer
bootstrap.

| Arm | Last epoch [95% CI] | Plain accuracy | Single-model ECE | Share of predictions to 130+ species (true 0.599) | Teacher agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| CE-25c | 0.710 [0.689, 0.733] | 0.739 | 0.077 | 0.598 | - |
| KD-25c | 0.712 [0.689, 0.736] | 0.752 | 0.032 | 0.612 | 0.801 |
| KDw-25c | **0.713** [0.692, 0.738] | 0.741 | **0.020** | 0.597 | 0.789 |

| Paired difference, last epoch | Delta [95% CI] | Seeds won |
| --- | --- | ---: |
| **KDw-25c - CE-25c** (primary) | **+0.003 [+0.001, +0.007]** | 5/5 |
| KDw-25c - KD-25c | +0.001 [-0.004, +0.009] | 4/5 |
| KD-25c - CE-25c | +0.002 [-0.005, +0.008] | 3/5 |

By training count, per-species accuracy averaged over seeds. Intervals are
the observer bootstrap of the paired difference, 1,000 resamples:

| Training images | Species | CE-25c | KD-25c | KDw-25c | KDw - CE | KDw - KD | KD - CE |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| under 40 | 23 | 0.631 | 0.603 | 0.643 | +0.012 [+0.004, +0.022] | **+0.041 [+0.021, +0.072]** | -0.029 [-0.058, -0.009] |
| 40-79 | 28 | 0.666 | 0.659 | 0.666 | +0.000 [-0.005, +0.008] | +0.007 [-0.002, +0.019] | -0.007 [-0.020, +0.004] |
| 80-129 | 17 | 0.742 | 0.741 | 0.744 | +0.002 [-0.004, +0.010] | +0.003 [-0.006, +0.012] | -0.000 [-0.009, +0.008] |
| 130 and over | 42 | 0.769 | 0.795 | 0.770 | +0.001 [-0.003, +0.004] | -0.025 [-0.030, -0.020] | **+0.026 [+0.021, +0.031]** |

- **The KD-25 pattern replicates on the cache.** Unweighted KD again takes
  accuracy from the thin species (-0.029) and gives it to the common ones
  (+0.026).
- **Weighting undoes the reallocation, as designed.** Thin species gain
  +0.041 over unweighted KD, and the prediction share returns to the true
  0.60.
- **But it also removes the common-species gain.** What is left over CE is a
  small, even gain: +0.012 on thin species, about 0 elsewhere, net +0.003.
  At 25 epochs the teacher moves accuracy between species far more than it
  adds.

## Decision, applied as written

| Rule | Fires? |
| --- | --- |
| KDw-25c - CE-25c: CI excludes 0, positive: class-balanced distillation is in the student recipe | **yes**, narrowly: +0.003 [+0.001, +0.007], 5/5 |
| KDw-25c - KD-25c on the under-40 bin: CI excludes 0, positive: KDw is the KD form from here on | **yes**: +0.041 [+0.021, +0.072] |
| KDw-25c within +-0.01 of CE-25c: weighting kept; the teacher is not the lever at 25 epochs | **yes**: the gain is real but a third of a point |
| Which KD variant runs at 100 epochs: the higher, or both if the CI on their difference covers 0 | **both**: +0.001 [-0.004, +0.009] covers 0. KD-100c job 18016920, KDw-100c job 18016921 (CE-100c, job 18016204, already running) |
| A run diverges or fails | no: seed SD 0.003-0.006 |

The first and third rows both fire, and they agree. Class-balanced
distillation is kept because it is significantly, if barely, better than
CE, and it is better calibrated. The effect is far too small to close the
0.11 gap to the teacher. The 100-epoch arms and the transfer set are what
remain.

The second row names KDw the KD form, but the selection row, written for
this exact decision, sends both variants to 100 epochs. Both run, as
written. The 100-epoch KD variant carried into the transfer set is chosen
by the rule there, the higher of the two.

## Against the prior

| Prior | Outcome |
| --- | --- |
| CE-25c 0.712 (0.70-0.725) | held: 0.710 |
| KD-25c 0.711 (0.70-0.72) | held: 0.712 |
| KDw-25c 0.72 (0.705-0.735) | held low: 0.713 |
| KDw - CE, under 40: within +-0.015 | held: +0.012 |
| KDw - CE, 130+: +0.005 to +0.015 | **missed**: +0.001; weighting removed all of the common-species gain |
| Prediction share back to about 0.60 | held: 0.597 |
| KDw ECE between KD (0.030) and CE (0.079) | **missed, favourably**: 0.020, the best of the three |

## Other diagnostics

- **The teacher is right about every training image.** The teacher's top-1
  disagrees with the label on none of them (0.000, all five seeds). So on
  training images its targets carry no correction of label noise, only the
  shape of the rest of the distribution. This supports the transfer-set
  reading.
- **The two weightings nearly agree on training images.** The mean
  |w_label - w_expected| is 0.091 against weights of 0.5-4.4. So the
  expected weight fixed for unlabeled photos is close to what KDw used.
- **Fidelity:** KDw agrees with the teacher slightly less than KD does (0.789
  vs 0.801), and its unweighted KL is higher (0.558 vs 0.459). It matches the
  teacher less on the common species it no longer favours.
