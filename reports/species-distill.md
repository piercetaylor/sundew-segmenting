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

# CE, KD and KDw at 100 epochs, on the cache

Pre-registered in `docs/species-classifier-plan.md`, "Class-weighted KD"
(decision rule) and "Distillation" (carried-forward recipe, release floor).
Three arms, five paired seeds each, 100 epochs, patience off:

- CE-100c (`dinov2-s-e100-c576`, job 18016204), 50-56 min per task;
- KD-100c (`dinov2-s-kd-e100-c576`, job 18016920), 2 h 38-44 min;
- KDw-100c (`dinov2-s-kdw-e100-c576`, job 18016921), 2 h 38-42 min.

No task was preempted or failed. Generated tables:
`species-distill/e100-results.md`; diagnostics:
`species-distill/e100-diagnostics.json`.

Reproduce with
`sbatch --array=0-4 --time=08:00:00 --export=ALL,MODEL=dinov2-s,EPOCHS=100,CACHE=1 scripts/hellbender_species_finetune.slurm`,
the same with `KD=1` and with `KD=1,KD_WEIGHT=label`, then
`python scripts/summarize_species_finetune.py --models dinov2-s-kdw-e100-c576 dinov2-s-kd-e100-c576 dinov2-s-e100-c576 --arms full --title "Distillation, 100 epochs, cache" --report reports/species-distill.md --out-md reports/species-distill/e100-results.md --out-json reports/species-distill/e100-summary.json`
and
`python scripts/diagnose_species_distill.py dinov2-s-kdw-e100-c576 dinov2-s-kd-e100-c576 dinov2-s-e100-c576 dinov2-s-kdw-c576 dinov2-s-c576 > reports/species-distill/e100-diagnostics.json`.
The diagnostics script reproduces every 25-epoch diagnostic in the section
above to the third decimal place.

## Answer

Validation balanced accuracy, seed mean at the last epoch, observer
bootstrap. Teacher (5-seed DINOv2-L ensemble): 0.834.

| Arm | Last epoch [95% CI] | Plain accuracy | Top-5 | Single-model ECE (after temperature) | Share of predictions to 130+ species (true 0.599) | Teacher agreement |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| CE-100c | 0.719 [0.697, 0.744] | 0.758 | 0.924 | 0.160 (0.036) | 0.612 | 0.802 |
| KD-100c | 0.731 [0.707, 0.759] | 0.775 | 0.946 | 0.082 (0.026) | 0.627 | 0.822 |
| KDw-100c | **0.735** [0.713, 0.760] | 0.773 | 0.942 | 0.089 (0.027) | 0.616 | 0.820 |

| Paired difference, last epoch | Delta [95% CI] | Seeds won |
| --- | --- | ---: |
| **KDw-100c - CE-100c** | **+0.016 [+0.011, +0.021]** | 5/5 |
| **KD-100c - CE-100c** | **+0.012 [+0.005, +0.019]** | 5/5 |
| KDw-100c - KD-100c | +0.004 [-0.001, +0.009] | 5/5 |

The 5-seed ensembles are closer: KDw 0.751, KD 0.749, CE 0.744; KDw - CE
+0.007 [-0.001, +0.017].

By training count, per-species accuracy averaged over seeds. Intervals are
the observer bootstrap of the paired difference, 1,000 resamples:

| Training images | Species | CE-100c | KD-100c | KDw-100c | KDw - CE | KDw - KD | KD - CE |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| under 40 | 23 | 0.610 | 0.613 | 0.634 | **+0.024 [+0.005, +0.043]** | +0.021 [-0.000, +0.040] | +0.003 [-0.014, +0.028] |
| 40-79 | 28 | 0.669 | 0.667 | 0.679 | +0.010 [+0.001, +0.021] | +0.013 [+0.002, +0.025] | -0.003 [-0.015, +0.010] |
| 80-129 | 17 | 0.759 | 0.779 | 0.777 | +0.018 [+0.009, +0.028] | -0.002 [-0.009, +0.006] | +0.019 [+0.009, +0.031] |
| 130 and over | 42 | 0.795 | 0.819 | 0.810 | **+0.015 [+0.011, +0.019]** | -0.008 [-0.013, -0.003] | +0.023 [+0.019, +0.028] |

- **At length, the teacher adds accuracy instead of only moving it.** At 25
  epochs KDw beat CE by +0.003, all of it on thin species. At 100 it wins by
  +0.016, and in every bin.
- **The long schedule alone costs the thin species.** From CE-25c to
  CE-100c, species with under 40 images drop 0.021 [-0.042, -0.001] while
  those with 130+ gain 0.026. Longer CE training over-fits the thin
  species. KDw holds them at 0.634 (25 epochs: 0.643, -0.009
  [-0.028, +0.009]).
- **Unweighted KD no longer hurts thin species against CE** (+0.003), but
  only because CE-100c fell to its level. Weighting still helps them over KD:
  +0.021, with the interval touching 0.
- **Schedule gain, CE-25c to KDw-100c: +0.025** (0.710 to 0.735). About a
  third is the schedule (CE-25c to CE-100c +0.009); the rest is the teacher.
  The gap to the teacher is now 0.099, down from 0.121.

## Decision, applied as written

| Rule | Fires? |
| --- | --- |
| KD(w)-100c - CE-100c < +0.01: distillation on the training set is closed | **no**: KDw +0.016 [+0.011, +0.021], KD +0.012 [+0.005, +0.019]. Both point estimates exceed +0.01, and so does the whole KDw interval |
| KD-100 - CE-100: CI excludes 0, positive: the teacher helps at the long schedule too | **yes**, for both variants, 5/5 seeds |
| Carried-forward recipe: the highest last-epoch seed mean among the cache arms | **KDw-100c**, 0.735. Its lead over KD-100c covers 0, but either reading gives KDw: it has the higher mean, and the 25-epoch rule made KDw the KD form |
| KD variant for the transfer set: the higher of the two at 100 epochs | **KDw**. The transfer-set arm is `dinov2-s-kdw-e100-t-c576`, with expected weights on transfer rows |
| Release floor: carried-forward seed mean >= 0.75 and top-5 >= 0.95 | **not met**: 0.735 and 0.942. Nothing ships until the transfer-set run. Seed 17 alone is 0.732, top-5 0.943 |
| A run diverges or fails | no: seed SD 0.004-0.006 |

## Against the prior

| Prior | Outcome |
| --- | --- |
| CE-100c 0.725 (0.71-0.735) | held low: 0.719 |
| KD-100c 0.73 (0.715-0.75) | held: 0.731 |
| KDw-100c 0.74 (0.72-0.76) | held low: 0.735 |
| Transfer set: absolute point about 0.76 "if KD(w)-100c lands near 0.74" | the base is 0.005 lower, so the corresponding point is about 0.755 |

## Other diagnostics

- **Calibration is worse at 100 epochs, and in the other direction.** All
  three arms are under-confident: mean confidence is 0.60-0.69 against a
  plain accuracy of 0.76-0.77. The fitted temperatures are 0.72-0.81, so
  predictions need sharpening. The single-model ECE is 0.160 for CE and
  0.082-0.089 for KD, against 0.020-0.077 at 25 epochs. After one
  temperature all three fall to 0.026-0.036. Shipping needs that temperature.
  The 25-epoch point that KDw is the best calibrated no longer holds; KD and
  KDw are equal here.
- **Every arm leans toward the common species at 100 epochs**, CE included:
  a share of 0.612-0.627 against the true 0.599 (0.597-0.612 at 25 epochs).
  Weighting moderates this but no longer holds the share at the true value.
- **Fidelity rises with length.** Agreement with the teacher is 0.820-0.822
  for KD and KDw (0.789-0.801 at 25 epochs), and KL on validation falls
  from 0.46-0.56 to 0.30-0.31. By bin, the KDw KL is 0.41 on thin species
  and 0.27 on common ones.
- **Most runs were still near their best at 100 epochs.** KD best epochs are
  71-99 and KDw 82-92; CE peaks earlier (61-89). The last-epoch mean is
  within 0.003 of the best-epoch mean for KDw and within 0.005 for CE.
- **Not yet run:** int8 export of seed 17 of the carried-forward arm. Because
  the release floor is not met, it moves to the transfer-set run's winner.

# KDw-100c+T: the transfer set

Pre-registered in `docs/species-classifier-plan.md`, "Transfer set (written
2026-09-27, before any acquisition or run)", with its 2026-09-28 amendment.
KDw-100c+T (`dinov2-s-kdw-e100-t-c576`, job 18068909) trains on the training
set plus the 12,126 transfer photos, batches drawn uniformly from the union.
Transfer rows enter the KD term only, with the expected class weight. It runs
17,300 steps, the control's 100 epochs of 173, with the same warmup and cosine.
Five seeds, paired with KDw-100c on seeds and starting weights. Tasks took
2 h 40-41 min (93-94 s per 173 steps); none was preempted. The transfer share
of batches was 0.515-0.530 (0.522 in the union). Generated tables:
`species-distill/t-results.md`; diagnostics: `species-distill/t-diagnostics.json`.

Reproduce with
`sbatch --array=0-4 --time=08:00:00 --export=ALL,MODEL=dinov2-s,KD=1,KD_WEIGHT=label,EPOCHS=100,CACHE=1,TRANSFER=1 scripts/hellbender_species_finetune.slurm`,
then
`python scripts/summarize_species_finetune.py --models dinov2-s-kdw-e100-t-c576 dinov2-s-kdw-e100-c576 dinov2-s-e100-c576 --arms full --title "Transfer set, KDw-100c+T" --report reports/species-distill.md --out-md reports/species-distill/t-results.md --out-json reports/species-distill/t-summary.json`
and
`python scripts/diagnose_species_distill.py dinov2-s-kdw-e100-t-c576 dinov2-s-kdw-e100-c576 dinov2-s-e100-c576 > reports/species-distill/t-diagnostics.json`.
The diagnostics reproduce the KDw-100c and CE-100c rows of the 100-epoch
section.

## Answer

Validation balanced accuracy, seed mean at the last epoch, observer
bootstrap. Teacher (5-seed DINOv2-L ensemble): 0.834.

| Arm | Last epoch [95% CI] | Plain accuracy | Top-5 | Single-model ECE (after temperature) | Share of predictions to 130+ species (true 0.599) | Teacher agreement |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| CE-100c | 0.719 [0.697, 0.744] | 0.758 | 0.924 | 0.160 (0.036) | 0.612 | 0.802 |
| KDw-100c | 0.735 [0.713, 0.760] | 0.773 | 0.942 | 0.089 (0.027) | 0.616 | 0.820 |
| **KDw-100c+T** | **0.768** [0.747, 0.794] | 0.803 | **0.959** | 0.101 (0.024) | 0.621 | 0.856 |

| Paired difference, last epoch | Delta [95% CI] | Seeds won |
| --- | --- | ---: |
| **KDw-100c+T - KDw-100c** (primary) | **+0.033 [+0.026, +0.042]** | 5/5 |
| KDw-100c+T - CE-100c | +0.049 [+0.042, +0.059] | 5/5 |

The 5-seed ensembles: +T 0.780 (top-5 0.964), KDw 0.751, CE 0.744; +T - KDw
+0.029 [+0.017, +0.040]. Seed SD fell from 0.006 to 0.002 (best epoch).

By training count, per-species accuracy averaged over seeds. Intervals are
the observer bootstrap of the paired difference, 1,000 resamples:

| Training images | Species | CE-100c | KDw-100c | KDw-100c+T | +T - KDw | +T - CE |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| under 40 | 23 | 0.610 | 0.634 | 0.677 | **+0.042 [+0.017, +0.077]** | +0.066 [+0.041, +0.099] |
| 40-79 | 28 | 0.669 | 0.679 | 0.719 | +0.040 [+0.023, +0.056] | +0.050 [+0.031, +0.067] |
| 80-129 | 17 | 0.759 | 0.777 | 0.792 | +0.014 [+0.001, +0.028] | +0.032 [+0.016, +0.049] |
| 130 and over | 42 | 0.795 | 0.810 | 0.842 | +0.032 [+0.025, +0.039] | +0.047 [+0.039, +0.055] |

- **The transfer set is the largest single gain in the student line.**
  +0.033, twice the teacher's gain over CE at 100 epochs (+0.016). The gap to
  the teacher falls from 0.099 to 0.066.
- **The thin species gained most, against the prior.** The pre-registration
  expected little for them (+0.00 to +0.015), since they got only 1-15 new
  photos each. They gained +0.042. The gain is therefore not mainly more
  photos of the thin species: matching the teacher on more photos of other
  species also sharpens the boundaries around the thin ones. This is the
  fidelity reading of the KD-25 section (Stanton et al. 2021): with the
  training images alone, the student could fit the teacher only where the
  teacher had already fitted the labels.
- **Fidelity rose, over-fitting fell.** Validation agreement with the teacher
  0.856 (KDw-100c 0.820), KL 0.21 (0.31), and 0.28 on thin species (0.41).
  The final training loss is 0.12 against 0.05 for KDw-100c: half of each
  batch is transfer photos, so each training photo is seen about 48 times, not 100.

## Decision, applied as written

| Rule | Fires? |
| --- | --- |
| +T - KD(w)-100c: CI excludes 0, positive: the transfer set is in the student recipe; +T is carried forward | **yes**: +0.033 [+0.026, +0.042], 5/5 |
| gain < +0.01 (amended 2026-09-28): the transfer set is dropped | no |
| Teacher agrees with the iNaturalist ID on under 0.6 of captive photos: "captive targets suspect" | no: 0.681 on the 659 captive photos (below) |
| Carried-forward recipe: the highest last-epoch seed mean | **KDw-100c+T**, 0.768 |
| Shipped model: seed 17 | seed 17 of KDw-100c+T: 0.763, top-5 0.959 (last epoch) |
| Release floor: seed mean >= 0.75 and top-5 >= 0.95 | **met**: 0.768 and 0.959 |
| A run diverges or fails | no: seed SD 0.002-0.003 |

The floor is met on validation. By the test-protocol amendment of
2026-09-29, what remains before release is, in order: the int8 ONNX export
of seed 17 with its accuracy change against fp32; its temperature, fitted on
validation; and the single scoring of the test split.

## Against the prior

| Prior | Outcome |
| --- | --- |
| overall +0.02 (+0.005 to +0.035) | held high: +0.033 |
| under 40: +0.00 to +0.015 | **missed, favourably**: +0.042 |
| 130 and over: +0.01 to +0.03 | missed narrowly: +0.032 |
| absolute about 0.755 (restated after KDw-100c) | missed, favourably: 0.768 |

## Other diagnostics

- **Calibration**: still under-confident (mean confidence 0.70 against plain
  accuracy 0.80). Fitted temperature 0.76; ECE 0.101 before, 0.024 after,
  the best of the three arms after temperature.
- **Prediction share**: 0.621 to the commonest 42 species against a true
  0.599, the same lean as KDw-100c (0.616).
- **Best epochs** 79-88; the last-epoch mean is within 0.004 of the
  best-epoch mean.

## The transfer set under the teacher

Pre-registered for this arm. The teacher (5-seed ensemble, tau 1) and the
shipped student (seed 17, last epoch) run over the 12,126 transfer photos
with the validation transform (224 px). The iNaturalist ID is the record's
taxon; 10 photos of 6 infraspecific taxa are mapped to their species. The
teacher reproduces its validation score (0.834) and its stored validation
targets exactly. Job 18079959, 2.5 min on one A100. Output:
`species-distill/t-transfer-diagnostics.json`; per-photo top-1, max-probability
and probabilities in `transfer-diagnostics.npz` beside the seed-17 checkpoint
(not in Git). Reproduce with
`sbatch scripts/hellbender_transfer_diagnostics.slurm`
(`scripts/diagnose_transfer_set.py`).

| Source | Photos | Species | Teacher max-prob | Teacher = iNat ID | Student = teacher | Student = iNat ID | KL(teacher \|\| student) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) wild, research grade | 7,970 | 97 | 0.771 | 0.898 | 0.953 | 0.895 | 0.077 |
| (b) captive | 659 | 48 | 0.613 | **0.681** | 0.882 | 0.675 | 0.115 |
| (c) needs-ID or casual | 3,497 | 107 | 0.685 | 0.744 | 0.908 | 0.747 | 0.116 |
| all | 12,126 | 110 | 0.738 | 0.842 | 0.936 | 0.840 | 0.091 |

By the training count of the iNaturalist-ID species. The share columns are of
the whole set:

| Training images | Photos | Share (a) | Share (b) | Share (c) | Teacher max-prob | Teacher = iNat ID | Student = teacher | KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| under 40 | 201 | 0.005 | 0.001 | 0.011 | 0.716 | 0.697 | 0.910 | 0.104 |
| 40-79 | 417 | 0.010 | 0.001 | 0.024 | 0.744 | 0.801 | 0.906 | 0.115 |
| 80-129 | 826 | 0.032 | 0.002 | 0.034 | 0.728 | 0.793 | 0.925 | 0.099 |
| 130 and over | 10,682 | 0.610 | 0.051 | 0.220 | 0.738 | 0.850 | 0.939 | 0.089 |

- **Captive rule: does not fire.** The teacher agrees with the iNaturalist ID
  on 0.681 of captive photos, above the 0.6 threshold. The captive targets are
  not recorded as suspect. They are still the least reliable source. The
  teacher's mean max-probability there is 0.613, against 0.771 on wild
  research-grade photos. Captive agreement is lowest on the few captive photos
  of mid-count species (80-129: 0.318 on 22 photos), and those cells are
  small.
- **Confidence and agreement follow the source.** Research-grade (a) is the
  teacher's domain: 0.898 agreement, against its plain accuracy of 0.856 on
  validation (mean max-probability 0.742 there, 0.771 here). Needs-ID (c) is lower (0.744, max-prob 0.685). The
  disagreements with the iNaturalist ID are partly teacher errors and partly
  wrong community IDs, which the rows cannot tell apart. For (b) and (c)
  neither the IDs nor the teacher is ground truth.
- **The student fits the teacher on the transfer set**: 0.936 top-1 agreement
  and KL 0.091, against 0.856 and 0.21 on validation. These are training
  photos for the student, so the gap is fit, not generalisation. It fits worst
  where the teacher is least sure: (b) 0.882, (c) 0.908.
- **Few photos of the thin species.** Species under 40 training images have
  201 transfer photos (0.017 of the set; 1-15 per species), 131 of them from
  needs-ID (c). The teacher agrees with those IDs least often (0.697; 0.603
  on (c)). 0.881 of the set belongs to the 42 species with 130 and over.
- **This fits the fidelity reading of the thin-species gain.** The teacher
  puts 0.097 of its probability mass on transfer photos onto the 23 thin
  species, about 1,180 photo-equivalents, against 201 photos with a thin ID.
  Its top-1 lands on a thin species for 0.026 of photos. Most of the
  thin-species KD signal therefore comes from the teacher's soft targets on
  photos of other species, not from new photos of the thin ones. This is
  consistent with the +0.042 gain there. It does not prove it.
