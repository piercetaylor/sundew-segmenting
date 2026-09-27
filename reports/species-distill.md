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
