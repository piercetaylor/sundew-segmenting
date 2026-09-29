# Distillation: from DINOv2-L into DINOv2-S

**In short.** The student learns to copy the teacher's predicted
probabilities (knowledge distillation, KD) instead of, or besides, the
labels. Four stages, each paired over five seeds against a student trained on
labels alone (CE), validation balanced accuracy at the last epoch:

| Stage | Change | Gain [95% CI] | Student |
| --- | --- | --- | ---: |
| KD-25 | KD, 25 epochs | -0.001 [-0.008, +0.006] vs CE-25 | 0.711 |
| KDw-25c | KD weighted by class, on a 576 px frame cache | +0.003 [+0.001, +0.007] vs CE-25c | 0.713 |
| KDw-100c | 100 epochs | +0.016 [+0.011, +0.021] vs CE-100c | 0.735 |
| KDw-100c+T | plus 12,126 unlabelled transfer photos | **+0.033 [+0.026, +0.042] vs KDw-100c** | **0.768** |

KDw-100c+T is the shipped recipe. It meets the release floor (seed mean >=
0.75, top-5 >= 0.95) and narrows the gap to the teacher (0.834) from 0.121
(KDw-25c) to 0.066. Unweighted KD shifts accuracy from rare to common species;
weighting by class stops that; the long schedule and the transfer set are what add accuracy.

Teacher throughout: the DINOv2-L `full` 5-seed ensemble (best-epoch
checkpoints), 0.834 on validation. Student: `dinov2-s`, `full` arm, 224 px,
`split-110-test`. All intervals are observer-grouped bootstraps; per-bin
intervals use 1,000 resamples and are not corrected for four comparisons.
"Bins" group the 110 species by training images: under 40 (23 species),
40-79 (28), 80-129 (17), 130 and over (42). Each stage was pre-registered in
the [plan](../species-classifier-plan.md) before it ran.

## 1. KD-25: no net gain; accuracy moves to common species

Plan: "Distillation into DINOv2-S (written 2026-09-27, before any run)". KL
to the teacher's mean softmax, tau 1, no label term, no class weights, 25
epochs; everything else, including batches and starting weights, matches the
bake-off's CE-25 seed for seed. Job 18012985, 5 A100 tasks of 50-52 min,
2026-09-27; the rebuilt teacher scored 0.8341 (required 0.834 +- 0.002). Generated table:
[kd25-results.md](species-distill/kd25-results.md).

Reproduce with
`sbatch --array=0-4 --export=ALL,MODEL=dinov2-s,KD=1 scripts/hellbender_species_finetune.slurm`,
then
`python scripts/summarize_species_finetune.py --models dinov2-s-kd dinov2-s --arms full --title "Distillation KD-25" --report docs/reports/species-distill.md --out-md docs/reports/species-distill/kd25-results.md --out-json docs/reports/species-distill/kd25-summary.json`.

| Arm | Last epoch [95% CI] | Plain accuracy | Single-model ECE | 5-seed ensemble | Top-5 (ensemble) |
| --- | --- | ---: | ---: | --- | ---: |
| CE-25 (`dinov2-s`) | 0.712 [0.691, 0.736] | 0.741 | 0.079 | 0.735 [0.712, 0.761] | 0.946 |
| KD-25 (`dinov2-s-kd`) | 0.711 [0.688, 0.736] | 0.750 | **0.030** | 0.732 [0.708, 0.759] | 0.950 |

**KD-25 - CE-25: -0.001 [-0.008, +0.006]**, 0 of 5 seeds won (best epoch
-0.002 [-0.008, +0.004]; ensembles -0.003 [-0.013, +0.005]).

| Training images | Teacher | CE-25 | KD-25 | KD - CE [95% CI] |
| --- | ---: | ---: | ---: | --- |
| under 40 | 0.758 | 0.639 | 0.601 | **-0.039 [-0.066, -0.022]** |
| 40-79 | 0.792 | 0.660 | 0.660 | +0.000 [-0.015, +0.016] |
| 80-129 | 0.854 | 0.750 | 0.746 | -0.003 [-0.014, +0.007] |
| 130 and over | 0.868 | 0.771 | 0.791 | **+0.020 [+0.015, +0.024]** |

**KD moved accuracy from the rare species to the common ones.** The teacher
does well on thin species (0.758), so the loss is not inherited. It comes
from the unweighted KD term: each image counts once, so common species
dominate, while CE re-weights toward rare ones. Plain accuracy rose (+0.009)
and the share of predictions going to the 42 commonest species rose from
0.596 to 0.612 (true share 0.599). A second reading: the student
matches the teacher far better on training images (KL 0.26) than on unseen
ones (0.46), so fidelity is limited by the transfer data (Stanton et al.
2021, arXiv:2106.05945); more images or a longer schedule are the remedies
(Beyer et al. 2022, arXiv:2106.05237).

KD did improve calibration (ECE 0.030 against 0.079), not an input to the rule.

| Rule | Fires? |
| --- | --- |
| KD-25 - CE-25: CI excludes 0, positive | **no**: -0.001 [-0.008, +0.006] |
| KD-100 - CE-100 | not run yet |
| Both differences < +0.01 | half known: KD-25 is below +0.01; the rule needs KD-100 too |
| A run diverges (seed SD > 0.02) or fails | no: all 25 epochs, SD 0.004 |

| Prior | Outcome |
| --- | --- |
| KD-25 0.735 (0.72-0.75) | **missed**: 0.711, below the range |
| A single student 5-9 points below the teacher | missed: 11 points below (0.824) |

The pre-registration named the follow-up for this outcome: class-weighted KD.
An amendment made after this result (2026-09-27) added a fast 576 px frame
cache, the 100-epoch arms on it, and a transfer set. The cache was adopted by
its equivalence rule: CE-25 on the cache scores 0.710, a difference of -0.002
[-0.006, +0.001], inside [-0.015, +0.015]
([cache-equivalence.md](species-distill/cache-equivalence.md)).

## 2. KDw-25c: class weighting undoes the shift, small net gain

Plan: "Class-weighted KD (written 2026-09-27, before any run)". Three arms on
the cache, five paired seeds, 25 epochs: CE-25c (`dinov2-s-c576`, job
18014847), KD-25c (`dinov2-s-kd-c576`, job 18015109), KDw-25c
(`dinov2-s-kdw-c576`, job 18015110; each image's KL weighted by its label's
class weight, normalised as the weighted CE is). Two KD-25c tasks were
preempted and resumed. Generated table: [kdw25-results.md](species-distill/kdw25-results.md).

Reproduce with
`sbatch --array=0-4 --export=ALL,MODEL=dinov2-s,KD=1,CACHE=1 scripts/hellbender_species_finetune.slurm`,
the same with `KD_WEIGHT=label`, then
`python scripts/summarize_species_finetune.py --models dinov2-s-kdw-c576 dinov2-s-kd-c576 dinov2-s-c576 --arms full --title "Class-weighted KD, 25 epochs, cache" --report docs/reports/species-distill.md --out-md docs/reports/species-distill/kdw25-results.md --out-json docs/reports/species-distill/kdw25-summary.json`.

| Arm | Last epoch [95% CI] | Plain accuracy | Single-model ECE | Share to 130+ species (true 0.599) | Teacher agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| CE-25c | 0.710 [0.689, 0.733] | 0.739 | 0.077 | 0.598 | - |
| KD-25c | 0.712 [0.689, 0.736] | 0.752 | 0.032 | 0.612 | 0.801 |
| KDw-25c | **0.713** [0.692, 0.738] | 0.741 | **0.020** | 0.597 | 0.789 |

| Paired difference | Delta [95% CI] | Seeds won |
| --- | --- | ---: |
| **KDw-25c - CE-25c** (primary) | **+0.003 [+0.001, +0.007]** | 5/5 |
| KDw-25c - KD-25c | +0.001 [-0.004, +0.009] | 4/5 |
| KD-25c - CE-25c | +0.002 [-0.005, +0.008] | 3/5 |

| Training images | CE-25c | KD-25c | KDw-25c | KDw - CE | KDw - KD |
| --- | ---: | ---: | ---: | --- | --- |
| under 40 | 0.631 | 0.603 | 0.643 | +0.012 [+0.004, +0.022] | **+0.041 [+0.021, +0.072]** |
| 40-79 | 0.666 | 0.659 | 0.666 | +0.000 [-0.005, +0.008] | +0.007 [-0.002, +0.019] |
| 80-129 | 0.742 | 0.741 | 0.744 | +0.002 [-0.004, +0.010] | +0.003 [-0.006, +0.012] |
| 130 and over | 0.769 | 0.795 | 0.770 | +0.001 [-0.003, +0.004] | -0.025 [-0.030, -0.020] |

The KD-25 shift replicates on the cache (KD - CE: under 40 -0.029
[-0.058, -0.009], 130+ +0.026 [+0.021, +0.031]). Weighting undoes it (thin species
+0.041 over KD, prediction share back to 0.60), but also removes the
common-species gain. What remains over CE is small: +0.003 net. At 25 epochs
the teacher moves accuracy between species far more than it adds. The
teacher's top-1 matches the label on every training image, so there its
targets correct no labels: another pointer to the transfer set.

| Rule | Fires? |
| --- | --- |
| KDw-25c - CE-25c: CI excludes 0, positive: class-balanced distillation is in the student recipe | **yes**, narrowly: +0.003 [+0.001, +0.007], 5/5 |
| KDw-25c - KD-25c on the under-40 bin: CI excludes 0, positive: KDw is the KD form from here on | **yes**: +0.041 [+0.021, +0.072] |
| KDw-25c within +-0.01 of CE-25c: weighting kept; the teacher is not the lever at 25 epochs | **yes**: the gain is real but a third of a point |
| Which KD variant runs at 100 epochs: the higher, or both if the CI on their difference covers 0 | **both**: +0.001 [-0.004, +0.009] covers 0. KD-100c job 18016920, KDw-100c job 18016921 (CE-100c job 18016204) |
| A run diverges or fails | no: seed SD 0.003-0.006 |

The second row names KDw the KD form, but the selection row, written for this
decision, sends both variants to 100 epochs; both ran as written.

| Prior | Outcome |
| --- | --- |
| CE-25c 0.712 (0.70-0.725) | held: 0.710 |
| KD-25c 0.711 (0.70-0.72) | held: 0.712 |
| KDw-25c 0.72 (0.705-0.735) | held low: 0.713 |
| KDw - CE, under 40: within +-0.015 | held: +0.012 |
| KDw - CE, 130+: +0.005 to +0.015 | **missed**: +0.001; weighting removed all of the common-species gain |
| Prediction share back to about 0.60 | held: 0.597 |
| KDw ECE between KD (0.030) and CE (0.079) | **missed, favourably**: 0.020, the best of the three |

## 3. 100 epochs: the teacher now adds accuracy

Plan: "Class-weighted KD" (decision rule) and "Distillation" (carried-forward
recipe, release floor). Three arms, five paired seeds, 100 epochs, patience
off, on the cache, 2026-09-28: CE-100c (`dinov2-s-e100-c576`, job 18016204,
50-56 min per task), KD-100c (`dinov2-s-kd-e100-c576`, job 18016920, 2 h
38-44 min), KDw-100c (`dinov2-s-kdw-e100-c576`, job 18016921, 2 h 38-42 min).
None preempted or failed. Generated: [e100-results.md](species-distill/e100-results.md),
`species-distill/e100-diagnostics.json`.

Reproduce with
`sbatch --array=0-4 --time=08:00:00 --export=ALL,MODEL=dinov2-s,EPOCHS=100,CACHE=1 scripts/hellbender_species_finetune.slurm`,
the same with `KD=1` and with `KD=1,KD_WEIGHT=label`, then
`python scripts/summarize_species_finetune.py --models dinov2-s-kdw-e100-c576 dinov2-s-kd-e100-c576 dinov2-s-e100-c576 --arms full --title "Distillation, 100 epochs, cache" --report docs/reports/species-distill.md --out-md docs/reports/species-distill/e100-results.md --out-json docs/reports/species-distill/e100-summary.json`
and
`python scripts/diagnose_species_distill.py dinov2-s-kdw-e100-c576 dinov2-s-kd-e100-c576 dinov2-s-e100-c576 dinov2-s-kdw-c576 dinov2-s-c576 > docs/reports/species-distill/e100-diagnostics.json`
(it reproduces the 25-epoch diagnostics to the third decimal).

| Arm | Last epoch [95% CI] | Plain accuracy | Top-5 | Single-model ECE (after temperature) | Teacher agreement |
| --- | --- | ---: | ---: | --- | ---: |
| CE-100c | 0.719 [0.697, 0.744] | 0.758 | 0.924 | 0.160 (0.036) | 0.802 |
| KD-100c | 0.731 [0.707, 0.759] | 0.775 | 0.946 | 0.082 (0.026) | 0.822 |
| KDw-100c | **0.735** [0.713, 0.760] | 0.773 | 0.942 | 0.089 (0.027) | 0.820 |

| Paired difference | Delta [95% CI] | Seeds won |
| --- | --- | ---: |
| **KDw-100c - CE-100c** | **+0.016 [+0.011, +0.021]** | 5/5 |
| **KD-100c - CE-100c** | **+0.012 [+0.005, +0.019]** | 5/5 |
| KDw-100c - KD-100c | +0.004 [-0.001, +0.009] | 5/5 |

5-seed ensembles: KDw 0.751, KD 0.749, CE 0.744; KDw - CE +0.007 [-0.001, +0.017].

| Training images | CE-100c | KD-100c | KDw-100c | KDw - CE |
| --- | ---: | ---: | ---: | --- |
| under 40 | 0.610 | 0.613 | 0.634 | **+0.024 [+0.005, +0.043]** |
| 40-79 | 0.669 | 0.667 | 0.679 | +0.010 [+0.001, +0.021] |
| 80-129 | 0.759 | 0.779 | 0.777 | +0.018 [+0.009, +0.028] |
| 130 and over | 0.795 | 0.819 | 0.810 | **+0.015 [+0.011, +0.019]** |

- **At length, KDw beats CE in every bin** (+0.016 overall, against +0.003 at
  25 epochs).
- **Longer CE training over-fits the thin species**: CE-25c to CE-100c, under
  40 drops 0.021 [-0.042, -0.001] while 130+ gains 0.026. KDw holds the thin
  species (0.634, against 0.643 at 25 epochs).
- **CE-25c to KDw-100c: +0.025** (0.710 to 0.735); about a third is the
  schedule (CE +0.009), the rest the teacher. Gap to the teacher: 0.099.
- **All arms are under-confident at 100 epochs** (mean confidence 0.60-0.69
  against plain accuracy 0.76-0.77). One temperature (0.72-0.81) brings ECE to
  0.026-0.036; shipping needs it.
- Best epochs: KD 71-99, KDw 82-92, CE 61-89; last-epoch means are within
  0.003 (KDw) and 0.005 (CE) of best-epoch means.

| Rule | Fires? |
| --- | --- |
| KD(w)-100c - CE-100c < +0.01: distillation on the training set is closed | **no**: KDw +0.016 [+0.011, +0.021], KD +0.012 [+0.005, +0.019]. Both point estimates exceed +0.01, and so does the whole KDw interval |
| KD-100 - CE-100: CI excludes 0, positive: the teacher helps at the long schedule too | **yes**, for both variants, 5/5 seeds |
| Carried-forward recipe: the highest last-epoch seed mean among the cache arms | **KDw-100c**, 0.735. Its lead over KD-100c covers 0, but either reading gives KDw |
| KD variant for the transfer set: the higher of the two at 100 epochs | **KDw**: the arm is `dinov2-s-kdw-e100-t-c576`, with expected weights on transfer rows |
| Release floor: carried-forward seed mean >= 0.75 and top-5 >= 0.95 | **not met**: 0.735 and 0.942. Nothing ships until the transfer-set run. Seed 17 alone is 0.732, top-5 0.943 |
| A run diverges or fails | no: seed SD 0.004-0.006 |

| Prior | Outcome |
| --- | --- |
| CE-100c 0.725 (0.71-0.735) | held low: 0.719 |
| KD-100c 0.73 (0.715-0.75) | held: 0.731 |
| KDw-100c 0.74 (0.72-0.76) | held low: 0.735 |
| Transfer set: about 0.76 "if KD(w)-100c lands near 0.74" | the base is 0.005 lower, so the corresponding point is about 0.755 |

## 4. KDw-100c+T: the transfer set is the largest gain

Plan: "Transfer set (written 2026-09-27, before any acquisition or run)" and
its 2026-09-28 amendment. KDw-100c+T (`dinov2-s-kdw-e100-t-c576`, job
18068909) trains on the training set plus 12,126 transfer photos (iNaturalist
photos of the 110 species the teacher never trained on), batches drawn
uniformly from the union (transfer share 0.515-0.530). Transfer photos enter
the KD term only, with the expected class weight. 17,300 steps, the control's
100 epochs of 173, same warmup and cosine; five seeds paired with KDw-100c.
Tasks took 2 h 40-41 min; none preempted. Generated:
[t-results.md](species-distill/t-results.md), `species-distill/t-diagnostics.json`.

Reproduce with
`sbatch --array=0-4 --time=08:00:00 --export=ALL,MODEL=dinov2-s,KD=1,KD_WEIGHT=label,EPOCHS=100,CACHE=1,TRANSFER=1 scripts/hellbender_species_finetune.slurm`,
then
`python scripts/summarize_species_finetune.py --models dinov2-s-kdw-e100-t-c576 dinov2-s-kdw-e100-c576 dinov2-s-e100-c576 --arms full --title "Transfer set, KDw-100c+T" --report docs/reports/species-distill.md --out-md docs/reports/species-distill/t-results.md --out-json docs/reports/species-distill/t-summary.json`
and
`python scripts/diagnose_species_distill.py dinov2-s-kdw-e100-t-c576 dinov2-s-kdw-e100-c576 dinov2-s-e100-c576 > docs/reports/species-distill/t-diagnostics.json`.

| Arm | Last epoch [95% CI] | Plain accuracy | Top-5 | Single-model ECE (after temperature) | Teacher agreement |
| --- | --- | ---: | ---: | --- | ---: |
| KDw-100c (section 3) | 0.735 [0.713, 0.760] | 0.773 | 0.942 | 0.089 (0.027) | 0.820 |
| **KDw-100c+T** | **0.768** [0.747, 0.794] | 0.803 | **0.959** | 0.101 (0.024) | 0.856 |

| Paired difference | Delta [95% CI] | Seeds won |
| --- | --- | ---: |
| **KDw-100c+T - KDw-100c** (primary) | **+0.033 [+0.026, +0.042]** | 5/5 |
| KDw-100c+T - CE-100c | +0.049 [+0.042, +0.059] | 5/5 |

5-seed ensembles: +T 0.780 (top-5 0.964), KDw 0.751, CE 0.744; +T - KDw
+0.029 [+0.017, +0.040]. Seed SD fell from 0.006 to 0.002 (best epoch).

| Training images | KDw-100c | KDw-100c+T | +T - KDw |
| --- | ---: | ---: | --- |
| under 40 | 0.634 | 0.677 | **+0.042 [+0.017, +0.077]** |
| 40-79 | 0.679 | 0.719 | +0.040 [+0.023, +0.056] |
| 80-129 | 0.777 | 0.792 | +0.014 [+0.001, +0.028] |
| 130 and over | 0.810 | 0.842 | +0.032 [+0.025, +0.039] |

- **+0.033, twice the teacher's gain at 100 epochs** (+0.016). The gap to the
  teacher falls from 0.099 to 0.066.
- **The thin species gained most, against the prior** (+0.042; expected
  +0.00 to +0.015), though they got only 1-15 new photos each. Matching the
  teacher on many photos of other species also sharpens the boundaries around
  the thin ones (the fidelity reading from stage 1).
- **Fidelity up**: agreement with the teacher 0.856 (from 0.820), validation
  KL 0.21 (from 0.31).
- Calibration: fitted temperature 0.76, ECE 0.101 -> 0.024. Best epochs
  79-88; last-epoch mean within 0.004 of best.

| Rule | Fires? |
| --- | --- |
| +T - KD(w)-100c: CI excludes 0, positive: the transfer set is in the student recipe; +T is carried forward | **yes**: +0.033 [+0.026, +0.042], 5/5 |
| gain < +0.01 (amended 2026-09-28): the transfer set is dropped | no |
| Teacher agrees with the iNaturalist ID on under 0.6 of captive photos: "captive targets suspect" | no: 0.681 on the 659 captive photos (below) |
| Carried-forward recipe: the highest last-epoch seed mean | **KDw-100c+T**, 0.768 |
| Shipped model: seed 17 | seed 17 of KDw-100c+T: 0.763, top-5 0.959 (last epoch) |
| Release floor: seed mean >= 0.75 and top-5 >= 0.95 | **met**: 0.768 and 0.959 |
| A run diverges or fails | no: seed SD 0.002-0.003 |

What remained before release (test-protocol amendment of 2026-09-29): int8
export and temperature ([species-release.md](species-release.md)), then one
scoring of the test split ([species-test.md](species-test.md)).

| Prior | Outcome |
| --- | --- |
| overall +0.02 (+0.005 to +0.035) | held high: +0.033 |
| under 40: +0.00 to +0.015 | **missed, favourably**: +0.042 |
| 130 and over: +0.01 to +0.03 | missed narrowly: +0.032 |
| absolute about 0.755 (restated after KDw-100c) | missed, favourably: 0.768 |

### The transfer set under the teacher

Pre-registered for this arm. The teacher (5-seed ensemble, tau 1) and the
shipped student (seed 17, last epoch) over the 12,126 transfer photos at 224
px. The iNaturalist ID is the record's taxon (10 photos of 6 infraspecific
taxa mapped to their species). Job 18079959, 2.5 min on one A100. Output:
`species-distill/t-transfer-diagnostics.json`; per-photo arrays in
`transfer-diagnostics.npz` beside the seed-17 checkpoint (not in Git).
Reproduce with `sbatch scripts/hellbender_transfer_diagnostics.slurm`
(`scripts/diagnose_transfer_set.py`).

| Source | Photos | Species | Teacher max-prob | Teacher = iNat ID | Student = teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| (a) wild, research grade | 7,970 | 97 | 0.771 | 0.898 | 0.953 |
| (b) captive | 659 | 48 | 0.613 | **0.681** | 0.882 |
| (c) needs-ID or casual | 3,497 | 107 | 0.685 | 0.744 | 0.908 |
| all | 12,126 | 110 | 0.738 | 0.842 | 0.936 |

- **Captive rule does not fire** (0.681 > 0.6), but captive photos are the
  least reliable source (teacher max-probability 0.613). For (b) and (c)
  neither the community ID nor the teacher is ground truth.
- **Few photos of the thin species**: 201 (0.017 of the set; 1-15 per
  species), 131 of them needs-ID. 0.881 of the set belongs to the 42 commonest
  species.
- **Most of the thin-species signal comes from other species' photos.** The
  teacher puts 0.097 of its probability mass on the 23 thin species, about
  1,180 photo-equivalents, against 201 photos with a thin ID. This is
  consistent with the +0.042 gain there; it does not prove it.
