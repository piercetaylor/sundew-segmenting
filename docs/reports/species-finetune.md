# Fine-tuned teachers: DINOv2-L leads, and the crop stops helping

**In short.** Fine-tuned end to end, DINOv2-L reaches 0.824 validation
balanced accuracy on full frames (5-seed ensemble 0.835), against 0.499 for
the ResNet-18 anchor. It beats BioCLIP-2 in 5 of 5 seeds and becomes the
teacher. Cropping to the plant no longer helps, so segmentation comes off the
species critical path: the classifier takes the whole photo.

Stage 2 of the [plan](../species-classifier-plan.md): the two leaders of the
[frozen screen](species-backbone-screen.md), `bioclip-2` and `dinov2-l-reg`,
fine-tuned at 224 px on 110 species, three arms (`full`, `crop`,
`full-square`), five paired seeds. Training split `split-110-test` (11,118
images; the test split is not read), validation unchanged (3,959 images, 308
observers). Generated tables: [results.md](species-finetune/results.md);
per-species errors and calibration:
[teacher-analysis.md](species-finetune/teacher-analysis.md).

Reproduce with
`sbatch --array=0-14 --export=ALL,MODEL=bioclip-2 scripts/hellbender_species_finetune.slurm`,
the same with `MODEL=dinov2-l-reg`, then
`python scripts/summarize_species_finetune.py --out-md docs/reports/species-finetune/results.md --out-json docs/reports/species-finetune/summary.json`
and
`python scripts/analyze_species_teacher.py --out-md docs/reports/species-finetune/teacher-analysis.md --out-json docs/reports/species-finetune/teacher-analysis.json`;
the contamination check with
`python scripts/check_bioclip_contamination.py --out-md docs/reports/species-finetune/contamination.md --out-json docs/reports/species-finetune/contamination.json`.
Jobs 17943517 (`bioclip-2`) and 17943518 (`dinov2-l-reg`), 30 A100 tasks of
27-53 min, 2026-09-25. Recipe: AdamW lr 5e-5, 2 warmup epochs, 25 epochs,
patience 8, layer decay 0.85, drop-path 0.2 (DINOv2-L only; open_clip has
none), head lr x10, weight decay 0.05, label smoothing 0.05, bf16. All 30
tasks completed; one (`dinov2-l-reg` crop, seed 17) stopped early at epoch 23.

## Result

Validation balanced accuracy, mean of five seeds at the best epoch. Anchor:
ResNet-18 on the same split, **full 0.499, crop 0.516**.

Intervals on differences are observer-grouped bootstraps: each resample of
validation observers rescores all five seeds, so they cover both seed noise
and which photos landed in validation. (Changed 2026-09-26, after the audit
below; the first version used a seed-paired t, which holds validation fixed
and is narrower. It stays in `results.md` as a secondary check.)

| Model | `full` | `crop` | `full-square` | Seed SD | 5-seed ensemble, `full` [95% CI] | Top-5 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `dinov2-l-reg` | **0.824** | 0.820 | 0.811 | 0.002 | **0.835** [0.814, 0.857] | 0.975 |
| `bioclip-2` | 0.802 | 0.799 | 0.795 | 0.006-0.008 | 0.816 [0.798, 0.836] | 0.967 |

- **Fine-tuning a large backbone adds +0.31** over the anchor on the same data.
- **DINOv2-L is the teacher.** It beats BioCLIP-2 on every arm in 5 of 5
  paired seeds, by +0.015 to +0.023, every interval above zero (`full`
  +0.022 [+0.013, +0.030]; ensemble +0.019 [+0.008, +0.030]). Frozen, the
  order was reversed (0.750 vs 0.729). DINOv2-L is also more stable (seed SD
  0.002 against 0.006-0.008) and has no contamination caveat.
- **Fine-tuning gained more for DINOv2-L**: crop arm 0.729 -> 0.820 (+0.091)
  against 0.750 -> 0.799 (+0.049), on 19% less training data than the probes.
- **Ensembling seeds helps a little, mixing models does not.** Five DINOv2-L
  seeds add about 0.011; averaging both models' ensembles gives 0.836.
- The DINOv2-L ensemble has the right species in its top five 97.5% of the time.

## The crop question: answered, no

| Seed-mean difference (best epoch), bootstrap 95% CI, seeds won | DINOv2-L | BioCLIP-2 |
| --- | --- | --- |
| `crop` - `full` | -0.004 [-0.012, +0.006], 1/5 | -0.002 [-0.011, +0.006], 3/5 |
| `crop` - `full-square` | +0.010 [-0.004, +0.022], 5/5 | +0.004 [-0.011, +0.017], 3/5 |
| `full` - `full-square` | +0.013 [+0.002, +0.022], 5/5 | +0.006 [-0.004, +0.016], 4/5 |

Once fine-tuned, there is no evidence the crop helps a strong backbone; for
DINOv2-L any benefit is at most about +0.006. The crop's +0.017 belonged to
the weak ResNet-18.

`full-square` ranks lowest, but its training transform was handicapped: it
squashed each photo to 224 px *before* the random crop, so it trained on
upsampled sub-crops. No decision rests on it.

**Segmentation comes off the species critical path.** The classifier takes
the whole photo, resized and centre-cropped. The segmenter remains a result in
its own right.

## Teacher errors and calibration

DINOv2-L `full`, 5-seed last-epoch ensemble, validation
([teacher-analysis.md](species-finetune/teacher-analysis.md)).

- **Under-confident.** Mean confidence 0.74 against plain accuracy 0.857; ECE
  (expected calibration error: the gap between confidence and accuracy) 0.118.
  One temperature, T = 0.72, brings ECE to 0.019 and NLL from 0.639 to 0.539
  (fitted and scored on validation). Class weighting, label smoothing and
  seed averaging all push confidence down, so this is expected.
- **Errors are mostly near-misses.** 68% of 566 errors stay within the true
  section. Top pairs are known look-alikes: *macrantha* / *planchonii* (16),
  *aberrans* / *whittakeri* (11), *dielsiana* / *natalensis* (11), *gunniana*
  / *hookeri* (10), *hirsuta* -> *macrantha* (10), and the *peltata* /
  *auriculata* / *lunata* complex, where some iNaturalist labels are likely
  wrong themselves.
- **Thin classes are weaker, loosely.** Species with under 40 training images
  average 0.76; those with 130 or more, 0.88 (Spearman 0.22). Worst: *D.
  dielsiana* 0.15 (32 training, 13 validation images), mostly called
  *natalensis*. Weakest sections: *Brasiliae* 0.71, *Luniferae* 0.76,
  *Macrantha* 0.77.
- **Balanced accuracy assumes every species is equally common.** App users
  mostly photograph common species, so field accuracy will differ. Plain
  accuracy (0.857) is reported beside it from here on.

## Against expectations

The plan fixed no prior for this stage beyond the choice of models and arms.
The stated expectations:

| Expectation | Outcome |
| --- | --- |
| Teacher at 0.80-0.85 top-1 balanced accuracy | held: 0.824 seed mean, 0.835 ensemble |
| Crop - full gap gone for strong backbones (frozen-screen reading) | held: on zero for both models |
| BioCLIP-2 and DINOv2-L close | held, but DINOv2-L now clearly ahead |

## What it means for the student

The small-backbone rule picked `dinov2-s` on the crop arm, where it leads
TinyViT-21M by 0.017. With a full-frame teacher, the app has no reason to
segment first, and on full frames the two students tie (frozen 0.616 vs
0.614). So the two were fine-tuned head to head before distillation:
[species-student-bakeoff.md](species-student-bakeoff.md).

## BioCLIP-2 contamination: bounded, not detected

BioCLIP-2's training corpus, TreeOfLife-200M, takes iNaturalist photos from a
GBIF download of May 2024 (doi:10.15468/dl.bfv433) and EOL media accessed
August 2024 (Gu et al. 2025, arXiv:2505.23883). No validation photo observed
from 2025 on can be in it (the boundary is set 4-7 months later to absorb
mis-entered dates). Photos observed up to 2022 are the ones it may have seen.
If it memorised them, its advantage over DINOv2-L would be larger on old
photos than on unseen ones. This difference of differences (DiD) is the test.
Generated table: [contamination.md](species-finetune/contamination.md).

| Strata | Images | Observers | Species |
| --- | ---: | ---: | ---: |
| old, observed <= 2022-12-31 | 1,606 | 215 | 110 |
| middle, 2023-2024 (not in the test) | 1,192 | 206 | 109 |
| unseen, observed >= 2025-01-01 | 1,161 | 200 | 106 |

| BioCLIP-2 - DINOv2-L | Old | Unseen | Old - unseen (DiD) [95% CI] |
| --- | ---: | ---: | --- |
| fine-tuned, `full`, seed mean, last epoch | -0.026 | -0.020 | **-0.006 [-0.032, +0.017]** |
| fine-tuned, `crop` | -0.029 | -0.030 | +0.002 [-0.020, +0.029] |
| frozen linear probe, `crop` | +0.028 | +0.011 | +0.017 [-0.030, +0.061] |
| frozen linear probe, `full` | +0.024 | +0.045 | -0.021 [-0.049, +0.032] |

- **No inflation detected, but only bounded**: at most about +0.017 for the
  fine-tuned `full` arm, +0.029 on `crop`.
- **The frozen screen is bounded loosely** (up to +0.061 on `crop`), so its
  0.021 BioCLIP-2 lead could in principle be contamination. The fine-tuned
  ordering, which reverses it, is the one the plan acts on.
- **The teacher choice stands either way.** Contamination can only flatter
  BioCLIP-2, and DINOv2-L wins regardless.
- The same holds on the 106 species present in both strata and with a later
  boundary (>= 2025-07-01, 961 images). The DiD cancels differences in
  species and observer mix that affect both models alike, not ones that affect
  one more. Intervals are observer bootstraps over all of validation; they are
  skewed because each stratum has few images per species.

## Caveats

- **Validation picked these models** (the backbones in the screen, the best
  epoch here). Last-epoch seed means are within 0.004 of best-epoch ones; the
  unbiased number is the held-out test, scored once at the end.
- **Recomputed balanced accuracy differs from the training log by up to
  0.0008** per run, from ties in bf16 logits and an unstable `argsort` in the
  training-time metric (now fixed). No comparison here is that close.
- **The recipes tilt against BioCLIP-2.** timm puts DINOv2-L's last block at
  lr scale 1.0, while `open_clip_groups` puts BioCLIP-2's at 0.85, so every
  BioCLIP-2 block trains at 0.85x the lr of its counterpart; BioCLIP-2 also
  gets no drop-path (0.2 for DINOv2-L). The lead holds for this recipe, not
  necessarily for tuned ones. (Verified 2026-09-26; an earlier version said the
  decay matched.) Neither model had an lr or layer-decay search.
- **The ResNet-18 anchor kept hue jitter 0.05**; the fine-tunes turned it off.
- **224 px only.**

## Audit, 2026-09-26

An independent read-only review (a second model) checked the split and the
code. Verified: no observer, observation, identical-file (sha256) or
near-duplicate (dHash distance <= 6) overlap between any two of train,
validation and test; one photo per observation; test split unread. Fixed
after it, in `scripts/finetune_species_backbone.py`:

- `img_size` was passed to any timm name containing `vit_`, which crashes on
  TinyViT and FastViT. Now plain ViTs only.
- `full-square` now squashes to 268 px before the random crop, so crops are
  not upsampled. Runs already done keep the old transform.
- A task requeued after it finished no longer retrains from scratch.
