# Student bake-off: DINOv2-S is the student

The two small backbones the frozen screen could not separate on full frames,
`dinov2-s` and `tinyvit-21m-in22k`, fine-tuned without a teacher, to pick the
student that distillation targets. Design, prior and decision rule were written
before any run: `docs/species-classifier-plan.md`, "Student bake-off (written
2026-09-26, before any run)". The generated table is
`species-student-bakeoff/results.md`.

Reproduce with
`sbatch --array=0-4 --export=ALL,MODEL=dinov2-s scripts/hellbender_species_finetune.slurm`,
the same with `MODEL=tinyvit-21m-in22k`, then
`python scripts/summarize_species_finetune.py --models dinov2-s tinyvit-21m-in22k --arms full --title "Student bake-off" --report docs/reports/species-student-bakeoff.md --out-md docs/reports/species-student-bakeoff/results.md --out-json docs/reports/species-student-bakeoff/summary.json`.
Jobs 17987233 (`dinov2-s`) and 17987234 (`tinyvit-21m-in22k`), 10 A100 tasks
of 44-47 min (7.4 A100-h), 2026-09-26 18:46 to 2026-09-27 01:05. All ran 25
epochs; none failed or was requeued. Peak GPU memory 3.3 GiB (DINOv2-S),
5.2 GiB (TinyViT).

## Answer

`full` arm, 224 px, `split-110-test`, validation balanced accuracy, seed mean
at the **last epoch** (the pre-registered metric), observer-grouped bootstrap.

| Model | Params | Last epoch [95% CI] | Seed SD | Frozen probe, `full` | 5-seed ensemble | Top-5 (ensemble) |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `dinov2-s` | 22M | **0.712** [0.691, 0.736] | 0.004 | 0.616 | 0.735 [0.712, 0.761] | 0.946 |
| `tinyvit-21m-in22k` | 21M | 0.622 [0.602, 0.651] | 0.003 | 0.614 | 0.625 [0.604, 0.655] | 0.906 |

For scale: teacher DINOv2-L `full` 0.824 (ensemble 0.835); ResNet-18 anchor
0.499.

**`dinov2-s` - `tinyvit-21m-in22k`: +0.090 [+0.075, +0.102]**, P(delta <= 0)
= 0.000, 5 of 5 paired seeds. The seed-paired t agrees
(+0.084 to +0.097).

## Decision, applied as written

| Rule | Fires? |
| --- | --- |
| CI on the difference excludes 0: the higher model is the student | **yes**: `dinov2-s` |
| CI includes 0: int8 ONNX latency decides | no, so the latency test is not needed for this decision |
| Better student < 0.70: distillation mandatory before release | **no, narrowly**: 0.712, lower CI bound 0.691 |
| Better student < 0.65: stop on-device, serve the teacher | no |
| A model fails or diverges (seed SD > 0.02) | no: SD 0.003-0.004 |

**DINOv2-S is the student.** It clears the 0.70 floor by 0.012, and its
interval reaches below it, so distillation is not mandatory by the rule but
is plainly needed in practice: the student sits 0.11 below the teacher.

## Against the prior

| Prior | Outcome |
| --- | --- |
| `dinov2-s` 0.73-0.77 | **missed low**: 0.712 |
| `tinyvit-21m-in22k` 0.72-0.78 | **wrong**: 0.622 |
| Difference within +-0.02, likely a tie | **wrong**: +0.090, decisive |

## Why TinyViT lost: under-trained by this recipe, most likely

TinyViT barely moved from its frozen probe (0.614 -> 0.622, +0.008), while
DINOv2-S gained +0.096 (0.616 -> 0.712). Its training loss stalls well above
DINOv2-S's (seed 17):

| Epoch | 1 | 5 | 10 | 15 | 20 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dinov2-s` train loss | 5.12 | 1.99 | 1.38 | 1.11 | 0.98 | 0.94 |
| `tinyvit-21m-in22k` train loss | 4.78 | 2.56 | 1.82 | 1.55 | 1.45 | 1.44 |

That is under-fitting, not over-fitting. The shared recipe (lr 5e-5, layer
decay 0.75) was set for ViT fine-tuning, and timm's grouping puts every
TinyViT block at 0.75x the lr of its DINOv2-S counterpart (see the plan's
layer-decay table), which the pre-registration recorded as an accepted
handicap. So the result reads: **under the teachers' recipe, DINOv2-S is the
far better student**. It does not show TinyViT is a worse architecture; a
TinyViT with its own lr would likely close some of the gap. The rule says
TinyViT is not retuned, and nothing here argues for reopening it: DINOv2-S
also shares the teacher's lineage and patch features, which feature
distillation can use.

DINOv2-S's own loss is still falling slowly at epoch 25, and its best epochs
(21-25) sit at the end of the schedule, as for the teachers. A longer
schedule may add a little; it is a recipe question for the distillation run,
not for this decision.

## What this does not decide

- **Distillation**: how much the DINOv2-L ensemble lifts DINOv2-S. That is the
  next pre-registered run, paired against these five seeds as the no-teacher
  baseline.
- **Latency and export**: the int8 ONNX latency test was only a tiebreaker
  and was not run. Export, size and real-device browser speed are measured on
  DINOv2-S alone, after distillation.
- **Calibration** of the student, fitted later on validation.

## Next

1. **Pre-register distillation** of the DINOv2-L `full` ensemble into
   `dinov2-s` (KD temperature and weight, schedule length, win rule against
   0.712), then run it (~15 A100-h).
2. **DINOv2-L at 384 px** (~25 A100-h), which may change the teacher the
   student learns from.
3. Open-set images and the abstain threshold; then the test split, once.
