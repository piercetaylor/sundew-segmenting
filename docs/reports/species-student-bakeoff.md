# Student bake-off: DINOv2-S is the student

**In short.** Fine-tuned without a teacher on full frames, DINOv2-S reaches
0.712 validation balanced accuracy and TinyViT-21M 0.622. DINOv2-S wins by
+0.090 [+0.075, +0.102], 5 of 5 seeds, and is the student that distillation
targets. Its 0.712 is the no-teacher baseline (CE-25) for the
[distillation runs](species-distill.md).

The frozen screen could not separate `dinov2-s` and `tinyvit-21m-in22k` on
full frames ([species-backbone-screen.md](species-backbone-screen.md)), so both
were fine-tuned with the teacher's recipe. Design, prior and rule were written
before any run: [plan](../species-classifier-plan.md), "Student bake-off
(written 2026-09-26, before any run)". Generated table:
[species-student-bakeoff/results.md](species-student-bakeoff/results.md).

Reproduce with
`sbatch --array=0-4 --export=ALL,MODEL=dinov2-s scripts/hellbender_species_finetune.slurm`,
the same with `MODEL=tinyvit-21m-in22k`, then
`python scripts/summarize_species_finetune.py --models dinov2-s tinyvit-21m-in22k --arms full --title "Student bake-off" --report docs/reports/species-student-bakeoff.md --out-md docs/reports/species-student-bakeoff/results.md --out-json docs/reports/species-student-bakeoff/summary.json`.
Jobs 17987233 (`dinov2-s`) and 17987234 (`tinyvit-21m-in22k`), 10 A100 tasks
of 44-47 min (7.4 A100-h), 2026-09-26 to 2026-09-27. All ran 25 epochs; none
failed.

## Result

`full` arm, 224 px, `split-110-test`, validation balanced accuracy, seed mean
at the **last epoch** (the pre-registered metric), observer-grouped bootstrap.

| Model | Params | Last epoch [95% CI] | Seed SD | Frozen probe, `full` | 5-seed ensemble | Top-5 (ensemble) |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `dinov2-s` | 22M | **0.712** [0.691, 0.736] | 0.004 | 0.616 | 0.735 [0.712, 0.761] | 0.946 |
| `tinyvit-21m-in22k` | 21M | 0.622 [0.602, 0.651] | 0.003 | 0.614 | 0.625 [0.604, 0.655] | 0.906 |

For scale: teacher DINOv2-L `full` 0.824 (ensemble 0.835); ResNet-18 anchor 0.499.

**`dinov2-s` - `tinyvit-21m-in22k`: +0.090 [+0.075, +0.102]**, P(delta <= 0)
= 0.000, 5 of 5 paired seeds (seed-paired t: +0.084 to +0.097).

## Decision, applied as written

| Rule | Fires? |
| --- | --- |
| CI on the difference excludes 0: the higher model is the student | **yes**: `dinov2-s` |
| CI includes 0: int8 ONNX latency decides | no, so the latency test was not needed |
| Better student < 0.70: distillation mandatory before release | **no, narrowly**: 0.712, lower CI bound 0.691 |
| Better student < 0.65: stop on-device, serve the teacher | no |
| A model fails or diverges (seed SD > 0.02) | no: SD 0.003-0.004 |

DINOv2-S clears the 0.70 floor by only 0.012, and sits 0.11 below the
teacher, so distillation is not required by the rule but is plainly needed.

## Against the prior

| Prior | Outcome |
| --- | --- |
| `dinov2-s` 0.73-0.77 | **missed low**: 0.712 |
| `tinyvit-21m-in22k` 0.72-0.78 | **wrong**: 0.622 |
| Difference within +-0.02, likely a tie | **wrong**: +0.090, decisive |

## Why TinyViT lost: most likely under-trained by this recipe

TinyViT barely moved from its frozen probe (0.614 -> 0.622), while DINOv2-S
gained +0.096. TinyViT's training loss stalls well above DINOv2-S's (seed 17):

| Epoch | 1 | 5 | 10 | 15 | 20 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dinov2-s` train loss | 5.12 | 1.99 | 1.38 | 1.11 | 0.98 | 0.94 |
| `tinyvit-21m-in22k` train loss | 4.78 | 2.56 | 1.82 | 1.55 | 1.45 | 1.44 |

That is under-fitting. The shared recipe (lr 5e-5, layer decay 0.75) was set
for ViTs, and timm's layer grouping gives every TinyViT block 0.75x the lr of
its DINOv2-S counterpart, a handicap the pre-registration accepted. So the
result is: **under the teacher's recipe, DINOv2-S is the far better student**.
It does not show TinyViT is a worse architecture. The rule says TinyViT is not
retuned, and DINOv2-S also shares the teacher's lineage.

DINOv2-S's loss is still falling at epoch 25 and its best epochs (21-25) are
at the end, which is why a longer schedule was tried in distillation.

Not decided here: how much the teacher helps (distillation), latency and
export (measured on DINOv2-S only, in [species-release.md](species-release.md)),
and calibration.
