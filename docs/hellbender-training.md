# Training on Hellbender

Every result in the README was produced by the SLURM jobs below, on the University of Missouri's Hellbender cluster. This page lists them in the order they run.

**Citation.** The Hellbender wiki (https://itrss-wiki.rnet.missouri.edu/pub/hpc/hellbender) asks every publication that uses the cluster to include: "The computation for this work was performed on the high performance computing infrastructure operated by Research Support Solutions in the Division of IT at the University of Missouri, Columbia MO DOI: https://doi.org/10.32469/10355/97710", and to send a copy to muitrss@missouri.edu. The README carries it.

## Environment

The stack is defined in `environment.yml` (the `sundew-seg` environment) and built with the cluster's `mamba` module. Torch comes from the PyTorch CUDA index, not conda-forge: those wheels bundle their own CUDA runtime, so jobs don't depend on the cluster's `cuda` modules, which stop at 11.8 and are too old for Hellbender's A100/H100/L40S mix. Environments land in the `envs_dirs` from `~/.condarc`, on project storage rather than the 50 GB `/home` quota.

Build it once, from a compute node. Cluster policy is that code never runs on the login node, and a full torch install is heavy:

```bash
mkdir -p logs          # Slurm writes logs here and fails the job if it is missing
sbatch scripts/hellbender_setup.slurm
```

`hellbender_setup.slurm` (`general`, 8 CPUs, 2 h) creates or updates the environment, installs the repo package with `pip install --no-deps -e .`, verifies and unpacks `dist/sundew-segmenting-reviewed-v0.3.0.zip`, and warms the segmentation encoder weights under `sundew-cache/` so training doesn't need the network. Paths can be overridden with `REPO_DIR`, `DATA_ROOT`, `CACHE_ROOT` and `ENV_NAME`. The species backbones' weights were fetched into the same cache (`HF_HOME`) on the login node beforehand, so the species jobs also run offline.

To use the environment interactively on a compute node:

```bash
module load mamba/v2.4.0
eval "$(conda shell.bash hook)"
conda activate sundew-seg
```

## Partitions

GPU jobs use `gpu_requeue`. The plain `requeue` partition rejects GPU jobs, and `gpu` doesn't preempt but has often been backlogged by more than a day. Jobs on `gpu_requeue` set `--requeue`, and the long ones resume from their last checkpoint. Pass `--partition=gpu` if a run must not be interrupted. CPU-only jobs use `general`. Walltimes are sized from measured runs, so jobs schedule sooner and leave GPUs free for others.

Outputs go to `models/` and logs to `logs/`; both are gitignored. Monitor with `squeue -u "$USER"`.

## Segmentation

All of these train on the frozen v0.3.0 split (144 train, 19 validation), pick checkpoints by validation IoU, and leave the 28 test images alone. Each GPU task asks for one A100.

| Job | What it does | Partition, time per task | Report |
| --- | --- | --- | --- |
| `hellbender_seed_sweep.slurm` | U-Net vs SegFormer-B0 over five seeds (array 0-9, five at a time) | `gpu_requeue`, 30 min | [baseline-seed-sweep](reports/baseline-seed-sweep.md) |
| `hellbender_data_curve.slurm` | SegFormer-B0 on 25/50/75/100% of the training split, three seeds each | `gpu_requeue`, 30 min | [segmentation-recipe](reports/segmentation-recipe.md) |
| `hellbender_recipe_sweep.slurm` | Six training recipes by two seeds, to tell a data ceiling from a recipe ceiling | `gpu_requeue`, 1 h | [segmentation-recipe](reports/segmentation-recipe.md) |
| `hellbender_field_compare.slurm` | Recipe variant 5 with and without the 30 field masks, five paired seeds | `gpu_requeue`, 90 min | [field-compare](reports/field-compare.md) |

```bash
sbatch scripts/hellbender_seed_sweep.slurm
sbatch scripts/hellbender_data_curve.slurm
sbatch scripts/hellbender_recipe_sweep.slurm
sbatch scripts/hellbender_field_compare.slurm
```

The field-trained SegFormer-B0, seed 101, is the model that crops the species photos. The earlier single-run comparison job, `hellbender_train.slurm`, has been retired to `deprecated/scripts/`.

## Species classifier

Run in this order. The design is in [the species classifier doc](species-classifier.md) and the rules in [the plan](species-classifier-plan.md).

**1. Data** (all CPU, `general`; each is restartable and skips finished files).

| Job | What it does | Resources |
| --- | --- | --- |
| `hellbender_species_corpus.slurm` | Downloads the 110-species corpus: up to 300 photos a species, at most 5 per observer, CC0/CC-BY/CC-BY-NC only | 4 CPUs, 12 h (about 4 h in practice) |
| `hellbender_species_110_crops.slurm` | Predicted crop boxes for all 17,678 photos with the seed-101 field model, in eight shards (array 0-7) | 8 CPUs, 3 h per shard |
| `hellbender_full_cache.slurm` | Caches full frames at short side 576 for train, validation and test; read by `CACHE=1` | 16 CPUs, 3 h |
| `hellbender_transfer_set.slurm` | Downloads the 12,179 unlabelled transfer photos, drops duplicates of any split (12,126 kept) and caches them at 576 px; read by `TRANSFER=1` | 8 CPUs, 12 h (about 3 h in practice) |

The held-out test split was carved from the corpus by `scripts/carve_test_split.py` (split `split-110-test`).

**2. ResNet-18 anchor.** `hellbender_species_110.slurm` (`gpu_requeue`, 3 h) is the ResNet-18 baseline; rerun on the test-carved split with `--export=ALL,SPLIT_NAME=split-110-test,OUT_ROOT=models/species-110-test/resnet18`. Report: [species-110-baseline](reports/species-110-baseline.md).

**3. Backbone screen.** `hellbender_backbone_screen.slurm` (`gpu_requeue`, 90 min per task) scores a frozen linear probe, k-NN and, for CLIP models, zero-shot, one backbone per array task. `hellbender_backbone_screen_summary.slurm` (`general`, 30 min) tabulates the results once every task succeeds. `MODEL_SET=small` screens the on-device student candidates. Report: [species-backbone-screen](reports/species-backbone-screen.md).

```bash
jid=$(sbatch --parsable scripts/hellbender_backbone_screen.slurm)
sbatch --dependency=afterok:$jid scripts/hellbender_backbone_screen_summary.slurm
```

**4. Fine-tune and distil.** `hellbender_species_finetune.slurm` (`gpu_requeue`, 5 h, resumes from `last.pt` on preemption) fine-tunes one backbone per submission. Array indices 0-4 are the full-frame arm with seeds 17/101/202/303/404, 5-9 the crop arm. Options, all passed with `--export=ALL,...`:

| Variable | Effect | Output tag |
| --- | --- | --- |
| `MODEL` (required) | Backbone tag from `screen_species_backbones.MODELS`, e.g. `dinov2-l-reg`, `dinov2-s` | `<model>` |
| `KD=1` | Distil from the mean softmax of the five `dinov2-l-reg` full-arm checkpoints | `-kd` |
| `KD_WEIGHT=label` or `expected` | With `KD=1`: weight each image's KD loss by its label's class weight, or by the teacher's expected class weight | `-kdw` / `-kdwe` |
| `EPOCHS=N` | Longer schedule (default 25); pass `--time=08:00:00` for 100 | `-eN` |
| `CACHE=1` | Read full frames from the 576 px cache | `-c576` |
| `TRANSFER=1` | With `KD=1` and `CACHE=1`: add the transfer set to the KD term at the control's step count | `-t` |

Any run with `KD=1` or a non-default `EPOCHS` turns early stopping off. The teacher and the shipped recipe:

```bash
sbatch --array=0-4 --export=ALL,MODEL=dinov2-l-reg scripts/hellbender_species_finetune.slurm
sbatch --array=0-4 --time=08:00:00 \
  --export=ALL,MODEL=dinov2-s,KD=1,KD_WEIGHT=label,EPOCHS=100,CACHE=1,TRANSFER=1 \
  scripts/hellbender_species_finetune.slurm          # -> dinov2-s-kdw-e100-t-c576
python scripts/summarize_species_finetune.py --help
```

The student was chosen by fine-tuning `dinov2-s` and `tinyvit-21m-in22k` without a teacher. Reports: [species-finetune](reports/species-finetune.md), [species-student-bakeoff](reports/species-student-bakeoff.md), [species-distill](reports/species-distill.md).

**5. Transfer-set diagnostics.** `hellbender_transfer_diagnostics.slurm` (`gpu_requeue`, 1 h) runs the teacher ensemble and student seed 17 over the transfer photos, by source and training-count bin. It never reads the test split.

**6. Export.** `hellbender_species_export.slurm` (`general`, 8 CPUs, 3 h, CPU only) exports seed 17 of the shipped recipe to fp32 and int8 ONNX, measures the int8 accuracy change, fits the temperature and times latency on one thread, all on validation. It writes `release.json` under `models/species-110/release/`. Report: [species-release](reports/species-release.md).

**7. Held-out test.** `hellbender_species_test.slurm` (`gpu_requeue`, 16 CPUs, 3 h) runs `scripts/score_species_test.py`.

```bash
sbatch --export=ALL,SPLIT=val  scripts/hellbender_species_test.slurm   # validation only; never opens a test image
sbatch --export=ALL,SPLIT=test scripts/hellbender_species_test.slurm   # the single test reading
```

`SPLIT=test` first repeats the validation pass and stops, test unread, if that fails. It also has a read-once guard: it refuses to start if `docs/reports/species-test/test-results.json` exists. That file is written last and atomically, so a preempted task has written nothing and simply reruns; once it exists, nothing reruns. The test split has been read (job 18082949). Report: [species-test](reports/species-test.md).
