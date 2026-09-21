# Returning the field-probe masks to Hellbender

Forty-seven masks came back from the fifty scraped images described in
`reports/scrape-probe-50.md`. This is the transfer and the two jobs that follow,
in the order the annotation handoff asked for: score first, retrain second.
Everything from the transfer onward runs on the cluster.

## What was exported, and from where

The annotations lived only in the local Label Studio database — project 2,
`Sundew Field Probe – 2026-09-19`, all fifty tasks reviewed. They had never been
exported to a file, so nothing had left the laptop.

`scripts/export_reviewed_masks.py` could not do this job. It reads from a
running Label Studio over HTTP, takes its project id from
`.tools/label-studio-project.json` (which points at the curated project), and
deletes any mask it does not expect — aimed at project 2 it would have removed
the 191 curated masks. `scripts/export_field_probe_masks.py` reads the database
read-only instead and writes only inside its own output directory. It decodes
brush RLE with the same `rgba.max(axis=2) >= 128` threshold as the curated
export, so field and curated masks agree on anti-aliased edges.

```bash
python scripts/export_field_probe_masks.py --project 2 \
  --provenance ../sundew-scrape-probe/metadata.jsonl
```

| | |
| --- | --- |
| reviewed | 50 |
| masks written | 47 — 41 `complete`, 6 `ambiguous` |
| rejected, no mask | 3 — `inat_310902349`, `inat_414839467`, and `inat_329881443` (plant dead) |
| mean foreground | 18.5% ground truth, against 18.1% predicted |
| empty masks | none |
| species | 29 |
| licence | 37 CC BY, 10 CC0, attribution present for all 47 |

Both `complete` and `ambiguous` masks are written and the quality choice is
recorded per row, so filtering stays a downstream decision. Discarding the six
at export time would not have been recoverable.

`inat_329881443`, the hand-held *D. gunniana* the scrape-probe report named as
its clearest failure, came back rejected because the plant was dead. That is a
genuine rejection rather than a hard trace, so the batch no longer contains that
particular diagnostic. The other named failure, `inat_686282548` — the mask that
landed on a green tape measure — was annotated and is held out below.

Every mask was checked pixel-for-pixel against its image dimensions, and five
were rendered as overlays (`data/reports/field-probe-mask-check.jpg`) to confirm
the decode lands on plant rather than background. Worth noting from that check:
`inat_453365067` is a flower close-up held in a hand, and the mask correctly
covers the flower while excluding the fingers — exactly the case the batch was
collected for.

## The split, and why part of the batch is held back

All 47 into training would have destroyed the only field-distribution number
available. The retrained model's field score would be measured on its own
training data, and telling whether the fix worked would need a second annotation
round before step 3 could run at all.

`scripts/assign_field_probe_split.py` holds 17 back. Observers stay on one side
(45 distinct observers across the 47, two contributing two images each, so
grouping costs almost nothing here); the six `ambiguous` reviews go to training
only, because an evaluation label the annotator was unsure of muddies the number
it exists to produce; and `inat_686282548` is pinned to evaluation, because
fixing a named failure on data the model trained on would prove nothing.

```bash
python scripts/assign_field_probe_split.py
```

| split | images | observers | species | quality | mean foreground |
| --- | --- | --- | --- | --- | --- |
| `field-train` | 30 | 28 | 20 | 24 complete, 6 ambiguous | 19.8% |
| `field-eval` | 17 | 17 | 13 | 17 complete | 16.1% |

No observer appears on both sides; the script fails rather than reporting a
leak. Assignment is seeded (`20260921`) and recorded in
`data/reports/field-probe-split.json`, so it reproduces.

One caveat worth stating before the retrain: 30 field images added to 144
curated is roughly a 20% addition. It may not move the needle much, and "not
enough data yet" is a real possible outcome — that result is what would justify
the next annotation round rather than a disappointment.

## Send it

`rsync` is not installed in this Git Bash, so this uses `scp`. The bundle is
1.9 MB, because only masks and manifests travel — the fifty JPEGs are already on
the cluster at `sundew-scrape-probe/images/`.

```bash
scp dist/field-probe-v1.tar.gz dist/field-probe-v1.tar.gz.sha256 pmt5gt@hellbender.rnet.missouri.edu:/cluster/VAST/mendozacozatld-lab/PierceTaylor/
```

Then on the cluster, build an evaluation root. `paired_samples` pairs strictly
by `<root>/images/<split>/<stem>.jpg` against `<root>/masks/<split>/<stem>.png`,
so the images have to appear under matching split directory names:

```bash
cd /cluster/VAST/mendozacozatld-lab/PierceTaylor
sha256sum -c field-probe-v1.tar.gz.sha256
tar -xzf field-probe-v1.tar.gz
for split in field-train field-eval; do
  mkdir -p "field-probe-eval/images/$split"
  while read -r name; do
    ln -sfn "$PWD/sundew-scrape-probe/images/$name" "field-probe-eval/images/$split/$name"
  done < "field-probe-v1/$split-images.txt"
done
ln -sfn "$PWD/field-probe-v1/masks" field-probe-eval/masks
ls field-probe-eval/images/field-eval | wc -l    # expect 17
ls field-probe-eval/masks/field-eval | wc -l     # expect 17
```

Worth one check before trusting the pairing: `field-probe-records.jsonl` carries
a `sha256` per image and the cluster copies came from the same scrape. A file
re-encoded or rotated in transit would still be the right size and would
misalign silently.

## Step 1 — score the current model

This is the first honest field number and it needs no further decisions. The
relevant checkpoint is the `combined` recipe from the sweep, not the one under
`models/hellbender/`:

```bash
cd /cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting
find models -name '*best.pt' -path '*combined*'
```

Run it against the held-out half. That is the number the retrain gets compared
against:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint <path from above> \
  --split field-eval \
  --images /cluster/VAST/mendozacozatld-lab/PierceTaylor/field-probe-eval/images \
  --masks /cluster/VAST/mendozacozatld-lab/PierceTaylor/field-probe-eval/masks \
  --metadata /cluster/VAST/mendozacozatld-lab/PierceTaylor/field-probe-v1/field-probe-records.jsonl \
  --output data/reports/field-probe-evaluation
```

Repeating it with `--split field-train` gives the descriptive number over the
other 30 while they are still unseen — useful now, meaningless after the
retrain. Compare both against the 0.683 mean IoU measured on curated validation.
That gap is the quantity this batch exists to produce, and the overlay sheet the
script writes is sorted worst-first, which is where the interesting cases are.

## Step 2 — retrain

**Which model.** `combined` on `segformer-b0`. The 0.683 checkpoint that ran the
probe and produced the hand and tape-measure failures is that one, and the
question being asked is whether field masks fix *those* failures, so that is the
model to retrain. `unet-resnet34` is the other arm of the
`hellbender_train.slurm` array rather than the current best — running it too
costs one extra GPU half-hour on `gpu_requeue` and gives a free second opinion
on whether field data helps generally or only one architecture, but it is the
extra, not the result.

Note that `hellbender_train.slurm` is **not** the `combined` recipe — it is the
two-arm architecture comparison at 30 epochs and 768 px. `combined` is variant 5
of the twelve-way `hellbender_recipe_sweep.slurm` (`--image-size 1024 --epochs
80 --patience 15 --learning-rate 3e-4 --loss bce-tversky`, seeds 17 and 101,
`--time=01:00:00`), which is array indices 10 and 11.

**Build a combined dataset root.** `train_baseline.py` reads only the `train` and
`validation` split directories, so a `field-train/` directory is invisible to
it; the field masks have to land in `train/`. Do not mutate the frozen
`sundew-segmenting-reviewed-v0.3.0/` — build a new root beside it:

```bash
cd /cluster/VAST/mendozacozatld-lab/PierceTaylor
FROZEN=sundew-data/sundew-segmenting-reviewed-v0.3.0
mkdir -p field-train-root/{images,masks}/{train,validation}
for split in train validation; do
  ln -sfn "$PWD/$FROZEN/images/$split"/* "field-train-root/images/$split/"
  ln -sfn "$PWD/$FROZEN/masks/$split"/*  "field-train-root/masks/$split/"
done
while read -r name; do
  ln -sfn "$PWD/sundew-scrape-probe/images/$name" "field-train-root/images/train/$name"
  ln -sfn "$PWD/field-probe-v1/masks/field-train/${name%.jpg}.png" "field-train-root/masks/train/${name%.jpg}.png"
done < field-probe-v1/field-train-images.txt
ls field-train-root/images/train | wc -l   # expect 174 = 144 curated + 30 field
cat "$FROZEN/metadata/training-records.jsonl" field-probe-v1/field-probe-records.jsonl \
  > field-train-root/training-records.jsonl
```

The concatenated manifest is fine: the loader only reads `curated_path`'s
basename to look up `growth_form`, and the field rows carry both.

**Turn off growth-form balancing.** `--balanced-growth-forms` is on by default
and weights by inverse frequency. Nine of the 30 field-train records carry
`growth_form: unknown`, which would become its own upweighted bucket and
oversample a group that means nothing morphologically. Pass
`--no-balanced-growth-forms`. Resolving those nine with
`scripts/update_growth_form_metadata.py` first would be better science; turning
balancing off is the faster controlled option.

**Run the baseline again under the same conditions.** Comparing a field-augmented
run against a number from a different day's sweep is not a controlled
comparison — the baseline has to be rerun with the same seed and the same flags,
including `--no-balanced-growth-forms`. That means two jobs, one per dataset
root, not one.

```bash
cd /cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting
mkdir -p logs
for name in frozen field; do
  case $name in
    frozen) ROOT=/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-data/sundew-segmenting-reviewed-v0.3.0
            META=$ROOT/metadata/training-records.jsonl ;;
    field)  ROOT=/cluster/VAST/mendozacozatld-lab/PierceTaylor/field-train-root
            META=$ROOT/training-records.jsonl ;;
  esac
  sbatch --job-name="sundew-$name" --partition=gpu_requeue --gres=gpu:A100:1 \
    --cpus-per-task=8 --mem=32G --time=01:30:00 \
    --output="logs/%x-%j.out" --error="logs/%x-%j.err" \
    --wrap "module load mamba/v2.4.0 && eval \"\$(conda shell.bash hook)\" && conda activate sundew-seg && \
      export TORCH_HOME=/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-cache/torch && \
      export HF_HOME=/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-cache/huggingface && \
      python scripts/train_baseline.py --model segformer-b0 \
        --images $ROOT/images --masks $ROOT/masks --metadata $META \
        --output models/field-compare/$name \
        --image-size 1024 --epochs 80 --patience 15 --learning-rate 3e-4 \
        --loss bce-tversky --batch-size 8 --workers 7 --seed 17 \
        --no-allow-partial --no-balanced-growth-forms"
done
```

`--no-allow-partial` is deliberate and is why only the 30 field masks go in — all
50 images linked into `train/` would hard-error on the three without masks.
Walltime is raised to 90 minutes because 80 epochs at 1024 px is well past the
30 minutes the 768 px array job needs.

Then score both checkpoints against `field-eval` with the step-1 command and
compare. The curated test split stays untouched.

## Housekeeping

`HANDOFF.md` in `sundew-scrape-probe/` asked to be deleted once the masks came
back. The masks are back.
