# Data layout

**In short.** No photographs, masks or derived images are in Git. The
segmentation data lives under `data/` locally; the species corpus and the
transfer set live next to the repository on Hellbender. Accepted licences are
CC0 and CC BY for the segmentation set, and CC0, CC BY and CC BY-NC for the
species data ([`docs/licence-policy.md`](../docs/licence-policy.md)). Exact
coordinates are never requested or stored.

What each set contains, and its counts, is in the
[dataset card](../docs/reports/dataset-card.md).

## Segmentation data (in the repository tree, not tracked)

```text
data/
├── raw/inaturalist/
│   ├── images/              # original licensed photographs
│   ├── metadata.jsonl       # image-level provenance and attribution
│   ├── rejected.jsonl       # failed, duplicate, or invalid downloads
│   └── acquisition.json     # query and run summary
├── curated/
│   ├── images/{train,validation,test}/
│   ├── masks/{train,validation,test}/   # human-reviewed complete masks
│   ├── metadata.jsonl       # provenance, grouped split, derivative details
│   ├── curation.jsonl       # decision for all 500 reviewed candidates
│   └── summary.json
├── annotations/             # Label Studio tasks, exports, SAM proposals
└── reports/                 # contact sheets and audit output
```

Only photos with an image-level `cc0` or `cc-by` licence are in this set. The
train, validation and test split is grouped by iNaturalist observer. The frozen
training snapshot, v0.3.0, is built from here into `dist/`
([`docs/dataset-release.md`](../docs/dataset-release.md)).

`growth_form` in the curated manifest is a morphology prior for balanced
sampling and evaluation. It is not ground truth for each photo and not part of
the binary segmentation target.

## Species corpus (`../sundew-species-corpus/`)

Built by `scripts/acquire_species_corpus.py`
(`sbatch scripts/hellbender_species_corpus.slurm`).

```text
sundew-species-corpus/
├── acquisition.json         # query, filters, seed, per-species counts
├── Drosera_<species>/
│   ├── selection.jsonl      # the sample, written before any download
│   ├── metadata.jsonl       # one row per downloaded photo
│   └── images/inat_<photo_id>.jpg
├── split-110-test/
│   ├── species-records.jsonl  # every photo with its label and split
│   └── labels.json            # the 110 class names
├── full-s576/inat_<photo_id>.jpg   # full frames, short side 576 px
└── crops/                   # predicted crops from the segmenter
```

- **Resume.** `selection.jsonl` is written first and replayed on resume, so a
  job restarted after its walltime reproduces the same sample. A row in
  `metadata.jsonl` means the file landed; `scripts/verify_corpus.py` checks
  that images, licences, attribution and coordinate fields match the policy.
- **The 576 px cache.** `scripts/make_full_cache.py`
  (`sbatch scripts/hellbender_full_cache.slurm`) writes each frame once with
  its short side at 576 px (LANCZOS, JPEG quality 95, 4:4:4 chroma, no EXIF
  rotation, matching the training loader). Train and validation only by
  default: the test split is not decoded until it is scored.
- `data/species-110-sections.json` (tracked) maps each species to its subgenus
  and section, from the iNaturalist taxonomy API. It is used only to score
  near-misses.

## Transfer set (`../sundew-transfer-set/`)

Built by `scripts/acquire_transfer_set.py`
(`sbatch scripts/hellbender_transfer_set.slurm`) in three modes: a dry run
(metadata only), `--download`, then `--dedup`.

```text
sundew-transfer-set/
├── pull.json                # observation-id ceiling and query time
├── pool/<species>/<source>.jsonl   # candidates per source (cached)
├── selection.jsonl          # the planned photos
├── transfer-plan.{json,md}  # dry-run counts (copy in docs/reports/species-distill/)
├── Drosera_<species>/{metadata.jsonl,images/}
├── transfer-records.jsonl   # kept photos after deduplication
├── transfer-dedup.json      # what deduplication removed
└── full-s576/               # 576 px cache, as for the corpus
```

## Rules

- Do not add images by hand without the same provenance fields in the manifest
  (creator, attribution, licence code and URL, source page, checksum).
- Keep each photo's licence code, so the CC0/CC BY subset can always be
  separated from CC BY-NC.
