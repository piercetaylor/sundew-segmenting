---
pretty_name: Sundew Segmentation Core
task_categories:
  - image-segmentation
  - image-classification
language:
  - en
size_categories:
  - n<1K
  - 10K<n<100K
---

# Sundew Segmentation Core

**In short.** Three sets of iNaturalist photos of sundews (*Drosera*): 191
reviewed segmentation masks (v0.3.0) plus 47 field masks; a species corpus of
17,678 photos of 110 species; and 12,126 unlabelled photos used only for
distillation. Every split is grouped by photographer. No images are in Git,
and exact coordinates are never stored. Use is personal and noncommercial.

Every image keeps its own creator, licence, and source page, and none is
relicensed by this project. iNaturalist's terms apply on top: its data must not
be used for commercial AI or machine-learning training. Re-check the source
terms and each image's licence before redistributing anything. The accepted
licences and the reasons are in [`docs/licence-policy.md`](../licence-policy.md).

## Segmentation set

Binary masks of visible sundew tissue in field RGB photos.

### Source and curation

500 research-grade, wild iNaturalist observations, CC0 or CC BY only, were
screened by eye in indexed contact sheets. Images dominated by flowers or seed
stalks, distant plants, blur, heavy obstruction, or too little sundew tissue
were rejected. A seeded selection then capped common species at 25 and
photographers at 8 where possible. Result: a 250-image core (56 taxa, 179
photographers; 201 CC BY, 49 CC0), 64 reserves, and 186 rejects. Originals are
kept, so every decision can be reversed.

No exact SHA-256 duplicates or identical 64-bit difference-hash groups were
found. All 500 source rows have a creator and source page. Derived JPEGs apply
EXIF orientation, convert to RGB, and resize the longest side to at most 1600
px. Provenance for each image (observation and photo IDs, source page, creator,
licence, dimensions, checksum, review decision, split) is in
`data/curated/metadata.jsonl`.

**The screen shapes the model.** Photos with hands, bright artificial objects,
heavy blur, or distant plants were screened out, so the core has almost no
examples teaching that these are background. Models trained on it fail
confidently on such photos
([`deprecated/reports/scrape-probe-50.md`](../../deprecated/reports/scrape-probe-50.md)).

### Splits

| Split | Curated tasks | Accepted pairs (v0.3.0) |
|---|---:|---:|
| Train | 184 | 144 |
| Validation | 26 | 19 |
| Test | 40 | 28 |
| **Total** | **250** | **191** |

Of the 250 reviewed tasks, 191 were accepted, 47 marked ambiguous, and 12
rejected. The accepted pairs span 51 taxa and 139 photographers, 158 CC BY and
33 CC0. See [dataset-freeze-v0.3.0.md](dataset-freeze-v0.3.0.md).

All photos from one observer are in one split, which limits leakage from a
photographer's equipment, processing and habitats. It does not guarantee
independence by plant or site, because precise locations and plant identifiers
are not available.

### Growth forms

Each image carries a taxon-level growth-form prior, used for annotation order,
balanced sampling and error analysis. It is a sampling hint, not a label
class: dense growth is a property of the photo, and young plants can differ
from the taxon prior.

| Growth form | Core (250) | Accepted (191) |
| --- | ---: | ---: |
| rosette | 169 | 131 |
| erect or branching | 45 | 32 |
| linear or forked | 23 | 17 |
| dense mat (likely) | 13 | 11 |

### Labels

Masks follow [`data/annotation-policy.md`](../../data/annotation-policy.md).
Model-generated or SAM-assisted masks are proposals until a person corrects and
accepts them. The public metadata release contains no masks. The policy asks
for 10% of the core to be double-labelled, with inter-annotator Dice and IoU
reported, before a final benchmark.

### Field masks

To measure the model on photos like those it will actually see, 50 uncurated
scraped iNaturalist photos were annotated. 47 masks came back (41 complete, 6
ambiguous; 3 rejected), spanning 29 species, 37 CC BY and 10 CC0. They are
split by observer into 30 `field-train` and 17 `field-eval`; ambiguous masks go
to training. Adding the 30 raised field IoU from 0.552 to 0.613
([field-compare.md](field-compare.md)).

## Species corpus

Labelled photos for the 110-species classifier.

- **Source.** Research-grade, wild iNaturalist observations
  (`scripts/acquire_species_corpus.py`, seed 20260921). A species is included
  if it has at least 50 available observations; each species is capped at 300
  photos and each observer at 5 photos per species.
- **Licences.** CC0, CC BY and CC BY-NC. ShareAlike and NoDerivatives are
  excluded ([`docs/licence-policy.md`](../licence-policy.md)).
- **Exclusions.** Photos in the v0.3.0 segmentation set or the field-probe set
  are excluded, so no crop comes from an image the segmenter trained on.
- **Size.** 17,678 photos of 110 species.
- **Split by observer.** Train 11,118, validation 3,959, test 2,601, from 4,650,
  308 and 1,095 observers with none shared. The test split holds 15% of each
  species (6-45 photos, median 23), carved from the training side and scored
  once. One species, *D. barbigera*, is one photo under its test quota (8 of
  9). Per-species counts:
  [`species-test/split-110-test.txt`](species-test/split-110-test.txt).
- **Duplicates.** Exact (SHA-256) duplicates are rejected at download. An
  independent audit on 2026-09-26 found no observer, observation, identical
  file, or near-duplicate (dHash distance <= 6) shared between any two splits,
  and one photo per observation
  ([species-finetune.md](species-finetune.md#audit-2026-09-26)).
- **Labels** are iNaturalist community identifications and may contain errors.

## Transfer set

Unlabelled photos of the same 110 species, used only as distillation inputs
for the on-device student. Their iNaturalist labels are not used in training.

- **Sources** (`scripts/acquire_transfer_set.py`, seed 20260927, observation-id
  ceiling 404031620, queried 2026-09-27): (a) wild research-grade photos beyond
  the corpus caps; (b) captive or cultivated photos, any grade; (c) wild
  needs-ID or casual photos with a species-level ID. Caps: 100 per species for
  (b) and (c), 300 per species in total, 5 per observer per species. Same
  licences as the corpus.
- **Exclusions.** Every validation and test observer (1,403 logins), every
  observation already in any split, every photo in the corpus or the
  segmentation sets. Other *Drosera* and other genera are never included; they
  are kept for open-set evaluation.
- **Size.** 12,179 planned; 12,165 downloaded, 14 failed. Deduplication removed
  39 (21 byte-identical to a split photo, 13 within dHash 6 of one, 5 within
  the set). **12,126 kept**: (a) 7,970, (b) 659, (c) 3,497.
- **Skew.** 88% of the planned set is common species; each of the 23 species
  with under 40 training photos gets 1-15. Plan:
  [`species-distill/transfer-plan.md`](species-distill/transfer-plan.md).

## Limitations

- iNaturalist sampling is opportunistic and does not reflect how common each
  sundew is.
- Taxon labels are community identifications and may be wrong.
- Observer-grouped splits cannot remove all geographic or individual-plant
  leakage.
- There is no calibrated ground sample distance, no controlled field-trial
  treatment, and no UAV flight metadata.
- The visual screen and every mask need human confirmation.
