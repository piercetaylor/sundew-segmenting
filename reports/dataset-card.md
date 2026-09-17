---
pretty_name: Sundew Segmentation Core
task_categories:
  - image-segmentation
language:
  - en
size_categories:
  - n<1K
---

# Sundew Segmentation Core

## Dataset description

This working dataset supports binary semantic segmentation of visible sundew
(`Drosera`) tissue in field RGB photographs. It contains a 250-image core selected
from 500 research-grade, wild iNaturalist observations. The core spans 56 reported
taxa and 179 photographers. Its purpose is a small, noncommercial research and
portfolio project.

The source images are not committed to the software repository. Each local image
has a manifest row containing its iNaturalist observation and photo IDs, source
page, creator, image-level license, dimensions, checksum, review decision, and split.

## Splits

| Split | Images |
|---|---:|
| Train | 184 |
| Validation | 26 |
| Test | 40 |

All photographs from one observer are assigned to one split. This reduces leakage
from photographer-specific equipment, processing, and habitat preferences. It does
not guarantee plant- or site-independent evaluation because precise location and
individual-plant identifiers are unavailable.

## Licenses and provenance

The core includes 201 CC BY photographs and 49 CC0 photographs. Every image retains
its individual creator, attribution, license URL, and source page in
`data/curated/metadata.jsonl`. The photographs are not relicensed by this project.
Derived JPEGs apply EXIF orientation, convert to RGB, and resize the longest side to
at most 1600 pixels.

iNaturalist data is used subject to iNaturalist's terms. This dataset is limited to
personal, noncommercial research; it must not be used for commercial AI or machine
learning training. Users must re-check the source terms and each image license
before redistribution.

## Curation

The 500-image candidate pool was inspected in indexed contact sheets. Images
dominated by flowers or seed stalks, distant plants, blur, severe obstruction, or
too little useful sundew tissue were rejected. Of the remaining images, a seeded
selection capped common species at 25 and photographers at 8 where possible. This
yielded 250 core images, 64 reserves, and 186 rejects. Originals are retained so
every decision is reversible.

No exact SHA-256 duplicates or identical 64-bit difference-hash groups were found.
All 500 source rows have a creator and source page. Exact coordinates are neither
acquired nor published.

## Labels

The images are prepared for annotation but do not yet have reviewed ground-truth
masks. Follow `data/annotation-policy.md`. Model-generated or SAM-assisted masks are
proposals until a person corrects and accepts them. Double-label 10% of the core and
report inter-annotator Dice and IoU before training the final benchmark.

## Limitations

- iNaturalist sampling is opportunistic and does not represent sundew prevalence.
- Taxon labels are community identifications and may contain errors.
- Observer-grouped splits cannot remove all geographic or individual-plant leakage.
- The dataset lacks calibrated ground sample distance, controlled field-trial
  treatments, and UAV flight metadata.
- The visual suitability review and all future masks require human confirmation.
