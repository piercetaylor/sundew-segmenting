# Sundew Segmentation

This project builds a reproducible dataset and model for segmenting visible sundew tissue in RGB photographs. The initial model comparison will use U-Net with a ResNet-34 encoder and SegFormer-B0, followed by optional Detectron2 instance segmentation if overlapping rosettes require it.

![Twelve CC0 sundew examples](assets/dataset-preview.jpg)

## Current status

- Project and reference-repository research are documented in `docs/`.
- The acquisition pipeline downloads only Research Grade, non-captive iNaturalist photographs whose individual photo license is CC0 or CC BY.
- Exact coordinates are deliberately excluded.
- 500 licensed candidates have been acquired and visually screened.
- A 250-image core set is normalized and split by observer for mask annotation.
- Downloaded images remain outside Git; source, attribution, curation, and split manifests are reproducible.

## Acquire candidate images

Use Python 3.10 or newer with Pillow installed:

```powershell
$env:PYTHONPATH = "src"
python scripts/acquire_inaturalist.py --count 500
python scripts/make_contact_sheets.py --output data/reports/curation_sheets --page-size 30 --columns 5 --thumb-size 256 --font-size 16
python scripts/build_curated_dataset.py
python scripts/export_label_studio_tasks.py
python scripts/audit_dataset.py
python scripts/validate_dataset.py --verify-hashes
```

The downloader records the creator, attribution, license, source page, source photo ID, dimensions, checksum, and acquisition time for every image. It caps repeated species and photographers and keeps one photo per observation.

The curation step retains every original, marks unsuitable frames, creates a diverse 250-image core set, and assigns all images from a photographer to the same train, validation, or test split. Its JPEG derivatives apply EXIF orientation and resize the longest side to at most 1600 pixels. The next data milestone is human correction of model-assisted masks; generated mask proposals must not be treated as ground truth.

## Usage constraint

This is a personal, noncommercial research portfolio. iNaturalist's terms prohibit using iNaturalist data for commercial AI or machine-learning training. Third-party photographs retain their individual licenses; the software license does not relicense them.

See [the project plan](docs/project-plan.md) and [the source and licensing audit](docs/free-image-sources.md).

Annotation instructions are in [the annotation workflow](docs/annotation-workflow.md),
the label definition is in [the annotation policy](data/annotation-policy.md), and
the release-ready documentation starts with [the dataset card](reports/dataset-card.md).
