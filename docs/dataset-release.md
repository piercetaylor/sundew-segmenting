# Dataset release procedure

**In short.** Acquire and curate the photos, annotate them, audit the masks,
then freeze the accepted pairs as v0.3.0. The frozen snapshot stays local,
because every photo has its own creator and licence; anything shared publicly
is metadata only.

Run everything from the repository root.

## 1. Acquire and curate

```powershell
$env:PYTHONPATH = "src"
python scripts/acquire_inaturalist.py --count 500
python scripts/build_curated_dataset.py
python scripts/export_label_studio_tasks.py
python scripts/validate_dataset.py --verify-hashes
```

## 2. Annotate and audit

Follow [annotation-workflow.md](annotation-workflow.md). Before freezing, every
task needs a review decision and the audits must pass:

```powershell
.tools\label-studio-venv\Scripts\python.exe scripts\audit_annotations.py
.tools\label-studio-venv\Scripts\python.exe scripts\export_reviewed_masks.py
.tools\label-studio-venv\Scripts\python.exe scripts\audit_reviewed_masks.py
```

## 3. Freeze

```powershell
python scripts/freeze_reviewed_dataset.py
```

This writes `dist/sundew-segmenting-reviewed-v0.3.0/` and a matching ZIP. The
snapshot holds only the accepted image-mask pairs, their per-image licence and
attribution, every review decision, the annotation and mask audits, the
annotation policy, the dataset card, and SHA-256 checksums. It refuses to
overwrite an existing snapshot. The v0.3.0 digest and counts are in
[reports/dataset-freeze-v0.3.0.md](reports/dataset-freeze-v0.3.0.md).

Keep the test split locked until the model and threshold are chosen.

## Sharing

`python scripts/package_dataset_release.py` builds a metadata-only package:
the raw and curated provenance manifests, the curation summary, the Label
Studio task definitions, the dataset card, the annotation policy and the
annotation workflow, with a `release-manifest.json` listing each file's size and SHA-256.
It contains no images, annotations or masks.

Before sharing any images or masks, re-check each source page and licence, keep
the per-image attribution fields, and get permission where the source terms
require it. Model-generated or SAM-assisted masks are proposals until a person
reviews and accepts them.
