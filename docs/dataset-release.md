# Dataset release procedure

The public release is metadata-only. Downloaded photographs and future masks
remain outside Git because each image has its own creator and license terms.
The release records enough provenance to reacquire the same candidates and
rebuild the curated split after checking the current source terms.

From the repository root, first acquire and curate the data, then build the
release directory:

```powershell
$env:PYTHONPATH = "src"
python scripts/acquire_inaturalist.py --count 500
python scripts/build_curated_dataset.py
python scripts/export_label_studio_tasks.py
python scripts/validate_dataset.py --verify-hashes
python scripts/package_dataset_release.py
```

`dist/sundew-segmenting-metadata-v0.1.0/release-manifest.json` lists every released file, its byte
count, and SHA-256 digest. The default package includes the raw and curated
provenance manifests, curation summary, Label Studio task definitions, and the
dataset card. It intentionally contains no images or segmentation masks.

Before sharing any image payload, re-check each source page and license, preserve
the per-image attribution fields, and obtain permission where the source terms
require it. Model-generated or SAM-assisted masks are proposals until a person
reviews and accepts them. A valid benchmark release should include accepted
masks, the annotation export, annotation policy/version, and the split summary.
