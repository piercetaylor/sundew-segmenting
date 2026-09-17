# Annotation workflow

The prepared Label Studio project uses a brush mask plus a required quality flag.
The task list points at the normalized files under `data/curated` and carries the
split, species, source, creator, and license metadata into the labeling interface.

## Prepare tasks

```powershell
$env:PYTHONPATH = "src"
python scripts/export_label_studio_tasks.py
```

## Configure Label Studio

1. Install and start Label Studio locally.
2. Set `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`.
3. Set `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` to the absolute `data/curated`
   directory before starting Label Studio.
4. Create a project and paste `annotation/label-config.xml` into its labeling setup.
5. Import `data/annotations/label-studio-tasks.json` as tasks. Do not sync the image
   directory as a second set of tasks.
6. Follow `data/annotation-policy.md` and export a versioned snapshot after each
   labeling session.

Label Studio documents the local-file URL form used here and recommends limiting
the document root to the image directory:
https://labelstud.io/guide/storage_local

Its Segment Anything integration can provide interactive proposals, but every
proposal must be inspected and corrected before it becomes a training label:
https://labelstud.io/guide/ml_tutorials/segment_anything_model

Keep annotation exports under `data/annotations/`; they are excluded from Git until
their licensing, privacy, and quality checks are complete.
