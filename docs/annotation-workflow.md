# Annotation workflow

**In short.** Masks are drawn in a local Label Studio project. Each task opens
with a MobileSAM proposal guided by a CLIPSeg text prompt; the annotator
corrects it, sets a quality flag, and submits. Only masks marked `complete` are
exported for training, and they are audited before a dataset version is
frozen. What counts as sundew is set by
[`data/annotation-policy.md`](../data/annotation-policy.md).

The task list points at the normalised files under `data/curated` and carries
split, species, source, creator and licence into the labelling interface.

## Set up

Export the tasks:

```powershell
$env:PYTHONPATH = "src"
python scripts/export_label_studio_tasks.py
```

Install and start the stack prepared by this repository:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_annotation_stack.ps1 -Python "C:\path\to\python.exe"
powershell -ExecutionPolicy Bypass -File scripts/start_label_studio.ps1
powershell -ExecutionPolicy Bypass -File scripts/start_mobilesam_backend.ps1
.tools\label-studio-venv\Scripts\python.exe scripts/initialize_label_studio_project.py
.tools\label-studio-venv\Scripts\python.exe scripts/smoke_test_annotation_stack.py
```

To configure Label Studio by hand instead:

1. Set `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true`.
2. Set `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` to the absolute `data/curated`
   directory before starting Label Studio.
3. Create a project and paste `annotation/label-config.xml` into its labelling
   setup.
4. Import `data/annotations/label-studio-tasks.json` as tasks. Do not also sync
   the image directory, which would create a second set of tasks.

Label Studio documents the local-file URL form used here and recommends
limiting the document root to the image directory:
https://labelstud.io/guide/storage_local

## Prefill every task

```powershell
.tools\label-studio-venv\Scripts\python.exe scripts/generate_sam_preannotations.py --upload --skip-reviewed
```

This writes one provisional mask per task. The first run downloads
`CIDAS/clipseg-rd64-refined` into `.tools/huggingface`. PNGs go to
`data/annotations/sam-proposals/` and are uploaded to Label Studio as model
predictions; training never reads either. `--replace` replaces earlier
predictions from this generator and keeps human annotations. A quality report
and an overlay audit go to `data/reports/sam-preannotations/`.

The generator keeps up to five separate high-confidence SAM regions and merges
them, so separated plants are still foreground. The CLIPSeg prompt uses the
task's species, with negative prompts for pitcher plants, moss, grass and
substrate. Before switching to a revised proposal method, compare it against
completed human reviews:

```powershell
.tools\label-studio-venv\Scripts\python.exe scripts/evaluate_sam_preannotations.py
```

Every proposal must be inspected and corrected before it becomes a training
label (https://labelstud.io/guide/ml_tutorials/segment_anything_model).

## Review loop

Open `http://127.0.0.1:8080/projects/1/data` once both services are running.
The project uses smart point and smart rectangle prompts, not the Magic Wand,
so `M` does nothing here.

1. Enable **Auto-Annotation** at the bottom of the labelling screen.
2. With the smart point tool, click once inside every visible sundew.
3. Hold `Alt` and click moss, grass or substrate that was wrongly included.
4. For branching plants, draw a tight smart rectangle around the whole plant,
   then add positive points on missed branches.
5. Accept the proposal, fix remaining edges with the brush, choose a quality
   value, and press `Ctrl+Enter` to submit.

Other shortcuts: `Ctrl+Z` undo, `Ctrl+Shift+Z` redo, `Backspace` delete the
selected region, `Esc` leave the active selection, `U` unselect, `Ctrl+Space`
skip. The gear icon shows the mappings for the installed version.

Export a versioned snapshot after each labelling session.

## Review order by growth form

`growth_form` is a taxon-level hint with four values: `rosette`,
`erect_or_branching`, `linear_or_forked` and `dense_mat`. It does not change
the target: include every visible sundew. `dense_mat` describes how a photo
looks, so override it by eye when a pygmy sundew stands alone.

Sync the metadata and rank the remaining work:

```powershell
$env:PYTHONPATH = "src"
.tools\label-studio-venv\Scripts\python.exe scripts\sync_label_studio_metadata.py
.tools\label-studio-venv\Scripts\python.exe scripts\annotation_priority.py
```

The ranking goes to `data/reports/annotation-priority.csv`. Review the
validation and test examples of branching, linear and dense forms first, then
the rest of train.

## Export, audit and freeze

```powershell
$env:PYTHONPATH = "src"
.tools\label-studio-venv\Scripts\python.exe scripts\audit_annotations.py
.tools\label-studio-venv\Scripts\python.exe scripts\export_reviewed_masks.py
.tools\label-studio-venv\Scripts\python.exe scripts\audit_reviewed_masks.py
```

- The annotation audit reports missing and duplicate submissions.
- The exporter takes only `complete` annotations, merges all brush regions,
  checks dimensions, and writes binary PNGs to `data/curated/masks/<split>/`.
  Ambiguous and rejected reviews stay in Label Studio.
- The mask audit checks binary values, dimensions, empty masks and extreme
  foreground fractions. Look at its `foreground-extremes.jpg` contact sheet
  before freezing.

Then freeze the dataset ([dataset-release.md](dataset-release.md)). Train on
`train`, pick checkpoints on `validation`, and keep `test` locked until the
model, resolution, loss and threshold are chosen. `scripts/evaluate_checkpoint.py`
draws validation overlays: green is a correct sundew pixel, yellow a false
positive, magenta a missed sundew pixel.

Keep annotation exports under `data/annotations/`. They stay out of Git until
their licensing, privacy and quality checks are complete.
