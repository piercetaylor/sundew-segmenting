# Annotation workflow

The prepared Label Studio project uses SAM-aware point and box prompts, a brush mask,
and a required quality flag.
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

For the local project environment prepared by this repository, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_annotation_stack.ps1 -Python "C:\path\to\python.exe"
powershell -ExecutionPolicy Bypass -File scripts/start_label_studio.ps1
powershell -ExecutionPolicy Bypass -File scripts/start_mobilesam_backend.ps1
.tools\label-studio-venv\Scripts\python.exe scripts/initialize_label_studio_project.py
.tools\label-studio-venv\Scripts\python.exe scripts/smoke_test_annotation_stack.py
```

## Prefill every task

Generate one provisional mask for each task with a CLIPSeg text prompt guiding
MobileSAM toward sundew tissue:

```powershell
.tools\label-studio-venv\Scripts\python.exe scripts/generate_sam_preannotations.py --upload --skip-reviewed
```

The first run downloads `CIDAS/clipseg-rd64-refined` into `.tools/huggingface`.
Generated PNGs go to `data/annotations/sam-proposals/`, and Label Studio receives
the same masks as model predictions. Neither location is used by the baseline
training scripts. Use `--replace` to replace earlier predictions from this
generator while preserving human annotations. A JSON quality report and a small
overlay audit are written under `data/reports/sam-preannotations/`.

The generator can retain as many as five spatially distinct, high-confidence SAM
regions and merge them into one binary mask, so separated sundew plants are still
foreground. It uses the task's species metadata in the CLIPSeg prompt and negative
prompts for pitcher plants, moss, grass, and substrate. Compare a revised proposal
method with completed human reviews before replacing predictions:

```powershell
.tools\label-studio-venv\Scripts\python.exe scripts/evaluate_sam_preannotations.py
```

Open `http://127.0.0.1:8080/projects/1/data` after both services are ready. Select
the smart point or smart rectangle tool and place a prompt on a rosette. MobileSAM
returns a brush-mask proposal. Batch predictions appear when the task opens.
Inspect the edges, correct missed or extra tissue, choose a quality value, and
submit it. Hold `Alt` while placing a point to mark a negative prompt.

### Fast review loop and shortcuts

`M` is not the auto-selection key for this project. The interface uses smart
point and smart rectangle prompts rather than the separate Magic Wand control.

1. Enable **Auto-Annotation** at the bottom of the labeling screen.
2. Select the smart point tool and click once inside every visible sundew plant.
3. Hold `Alt` and click moss, grass, or substrate that was incorrectly included.
4. For branching plants, draw a tight smart rectangle around the entire plant,
   then add positive points on missed branches.
5. Accept the proposal, repair the remaining boundary errors with the brush,
   choose a quality value, and press `Ctrl+Enter` to submit.

Useful defaults are `Ctrl+Z` to undo, `Ctrl+Shift+Z` to redo, `Backspace` to
delete the selected region, `Esc` to exit the active selection, `U` to unselect,
`Ctrl+Enter` to submit, and `Ctrl+Space` to skip. The gear icon in the labeling
screen shows the mappings active in the installed Label Studio version.

## Morphology-aware review order

The `growth_form` field is a taxon-level review and sampling hint with four
values: `rosette`, `erect_or_branching`, `linear_or_forked`, and `dense_mat`.
It does not change the binary target: continue to include every visible sundew.
`dense_mat` is an image-level presentation, so override that expectation by eye
when a pygmy sundew is isolated.

Update the local metadata and rank the remaining work with:

```powershell
$env:PYTHONPATH = "src"
.tools\label-studio-venv\Scripts\python.exe scripts\update_growth_form_metadata.py
.tools\label-studio-venv\Scripts\python.exe scripts\sync_label_studio_metadata.py
.tools\label-studio-venv\Scripts\python.exe scripts\annotation_priority.py
```

The priority CSV goes to `data/reports/annotation-priority.csv`. Review the
validation and test examples of branching, linear, and dense forms first, then
return to the remaining training examples.

## Export reviewed masks

Only annotations marked `complete` are eligible for model training:

```powershell
.tools\label-studio-venv\Scripts\python.exe scripts\export_reviewed_masks.py
```

The exporter unions all brush regions in an annotation, validates dimensions,
and writes binary PNGs to `data/curated/masks/<split>/`. Ambiguous and rejected
reviews stay in Label Studio and are excluded.

## Freeze and audit a completed labeling pass

Run the task-level and pixel-level audits before training:

```powershell
$env:PYTHONPATH = "src"
.tools\label-studio-venv\Scripts\python.exe scripts\audit_annotations.py
.tools\label-studio-venv\Scripts\python.exe scripts\export_reviewed_masks.py
.tools\label-studio-venv\Scripts\python.exe scripts\audit_reviewed_masks.py
```

The annotation audit reports missing and duplicate submissions. The mask audit
checks binary values, dimensions, empty masks, and extreme foreground fractions.
Its local `foreground-extremes.jpg` contact sheet should be inspected before the
dataset version is frozen.

After the audit passes, train on `train`, select checkpoints using `validation`,
and keep `test` locked until the model family, resolution, loss, and threshold
have been chosen. Use `scripts/evaluate_checkpoint.py` for validation overlays;
green is a correct foreground pixel, yellow is a false positive, and magenta is
a missed sundew pixel.

Label Studio documents the local-file URL form used here and recommends limiting
the document root to the image directory:
https://labelstud.io/guide/storage_local

Its Segment Anything integration can provide interactive proposals, but every
proposal must be inspected and corrected before it becomes a training label:
https://labelstud.io/guide/ml_tutorials/segment_anything_model

Keep annotation exports under `data/annotations/`; they are excluded from Git until
their licensing, privacy, and quality checks are complete.
