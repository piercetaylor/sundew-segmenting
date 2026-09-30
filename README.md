# Sundew Segmentation

This project does two separate jobs on photos of sundews (*Drosera*). A SegFormer-B0 model outlines the plant, and a 22 MB int8 DINOv2-S model names the species in about 120 ms per photo on one CPU thread. It knows 110 species: those with at least 50 research-grade, wild, openly licensed observations on iNaturalist, out of the 220+ *Drosera* species. The species model reads the whole photo, so it does not need the segmenter. It is built on licensed iNaturalist photographs and trained on Mizzou's Hellbender HPC cluster. Every comparison uses five paired seeds, with its decision rule written down before the run.

![Twelve licensed sundew examples](assets/dataset-preview.jpg)

The preview's credits and licences are in [its attribution record](assets/dataset-preview-attribution.md).

## Where it stands

**Data.** 500 hand-screened iNaturalist photos became 191 reviewed plant masks (frozen as [v0.3.0](docs/reports/dataset-freeze-v0.3.0.md)), plus 30 field masks, initially assuming that the models would work better on segmented photos. The current species corpus is larger: 17,678 photos of 110 species, split by photographer, with 2,601 held back as a test set. A further 12,126 unlabelled photos form a transfer set for distillation. Exact coordinates are never stored. Provenance, splits and limitations are in the [dataset card](docs/reports/dataset-card.md).

**Segmentation.** SegFormer-B0 beat a U-Net on all five seeds (validation IoU 0.632 vs 0.610). Adding the field masks raised IoU on messy real-world photos from 0.552 to 0.613. Only 2.4% of uncurated photos got a crop that missed the plant.

**Species classifier.** Balanced accuracy over 110 species:

| Model | Validation | Test |
| --- | ---: | ---: |
| ResNet-18 baseline | 0.50 | 0.52 |
| DINOv2-S, fine-tuned | 0.72 | 0.74 |
| DINOv2-S, distilled from DINOv2-L | 0.735 | - |
| DINOv2-S, distilled, plus the transfer set | 0.768 | 0.783 |
| **Shipped: that model's seed 17, int8 ONNX** | **0.764** | **0.775** (top-5 0.962) |
| DINOv2-L teacher, 5-model ensemble | 0.834 | 0.842 |

The large DINOv2-L model is the most accurate, but it's too big to run on a phone, so it teaches the small DINOv2-S instead. Weighting that teaching by species keeps the rare species from being crowded out, and adding 12,126 unlabelled photos for the teacher to label gave the largest single gain (+0.033). The shipped model is 22 MB, runs in about 120 ms on one CPU thread, and clears the release floor (0.75, top-5 0.95). The test set was scored once, after everything else was fixed.

The classifier reads the whole photo, not a crop around the plant. Cropping helped the old ResNet-18 (0.516 against 0.499), but once the DINOv2 models were fine-tuned it made no difference (DINOv2-L: -0.004 [-0.012, +0.006]), so every DINOv2 model here, including the shipped one, trains and runs on full frames ([why](docs/reports/species-finetune.md)). Segmentation stands on its own: it finds and outlines the plant, and it is not a step in species identification.

**Known limits.** Species with under 40 training photos are the weak spot (0.65 on test, against 0.84 for common ones). The model always names one of the 110 species, so it has no answer yet for other *Drosera* or other plants. Browser preprocessing and int8 behaviour in WebAssembly are not yet measured.

How the classifier works and why it is built this way is in [the species classifier design](docs/species-classifier.md); the pre-registered rules are in [the plan](docs/species-classifier-plan.md). Every number above comes from a report in [`docs/reports/`](docs/reports/), for example [distillation](docs/reports/species-distill.md), [export](docs/reports/species-release.md) and [the held-out test](docs/reports/species-test.md).

## Running it

Python 3.10+. Downloaded images and model weights stay out of Git, but the manifests, splits and attribution records are enough to rebuild them. The cluster environment is in `environment.yml`, and [Hellbender training](docs/hellbender-training.md) lists every job that reproduces the results. `pip install -e ".[species]"` installs the species stack locally, and `python -m unittest discover -s tests` runs the repository checks.

**Build the dataset.** The downloader records creator, licence, source and checksum for every photo:

```powershell
$env:PYTHONPATH = "src"
python scripts/acquire_inaturalist.py --count 500
python scripts/make_contact_sheets.py --output data/reports/curation_sheets
python scripts/build_curated_dataset.py
python scripts/export_label_studio_tasks.py
python scripts/validate_dataset.py --verify-hashes
```

**Annotate.** Label Studio with MobileSAM proposals, which a person accepts or rejects (see [the annotation workflow](docs/annotation-workflow.md)):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_annotation_stack.ps1   # once
powershell -ExecutionPolicy Bypass -File scripts/start_label_studio.ps1
powershell -ExecutionPolicy Bypass -File scripts/start_mobilesam_backend.ps1
.tools\label-studio-venv\Scripts\python.exe scripts/initialize_label_studio_project.py
.tools\label-studio-venv\Scripts\python.exe scripts/generate_sam_preannotations.py --upload --skip-reviewed
```

**Train segmentation:**

```bash
python scripts/export_reviewed_masks.py && python scripts/freeze_reviewed_dataset.py
sbatch scripts/hellbender_seed_sweep.slurm      # U-Net vs SegFormer
sbatch scripts/hellbender_field_compare.slurm   # with and without field masks
```

**Train the species classifier.** Build the corpus, frame cache and transfer set, train the teacher, then distil. The model never sees the crops, but the training script refuses to run unless every photo also has one (so full-frame and crop runs always use the same photos), so the crops job still has to run first. `hellbender_species_finetune.slurm` takes the model and options as environment variables, as documented in its header:

```bash
sbatch scripts/hellbender_species_corpus.slurm
sbatch scripts/hellbender_species_110_crops.slurm # plant crops: checked for pairing, not used by the shipped model
sbatch scripts/hellbender_full_cache.slurm      # 576 px frames, used by CACHE=1
sbatch scripts/hellbender_transfer_set.slurm    # unlabelled photos, used by TRANSFER=1
sbatch --array=0-4 --export=ALL,MODEL=dinov2-l-reg scripts/hellbender_species_finetune.slurm   # teacher
sbatch --array=0-4 --time=08:00:00 \
  --export=ALL,MODEL=dinov2-s,KD=1,KD_WEIGHT=label,EPOCHS=100,CACHE=1,TRANSFER=1 \
  scripts/hellbender_species_finetune.slurm                                                   # shipped recipe
python scripts/summarize_species_finetune.py --help
```

**Export and score.** `hellbender_species_export.slurm` writes the fp32 and int8 ONNX models and a `release.json` (preprocessing, temperature, checksums) to `models/species-110/release/`. `hellbender_species_test.slurm` scores the held-out test split; it refuses to run a second time.

```bash
sbatch scripts/hellbender_species_export.slurm
sbatch --export=ALL,SPLIT=val scripts/hellbender_species_test.slurm    # validation only
```

## Repository layout

```text
src/            the sundew_segmentation package: acquisition, curation, segmentation baselines
scripts/        command-line steps and the Hellbender SLURM jobs
docs/           design, plan, cluster, annotation, release and licence guides
docs/reports/   one report per experiment, with its numbers and decision
release/        the published species model (int8 ONNX), its model card, licence and photo credits
data/           annotation policy and small tracked metadata (images are not in Git)
annotation/     the Label Studio labelling config
assets/         the README preview image and its attribution record
tests/          unit tests, run in CI
deprecated/     retired docs, reports and scripts, kept for the record
```

## Use and citation

This is a personal, noncommercial research project. [iNaturalist's terms](https://www.inaturalist.org/pages/terms) don't allow its data to be used for commercial AI training. Each photo keeps its own licence and attribution (see [the licence policy](docs/licence-policy.md)), and nothing else in this repository changes those terms. The code is released under the Apache License 2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)); that licence covers the code only, not the iNaturalist photographs, the masks derived from them, or any trained weights. The species model weights ([`release/species-v1.0.0/`](release/species-v1.0.0/)) are licensed under CC BY-NC 4.0, with credit for every training photo in its `ATTRIBUTION.md`. Cite this repository with the commit used (see [`CITATION.cff`](CITATION.cff)), and attribute each source image according to its licence.

**Computing acknowledgement.** Training ran on Hellbender. As [the Hellbender wiki](https://itrss-wiki.rnet.missouri.edu/pub/hpc/hellbender) asks, any publication using this work should include:

> The computation for this work was performed on the high performance computing infrastructure operated by Research Support Solutions in the Division of IT at the University of Missouri, Columbia MO DOI: https://doi.org/10.32469/10355/97710

The wiki also asks authors to email muitrss@missouri.edu and share a copy of the publication.

More detail: [species classifier design](docs/species-classifier.md), [all reports](docs/reports/), [dataset release guide](docs/dataset-release.md).
