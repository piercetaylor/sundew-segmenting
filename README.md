# Sundew Segmentation

This project finds the sundew (*Drosera*) in a photo, crops to it, and identifies the species. It is built on licensed iNaturalist photographs. Training runs on the Hellbender SLURM cluster, and every comparison uses five paired seeds with its decision rule written down before the run.

![Twelve CC0 sundew examples](assets/dataset-preview.jpg)

## Where it stands

**Data.** 500 hand-screened iNaturalist photos became 191 reviewed plant masks (frozen as v0.3.0), plus 30 field masks. The species corpus is larger: 17,678 photos of 110 species, split by photographer, with 2,601 held back as a test set that hasn't been scored yet. Exact coordinates are never stored.

**Segmentation.** SegFormer-B0 beat a U-Net on all five seeds (validation IoU 0.632 vs 0.610). Adding the field masks raised IoU on messy real-world photos from 0.552 to 0.613. Only 2.4% of uncurated photos got a crop that missed the plant.

**Species classifier.** Balanced accuracy on validation, 110 species:

| Model | Balanced accuracy |
| --- | ---: |
| ResNet-18 baseline | 0.50 |
| DINOv2-S, fine-tuned | 0.71 |
| DINOv2-S, distilled from DINOv2-L (100 epochs) | 0.735 |
| DINOv2-L teacher, 5-model ensemble | 0.834 |

The large DINOv2-L model is the most accurate, but it's too big to run on a phone. So it teaches the small DINOv2-S instead. Weighting that teaching by species keeps the rare species (under 40 training photos) from being crowded out. It beats plain training by +0.016 [+0.011, +0.021] on all five seeds. Cropping helped the old ResNet-18 but stopped mattering once the stronger models were fine-tuned.

Full results and the reasoning behind each decision are in `reports/` and [the species classifier plan](docs/species-classifier-plan.md).

## Running it

Python 3.10+. Downloaded images stay out of Git, but the manifests, splits and attribution records are enough to rebuild them. The cluster environment is in `environment.yml`, and [Hellbender training](docs/hellbender-training.md) covers setup.

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

**Train the species classifier.** Build the corpus and crops, then fine-tune. `hellbender_species_finetune.slurm` takes the model and options as environment variables, as documented in its header:

```bash
sbatch scripts/hellbender_species_corpus.slurm
sbatch scripts/hellbender_species_110_crops.slurm
sbatch scripts/hellbender_full_cache.slurm      # 576 px frames, used by CACHE=1
sbatch --array=0-4 --export=ALL,MODEL=dinov2-l-reg scripts/hellbender_species_finetune.slurm   # teacher
sbatch --array=0-4 --time=08:00:00 --export=ALL,MODEL=dinov2-s,KD=1,KD_WEIGHT=label,EPOCHS=100,CACHE=1 \
  scripts/hellbender_species_finetune.slurm                                                   # distilled student
python scripts/summarize_species_finetune.py --help
```

## Usage

This is a personal, noncommercial research project. iNaturalist's terms don't allow its data to be used for commercial AI training. Each photo keeps its own licence (see [the licence policy](docs/licence-policy.md)), and the software licence doesn't change that. More detail: [project plan](docs/project-plan.md), [dataset card](reports/dataset-card.md), [release guide](docs/dataset-release.md).
