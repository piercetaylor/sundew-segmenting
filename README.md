# Sundew Segmentation

This project curates licensed photographs of wild sundews and develops a model for segmenting visible plant tissue in RGB images. Its scripts cover image acquisition, annotation review, dataset validation, baseline training, and evaluation.

![Twelve licensed sundew examples](assets/dataset-preview.jpg)

The preview's image credits and licenses are listed in [its attribution record](assets/dataset-preview-attribution.md).

## Dataset and current result

The source is Research Grade, non-captive [iNaturalist](https://www.inaturalist.org/) observations with an image-level CC0 or CC BY license. The 250-image core was selected from 500 screened candidates. The [reviewed v0.3.0 snapshot](reports/dataset-freeze-v0.3.0.md) contains 191 accepted image-mask pairs: 144 train, 19 validation, and 28 locked test pairs. Forty-seven reviewed tasks were ambiguous and 12 were rejected. Photographs and masks are kept outside Git; the [dataset card](reports/dataset-card.md) documents provenance, splits, licenses, and limitations.

A five-epoch CPU development run with U-Net/ResNet-34 at 256 pixels used 143 training masks and 19 observer-held-out validation masks. Its mean per-image validation Dice was 0.749 and IoU was 0.609, as recorded in the [experiment report](reports/experiments/unet-resnet34-v0.1-dev-validation.json). The test split remains locked. Thin linear and forked forms were the weakest validation group; these figures do not establish final model performance.

## Reproduce the workflow

Python 3.10 or newer is required. The [project plan](docs/project-plan.md) describes acquisition and curation, and the [annotation workflow](docs/annotation-workflow.md) covers mask review. After the reviewed masks are available locally, the baseline can be trained with:

```sh
python -m pip install -e ".[baseline]"
python scripts/train_baseline.py --model unet-resnet34
```

The planned comparison also includes SegFormer-B0. [The training guide](docs/hellbender-training.md) covers the larger GPU run, and [the dataset release guide](docs/dataset-release.md) covers checksums and packaging. `pytest` runs the repository checks. The [archived README](docs/legacy-readme.md) retains the longer operational notes.

## Use and citation

This is a personal, noncommercial research project. [iNaturalist's terms](https://www.inaturalist.org/pages/terms) prohibit using its data for commercial AI and machine-learning training. Each photograph retains its own license and attribution; the repository does not relicense the images. Confirm the current source terms and individual photo licenses before reuse. Cite this repository with the commit used and cite or attribute each source image according to its license.
