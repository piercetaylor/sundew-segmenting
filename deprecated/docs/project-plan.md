# SundewVision project plan

## Project in one sentence

Build a reproducible PyTorch system that segments sundew plants in standardized overhead RGB images, converts the masks into useful plant traits, compares several modern model families, and serves the best model through a small containerized web application.

The project should answer a real research question rather than stop at a demo:

> How accurately can RGB imagery estimate sundew projected plant area and percent cover across plants, dates, and imaging conditions?

This is deliberately a **binary semantic segmentation** problem for the first release: sundew tissue versus background. Semantic masks are cheaper to label than per-plant instance masks and directly support projected-area and cover measurements. Instance segmentation and species classification remain later extensions.

## What to borrow from the reference project

The [SegmentationService repository](https://github.com/praxxus11/SegmentationService) has a sound product idea: find plant structures first, then run downstream analysis on the segmented region. Its implementation uses PointRend instance segmentation, a Swin V2 classifier, Flask, Redis/RQ, SQLite telemetry, and Docker Compose. Its public application can locate and classify multiple pitchers in one image.

Carry these ideas forward:

- Separate image analysis from the web interface.
- Load models once when the inference process starts.
- Return both a visual mask and structured results.
- Package inference in a container and provide a health check.
- Record latency, model version, and useful prediction metadata.

Change these parts for a small portfolio project:

- Include the missing data preparation, training, evaluation, and model-card workflow in the repository.
- Begin with synchronous inference in one FastAPI container. Redis and a worker queue solve demand that this project will not initially have.
- Use configuration rather than environment variables for scientific experiment choices.
- Test data splits, mask alignment, metrics, and the public inference interface.
- Compare models on a locked test set and publish failure cases instead of relying on anecdotal examples.

## Scope

### Minimum viable portfolio release

1. A documented image-collection and annotation protocol.
2. A versioned Hugging Face dataset with RGB images, binary masks, capture metadata, and a dataset card.
3. A color-threshold baseline and a trainable U-Net baseline.
4. A controlled comparison between U-Net with a ResNet-34 encoder and SegFormer-B0.
5. Evaluation by held-out plant identity using Dice, Intersection over Union, boundary F1, and projected-area error.
6. A phenotyping report containing projected plant area, percent cover, and an explicitly labeled RGB color proxy.
7. A FastAPI endpoint and simple upload page that return a mask overlay, measurements, model version, and inference time.
8. A CPU Docker image, automated checks, and deployment to Google Cloud Run.
9. A README with the research question, dataset design, experiment table, failure analysis, architecture diagram, and a short demo.

### Strong follow-up release

- Add DeepLabV3-ResNet50 as an optional third model under the same training and evaluation protocol.
- Use Segment Anything to accelerate annotation, while retaining human-reviewed masks as ground truth.
- Add tiled inference for large plot images or orthomosaics.
- Publish the trained model and model card on Hugging Face Hub.
- Add an optional Detectron2 PointRend or Mask R-CNN instance-segmentation experiment only if overlapping rosettes make connected components unreliable.

### Explicitly out of scope for the first release

- Species identification, pest diagnosis, or biological stress claims.
- A Redis queue, database server, Kubernetes, or multiple deployable microservices.
- Multispectral indices without a calibrated multispectral sensor.
- Claims of UAV experience or FAA certification that the work did not actually involve.

## Data and study design

### Collection target

Aim for a first useful dataset of 150–300 labeled images from at least 12–20 distinct plants, photographed on four or more dates. Stretch toward 300–600 images by adding independent plants and capture sessions rather than bursts of near-duplicates. Capture both easy and difficult cases: moss, wet substrate, neighboring pots, overlapping leaves, flower stalks, glare, shadows, different sundew coloration, and empty or non-sundew hard negatives.

Use a repeatable nadir capture setup:

- fixed or recorded camera height and approximate viewing angle;
- a scale marker in the plant plane for pixel-to-area conversion;
- a neutral gray or color reference when practical;
- unchanged original files with EXIF retained;
- one row of metadata for every image.

Recommended metadata fields are `image_id`, `plant_id`, `species`, `capture_date`, `block`, `treatment`, `camera`, `height_cm`, `scale_mm_per_pixel`, `lighting`, `annotator`, `mask_version`, and `split`.

If a small controlled study is feasible, use a non-destructive randomized block or repeated-measures design. Treat each plant as the experimental unit, block by tray or shelf position, randomize treatment placement, and image the same plants over time. Record environmental conditions and management events. The segmentation model is evaluated by plant identity; treatment comparisons are a separate analysis and must not treat repeated images of one plant as independent replicates.

### Split strategy

Create train, validation, and test assignments before model tuning. Group by `plant_id`, and group by plot or collection event where relevant. Images of the same plant must never cross splits. Keep the test set locked until the modeling choices are fixed. This is one of the most valuable agricultural-ML decisions in the project because a random image split would reward memorizing the same plant and background.

### Annotation and quality control

Use CVAT, Label Studio, or another polygon/mask tool. SAM-generated masks may be used as proposals, but a person should inspect every final mask. Define whether flower stalks, dead leaves, moss-covered leaves, and trapped insects count as sundew tissue. Review at least 10% of masks twice and report reviewer IoU or the number of corrected masks.

Store one lossless PNG mask per image, with `0` for background and `1` or `255` for sundew. Package image and mask paths plus metadata as a Hugging Face dataset. The official Datasets documentation supports paired image fields and metadata files, which fits this project without a custom dataset server.

### Remote-sensing content

The core experiment is close-range RGB remote sensing. Show that the imaging geometry, spatial resolution, illumination, and calibration affect the measurement. Convert pixel area to square millimeters only when a scale marker or validated ground sampling distance is present.

For a UAV extension, use only imagery in which sundews occupy enough pixels to resolve leaf edges. Preserve altitude, camera parameters, ground sampling distance, coordinate reference system, and plot boundaries; tile the orthomosaic with overlap; split by site or flight rather than tile. A short feasibility analysis that explains why normal flight altitude may be too coarse for individual sundews is more credible than an artificial drone claim.

## Modeling plan

### Baselines and comparisons

Run models in this order so each addition answers a question:

| Experiment | Purpose | Suggested implementation |
|---|---|---|
| RGB threshold | Establish a non-deep-learning floor and expose color/lighting failure modes | HSV or excess-green/red threshold plus morphology |
| U-Net | Establish a compact convolutional segmentation baseline | U-Net with a pretrained ResNet-18 or ResNet-34 encoder |
| SegFormer-B0 | Test a lightweight transformer architecture | Hugging Face Transformers |
| DeepLabV3, optional | Add a standard-library multi-scale sanity check if time and compute remain | Torchvision DeepLabV3-ResNet50 |
| SAM-assisted labels | Measure annotation-time savings; do not score proposal masks as ground truth | Human-corrected SAM proposals |

Train every learned model with the same grouped splits, image resolution, augmentation policy, and evaluation code. The required comparison is U-Net/ResNet34 versus SegFormer-B0; treat DeepLabV3 as optional. Do not add YOLO or instance segmentation unless the error analysis shows that individual plant separation is necessary.

### Machine-learning concepts to make visible

- Start with binary cross-entropy plus Dice loss; explain how pixel imbalance motivates the Dice term.
- Use AdamW, record weight decay, and compare at least one learning-rate schedule such as cosine decay or OneCycle.
- Apply modest geometric and photometric augmentations that preserve mask alignment.
- Use pretrained encoders, early stopping on validation Dice, deterministic seeds, and mixed precision when a GPU is available.
- Run one small ablation table: loss choice, augmentation on/off, or pretrained versus random initialization.
- Save the configuration, code revision, best checkpoint, metrics, and plots for each run.

### Evaluation

Report the following on the locked test set:

- Dice and Intersection over Union, with bootstrap confidence intervals by plant;
- precision and recall for the sundew class;
- boundary F1 for thin leaves and tentacle-rich edges;
- projected-area mean absolute error against manual masks;
- model size and CPU latency at the deployed resolution;
- performance slices by species, lighting, plant size, and background type when sample counts permit.

Always include a contact sheet of best, typical, and worst predictions. Document false positives on moss and substrate, missed thin leaves, glare, occlusion, and domain shifts. The strongest result is a defensible evaluation, even if the smallest model wins.

## Code design

Design a few deep modules with small interfaces:

| Module | Interface | What its implementation hides |
|---|---|---|
| Dataset | `load_dataset(config) -> DatasetBundle` | file layout, Hugging Face loading, grouped splits, paired augmentation, metadata validation |
| Segmenter | `predict(image) -> SegmentationResult` | preprocessing, device placement, model-specific output shapes, thresholds, resizing |
| Trainer | `train(config) -> RunSummary` | losses, optimizer, scheduler, checkpoints, logging, early stopping |
| Phenotyper | `measure(image, mask, calibration) -> TraitReport` | connected components, area conversion, cover, color summaries, quality flags |
| Analyzer | `analyze(image, metadata=None) -> AnalysisResult` | segmentation, postprocessing, phenotyping, overlay generation, timing, provenance |

The `Segmenter` seam is justified because U-Net, DeepLabV3, and SegFormer are real adapters with different preprocessing and outputs. The Dataset seam is justified once both local files and Hugging Face are supported. Keep cloud storage and queue interfaces out until a second adapter is genuinely needed.

Suggested repository layout:

```text
Sundew_Segmentation/
├── README.md
├── pyproject.toml
├── configs/
│   ├── unet.yaml
│   ├── deeplabv3.yaml
│   └── segformer.yaml
├── src/sundew_seg/
│   ├── data.py
│   ├── models/
│   ├── training.py
│   ├── metrics.py
│   ├── phenotyping.py
│   ├── inference.py
│   └── api.py
├── tests/
├── scripts/
├── notebooks/
├── docs/
│   ├── project-plan.md
│   ├── data-protocol.md
│   └── experiment-report.md
├── Dockerfile
└── .github/workflows/ci.yml
```

Keep notebooks for exploration and figures. Put reusable data, training, and inference logic in `src/` so the same code runs in tests, command-line experiments, and the deployed application.

## Deployment plan

Use one FastAPI container with these public routes:

- `GET /healthz` returns readiness and model version.
- `POST /v1/analyze` accepts one RGB image and returns measurements, latency, confidence/quality flags, and either a mask or overlay URL/data.
- `GET /` serves a small upload interface with an example image and a clear limitations statement.

Enforce file-type, file-size, and decoded-pixel limits. Do not retain uploaded images by default, and keep image contents out of logs.

Build and test the container in GitHub Actions, publish it to Google Artifact Registry, and deploy it to Cloud Run. Cloud Run accepts container images and can scale an idle service to zero, which is appropriate for an occasional portfolio demo, with cold-start latency documented. Do not deploy until local container inference and an end-to-end API test pass.

## Verification strategy

Prioritize tests that catch scientific or interface failures:

- image and mask remain aligned after every transform;
- grouped splits contain no shared `plant_id` or collection group;
- corrupt masks, unknown labels, missing metadata, and zero-area scale markers fail clearly;
- metric functions match hand-calculated examples;
- each model adapter returns the same `SegmentationResult` shape and coordinate system;
- area conversion is correct for a synthetic shape with a known scale;
- the API accepts a valid image, rejects invalid uploads, and reports the loaded model version;
- the built CPU container completes one representative inference.

## Six weekend-sized milestones

1. **Protocol and pilot:** define the mask policy, photograph a diverse pilot set, annotate 20 images, and test whether projected area is measurable.
2. **Dataset release:** collect and label the first dataset, run validation checks, freeze grouped splits, and write the dataset card.
3. **Baseline:** implement the RGB threshold and U-Net training path; produce the first test report and failure contact sheet.
4. **Modern model:** add SegFormer, run the controlled comparison, and complete one ablation; add DeepLabV3 only if time remains.
5. **Phenotyping and application:** add calibrated area/cover measurements, the upload interface, container, tests, and model card.
6. **Cloud and portfolio polish:** deploy to Cloud Run, capture a demo, finalize the experiment report, and rewrite the README around evidence and limitations.

If time is limited, stop after U-Net plus one modern model. A complete dataset-to-deployment story with honest evaluation is stronger than a long list of partially implemented architectures.

## How the finished repository demonstrates the target qualifications

| Qualification | Evidence in this project |
|---|---|
| Python and PyTorch | Package-quality training, evaluation, inference, and tests |
| Losses, regularization, optimization, schedules | Configured experiments and an ablation table |
| Agricultural research methods | Experimental unit, blocking, repeated measures, metadata, grouped validation |
| Remote sensing | Nadir acquisition protocol, scale/GSD, illumination controls, optional geospatial tiling |
| ResNet, U-Net, DeepLab, SegFormer, SAM, transformers | ResNet encoder, controlled model comparison, and SAM-assisted annotation |
| Large dataset management | Hugging Face dataset, dataset card, versioned metadata, automated validation |
| UAV imaging | Optional flight/orthomosaic extension with resolution feasibility; only claim hands-on experience if actually flown |
| Cloud platform | Tested container and Google Cloud Run deployment |
| Analytical problem solving | Leakage prevention, confidence intervals, error slices, failure analysis, size/latency tradeoffs |
| Collaboration | Clear issues, reproducible commands, contribution guide, dataset/model cards, and reviewable experiment records |

## Portfolio acceptance criteria

The first release is complete when another person can:

1. understand the research question and label policy from the documentation;
2. load the published dataset and reproduce the frozen splits;
3. train at least two models from configuration;
4. reproduce the final metrics and qualitative error figure;
5. run one local command to analyze an image;
6. build the container and exercise the API; and
7. visit a live demo that identifies its model version and limitations.

## Primary references

- [SegmentationService source](https://github.com/praxxus11/SegmentationService)
- [Nepenthes Species Identifier V2 technical overview](https://tomscarnivores.com/tech/nepenthes-species-identifier-v2/)
- [Detectron2 source and model zoo](https://github.com/facebookresearch/detectron2)
- [Torchvision DeepLabV3 documentation](https://docs.pytorch.org/vision/stable/models/deeplabv3.html)
- [Hugging Face SegFormer documentation](https://huggingface.co/docs/transformers/model_doc/segformer)
- [Hugging Face image dataset documentation](https://huggingface.co/docs/datasets/image_dataset)
- [Google Cloud Run container deployment](https://docs.cloud.google.com/run/docs/quickstarts/deploy-container)
- [FAA guidance for certificated remote pilots](https://www.faa.gov/uas/commercial_operators)
