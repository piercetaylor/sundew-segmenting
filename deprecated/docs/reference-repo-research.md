# Reference-repository research and a scoped plan for Sundew Segmentation

Research date: 2026-09-16
Reference reviewed: [`praxxus11/SegmentationService` at `fe41872`](https://github.com/praxxus11/SegmentationService/tree/fe41872212fb088cd6ea9dd9541235114ac970f7)

## Recommendation

Build a small, complete **sundew phenotyping and segmentation system**, rather than a broad model zoo. The first release should segment sundew tissue from field or greenhouse RGB images, return an overlay and plant-cover measurements, and document a credible agricultural experiment. Compare two deliberately chosen models:

1. **U-Net with an ImageNet-pretrained ResNet34 encoder** as the reproducible CNN baseline. This single model demonstrates both ResNet transfer learning and U-Net; `segmentation_models_pytorch` exposes this combination directly, including pretrained encoders and a one-channel mask head ([official model API](https://segmentation-models-pytorch.readthedocs.io/en/latest/models.html#unet)).
2. **SegFormer-B0** as the compact transformer challenger. SegFormer combines a hierarchical transformer encoder with a lightweight MLP decoder and accepts variable input sizes subject to its patch-size constraints ([official Transformers documentation](https://huggingface.co/docs/transformers/model_doc/segformer)).

Use **SAM only to propose annotations**, with every mask reviewed by a human. Meta's official implementation supports point/box-prompted masks and automatic mask generation, including predicted-IoU and stability filters ([official repository](https://github.com/facebookresearch/segment-anything), [automatic-mask generator](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py)). Add instance segmentation or species classification only after the binary segmentation release works and the data can support the harder question.

This scope demonstrates Python, PyTorch, optimization, regularization, CNNs, vision transformers, dataset management, agricultural experimental design, basic remote-sensing concepts, Docker, and cloud deployment. It remains small enough to finish and explain in an interview.

## What the reference repository actually does

### Architecture and request flow

The source implements a two-container application plus Redis:

```text
browser/client
    -> Flask gateway: validate/decode image, correct EXIF orientation, save JPEG
    -> Redis/RQ queue: return job ID and queue position
    -> inference worker
         -> PointRend instance segmentation (one foreground class)
         -> crop each detected pitcher with its mask
         -> Swin V2 Tiny species classification, top five probabilities
         -> base64 PNG mask + confidence per instance
         -> SQLite timing/prediction metadata
    <- client polls job status for the result
```

The gateway exposes `/upload`, `/status/<job_id>`, and `/healthcheck`; it saves a normalized RGB JPEG and enqueues the work rather than holding the HTTP request open ([gateway source](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/gateway/server/app.py)). RQ records queued, started, finished, and failure states and reports the queue position ([queue source](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/gateway/server/taskqueue.py)). Docker Compose joins the Flask gateway, one inference worker, and Redis on a shared network and mounts image, log, model, and database directories ([Compose file](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/docker-compose.yml)).

### Models and inference

The segmentation stage builds Detectron2's **PointRend R-CNN X-101-32x8d-FPN** configuration, changes both heads to one class, loads an external checkpoint, and forces CPU inference ([segmentation setup](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/inference/segmentation/setup.py)). It resizes the image so its shorter side is 700 pixels, runs instance segmentation, and retains masks whose detector score is at least `0.95` ([segmentation inference](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/inference/segmentation/inference.py)). Detectron2 is an appropriate framework for this problem class and officially includes PointRend among its detection and segmentation capabilities ([Detectron2 repository](https://github.com/facebookresearch/detectron2)).

The classification stage constructs **`torchvision.models.swin_v2_t()`**, replaces its head with the required number of species, and loads a second checkpoint ([classifier setup](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/inference/classification/setup.py)). Torchvision identifies Swin V2 as a shifted-window transformer family and provides tiny, small, and base builders ([official model documentation](https://docs.pytorch.org/vision/stable/models/swin_transformer.html)). Each detected mask is applied to the original image, the nonzero region is tightly cropped and resized to `256 x 256`, and the classifier returns softmax-ranked species probabilities ([classification inference](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/inference/classification/inference.py), [crop utility](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/inference/classification/utils.py)).

Models are created once when the worker imports the setup modules, not once per request. The orchestration layer then records segmentation duration, classification duration per object, mask count, top-three predictions, and confidences in SQLite ([orchestration](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/inference/inference.py), [storage schema](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/inference/storage/db.py)).

### Dataset and training workflow

The current repository contains **serving code only**. It has no raw-data schema, annotation scripts, training loop, split logic, evaluation report, model card, dataset card, tests, or checkpoint provenance. The architecture image describes photos flowing through labeling on UCSD resources and training on Vast.ai, while model artifacts are mounted into the inference container; these steps are not reproducible from this repository ([architecture image](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/metaimages/Design.png), [inference Dockerfile](https://github.com/praxxus11/SegmentationService/blob/fe41872212fb088cd6ea9dd9541235114ac970f7/inference/Dockerfile)). The associated project page points to PointRend, Swin V2, and SAM papers for segmentation, classification, and labeling respectively, but does not provide the training implementation ([project page](https://tomscarnivores.com/tech/nepenthes-species-identifier-v2/)).

## What is worth reusing

- **Separate localization from identification.** A segmentation model first removes distracting background; a classifier then reasons about each object. This is a useful second release if the sundew dataset eventually includes enough examples per species.
- **Return spatial outputs, not only a class label.** Masks and overlays make errors inspectable and support phenotypes such as projected plant area, vegetation cover, rosette count, and size distribution.
- **Load weights once and isolate inference.** A long-lived worker amortizes model initialization and keeps the web layer light.
- **Expose job status when inference is slow.** Queue position and explicit processing states are good user experience under constrained compute.
- **Containerize the service and persist operational metadata.** The Compose layout provides a clear boundary between the public API, queue, worker, and stored artifacts.
- **Correct camera orientation before inference.** Applying EXIF orientation avoids a common mismatch between what the user sees and what the model receives.

## Gaps to address in the sundew project

The reference is a useful serving example, but copying it directly would leave the most relevant portfolio evidence invisible.

| Gap | Why it matters | Sundew project response |
|---|---|---|
| No reproducible training code or experiment configuration | An interviewer cannot assess losses, regularization, optimizer choice, scheduling, or experimental reasoning | Commit configuration-driven training and evaluation, seed handling, checkpoint metadata, and a short experiment report |
| No dataset specification or provenance | Labels, licensing, leakage risk, class balance, and collection conditions are unknown | Publish a dataset card and manifest with source/license, plant/site/session grouping, sensor and capture metadata, and split assignments |
| No reported test metrics or error analysis | Confidence scores alone do not establish accuracy or generalization | Freeze a held-out test set; report Dice, IoU, precision/recall, uncertainty intervals, subgroup results, and an error gallery |
| Hard-coded inference choices | The `0.95` detector threshold and top-five behavior are not tied to validation results; the `topk` parameter is ignored in favor of a literal `5` | Tune thresholds on validation data, store them in the model artifact/config, and test API behavior |
| Reproducibility risk in dependencies | The image clones Detectron2's moving default branch instead of pinning a commit, and model files live outside the repository | Pin Python and system dependencies, record checkpoint hashes, and build a locked CPU image |
| Limited production safeguards | Upload limits, authentication/rate limiting, cleanup/retention, structured observability, and privacy handling are absent | For a public demo, enforce image type/size/pixel limits, avoid retaining uploads by default, and log aggregate latency without image contents |
| CPU-only, single-worker deployment | It is simple but slow and cannot batch requests | Benchmark first; keep the initial synchronous service small, then add a queue only if measured latency or concurrency requires it |
| Distorting masked crops | Tight crops are resized directly to a square, which can alter morphology | Pad to a square while preserving aspect ratio; include ablation results for masked versus unmasked crops if classification is added |

## Proposed scientific question and outputs

Use one sentence to keep the project honest:

> Given an RGB image collected under varied field or greenhouse conditions, how accurately can a compact model delineate sundew plant tissue from the surrounding substrate and vegetation?

Primary output: a binary mask for **sundew tissue versus background**. Secondary outputs derived without another learned model: projected pixel area, fractional image cover, connected-component count, and an overlay. If a scale marker or calibrated ground sample distance (GSD) is present, convert pixel area to physical area and state the calibration assumptions.

Do not begin with species classification. Classification requires enough independent plants per species and can easily learn pot, site, photographer, or background cues. Promote it to phase two only after the dataset contains at least several independent plants and collection sessions per species and the split is grouped by plant/site.

## Dataset and field protocol

### Sampling unit and split design

Define the independent experimental unit before collecting images. Recommended hierarchy:

```text
site -> plot or tray -> plant/rosette -> capture session -> image/tile
```

Assign **site/plot/plant groups**, not individual images or random tiles, to train/validation/test. All near-duplicates, bursts, augmentations, crops, and tiles from one parent image stay in the same split. This prevents a model from seeing the same plant or substrate texture during training and testing. Freeze the test groups before serious tuning.

If images come from a field trial, record block, treatment, replicate, sampling date, and plot ID. Report results by site, species, growth stage, lighting, camera, and plant-size band. This turns an image demo into a defensible agricultural experiment.

### Collection targets

An achievable first dataset is **300-600 independently useful images**, collected across at least three dates and varied conditions, with a smaller high-quality test set from plants or sites never used in training. More images from the same burst do not replace more independent plants and sessions.

Capture intentionally difficult cases: moss and green substrate, water glare, red and green phenotypes, overlapping leaves, flowering stems, partial plants at borders, tiny seedlings, blur, shadow, and non-sundew carnivorous plants as hard negatives. Keep empty-background images so false-positive behavior is measurable.

### Metadata contract

Store one row per source image in `metadata.parquet` or `metadata.jsonl`:

- `image_id`, `mask_id`, `split`, `parent_image_id`
- `site_id`, `plot_id`, `plant_id`, `capture_session_id`, `date_time`
- `species_label` and `label_certainty` when known
- `sensor`, `lens`, `image_width`, `image_height`, `view_angle`
- optional `altitude_agl_m`, `estimated_gsd_mm_px`, `orthomosaic_id`, and tile coordinates
- `annotator_id`, `reviewer_id`, `annotation_tool`, `annotation_version`
- `source`, `license`, `consent_or_permission`, and location-privacy status

Hugging Face dataset repositories can infer splits from directory structure and automatically load additional fields from CSV, JSONL, or Parquet metadata ([official image-dataset guide](https://huggingface.co/docs/hub/en/datasets-image)). Its semantic-segmentation guide defines the task as per-pixel classification ([official Datasets guide](https://huggingface.co/docs/datasets/v4.7.0/en/semantic_segmentation)). Use a private dataset while permissions, rare-plant location privacy, or image licenses are unresolved; publish a de-identified subset and dataset card when possible.

### Annotation and quality control

1. Write a one-page label policy covering leaf hairs, flower stalks, dead tissue, water reflections, occlusion, image borders, and ambiguous seedlings.
2. Use SAM point/box prompts for a draft mask, then correct boundaries manually. SAM's automatic generator filters masks using predicted IoU, stability, and duplicate suppression, which makes it a useful annotation accelerator rather than ground truth ([official implementation](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py)).
3. Double-label 10% of images. Report inter-annotator Dice/IoU and adjudicate large disagreements.
4. Run automated checks for mask dimensions, allowed values, empty masks, disconnected specks, and image/mask pairing.
5. Version the manifest and annotation policy with the code. Store large image assets in a dataset repository or object storage rather than ordinary Git history.

## Modeling and experiment plan

### Data pipeline

Use paired, geometry-aware transforms so every crop, flip, rotation, or resize is applied identically to the image and its mask. Torchvision v2 accepts images and segmentation masks together and documents geometric and photometric augmentation primitives ([official transforms documentation](https://docs.pytorch.org/vision/stable/transforms.html)). Apply color changes to the image only. Start conservatively:

- random resized crop or fixed tiles, with a minimum positive-mask fraction for some batches;
- horizontal/vertical flips and rotations appropriate for top-down plants;
- mild brightness, contrast, saturation, blur, and compression variation;
- no augmentation on validation or test beyond deterministic resize/pad/normalize.

Inspect augmented image-mask pairs in tests. Record channel mean/std or use the pretrained encoder's expected normalization.

### Baseline and challenger

| Model | Purpose | What it demonstrates | Stop rule |
|---|---|---|---|
| U-Net + ResNet34 encoder | Main baseline and likely deployable model | CNNs, residual transfer learning, encoder-decoder segmentation | Ship it if its test IoU is close to the challenger and it is materially smaller/faster |
| SegFormer-B0 | Transformer comparison | Vision transformers and modern semantic segmentation | Keep it only if subgroup/generalization results justify added complexity |
| DeepLabV3-ResNet50 (optional) | Standard-library sanity baseline | Atrous segmentation and torchvision model APIs | Run only if compute/time remains; torchvision provides pretrained DeepLabV3 builders ([official models page](https://docs.pytorch.org/vision/main/models.html#semantic-segmentation)) |

PointRend, Mask R-CNN, or YOLO segmentation should wait until the product requires distinct overlapping rosettes. Semantic segmentation is sufficient for plant-cover and projected-area phenotypes; instance segmentation adds annotation and evaluation complexity without improving the first scientific answer.

### Loss, optimization, and regularization

For the binary mask, train from logits with:

```text
loss = 0.5 * BCEWithLogitsLoss(pos_weight=estimated_from_training_split)
     + 0.5 * soft_Dice_loss
```

`BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a numerically stable operation and supports positive-class weighting ([PyTorch loss documentation](https://docs.pytorch.org/cppdocs/api/nn/loss.html#bcewithlogitsloss)). Estimate class weights from the training split only and compare the combined loss against unweighted BCE in a small ablation.

Use AdamW, whose weight decay is decoupled from momentum and variance accumulation ([PyTorch optimizer documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)). A compact search is enough: learning rates `{1e-4, 3e-4}`, weight decay `{1e-4, 1e-2}`, batch size as memory allows, gradient clipping, mixed precision on GPU, and early stopping on validation IoU. Use one documented learning-rate schedule, such as cosine decay with warmup, and plot learning rate, training loss, validation loss, Dice, and IoU per epoch.

### Evaluation

Use validation data to select checkpoints and the probability threshold. Evaluate the frozen test set once the choices are fixed.

Required metrics:

- foreground Dice and IoU/Jaccard;
- pixel precision and recall;
- image-level false-positive rate on empty-background images;
- per-image inference latency, peak memory, parameter count, and model artifact size;
- metrics stratified by site/session, species, lighting, scale, and camera when sample counts permit;
- bootstrap confidence intervals over independent plant/site groups, not individual pixels.

TorchMetrics defines Jaccard as intersection divided by union and supports binary masks and explicit thresholds ([Jaccard documentation](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html)); its segmentation Dice metric can include or exclude background and aggregate sample-wise ([Dice documentation](https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html)). Include a qualitative panel of best, median, and worst predictions, plus categories for misses, false positives, boundary errors, and domain-shift failures.

## Service and cloud plan

Start with one Dockerized FastAPI or Flask service that loads a compact model at startup and serves synchronous `/predict` and `/health` endpoints. Return the binary PNG mask, an overlay, threshold, model version, and derived measurements. Add `/metadata` or include response fields for preprocessing version and latency. A simple service makes local reproduction and a live demo easy.

Add Redis/RQ only if a benchmark shows that inference regularly exceeds an acceptable HTTP request time or concurrent demo traffic needs backpressure. If that happens, reuse the reference repository's gateway/job/worker boundary while adding upload limits, job expiry, cleanup, and structured logs.

For the public portfolio demo, choose one deployment path:

- **Fastest:** a Gradio or Docker Hugging Face Space linked to the model and dataset cards. Spaces are intended for hosted ML demos and support Gradio and Docker SDKs ([official overview](https://huggingface.co/docs/hub/spaces-overview), [Docker Spaces](https://huggingface.co/docs/hub/main/en/spaces-sdks-docker)).
- **Cloud-platform evidence:** build the same CPU container in CI, push it to Amazon ECR, and deploy it to AWS App Runner. App Runner accepts a ready-to-deploy public or private ECR image and manages the service infrastructure ([AWS documentation](https://docs.aws.amazon.com/apprunner/latest/dg/service-source-image.html)). Keep infrastructure configuration in the repository and document cost controls and teardown.

One polished live deployment is stronger than several half-configured cloud targets.

## UAV and remote-sensing extension

UAV imagery is an optional field extension, not a prerequisite for the first release. Individual sundews may occupy only a few pixels at ordinary flight altitudes, so calculate whether expected GSD can resolve leaves before collecting data. Record altitude, sensor dimensions/focal length, image overlap, view angle, GSD, ground-control method, orthomosaic software/version, and tile-to-orthomosaic coordinates. USGS examples show why these quantities matter: UAS products are commonly orthorectified mosaics with an explicit GSD, and positional accuracy is not automatically equal to pixel GSD ([USGS UAS example](https://eros.usgs.gov/doi-remote-sensing-activities/2018/blm/mapping-cultural-resources-uas-imagery), [USGS calibration guidance](https://pubs.usgs.gov/of/2023/1033/ofr20231033.pdf)).

If actual flights are performed under Part 107, the FAA requires the person manipulating the controls to hold a Remote Pilot Certificate or be directly supervised by someone who does ([FAA Part 107 overview](https://www.faa.gov/newsroom/small-unmanned-aircraft-systems-uas-regulations-part-107)). A strong portfolio can still demonstrate remote-sensing literacy using previously collected, permitted imagery and a documented tiling/GSD/orthomosaic workflow.

## Suggested repository layout

```text
.
|-- README.md                       # problem, demo, headline results, limitations
|-- pyproject.toml                  # pinned package metadata and tool configuration
|-- configs/
|   |-- unet_resnet34.yaml
|   `-- segformer_b0.yaml
|-- data/
|   |-- README.md                   # download/access instructions; no private raw data
|   |-- annotation-policy.md
|   `-- sample/                     # a few licensed image/mask pairs for tests
|-- src/sundew_segmentation/
|   |-- data.py
|   |-- transforms.py
|   |-- models.py
|   |-- losses.py
|   |-- metrics.py
|   |-- train.py
|   |-- evaluate.py
|   `-- predict.py
|-- app/                            # API/demo and response schemas
|-- tests/                          # data pairing, transforms, metrics, API smoke test
|-- reports/
|   |-- dataset-card.md
|   |-- model-card.md
|   `-- evaluation.md
|-- Dockerfile
`-- .github/workflows/ci.yml
```

Keep notebooks for exploration, but make the reported run reproducible from command-line modules and configuration files.

## Milestones and acceptance criteria

### Milestone 1: problem and data contract (week 1)

- Write the label policy, metadata schema, licensing/privacy policy, and grouped split rules.
- Curate 30 representative images and manually label a seed set.
- Deliver a visual dataset audit showing positive area, image size, site/session counts, and difficult conditions.

**Done when:** another person can label five images consistently and explain which unit is held out.

### Milestone 2: data pipeline and baseline (weeks 2-3)

- Build image/mask loading, paired transforms, quality checks, and deterministic grouped splits.
- Train U-Net/ResNet34 with logged loss, optimizer, weight decay, scheduler, and seed.
- Produce validation curves and an initial error gallery.

**Done when:** one command reproduces training on a small sample, and a second command evaluates a checkpoint without notebook state.

### Milestone 3: controlled model comparison (week 4)

- Train SegFormer-B0 under the same splits and augmentation policy.
- Run one useful ablation: combined loss versus BCE, pretrained versus frozen/unfrozen encoder, or mask-positive sampling versus uniform sampling.
- Select a model using accuracy, subgroup robustness, speed, memory, and artifact size.

**Done when:** `reports/evaluation.md` explains the selection with a fixed test protocol and limitations.

### Milestone 4: demo and packaging (week 5)

- Add upload, overlay, measurement output, model version, and example images.
- Build a CPU Docker image, test the health/prediction endpoints, and scan dependencies.
- Publish model and dataset cards; keep sensitive locations and unlicensed images private.

**Done when:** a reviewer can clone the repository, run one documented command, and reproduce an example prediction.

### Milestone 5: cloud and field extension (week 6, optional)

- Deploy the same container to one cloud target and record latency/cost.
- Add one orthomosaic or calibrated plot image, tile it with overlap, stitch probabilities, and compare against a held-out field annotation.
- Document GSD, capture constraints, and whether the scale is sufficient to resolve the target.

**Done when:** the live demo is linked from GitHub and the remote-sensing example states its resolution and spatial assumptions.

## How the finished project demonstrates the target qualifications

| Qualification | Visible evidence in the repository |
|---|---|
| Python and PyTorch | Packaged training/evaluation/inference code, typed API schemas, and tests |
| Losses, regularization, optimization, scheduling | BCE-plus-Dice implementation, AdamW/weight-decay ablation, scheduler and learning-rate plots |
| Agricultural research and field trials | Explicit experimental unit, site/plot/block metadata, grouped splits, repeats across dates, subgroup analysis |
| Remote sensing | GSD and sensor metadata, orthomosaic tiling/stitching example, scale-aware limitations |
| ResNet and U-Net | Main CNN baseline with a pretrained ResNet encoder |
| SegFormer and vision transformers | Controlled transformer challenger on identical data splits |
| SAM | Human-reviewed annotation-assistance workflow and annotation-QA measurements |
| Large dataset management | Hugging Face dataset structure, dataset card, Parquet/JSONL metadata, versioned manifest |
| Cloud | Reproducible CPU container, CI build, one public cloud deployment, latency/cost note |
| Analytical problem solving | Baseline/challenger comparison, ablation, confidence intervals, subgroup results, error taxonomy |
| Collaboration across expertise | Annotation guide, model/dataset cards, issue templates, and a contribution workflow a botanist or field researcher can follow |

The central portfolio story should be: **a well-designed dataset and experiment produced a trustworthy, deployable phenotype measurement**. The architecture choices support that story; they are not the story by themselves.
