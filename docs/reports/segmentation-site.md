# Plant outline on the site: U-Net result

**Decision: the site launches without a plant outline.** The U-Net/ResNet-34
trained with the field recipe failed the pre-registered quality bar
([segmentation-site-prereg.md](segmentation-site-prereg.md)): five-seed mean
field-eval IoU **0.5055** against a bar of 0.593. SegFormer-B0 (0.6129) is not
used instead, because NVIDIA's licence limits it to research or evaluation.
Species identification is unaffected: it reads the whole photo and never used
the outline.

Scored with `scripts/score_field_unet.py` (same preprocessing as
`score_field_compare.py`; it reproduces SegFormer seed 101's 0.6315); raw
numbers in [segmentation-site-scores.json](segmentation-site-scores.json).
Training: `scripts/hellbender_field_unet.slurm`, job 18104757, 4-6 minutes per
seed on an A100 (early stopping after 20-35 epochs).

## Results

| Seed | U-Net curated val | U-Net field-eval @0.5 | SegFormer field-eval @0.5 | U-Net - SegFormer |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 0.6749 | 0.4652 | 0.6086 | -0.1434 |
| 101 | 0.6598 | 0.5086 | 0.6315 | -0.1229 |
| 202 | 0.6565 | 0.5148 | 0.6166 | -0.1017 |
| 303 | 0.6540 | 0.4870 | 0.6055 | -0.1184 |
| 404 | 0.6557 | 0.5519 | 0.6022 | -0.0503 |
| **mean** | | **0.5055** | 0.6129 | **-0.1073** |

The U-Net is worse than SegFormer on every seed. On curated validation it is
close (0.65-0.67 against SegFormer's 0.69-0.70); the gap opens on the messy
field photos, which are what users will upload. Its field-eval score is about
where SegFormer was before the 30 field masks were added (0.4994, the original
`combined` recipe).

Area error, the project's main output, is also about twice SegFormer's:

| Threshold | U-Net IoU | Median abs. area error | Mean signed area error |
| ---: | ---: | ---: | ---: |
| 0.4 (lowest area error) | 0.5134 | 28.5% | +17.4% |
| 0.5 | 0.5055 | 29.1% | +7.6% |
| 0.6 | 0.4964 | 31.9% | +0.2% |
| 0.7 | 0.4844 | 36.6% | -7.3% |
| 0.8 | 0.4636 | 36.8% | -17.4% |

SegFormer's median area error at its chosen threshold (0.6) is 16.1%. Rules 2
and 3 of the pre-registration would have picked seed 17 at threshold 0.4
(field-eval IoU 0.4734), but the quality bar failed first, so nothing was
exported.

## What was not tried

The pre-registration allowed exactly one recipe, SegFormer's, with only the
model swapped. That recipe (learning rate 3e-4, 1024 px, BCE + Tversky) was
tuned for SegFormer, so a U-Net recipe of its own might do better. Field-eval
has now been used to reject this U-Net; any further attempt should be judged on
new field masks, not the same 17 images.

## Options if an outline is still wanted

1. **Ask NVIDIA** whether a free, noncommercial public identification site
   counts as permitted use of SegFormer-B0 (or get a licence). If yes, the
   tuned SegFormer-B0 (15 MB, field IoU 0.613) is ready: export, parity check
   and model card as in the open-set-and-segmentation handoff.
2. **Tune a licence-clean model** (U-Net or another ImageNet-pretrained
   encoder without an NVIDIA clause) with its own recipe, then judge it on a
   fresh batch of field masks against a bar written down first.
3. **More field masks.** Thirty field masks moved SegFormer from 0.552 to
   0.613; the U-Net may gain more from them than from tuning.

## Licence notes

- The U-Net starts from smp-hub's `resnet34.imagenet`, torchvision's ImageNet
  ResNet-34. The cached copy carries no licence file; torchvision's code is
  BSD-3-Clause, and torchvision leaves it to users to check the pretrained
  weights for their use. No NVIDIA-style clause is known.
- SegFormer-B0: NVIDIA Source Code License for SegFormer, section 3.3: "The
  Work and any derivative works thereof only may be used or intended for use
  non-commercially ... 'non-commercially' means for research or evaluation
  purposes only." Not legal advice.
