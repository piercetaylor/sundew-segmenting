# U-Net vs SegFormer: five-seed comparison

**In short.** SegFormer-B0 beats U-Net / ResNet34 on 5 of 5 seeds, mean
validation IoU 0.632 vs 0.610 (paired difference +0.0215, significant at
p = 0.05). The gap is smaller than either model's own seed spread, so it only
shows when runs are paired by seed. SegFormer-B0 is the architecture used from
here on.

Both models were trained five times on the frozen v0.3.0 dataset (144 train,
19 validation) on Hellbender A100s, varying only the seed. The 28-image test
split was not touched. A single run could not settle this: the first pair
differed by 0.018 IoU, less than the seed spread of either model.

Reproduce with `sbatch scripts/hellbender_seed_sweep.slurm` (2 models x 5 seeds,
seeds 17/101/202/303/404, ~15 minutes of GPU time in total).

## Best validation IoU per seed

| Seed | U-Net / ResNet34 | SegFormer-B0 | Difference |
| ---: | ---: | ---: | ---: |
| 17 | 0.5876 | 0.6158 | +0.0281 |
| 101 | 0.6336 | 0.6503 | +0.0167 |
| 202 | 0.6190 | 0.6224 | +0.0034 |
| 303 | 0.6154 | 0.6516 | +0.0362 |
| 404 | 0.5965 | 0.6194 | +0.0228 |

| Model | Mean | SD | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| U-Net / ResNet34 | 0.6104 | 0.0183 | 0.5876 | 0.6336 |
| SegFormer-B0 | 0.6319 | 0.0176 | 0.6158 | 0.6516 |

## Result

Paired difference +0.0215 (SD 0.0124), 95% CI [+0.0061, +0.0368], paired
t = 3.88 on 4 df against a critical value of 2.78.

The effect is real but smaller than each model's seed spread (SD ~0.018). Seed
noise moves both models together, so the per-seed difference is much more
stable than either column alone. Any future comparison on this dataset should
be paired across seeds.

Secondary metrics (mean over seeds) agree: SegFormer leads on dice (0.7743 vs
0.7580) and precision (0.7530 vs 0.7246), recall is about tied (0.8003 vs
0.7958), and SegFormer converges in fewer epochs (14.2 vs 16.6).

## Per growth form

Mean validation IoU ± SD over the five seeds.

| Growth form | n | U-Net / ResNet34 | SegFormer-B0 | Winner |
| --- | ---: | ---: | ---: | --- |
| dense_mat | 1 | 0.7076 ± 0.0814 | 0.7545 ± 0.0453 | SegFormer (overlapping) |
| erect_or_branching | 4 | 0.6023 ± 0.0460 | 0.6970 ± 0.0347 | **SegFormer** |
| linear_or_forked | 4 | 0.5454 ± 0.0395 | 0.5277 ± 0.0573 | U-Net (overlapping) |
| rosette | 10 | 0.6365 ± 0.0245 | 0.6458 ± 0.0232 | SegFormer (overlapping) |

`erect_or_branching` is the only form where the models separate cleanly, and it
carries most of SegFormer's margin (+0.095).

`linear_or_forked` is the weakest form for both, and the only one where U-Net
leads, though the spreads overlap and it is four images. Thin filiform leaves
may be hard for a patch-based encoder, but this sweep does not show that.

## Caveats

- **19 validation images.** Each growth-form row rests on 1-10 images;
  `dense_mat` is a single image, an anecdote rather than a measurement. The
  headline is dominated by the 10 rosettes.
- **The training split is skewed** (104 of 144 rosette, 8 dense_mat). The
  inverse-frequency sampler balances training, but validation is unweighted,
  so the headline IoU is mostly a rosette score.
- **Five seeds is a small sample** for a t-test. The conclusion is "SegFormer
  is ahead", not a reliable estimate of by how much.
