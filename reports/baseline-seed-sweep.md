# U-Net vs SegFormer: five-seed comparison

Both baselines were trained five times each on the frozen reviewed dataset
(v0.3.0; 144 train / 19 validation) on Hellbender A100s, varying only the seed.
The test split (28 images) was not touched.

Reproduce with `sbatch scripts/hellbender_seed_sweep.slurm` (2 models x 5 seeds,
seeds 17/101/202/303/404, ~15 minutes of GPU time in total).

A single run could not settle this comparison: the first pair differed by 0.018
IoU, which is smaller than the seed-to-seed spread of either model.

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

SegFormer-B0 wins on **5 of 5 seeds**. Paired difference +0.0215 (SD 0.0124),
95% CI [+0.0061, +0.0368], paired t = 3.88 on 4 df against a critical value of
2.78 — significant at p = 0.05.

The effect is real but small, and it is smaller than each model's own seed
spread (SD ~0.018). That is precisely why the pairing matters: seed noise moves
both models together, so the per-seed difference is far more stable than either
column read alone. Any future comparison on this dataset should be run paired
across seeds rather than as a single run.

Secondary metrics (mean over seeds) agree: SegFormer leads on dice (0.7743 vs
0.7580) and precision (0.7530 vs 0.7246), with recall essentially tied (0.8003
vs 0.7958). SegFormer also converges in fewer epochs (14.2 vs 16.6).

## Per growth form

Mean validation IoU +/- SD over the five seeds.

| Growth form | n | U-Net / ResNet34 | SegFormer-B0 | Winner |
| --- | ---: | ---: | ---: | --- |
| dense_mat | 1 | 0.7076 +/- 0.0814 | 0.7545 +/- 0.0453 | SegFormer (overlapping) |
| erect_or_branching | 4 | 0.6023 +/- 0.0460 | 0.6970 +/- 0.0347 | **SegFormer** |
| linear_or_forked | 4 | 0.5454 +/- 0.0395 | 0.5277 +/- 0.0573 | U-Net (overlapping) |
| rosette | 10 | 0.6365 +/- 0.0245 | 0.6458 +/- 0.0232 | SegFormer (overlapping) |

`erect_or_branching` is the only growth form where the two models separate
cleanly, and it carries most of SegFormer's overall margin (+0.095).

`linear_or_forked` is the weakest class for both models and the one category
where U-Net leads, though the seed spreads overlap and it is only four images.
Thin filiform leaves are a plausible weak point for a patch-based encoder, but
this sweep does not establish that — it would need more annotated
`linear_or_forked` examples to test.

## Caveats

- **19 validation images.** Every per-growth-form row rests on 1-10 images;
  `dense_mat` is a single image and should be read as an anecdote, not a
  measurement. The aggregate is dominated by the 10 rosettes.
- **The training split is skewed** (104 of 144 rosette, 8 dense_mat). The
  inverse-frequency sampler compensates during training, but validation is
  unweighted, so the headline IoU is largely a rosette score.
- **Five seeds is a small sample** for a t-test. The interval is wide relative
  to the effect, and the conclusion is "SegFormer is ahead", not a trustworthy
  estimate of by how much.
