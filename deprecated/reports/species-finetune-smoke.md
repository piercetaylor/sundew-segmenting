# Fine-tune smoke test: the stage-2 script is ready to launch

`scripts/finetune_species_backbone.py` run end to end before the real array
(`scripts/hellbender_species_finetune.slurm`), so a crash or a broken resume
costs five minutes, not a night of GPU. Run with
`sbatch scripts/hellbender_species_finetune_smoke.slurm`; writes only to
`models/species-110/finetune-smoke/`, which is safe to delete.

## Answer: passed on the second attempt, job 17929407

| Attempt | Job | Finished | Result |
| --- | --- | --- | --- |
| 1 | 17928480 | 2026-09-24 02:47 | `RESUME FAILED`; peak memory printed as `0 MiB` |
| — | 17929406 | 2026-09-24 02:47 | cancelled before start (resubmitted after the fix) |
| 2 | 17929407 | 2026-09-24 05:55 | **`SMOKE PASSED`**, 4 min 41 s |

## What it covers

Every code path the real array can take, on 512 images per split, 2 epochs,
warmup 1, batch 64, split `split-110` (not `split-110-test`, so
`test_held_out=0` in the log; the real array reads `split-110-test`):

| Backbone | Library | Arm | Epoch time | Peak GPU memory |
| --- | --- | --- | ---: | ---: |
| `convnext-b-in22k` | timm, CNN | `full` | 13-14 s | 9.3 GiB |
| `dinov2-l-reg` | timm, ViT-L | `full-square` | 13-14 s | 20.7 GiB |
| `bioclip-2` | open_clip, ViT-L | `crop` | 5-7 s | 19.0 GiB |
| `dinov2-b` (resume check) | timm, ViT-B | `crop` | 4-5 s | 7.6 GiB |

- **Memory is not a constraint.** The largest, DINOv2-L at batch 64, peaks at
  20.7 GiB of the A100's 80 GiB.
- **The accuracies in the log mean nothing.** Two epochs on 512 images; the
  test is that the code runs, the loss falls and every metric is written
  (balanced, plain, top-5, and section-level balanced accuracy).

## Resume check

The real array runs on `gpu_requeue`, so a preempted task must pick up from
`last.pt` and not start over. The check starts a 3-epoch `dinov2-b` run, kills
it once epoch 1's state is on disk, restarts it, and requires
`resumed after epoch 1` and an epoch-3 line in the log. It passed:

```
resumed after epoch 1 (best 0.0246 at 1)
{"epoch": 2, ...}
{"epoch": 3, ...}
RESUME OK
last.pt cleaned up
```

**Why attempt 1 failed.** The restarted run began at epoch 1 with no
`resumed` line: the first run had been killed before any checkpoint was
written, so there was nothing to resume from. The kill was timed on the epoch
line in the log, and that line prints before the state is saved. The harness
now waits for the checkpoint file itself (`last-smoke.pt`) before killing,
kills the Python child as well as the backgrounded shell, and confirms nothing
survives before restarting. With the kill timed on the checkpoint, resume
worked first time.

**Peak memory.** Attempt 1 printed `0 MiB` per run. The script now reports
`torch.cuda.max_memory_allocated()` itself, on the summary line and as
`peak_gpu_gib` in the run's JSON.

Logs: `logs/sundew-finetune-smoke-{17928480,17929407}.out`,
`models/species-110/finetune-smoke/resume.log`.
