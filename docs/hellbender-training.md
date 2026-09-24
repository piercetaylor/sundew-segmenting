# Training on Hellbender

Use Hellbender for the final U-Net and SegFormer comparison. The reviewed
dataset snapshot is small, while 768-pixel training benefits substantially from
an NVIDIA GPU.

## Environment

The training stack is defined in `environment.yml` and built with the cluster's
`mamba` module. Torch is installed from the PyTorch CUDA index rather than from
`conda-forge`: those wheels bundle their own CUDA runtime, so the job does not
depend on the cluster's `cuda` modules, which only go up to 11.8 and are too old
for Hellbender's A100/H100/L40S mix.

Environments land in the `envs_dirs` from `~/.condarc`, which points at pixstor
rather than the 50 GB `/home` quota.

Build it once. Do not run this on the login node — cluster policy is explicit
that "under no circumstances should your code be running on the login node", and
a full torch install is heavy:

```bash
mkdir -p logs
sbatch scripts/hellbender_setup.slurm
```

That job creates (or updates) the `sundew-seg` environment, installs the repo
package, verifies and unpacks `dist/sundew-segmenting-reviewed-v0.3.0.zip`, and
warms the pretrained-encoder caches under `sundew-cache/` so the training run
does not depend on the network.

Paths are overridable via `REPO_DIR`, `DATA_ROOT`, `CACHE_ROOT`, and `ENV_NAME`.

## Submitting the comparison

```bash
sbatch scripts/hellbender_train.slurm
```

`logs/` must exist before submitting: Slurm opens the output files itself and
fails the job if the directory is missing.

The two-item job array requests one A100, eight CPU cores, 32 GB RAM, and 30
minutes per model.

### Partition

GPU jobs are **rejected** by the `requeue` partition — the scheduler returns
"GPU jobs are not allowed in the 'requeue' partition. Please use the
'gpu_requeue' partition instead." The script therefore uses `gpu_requeue`.

| Partition | Preemption | Notes |
| --- | --- | --- |
| `gpu_requeue` | preemptible (`--requeue` set) | schedules immediately; the default here |
| `gpu` | none | safer for long runs, but currently backlogged by more than a day |

Both satisfy the documented acceptable use for the GPU partitions, since these
jobs use the GPU for the majority of the run. A 30-epoch run takes about three
minutes at roughly 6.3 s/epoch on an A100, so preemption is unlikely to matter;
pass `--partition=gpu` if a run must not be interrupted anyway.

### Walltime

Walltime is sized from a measured run rather than guessed. Requesting only what
the job needs schedules sooner and leaves the GPUs free for other users.

## Results

Monitor with `squeue -u "$USER"`. Checkpoints and metrics are written under
`models/hellbender/` and logs under `logs/`; both are gitignored. Set
`MODEL_ROOT` to write elsewhere, which is useful for short trial runs:

```bash
sbatch --export=ALL,EPOCHS=2,MODEL_ROOT=models/smoke scripts/hellbender_train.slurm
```

Both models use the frozen training split (144 train / 19 validation), select
checkpoints using validation IoU, and leave the 28-image test split untouched.
