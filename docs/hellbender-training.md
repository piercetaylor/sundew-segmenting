# Training on Hellbender

Use Hellbender for the final U-Net and SegFormer comparison. The reviewed
dataset snapshot is small, while 768-pixel training benefits substantially from
an NVIDIA GPU.

Copy these items to your Hellbender storage:

- the repository;
- `dist/sundew-segmenting-reviewed-v0.3.0.zip`;
- its adjacent `.sha256` file.

On Hellbender, verify and unpack the archive, then create the environment once:

```bash
sha256sum --check sundew-segmenting-reviewed-v0.3.0.zip.sha256
unzip sundew-segmenting-reviewed-v0.3.0.zip
cd sundew-segmenting
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[baseline]'
```

Submit the controlled comparison from the repository directory:

```bash
mkdir -p logs
sbatch scripts/hellbender_train.slurm \
  "$PWD" \
  "$HOME/sundew-segmenting-reviewed-v0.3.0"
```

The two-item job array requests one A100 GPU, eight CPU cores, 32 GB RAM, and
eight hours per model from the preemptible `requeue` partition. Adjust the
partition or account flags to match your allocation. Both models use the frozen
training split, select checkpoints using validation IoU, and leave the test
split untouched.

Monitor the job with `squeue -u "$USER"`. Results are written under
`models/hellbender/` and logs under `logs/`.
