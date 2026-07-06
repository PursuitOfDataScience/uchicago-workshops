# Common sbatch mistakes (pre-flight checklist)

- **No `--account=`** — the job is rejected or charged to the wrong allocation.
  Fix: `#SBATCH --account=pi-<name>`.
- **No `--time=`** — falls back to a tiny default and gets killed early.
  Fix: `#SBATCH --time=HH:MM:SS`.
- **Wrong or missing `--partition=`** — e.g. asking for a GPU on a CPU-only partition.
  Fix: match the partition to the resource you need.
- **GPU job with no `--gres`/`--gpus`** — sits idle or fails on a GPU partition.
  Fix: `#SBATCH --gpus=1` (or `--gres=gpu:1`).
- **`--mem` too low** — the job OOMs mid-run.
  Fix: estimate real memory and set `#SBATCH --mem=<N>G`.
- **No `module load` / env activation in the body** — "command not found" at runtime.
  Fix: load modules or `source .../activate <env>` before the command.
- **Assuming internet on compute nodes** — many RCC compute nodes have no egress, so downloads hang.
  Fix: pre-stage data/models on a login node first.
- **`--ntasks` vs `--cpus-per-task` confusion** — MPI ranks vs threads.
  Fix: match the layout to how the program actually parallelizes.
