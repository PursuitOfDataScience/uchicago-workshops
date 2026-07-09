# Midway3 + Claude Code — project notes

Claude Code loads this file automatically at the start of a session in this folder. It records the
facts about this hands-on lab.

## What this is
A two-part hands-on lab that teaches you to drive Claude Code in plain English:

- **Part 1 — Navigate Midway3 (Tasks 1–8).** Ask Claude, in ordinary English, to answer real
  questions about *this* cluster — your disk quota, your jobs, your allocation balance, how busy
  the partitions are, how to get an interactive session or write a batch script, what software is
  available. Claude runs the underlying commands for you and explains the output.
- **Part 2 — Analyze a dataset (Tasks 9–17).** A small data-analysis project on the **Palmer
  Penguins** dataset: 344 field measurements of penguins from three islands near Palmer Station,
  Antarctica.

## Part 1 — the Midway3 environment
This session is running on a **Midway3** node (RCC, University of Chicago). The RCC and Slurm
command-line tools are available and can be run directly to answer questions:

- **Jobs:** `squeue`, `scontrol show job`, `sacct`
- **Cluster / nodes:** `sinfo`, `scontrol show node`, `rcchelp`
- **Accounts / allocations:** `accounts` (balance, allocations, usage, list, members, storage, qos)
- **Storage:** `quota`, `du`
- **Software:** `module` (this is **Environment Modules 4.x** — use `module avail` / `module load`
  / `module list`; there is **no** `module spider`)
- **Running work:** `sinteractive` (interactive session; takes sbatch options — for a GPU pass
  `--gres=gpu:N` **and** `--partition=gpu`, since the default `caslake` partition has no GPUs),
  `sbatch` (batch submit)

Cluster facts: the **default CPU partition is `caslake`**; GPU work goes to `gpu` or `beagle3`.
For any "my …" question, detect the current user with `whoami` first — never ask for the username.

### Safety — Part 1 is read-only exploration
Inspect and explain freely, and you may **write** files (e.g. an `sbatch` script) to this folder.
But do **not** submit, cancel (`scancel`), or modify jobs, and do **not** delete or move files,
unless the user explicitly asks and approves it. When a task asks for a command (e.g. `sinteractive`
or `sbatch`), hand over the command and explain it rather than running it.

## Part 2 — the dataset
Hosted online — nothing to download:

    https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv

Columns: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`,
`body_mass_g`, `sex`. Some rows have missing values. This environment has internet access and
pandas, so the file can be read directly from the URL.

## Working style
At the start of each task, first **restate it in one short line** — a plain-language summary of
what the prompt is asking — so the workshop audience can see which task this is. Then carry it out.
