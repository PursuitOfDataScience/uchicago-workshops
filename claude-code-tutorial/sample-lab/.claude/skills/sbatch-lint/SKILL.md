---
name: sbatch-lint
# TODO(you): write a description that makes Claude reach for this skill when someone is about to
# submit a job or asks whether their sbatch script is correct — WITHOUT them typing "/sbatch-lint".
# Say WHAT it does AND WHEN to use it, in the third person, with trigger words. This one line is
# the only thing the model matches on, so it is the difference between the skill firing or not.
# Example shape:  "Check a Slurm sbatch script for common mistakes before submission. Use when ..."
description: TODO — replace this line with a real trigger description
allowed-tools: Read
---

# sbatch pre-flight lint (read-only)

TODO(you): write the procedure below. A good one will:
  1. Read the sbatch script the user names.
  2. Check it against the common mistakes in `reference.md` (in this folder) — a bundled file you
     can tell the model to read.
  3. Report each problem you find and its one-line fix. Do NOT submit or modify anything.

Begin your report with the tag `[sbatch-lint]` on its own line so it's clear this skill handled it.
