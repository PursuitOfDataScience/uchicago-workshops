---
description: Diagnose a failed Slurm job from its script and error log
argument-hint: <path to sbatch script> <path to .err log>
allowed-tools: Read, Bash(sacct*), Bash(sinfo*)
---
A Slurm job failed. Diagnose the most likely cause and propose a concrete fix.

Context (injected automatically):
- Current queue for me: !`squeue -u $USER 2>/dev/null | head`
- Partitions available: !`sinfo -o "%P %a %l %D" 2>/dev/null | head`

Now read the sbatch script and error log the user pointed at (`$ARGUMENTS`), then:
1. State the single most likely root cause (OOM, wrong partition, bad `--gres`, time limit, module/env).
2. Show the exact lines to change in the sbatch script.
3. Note any RCC-specific gotcha (e.g., GPU constraint, account, internet egress on compute nodes).
