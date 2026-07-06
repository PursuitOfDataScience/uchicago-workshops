---
name: slurm-triage
description: Explain why a Slurm job is pending, stuck, or failed, and recommend the smallest safe fix. Use when the user asks why a job won't start, is stuck in the queue, got killed, or shares scontrol/squeue output. Strictly read-only — it never modifies jobs.
allowed-tools: Read, Bash(squeue:*), Bash(scontrol show:*), Bash(sacct:*)
---

# Slurm job triage (read-only)

Diagnose why a Slurm job is pending or failed and say what to do — never mutate anything.

## Procedure
1. Get the job's state:
   - a job id → `scontrol show job <id>`
   - a saved dump or an `.err` log the user points at → read the file
   - "my jobs" → `squeue -u $USER`
2. Find `JobState` and, when it's pending, the `Reason=(...)` code.
3. Look up the code in `reference.md` (in this folder) and explain it in plain language.
4. Recommend the *smallest safe* change — usually: wait, request fewer resources, or fix the script.
   NEVER extend a time limit, cancel a job, or edit an allocation: those need privileges most users do
   not have, and this skill does not perform them.

Begin every diagnosis with the tag `[slurm-triage]` on its own line so it's clear this skill handled it.
