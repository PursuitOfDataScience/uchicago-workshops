# Hands-on: drive Claude Code in plain English

This is the interactive half of the workshop. There is **no notebook, no subfolders, and nothing
to download** — just this folder of Markdown task cards and a `CLAUDE.md`. You open Claude Code
here and, in ordinary English, ask it to do real work; it figures out and runs the commands, then
reports back while you review.

The lab has **two parts**:

- **Part 1 · Navigate Midway3 (Tasks 1–8).** Ask Claude about *this* cluster — your disk quota,
  your jobs, your allocation balance, how busy the partitions are, how to get an interactive
  session or write a batch script, what software is available. Claude runs the RCC/Slurm commands
  for you (`quota`, `squeue`, `sinfo`, `accounts`, `module`, …) and explains the output. This is
  read-only exploration — nothing is submitted, cancelled, or deleted.
- **Part 2 · Analyze a dataset (Tasks 9–17).** A short data-analysis project on **Palmer
  Penguins** (344 measurements, 3 species, 3 Antarctic islands), read live from a URL. Claude
  writes and runs the analysis; you check the numbers.

Each task card holds **only the prompt** to paste. For Part 2, the reference solution — the Python
code and expected output — is in **`answers.py`**, so you can check Claude's numbers against a
known-correct answer. Part 1 has no answer key: it reads *live* cluster state (your quota, the
queue, node availability), which changes minute to minute.

## Before you start
- **Claude Code installed and logged in** — `claude --version` prints a 2.x build (see the
  top-level [`../README.md`](../README.md) for the no-sudo install and `/login`).
- **A Midway3 node.** Part 1 uses the RCC/Slurm commands, which are available on any login node.
  Part 2 also needs **internet egress** (it reads the dataset over the network) and pandas +
  matplotlib — activate the shared **`AI`** environment:
  `source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI`.

## Start
```bash
cd hands-on        # (or copy this folder somewhere writable and cd there)
claude
```

## Part 1 · Navigate Midway3
Do these in any order — each is self-contained. Paste the prompt and read Claude's answer against
what you already know about your account.

| # | Task | You ask Claude to… |
|---|------|--------------------|
| 1 | [Get your bearings](1-bearings.md) | say who/where you are and what this node is for |
| 2 | [Check your disk quota](2-quota.md) | show quotas + what's using space (nothing deleted) |
| 3 | [Your compute budget (SUs)](3-account.md) | list your allocations and how much is left |
| 4 | [How busy is the cluster?](4-cluster.md) | break down partitions, idle vs busy nodes, free GPUs |
| 5 | [Grab an interactive session](5-interactive.md) | hand you the exact `sinteractive` command, explained |
| 6 | [Write a batch job script](6-batch-script.md) | generate and explain a Slurm `sbatch` script |
| 7 | [Track your jobs](7-jobs.md) | summarize your jobs and diagnose a pending one |
| 8 | [Find and load software](8-software.md) | find a module and give the exact `module load` line |

## Part 2 · Analyze a dataset
Work these **in order, in one running session** — paste each prompt after the last one finishes and
let the conversation continue. Later tasks build on earlier ones (Task 15 sums up the rest), so
don't restart or `/clear` between them.

| # | Task | You ask Claude to… |
|---|------|--------------------|
| 9 | [First look](9-first-look.md) | load the data from the URL and describe it |
| 10 | [Summary by species](10-summary-by-species.md) | group and summarize body mass & flipper length |
| 11 | [The heaviest species](11-heaviest-species.md) | compare species and make a judgment call |
| 12 | [Flipper length vs body mass](12-flipper-vs-mass.md) | test a relationship — overall and within groups |
| 13 | [Data quality](13-data-quality.md) | find missing / implausible values (without dropping them) |
| 14 | [Make a figure](14-make-a-figure.md) | write and run a plotting script → a saved figure |
| 15 | [Write it up](15-write-it-up.md) | turn the numbers into a Results paragraph |
| B | [Bonus: run it headless](16-headless.md) | get the same analysis as one `claude -p` command |
| ★ | [Take-home: your own data](17-your-own-data.md) | point it at a CSV URL of your own |

Part 2 is about 40 minutes. Each task builds on the last, so go in order.

## The one rule: check its work
Claude reports by running commands and code you can read. On at least one task, ask it to **show
what it ran** and spot-check the result yourself — verify a quota figure against `quota`, or a
penguin number against **`answers.py`**. An agent can be confidently wrong; verify before you cite.
