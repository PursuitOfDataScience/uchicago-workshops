# Bonus A · Scale to a batch

**Goal:** run the same agent call over many files, unattended, with the bill printed at the end.
**You'll practise:** headless `claude -p` and the leap from one call to a Slurm job.

Everything so far was interactive. The same agent runs **headless**: one prompt in, one
machine-readable answer out — the building block of any pipeline.

## Try one headless call — in a terminal (in the project dir)

```bash
cat logs/charlie_2023-07-07.log | claude -p \
  "In one word — CALIBRATION, BATTERY, CONNECTION, or OK — classify the instrument problem \
   described on standard input. Reply with only the word." \
  --model haiku --output-format json
```

You get back a JSON object: the `result` text, `total_cost_usd`, `session_id`, `num_turns`,
`is_error`. That is a record you can parse, sum, and audit — not prose to eyeball.

## Now the batch
`classify_logs.sh` loops that one call over every file in `logs/` and totals the cost:

```bash
bash classify_logs.sh
```

`run.sh` submits the exact same batch as an unattended Slurm job (it preflights one cheap
call first, so a node with no internet fails in seconds, not after a long wait):

```bash
sbatch --export=ALL,CLAUDE_CONFIG_DIR=$HOME/.claude run.sh
```

## What to watch for
- **One call is the unit of a batch job.** Loop it, sum `total_cost_usd`, submit it.
- **Network:** the compute node needs egress to `api.anthropic.com`. On Midway3, login nodes
  and the `test`/`caslake` partitions have it; many other compute nodes do not.
- **Rate limits & secrets:** dozens of simultaneous calls from one key will throttle — stagger
  them or use per-user keys. Log in via `CLAUDE_CONFIG_DIR`, not a raw `ANTHROPIC_API_KEY`
  baked into the job, which would be visible via `scontrol`.

## Make it yours
- Point the loop at *your* logs, abstracts, or output files. Add `--json-schema` to force the
  reply into a validated schema you can drop straight into a table.
