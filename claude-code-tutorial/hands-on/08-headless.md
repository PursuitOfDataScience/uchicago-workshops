# Bonus · Run it headless

**Goal:** get the same kind of analysis as **one command** you could drop into a script or a
Slurm job — no interactive session.

Everything so far was a conversation. The same agent runs **headless**: one prompt in, one
machine-readable answer out. This is the building block of any pipeline (see the lecture's
"Scaling up" section).

## Try it — in a terminal (in this folder)

```bash
claude -p "Read the penguins CSV at https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv \
and reply with only the mean body_mass_g for each species, as compact JSON." \
  --output-format json --model haiku
```

You get back a JSON object: the `result` text plus `total_cost_usd`, `session_id`,
`num_turns`, and `is_error` — a record you can parse, log, and audit, not prose to eyeball.

## What to expect / check
- The `result` contains the three per-species means (Gentoo ~5076, Chinstrap ~3733,
  Adelie ~3701). The call needs internet on the node, just like the interactive tasks.
- Add `--json-schema schema.json` to force the reply into fields validated against a schema —
  ideal when the output feeds the next step of a pipeline.

## Make it yours
- Imagine the URL is one of hundreds of result files. A shell `for` loop over `claude -p`
  calls — summing `total_cost_usd` — is a costed batch job; wrap it in `sbatch` to run it
  unattended on the cluster.
