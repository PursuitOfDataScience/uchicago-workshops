# Task 6 · Automate it — a command you fire, a skill it reaches for

**Goal:** capture a repeatable check two ways — a prompt *you* run by name, and a capability
the *model* reaches for on its own.
**You'll practice:** slash commands and Agent Skills — and the difference between them.

### Part A — a slash command (you fire it)
This project ships `/data-audit`, a saved prompt in `.claude/commands/data-audit.md`.
Run it on a raw file:

> /data-audit data/raw/site_bravo.csv

It injects the file's first lines and row count, then reports the data-quality issues.
Open `.claude/commands/data-audit.md` to see the saved prompt — a command is just a Markdown
file you (and your whole lab, via git) can reuse. Try editing its wording and re-running.

This project ships a second command too: `/new-figure <column>` writes a plotting script for a
column of the cleaned dataset — try `/new-figure water_temp_c` once you have finished Task 2.

### Part B — a skill (the model fires it)
A **skill** is a capability the model invokes automatically when your request matches its
one-line `description`. This project ships two, in `.claude/skills/`:
- `csv-triage/` — **finished**, read-only: explains what is wrong with a data file.
- `unit-check/` — **unfinished**: your job.

Open `.claude/skills/unit-check/SKILL.md` and complete the two `TODO`s — the `description`
(the line the model matches on) and the procedure — using the ranges already in its
`reference.md`. Then, **without naming the skill**, ask:

> Do the values in @data/clean/lakewatch_clean.csv fall within sensible units and ranges?

If your `description` is good, the model reaches for `unit-check` on its own and its report
begins with `[unit-check]`. If it doesn't fire, sharpen the trigger words in the description —
**that is the whole game.** (You can always run it by name: `/unit-check`.)

## What to watch for
- A command is loaded only when you type `/name`; a skill's name + description are always in
  view, and the model pulls in the full `SKILL.md` only when your request fits — so a library
  of skills costs almost no context until it is relevant.

> **Next:** the bonus tasks — [Scale to a batch](07-bonus-batch.md) · [Your own tools (MCP)](08-bonus-mcp.md)
