# Task 5 · Data quality

**Goal:** have the agent audit the data — and hold it to the rule that it must not quietly
throw anything away.

## Paste this into Claude Code

> Find every row with a missing value, and flag any measurements that look implausible for a
> penguin. Summarize what a careful analyst should do about them — but do not change or drop
> any rows.

## What to expect / check
- **11 rows** have missing values: **2 rows are missing every measurement** (unusable — only
  species and island are present), and the other **9 are missing only `sex`** (still fully
  usable for size analysis).
- No measurement is wildly implausible — the values are real field data — so a good answer
  says so rather than inventing problems.
- The recommendation should be sensible and *conservative*: exclude the 2 empty rows from
  measurement analyses, keep the 9 (or impute/label `sex` if needed), and **document the
  choice** — never silently drop. `CLAUDE.md` tells it not to alter data; check that it obeyed.

## Make it yours
- This is exactly how you'd triage your own messy export before trusting it. Ask: *"write these
  findings into a short `data_notes.md`"* — a durable record of what you found.

> **Next:** [Task 6 · Make a figure](06-make-a-figure.md)
