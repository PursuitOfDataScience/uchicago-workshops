---
name: csv-triage
description: Inspect a CSV of tabular research data and explain its data-quality problems in plain language — missing values, inconsistent headers or units, impossible readings, and duplicate rows. Use when the user asks what is wrong with a data file, whether a CSV is clean, or why a dataset looks off. Strictly read-only — it never edits the data.
allowed-tools: Read, Bash(head:*), Bash(wc:*)
---

# CSV triage (read-only)

Explain the data-quality problems in a CSV so a researcher knows what to fix before
analysis. Never modify the file — describe and recommend only.

## Procedure
1. Read the file the user names (or `head -n 8` it first to see the shape).
2. Work through the checklist in `reference.md` (in this folder), category by category.
3. For each problem found, report: the column or row, what is wrong, and the smallest safe fix.
4. Finish with a one-line verdict: is this file ready for analysis, or does it need cleaning first?

Begin every report with the tag `[csv-triage]` on its own line so it is clear this skill handled it.
