---
description: Audit a CSV for data-quality problems before you trust it
argument-hint: <path to a .csv file>
allowed-tools: Read, Bash(head:*), Bash(wc:*)
---
You are auditing a data file for quality problems. The file is: `$ARGUMENTS`

Context (gathered for you):
- First lines: !`head -n 6 "$ARGUMENTS"`
- Row count: !`wc -l "$ARGUMENTS"`

Now read the file and report, as a short bulleted list:
1. The column names and the unit each column appears to use (look for units in the header).
2. Any missing values (blank cells, `NA`, `null`).
3. Any values that are physically impossible or clearly wrong (e.g. a pH outside 0–14,
   a negative concentration, an out-of-range temperature).
4. Any duplicate rows or inconsistent formatting (dates, capitalisation, stray spaces).

Do not change the file — this is a read-only audit. End with the single most important fix.
