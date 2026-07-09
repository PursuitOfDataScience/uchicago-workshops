# Task 2 · Tidy the messy data

**Goal:** turn three inconsistent raw exports into one clean, analysis-ready dataset.
**You'll practice:** the permission ladder — explore in **Plan mode** first, then let the
agent do the work in **Accept-edits** mode.

The three files in `data/raw/` each use different column names, units, and date formats,
and a few readings are missing or impossible. This is the everyday reality of research data.

## Step 1 — look before you leap (Plan mode)
Press **Shift+Tab** until the mode indicator reads **plan** (read-only), then paste:

> Compare @data/raw/site_alpha.csv, @data/raw/site_bravo.csv, and @data/raw/site_charlie.csv.
> How do their columns, units, and date formats differ, and which values are missing or
> impossible? Propose a plan to merge them into one tidy file — don't write anything yet.

Read the plan. In Plan mode the agent **cannot** change a single file, so this is a safe way
to think out loud together.

## Step 2 — let it do the work (Accept-edits mode)
Press **Shift+Tab** to switch to **acceptEdits**, then paste:

> Follow that plan. Write a standard-library-only script `src/clean_data.py` that reads the
> three raw files, normalizes them to the columns in @docs/data_dictionary.md (convert
> Fahrenheit to Celsius, parse the dates to ISO `YYYY-MM-DD`, add a `site` column), applies
> the cleaning rule in the data dictionary (blank out any missing value or impossible reading
> but keep the rest of the row; drop only exact duplicate rows), and writes
> `data/clean/lakewatch_clean.csv`. Then run it and show me the first few rows.

## What to watch for
- The agent reads `data/raw/` but **writes only to `data/clean/` and `src/`**. If it ever
  tries to "fix" a raw file in place, the deny rule `Edit(data/raw/**)` stops it — raw data
  is the system of record.
- Check its work: `git diff` shows the new script; open the clean CSV. Did it blank the
  impossible pH of 99.9 and the negative dissolved-oxygen at site_charlie, leave bravo's
  missing temperature (7/12) and dissolved-oxygen (6/21) empty, and drop the duplicate
  site_charlie row for 2023-06-23?

## Make it yours
- Ask it to add a short comment block to `clean_data.py` explaining each cleaning decision,
  so the next person understands *why* a row was dropped.

> **Next:** [Task 3 · Fix the bug — safely](03-fix-the-bug.md)
