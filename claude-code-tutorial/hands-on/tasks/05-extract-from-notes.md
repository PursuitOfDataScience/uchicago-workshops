# Task 5 · Turn field notes into data

**Goal:** convert a folder of free-text field logs into one structured table you can join
to the measurements.
**You'll practice:** structured extraction — the everyday "read a pile of documents and pull
out the fields" task (lit reviews, abstracts, survey responses, lab notebooks).

The `notes/` folder holds one Markdown field log per site visit. Buried in the prose is
information that explains anomalies in the data (a blank temperature, a wild pH, a negative reading).

## Do this — paste into Claude Code

> Read every Markdown file in @notes/. For each visit, extract: date, site, sampler,
> weather, whether there was an incident (yes/no), and a short description of the incident
> if any. Write the result as a CSV with those columns to `data/clean/field_log_summary.csv`,
> one row per note, sorted by date.

## What to watch for
- Open `data/clean/field_log_summary.csv`: one row per note, consistent columns — prose
  turned into a table you can analyse.
- Cross-check against Task 2: the field-log incidents explain the anomalies you handled — the
  **missing readings at bravo** (a dead logger battery, a probe that would not read) and the
  **impossible values at charlie** (a pH near 100, a negative dissolved-oxygen reading). The
  notes and the data tell the same story.

## Make it yours
- For a table you will feed to another program, you want *guaranteed* structure. Headless,
  that is `claude -p "..." --output-format json --json-schema schema.json`, which returns
  fields validated against a schema instead of prose you have to parse (see Bonus A).

> **Next:** [Task 6 · Automate it](06-automate.md)
