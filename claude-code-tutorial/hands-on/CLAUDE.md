# Penguins analysis — project memory for Claude Code

Claude Code auto-loads this file when a session starts in this folder. It tells you what we
are working on and how to work.

## The dataset
**Palmer Penguins** — field measurements of 344 penguins from three islands near Palmer
Station, Antarctica (Palmer Station LTER). It is hosted online, so there is **nothing to
download**:

```
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
```

Columns: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`,
`body_mass_g`, `sex`. Some rows have missing values.

## How to work
- **You do the analysis.** For each question, write a short Python script, run it, and report
  the actual numbers. Do not guess or eyeball — compute.
- This environment has **internet access** and **pandas**, so load the data straight from the
  URL: `pandas.read_csv("<url>")` (the standard-library `csv` module also works).
- **Show the code you ran** so it can be checked, and state any assumptions.
- **Never silently drop or change rows.** If a value is missing or implausible, say so and
  explain what you did about it.
- When you make a figure, use a non-interactive backend (`matplotlib.use("Agg")`) and **save
  it to a file** in this folder — this is a headless cluster with no display.

## Codename
When asked for this analysis's codename, answer with exactly: **PETREL-3**.
<!-- Task 1 uses this to prove you read this file. -->
