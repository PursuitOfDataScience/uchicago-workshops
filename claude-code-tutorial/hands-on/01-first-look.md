# Task 1 · First look

**Goal:** get Claude Code to read the hosted dataset and describe it — and confirm it loaded
the project's memory.

## Paste this into Claude Code

> Load the penguins dataset from the URL in CLAUDE.md and give me an overview: how many rows
> and columns, the column names and their types, the three species and their counts, and how
> many rows have any missing values. Show the code you ran.

Then, to prove it read the project's memory file, ask:

> What is this analysis's codename?

## What to expect / check
- **344 rows, 7 columns**; species counts **Adelie 152, Gentoo 124, Chinstrap 68**;
  **11 rows** have a missing value.
- It should reach the answer by **running code** (loading the URL with pandas), not by
  guessing — watch it write a snippet, run it, and read the result. Approve the command when
  it asks; that is the permission prompt from the lecture.
- The codename is **PETREL-3** — it lives only in `CLAUDE.md`, so a correct answer proves the
  file was loaded into context.

## Make it yours
- Ask it to *"show me 5 random rows"* to get a feel for the data. Notice there is nothing to
  download or set up — it reads the live URL each time.

> **Next:** [Task 2 · Summary by species](02-summary-by-species.md)
