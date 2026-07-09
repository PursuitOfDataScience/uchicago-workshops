# Task 4 · Document the project

**Goal:** finish the documentation the previous owner left half-written.
**You'll practise:** using the agent for the writing that surrounds research — not just code.

## Do this — paste into Claude Code

> Read @src/waterquality.py and @docs/data_dictionary.md. Fill in every `TODO` in the data
> dictionary: give each measurement column a realistic valid range and a one-line
> description, based on the code and on typical lake water-quality values. Update only
> @docs/data_dictionary.md.

Then improve the code's own documentation:

> Check that every public function in @src/waterquality.py has a clear one-line docstring
> describing what it returns. Improve any that are vague. Don't change what the functions do.

## What to watch for
- The agent grounds the ranges in the code and the domain, rather than inventing numbers —
  read them and sanity-check (pH is 0–14; dissolved oxygen is never negative).
- `git diff docs/data_dictionary.md` shows exactly what it wrote. You are the reviewer;
  keep what is right, correct what is not.

## Make it yours
- Ask it to draft a two-paragraph *"Methods: data cleaning"* note for a paper's supplement,
  citing the specific rules from `clean_data.py`. Generated prose is a first draft you edit —
  never a final artifact you ship unread.

> **Next:** [Task 5 · Turn field notes into data](05-extract-from-notes.md)
