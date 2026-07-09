---
name: unit-check
# ┌─ TASK 6 EXERCISE ───────────────────────────────────────────────────────────┐
# │ Replace the description below with a real one. This single line is the ONLY  │
# │ thing Claude matches on to decide whether to reach for this skill on its     │
# │ own — so say WHAT it does AND WHEN to use it, in the third person, with the  │
# │ trigger words a researcher would actually type.                              │
# │ Example shape: "Check that each numeric column of a dataset uses plausible   │
# │ values for its unit (…). Use when the user asks whether the units or ranges  │
# │ in a data file look right, or before trusting a cleaned dataset."            │
# └──────────────────────────────────────────────────────────────────────────────┘
description: TODO — replace this line with a real trigger description (see the note above)
allowed-tools: Read
---

# Unit & range check (read-only)

<!-- TASK 6 EXERCISE: write the procedure. A good one will: -->
<!--   1. Read the CSV the user names. -->
<!--   2. For each numeric column, look up its expected unit and valid range in reference.md. -->
<!--   3. Report every value that falls outside its valid range, with the row and the reason. -->
<!--   4. Do NOT edit the data — report only. -->

Begin your report with the tag `[unit-check]` on its own line so it is clear this skill handled it.
