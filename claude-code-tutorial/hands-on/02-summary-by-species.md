# Task 2 · Summary by species

**Goal:** the everyday first move on any dataset — group by a category and summarize.

## Paste this into Claude Code

> For each species, report the number of penguins and the mean and standard deviation of
> `body_mass_g` and `flipper_length_mm`. Present it as a clean table, and say how you handled
> the missing values.

## What to expect / check
- Mean body mass is roughly **Adelie ≈ 3700 g, Chinstrap ≈ 3733 g, Gentoo ≈ 5076 g** (standard
  deviations are in the 380–500 g range). **Gentoo is clearly the heaviest.**
- A careful answer notes that missing values were **skipped** in the averages (so the counts
  behind the means are 151 / 68 / 123, not 152 / 68 / 124) rather than silently dropped.
- Spot-check: ask it to *show the code*, and confirm the per-species counts add up to 344.

## Make it yours
- Ask for the same summary **by island** instead of species, or add `bill_length_mm`. One line
  changes; the agent rewrites and re-runs.

> **Next:** [Task 3 · The heaviest species](03-heaviest-species.md)
