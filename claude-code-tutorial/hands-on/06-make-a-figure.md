# Task 6 · Make a figure

**Goal:** get a real, saved artifact — the agent writes a plotting script, runs it, and
produces a figure you can drop into a talk.

## Paste this into Claude Code

> Write and run a Python script that saves a boxplot of `body_mass_g` by species to
> `body_mass_by_species.png` in this folder. Label the axes and give it a title. Then tell me
> what the plot shows.

## What to expect / check
- A file **`body_mass_by_species.png`** appears in the folder (`ls` to confirm; open it if you
  can). The Gentoo box sits clearly above Adelie and Chinstrap, which overlap.
- Because this is a headless cluster, the script must use a non-interactive backend
  (`matplotlib.use("Agg")`) and save rather than `show()` — `CLAUDE.md` reminds it to. If it
  errors on a display, that is the fix.
- The written description should match the picture (Gentoo heaviest; Adelie and Chinstrap
  similar).

## Make it yours
- Ask for a different plot — a scatter of flipper length vs body mass colored by species, or a
  histogram — and it re-runs. You are directing the analysis; it does the plotting.

> **Next:** [Task 7 · Write it up](07-write-it-up.md)
