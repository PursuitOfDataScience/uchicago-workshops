# Task 3 · The heaviest species

**Goal:** go past raw numbers to a comparison and a judgment — the kind of question you'd
actually ask of your data.

## Paste this into Claude Code

> Which species is heaviest on average, and by how much compared with the lightest? Is that
> difference large relative to the spread *within* each species? Give me the numbers and a
> one-sentence verdict.

## What to expect / check
- **Gentoo (~5076 g)** is heaviest; **Adelie (~3701 g)** is lightest — a gap of about
  **1350–1400 g**. Within-species standard deviations are ~400–500 g, so the between-species
  gap is **roughly three standard deviations** — a large, real difference.
- The verdict should be a clear "yes, Gentoo is substantially heavier," *grounded in the
  numbers* — not a vague statement.

## Make it yours
- Push it: *"is that difference statistically significant?"* and see it reach for a t-test.
  Then ask it to **state its assumptions** — a good habit to demand of any analysis, human or
  agent.

> **Next:** [Task 4 · Flipper length vs body mass](04-flipper-vs-mass.md)
