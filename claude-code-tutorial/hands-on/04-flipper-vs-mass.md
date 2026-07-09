# Task 4 · Flipper length vs body mass

**Goal:** test a relationship — and learn why you should always check it *within* groups, not
just overall.

## Paste this into Claude Code

> Is flipper length related to body mass? Report the correlation across all penguins, then the
> correlation within each species separately. Does the within-species picture differ from the
> overall one, and why might that be?

## What to expect / check
- **Overall correlation ≈ 0.87** — very strong. But **within each species it is weaker**:
  roughly **Adelie 0.47, Chinstrap 0.64, Gentoo 0.70**.
- The takeaway (a classic data-analysis trap): the strong overall number is inflated because
  the species differ in *both* flipper length and mass, so mixing them exaggerates the link.
  The honest picture is the within-group one. A good answer explains this.
- Spot-check: ask it to show the correlation code and confirm the overall figure is ~0.87.

## Make it yours
- Ask for a scatter plot colored by species (saved to a file) to *see* the effect — the three
  clusters line up along a shared trend.

> **Next:** [Task 5 · Data quality](05-data-quality.md)
