# Deck source — `claude-code-tutorial.pptx`

The slide deck is generated from these scripts so it can be re-edited and rebuilt
reproducibly (no manual PowerPoint fiddling). Edit the text in `build_deck.py`, rerun,
and the `.pptx` is regenerated.

## Files
- `build_deck.py` — the deck **content**: every slide's title, caption, bullets, table,
  and speaker notes, in reading order. This is the file you edit to change wording.
- `deck_engine.py` — the **design system**: 16:9 layout, palette (navy / blue / teal),
  Arial, and the slide constructors (`add_title`, `add_divider`, `add_content`,
  `add_image`, `add_two_column`, `add_table`, speaker `set_notes`). Edit for look-and-feel.
- `figures/` — the diagrams and terminal mockups embedded in the deck (the source of
  truth for the figures; the deck reuses these unchanged).
- `render_preview.py` — renders a `.pptx` to PNGs **without LibreOffice** (uses Pillow),
  for quickly checking layout/overflow. Handy on a cluster with no Office install.

## Build
```bash
# from this folder, with python-pptx + Pillow available (the workshop's AI conda env has both)
python build_deck.py figures ../claude-code-tutorial.pptx
```

## Preview (optional, no LibreOffice needed)
```bash
python render_preview.py ../claude-code-tutorial.pptx preview   # writes preview/slideNN.png + a contact sheet
```

## Notes
- The deck is **self-contained**: figures are embedded into the `.pptx` at build time,
  so the shipped file needs nothing from this folder.
- Design intent: accessible intro for a general research audience — day-to-day use first,
  then scaling up; ≤3 bullets per slide; captions complement (never re-narrate) the
  figures; HPC operational detail lives in the **speaker notes**.
- This folder is convenience tooling, not part of the workshop itself — safe to remove
  if you'd rather ship only the `.pptx`.
