# Deck source — `claude-code-tutorial.pptx`

The slide deck is generated from these scripts so it can be re-edited and rebuilt
reproducibly (no manual PowerPoint fiddling). Edit the text in `build_deck.py`, rerun,
and the `.pptx` is regenerated.

## Files
- `build_deck.py` — the deck **content**: every slide's title, caption, bullets, table,
  and speaker notes, in reading order. This is the file you edit to change wording.
- `deck_engine.py` — the **design system**: 16:9 layout, palette (navy / blue / teal),
  Arial, and the slide constructors (`add_title`, `add_agenda`, `add_divider`,
  `add_content`, `add_image`, `add_two_column`, `add_table`, `add_references`, and the
  `set_notes` / `source_tag` helpers). Edit for look-and-feel.
- `make_figures.py` — regenerates the three **generated** diagrams (`image3`, `image4`,
  `image13`) with matplotlib, so they track the content and the palette. The remaining
  figures are static, hand-built assets.
- `figures/` — the diagrams and terminal mockups embedded in the deck. Every file here
  is used by exactly one slide.
- `render_preview.py` — renders a `.pptx` to PNGs **without LibreOffice** (uses Pillow),
  for quickly checking layout/overflow. Handy on a cluster with no Office install.

## Build
```bash
# from this folder, with python-pptx + Pillow + matplotlib (the workshop's AI conda env has all three)
python make_figures.py figures                          # regenerate the generated diagrams
python build_deck.py figures ../claude-code-tutorial.pptx
```

## Preview (optional, no LibreOffice needed)
```bash
python render_preview.py ../claude-code-tutorial.pptx preview   # writes preview/slideNN.png + a contact sheet
```

## Notes
- The deck is **self-contained**: figures are embedded into the `.pptx` at build time,
  so the shipped file needs nothing from this folder.
- Design intent: an accessible, formal intro for a general research audience. Declarative
  slide titles (never opening with What / How / When); ≤3 bullets per slide, each reading
  as a sentence; clean section dividers with a progress bar (no oversized letters);
  captions complement (never re-narrate) the figures; concise, factual speaker notes with
  citations; a References section closes the deck.
- This folder is convenience tooling, not part of the workshop itself — safe to remove
  if you'd rather ship only the `.pptx`.
