#!/usr/bin/env python3
"""Design-system engine for the 'Claude Code on Midway' deck.

Clean, consistent, accessible. 16:9. Arial. Three-accent palette:
  navy (title), blue (structure / bullet lead-ins), teal (accents / inline code).
Slide constructors: title, divider, content (bullets), image, table, two-column.
Content spec lives in build_deck.py; this file is just the renderer.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from PIL import Image

# ---- palette -------------------------------------------------------------
NAVY  = RGBColor(0x0B, 0x2E, 0x6B)   # title text
BLUE  = RGBColor(0x11, 0x55, 0xCC)   # section letters, bullet lead-ins
TEAL  = RGBColor(0x00, 0x74, 0x82)   # accent bars, inline code, table header
INK   = RGBColor(0x20, 0x21, 0x24)   # body text
GRAY  = RGBColor(0x59, 0x59, 0x59)   # captions
FAINT = RGBColor(0x9A, 0xA0, 0xA6)   # page numbers
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ROWALT = RGBColor(0xEE, 0xF2, 0xFB)  # table zebra
ROWHDR = TEAL
DIVIDER_BG = RGBColor(0xF7, 0xF9, 0xFC)

FONT = "Arial"
FONT_MONO = "Arial"   # inline code stays Arial, colored teal (clean, no font-dep)

# ---- geometry (inches) ---------------------------------------------------
SW, SH = 10.0, 5.625
M = 0.55                      # side margin
TITLE_Y = 0.46
UNDERLINE_Y = 1.12
CAPTION_Y = 1.30
BODY_TOP = 1.95               # content region top (with caption)
BODY_TOP_NOCAP = 1.55         # content region top (no caption)
BODY_BOTTOM = 5.18            # above page number


def new_deck():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    return prs


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _txt(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    return tb, tf


def _run(p, text, size, color=INK, bold=False, italic=False, mono=False):
    r = p.add_run()
    r.text = text
    r.font.name = FONT_MONO if mono else FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return r


def _rich(p, text, size, color=INK, bold=False, italic=False, code_color=TEAL):
    """Inline markup: `code` -> teal code run; *emphasis* -> italic. Backticks are
    parsed first so a '*' inside a code span stays literal."""
    for i, seg in enumerate(text.split("`")):
        if seg == "":
            continue
        if i % 2 == 1:
            _run(p, seg, size, code_color, bold=bold, italic=False, mono=True)
        else:
            for j, sub in enumerate(seg.split("*")):
                if sub == "":
                    continue
                _run(p, sub, size, color, bold=bold, italic=(italic or j % 2 == 1))


def _accent_bar(slide, color, x=M, y=UNDERLINE_Y, w=1.35, h=0.045):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def _page_number(slide, n):
    tb, tf = _txt(slide, SW - 1.0, BODY_BOTTOM, 0.6, 0.3, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    _run(p, str(n), 9, FAINT)


def _title_block(slide, title, caption, accent=TEAL):
    tb, tf = _txt(slide, M, TITLE_Y, SW - 2 * M, 0.7)
    p = tf.paragraphs[0]
    _run(p, title, 23, INK, bold=True)
    _accent_bar(slide, accent)
    if caption:
        tb2, tf2 = _txt(slide, M, CAPTION_Y, SW - 2 * M, 0.55)
        p2 = tf2.paragraphs[0]
        _rich(p2, caption, 12.5, GRAY)
        return BODY_TOP
    return BODY_TOP_NOCAP


# ---- slide types ---------------------------------------------------------

def add_title(prs, kicker, title, subtitle, byline):
    s = _blank(prs)
    tb, tf = _txt(s, M, 0.42, SW - 2 * M, 0.4)
    _run(tf.paragraphs[0], kicker, 12, TEAL, bold=True)
    tb, tf = _txt(s, M, 1.55, SW - 2 * M, 1.5, anchor=MSO_ANCHOR.TOP)
    _run(tf.paragraphs[0], title, 40, NAVY, bold=True)
    tb, tf = _txt(s, M, 3.35, SW - 2 * M, 1.2)
    _rich(tf.paragraphs[0], subtitle, 16, GRAY)
    p = tf.add_paragraph(); p.space_before = Pt(14)
    _run(p, byline, 13, GRAY)
    _accent_bar(s, TEAL, x=M, y=3.15, w=1.6, h=0.05)
    return s


def add_divider(prs, letter, title, sub, accent=BLUE):
    s = _blank(prs)
    # soft full-bleed background
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = DIVIDER_BG
    bg.line.fill.background(); bg.shadow.inherit = False
    # left accent bar
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), prs.slide_height)
    bar.fill.solid(); bar.fill.fore_color.rgb = accent
    bar.line.fill.background(); bar.shadow.inherit = False
    # big letter + title share a vertical center line
    tb, tf = _txt(s, 0.72, 1.55, 1.75, 1.45, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    _run(p, letter, 64, accent, bold=True)
    tb, tf = _txt(s, 2.55, 1.55, 6.9, 1.45, anchor=MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], title, 30, INK, bold=True)
    # sub
    tb, tf = _txt(s, 2.58, 3.15, 6.9, 1.0)
    _rich(tf.paragraphs[0], sub, 14, GRAY)
    return s


def add_content(prs, title, caption, bullets, accent=TEAL, body_size=15.5, gap=13):
    """bullets: list of (lead, body) ; body may contain `code` spans.
    If lead is '' the paragraph is a plain rich line."""
    s = _blank(prs)
    top = _title_block(s, title, caption, accent)
    tb, tf = _txt(s, M, top, SW - 2 * M, BODY_BOTTOM - top, anchor=MSO_ANCHOR.MIDDLE)
    first = True
    for lead, body in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = 1.06
        if lead:
            _run(p, lead, body_size, BLUE, bold=True)
            if body:
                _run(p, "  —  ", body_size, GRAY)
        if body:
            _rich(p, body, body_size, INK)
    return s


def add_two_column(prs, title, caption, left, right, accent=TEAL):
    """left/right: dict(head=, head_color=, lines=[str with `code`])"""
    s = _blank(prs)
    top = _title_block(s, title, caption, accent)
    colw = (SW - 2 * M - 0.5) / 2
    for i, col in enumerate((left, right)):
        x = M + i * (colw + 0.5)
        tb, tf = _txt(s, x, top, colw, BODY_BOTTOM - top)
        p = tf.paragraphs[0]
        _rich(p, col["head"], 15, col.get("head_color", BLUE), bold=True)
        p.space_after = Pt(9)
        for ln in col["lines"]:
            pp = tf.add_paragraph(); pp.space_after = Pt(7); pp.line_spacing = 1.05
            _rich(pp, ln, 13.5, INK)
    return s


def _fit_box(img_path, bx, by, bw, bh):
    im = Image.open(img_path)
    iw, ih = im.size
    scale = min(bw / iw, bh / ih)
    w = iw * scale; h = ih * scale
    x = bx + (bw - w) / 2
    y = by + (bh - h) / 2
    return x, y, w, h


IMG_BOTTOM = 5.32          # image region reaches lower than text region


def add_image(prs, title, caption, img_path, accent=TEAL, pad_top=0.06,
              side_margin=0.35):
    s = _blank(prs)
    # tighter caption region so the figure gets more room
    top = _title_block(s, title, caption, accent)
    if caption:
        top -= 0.12
    bx, by = side_margin, top + pad_top
    bw, bh = SW - 2 * side_margin, IMG_BOTTOM - by
    x, y, w, h = _fit_box(img_path, bx, by, bw, bh)
    s.shapes.add_picture(img_path, Inches(x), Inches(y), Inches(w), Inches(h))
    return s


def _set_cell(cell, text, size, color=INK, bold=False, align=PP_ALIGN.LEFT,
              fill=None, mono=False):
    cell.margin_left = Inches(0.12); cell.margin_right = Inches(0.08)
    cell.margin_top = Inches(0.04); cell.margin_bottom = Inches(0.04)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill is not None:
        cell.fill.solid(); cell.fill.fore_color.rgb = fill
    else:
        cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
    tf = cell.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    _rich(p, text, size, color, bold=bold)


def _strip_table_style(tbl):
    # remove default banding style so our fills show cleanly
    el = tbl._tbl
    for pr in el.findall(qn('a:tblPr')):
        pr.set('firstRow', '0'); pr.set('bandRow', '0')


def add_table(prs, title, caption, headers, rows, colw, accent=TEAL,
              hdr_size=13, body_size=12.5, row_h=0.52):
    s = _blank(prs)
    top = _title_block(s, title, caption, accent)
    ncol = len(headers); nrow = len(rows) + 1
    total_w = SW - 2 * M
    scale = total_w / sum(colw)
    widths = [c * scale for c in colw]
    tblh = row_h * nrow
    by = top + 0.15
    gtf = s.shapes.add_table(nrow, ncol, Inches(M), Inches(by),
                             Inches(total_w), Inches(tblh))
    tbl = gtf.table
    _strip_table_style(tbl)
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)
    tbl.rows[0].height = Inches(row_h)
    for i, htxt in enumerate(headers):
        _set_cell(tbl.cell(0, i), htxt, hdr_size, WHITE, bold=True,
                  fill=accent, align=PP_ALIGN.LEFT)
    for r, row in enumerate(rows, start=1):
        tbl.rows[r].height = Inches(row_h)
        fill = ROWALT if (r % 2 == 0) else WHITE
        for c, val in enumerate(row):
            bold = (c == 0)
            _set_cell(tbl.cell(r, c), val, body_size, INK, bold=bold, fill=fill)
    return s


def set_notes(slide, text):
    """Attach speaker notes (holds detail moved off the slide to declutter)."""
    if text:
        slide.notes_slide.notes_text_frame.text = text
    return slide


def caption_line(slide, text, y=None, accent=None):
    """Add a small caveat line just below a table/figure (e.g. Bypass-mode note)."""
    tb, tf = _txt(slide, M, y if y is not None else 4.55, SW - 2 * M, 0.5)
    p = tf.paragraphs[0]
    _rich(p, text, 11, GRAY, italic=True)
    return slide


def finalize(prs, out_path, skip_numbers=()):
    for i, slide in enumerate(prs.slides, 1):
        if i in skip_numbers:
            continue
        _page_number(slide, i)
    prs.save(out_path)
