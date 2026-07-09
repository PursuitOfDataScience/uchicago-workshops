#!/usr/bin/env python3
"""Design-system engine for the 'Claude Code on Midway' deck.

Clean, accessible, formal. 16:9. Arial. A restrained three-accent palette:
  navy (titles), blue (structure / lead-ins), teal (accents / inline code).

Layout is engineered so nothing ever overlaps: the title sits in its own band, the
accent underline is placed with a clear gap below a single-line title, and body text
lives in a measured region above the footer. Bullets read as sentences (a bold lead
followed by ordinary text) — no dash-joined fragments.

Content lives in build_deck.py; this file is only the renderer.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from PIL import Image

# ---- palette -------------------------------------------------------------
NAVY  = RGBColor(0x0B, 0x2E, 0x6B)   # title / hero text
BLUE  = RGBColor(0x11, 0x55, 0xCC)   # bullet lead-ins, structure
TEAL  = RGBColor(0x00, 0x74, 0x82)   # accent bars, inline code, table header
INK   = RGBColor(0x1E, 0x21, 0x26)   # body text
GRAY  = RGBColor(0x55, 0x5A, 0x61)   # captions / secondary
FAINT = RGBColor(0x98, 0x9E, 0xA6)   # page numbers, rail-inactive
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ROWALT = RGBColor(0xEE, 0xF2, 0xFB)  # table zebra
DIVIDER_BG = RGBColor(0xF6, 0xF8, 0xFC)
RAIL_TRACK = RGBColor(0xDD, 0xE3, 0xEC)

FONT = "Arial"
FONT_MONO = "Arial"   # inline code stays Arial, colored teal — no font dependency

# ---- geometry (inches, 16:9) ---------------------------------------------
SW, SH = 10.0, 5.625
M = 0.6                        # side margin
TITLE_Y = 0.5
BAR_Y = 1.08                   # accent underline — clear gap below a single-line title
CAPTION_Y = 1.24
BODY_TOP = 1.92                # content region top (with caption)
BODY_TOP_NOCAP = 1.55          # content region top (no caption)
BODY_BOTTOM = 5.12             # above the page-number footer
IMG_BOTTOM = 5.28


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


def _run(p, text, size, color=INK, bold=False, italic=False, mono=False, spacing=None):
    r = p.add_run()
    r.text = text
    r.font.name = FONT_MONO if mono else FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    if spacing is not None:
        r.font._rPr.set("spc", str(int(spacing * 100)))  # letter-spacing in points
    return r


def _rich(p, text, size, color=INK, bold=False, italic=False, code_color=TEAL):
    """Inline markup: `code` -> teal code run; *emphasis* -> italic. Backticks parse
    first so a '*' inside a code span stays literal."""
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


def _hang(p, marL_in=0.3):
    """Give a bulleted paragraph a hanging indent so wrapped lines align under the
    lead text, not back at the bullet glyph."""
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', str(Inches(marL_in)))
    pPr.set('indent', str(Inches(-marL_in)))


def _rect(slide, x, y, w, h, color, line=None):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
    sp.shadow.inherit = False
    return sp


def _page_number(slide, n):
    tb, tf = _txt(slide, SW - 1.0, BODY_BOTTOM + 0.12, 0.6, 0.3, anchor=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    _run(p, str(n), 9, FAINT)


def _title_block(slide, title, caption, accent=TEAL):
    tb, tf = _txt(slide, M, TITLE_Y, SW - 2 * M, 0.62)
    _run(tf.paragraphs[0], title, 22, INK, bold=True)
    _rect(slide, M, BAR_Y, 1.15, 0.045, accent)
    if caption:
        tb2, tf2 = _txt(slide, M, CAPTION_Y, SW - 2 * M, 0.55)
        _rich(tf2.paragraphs[0], caption, 12.5, GRAY)
        return BODY_TOP
    return BODY_TOP_NOCAP


# ---- slide types ---------------------------------------------------------

def add_title(prs, kicker, title, subtitle, byline):
    s = _blank(prs)
    tb, tf = _txt(s, M, 0.62, SW - 2 * M, 0.4)
    _run(tf.paragraphs[0], kicker, 12, TEAL, bold=True, spacing=1.2)
    _rect(s, M, 3.02, 1.7, 0.06, TEAL)
    tb, tf = _txt(s, M, 1.7, SW - 2 * M, 1.3, anchor=MSO_ANCHOR.TOP)
    _run(tf.paragraphs[0], title, 40, NAVY, bold=True)
    tb, tf = _txt(s, M, 3.25, SW - 2 * M, 1.4)
    _rich(tf.paragraphs[0], subtitle, 16, GRAY)
    p = tf.add_paragraph(); p.space_before = Pt(16)
    _run(p, byline, 12.5, GRAY)
    return s


def add_agenda(prs, title, items, accent=TEAL, note=None):
    """items: list of (title, gloss). Rendered as a clean numbered list — no dashes."""
    s = _blank(prs)
    top = _title_block(s, title, None, accent)
    n = len(items)
    row_h = (BODY_BOTTOM - top - (0.5 if note else 0)) / n
    for i, (t, gloss) in enumerate(items):
        y = top + i * row_h
        # number
        tb, tf = _txt(s, M, y, 0.5, row_h, anchor=MSO_ANCHOR.MIDDLE)
        _run(tf.paragraphs[0], str(i + 1), 17, accent, bold=True)
        # title + gloss
        tb, tf = _txt(s, M + 0.52, y, SW - 2 * M - 0.52, row_h, anchor=MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        _run(p, t, 15.5, INK, bold=True)
        _run(p, "     ", 15.5, GRAY)
        _rich(p, gloss, 13.5, GRAY)
    if note:
        tb, tf = _txt(s, M, BODY_BOTTOM - 0.34, SW - 2 * M, 0.4)
        _rich(tf.paragraphs[0], note, 12, GRAY, italic=True)
    return s


def add_divider(prs, part_no, part_total, title, sub, accent=BLUE, kicker=None, frac=None):
    """A clean section divider: kicker + big title + subtitle + a slim progress bar.
    No oversized letters — the progress bar shows where you are in the arc.
    Pass `kicker` to override the auto 'PART n OF m' label; `frac` to set the bar fill."""
    s = _blank(prs)
    _rect(s, 0, 0, SW, SH, DIVIDER_BG)
    _rect(s, 0, 0, 0.22, SH, accent)
    kick = kicker if kicker is not None else (f"PART {part_no} OF {part_total}" if part_no else "")
    tb, tf = _txt(s, 1.0, 2.05, SW - 1.6, 0.4)
    _run(tf.paragraphs[0], kick, 12.5, accent, bold=True, spacing=1.6)
    tb, tf = _txt(s, 1.0, 2.5, SW - 1.6, 0.9)
    _run(tf.paragraphs[0], title, 30, INK, bold=True)
    tb, tf = _txt(s, 1.0, 3.42, SW - 1.6, 0.7)
    _rich(tf.paragraphs[0], sub, 14, GRAY)
    bar_x, bar_w, bar_y = 1.0, SW - 2.0, 4.5
    _rect(s, bar_x, bar_y, bar_w, 0.06, RAIL_TRACK)
    f = frac if frac is not None else ((part_no / part_total) if part_no else 1.0)
    _rect(s, bar_x, bar_y, bar_w * max(0.02, f), 0.06, accent)
    return s


def add_content(prs, title, caption, bullets, accent=TEAL, body_size=15, gap=13, lead_gap=True):
    """bullets: list of (lead, body). Rendered as a bulleted sentence:
    '•  <bold lead> <body>'. If lead is '' the line is plain rich text.
    Bodies may contain `code` spans and *emphasis*."""
    s = _blank(prs)
    top = _title_block(s, title, caption, accent)
    tb, tf = _txt(s, M, top + 0.28, SW - 2 * M, BODY_BOTTOM - top - 0.28, anchor=MSO_ANCHOR.TOP)
    first = True
    for lead, body in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = 1.08
        _hang(p, 0.3)
        _run(p, "•  ", body_size, accent, bold=True)
        if lead:
            _rich(p, lead, body_size, BLUE, bold=True)   # _rich so `code` in a lead never shows literal backticks
            if body:
                _run(p, "  ", body_size, INK)
        if body:
            _rich(p, body, body_size, INK)
    return s


def add_two_column(prs, title, caption, left, right, accent=TEAL):
    """left/right: dict(head=, head_color=, lines=[str with `code`])."""
    s = _blank(prs)
    top = _title_block(s, title, caption, accent)
    gaptop = top + 0.05
    colw = (SW - 2 * M - 0.55) / 2
    for i, col in enumerate((left, right)):
        x = M + i * (colw + 0.55)
        # subtle header rule
        _rect(s, x, gaptop, colw, 0.5, RGBColor(0xF2, 0xF5, 0xFA))
        tb, tf = _txt(s, x + 0.14, gaptop, colw - 0.24, 0.5, anchor=MSO_ANCHOR.MIDDLE)
        _rich(tf.paragraphs[0], col["head"], 14.5, col.get("head_color", BLUE), bold=True)
        tb, tf = _txt(s, x + 0.14, gaptop + 0.62, colw - 0.24, BODY_BOTTOM - gaptop - 0.62)
        firstp = True
        for ln in col["lines"]:
            pp = tf.paragraphs[0] if firstp else tf.add_paragraph()
            firstp = False
            pp.space_after = Pt(8); pp.line_spacing = 1.06
            _hang(pp, 0.24)
            _run(pp, "•  ", 13, accent, bold=True)
            _rich(pp, ln, 13, INK)
    return s


def _fit_box(img_path, bx, by, bw, bh):
    im = Image.open(img_path)
    iw, ih = im.size
    scale = min(bw / iw, bh / ih)
    w, h = iw * scale, ih * scale
    return bx + (bw - w) / 2, by + (bh - h) / 2, w, h


def add_image(prs, title, caption, img_path, accent=TEAL, pad_top=0.08, side_margin=0.4, source=None):
    s = _blank(prs)
    top = _title_block(s, title, caption, accent)
    if caption:
        top -= 0.14
    bx, by = side_margin, top + pad_top
    bw, bh = SW - 2 * side_margin, IMG_BOTTOM - by - (0.16 if source else 0)
    x, y, w, h = _fit_box(img_path, bx, by, bw, bh)
    s.shapes.add_picture(img_path, Inches(x), Inches(y), Inches(w), Inches(h))
    if source:
        source_tag(s, source)
    return s


def _set_cell(cell, text, size, color=INK, bold=False, align=PP_ALIGN.LEFT, fill=None):
    cell.margin_left = Inches(0.14); cell.margin_right = Inches(0.1)
    cell.margin_top = Inches(0.05); cell.margin_bottom = Inches(0.05)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.fill.solid(); cell.fill.fore_color.rgb = fill if fill is not None else WHITE
    tf = cell.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    _rich(p, text, size, color, bold=bold)


def _strip_table_style(tbl):
    el = tbl._tbl
    for pr in el.findall(qn('a:tblPr')):
        pr.set('firstRow', '0'); pr.set('bandRow', '0')


def add_table(prs, title, caption, headers, rows, colw, accent=TEAL,
              hdr_size=13, body_size=12.5, row_h=0.52, source=None):
    s = _blank(prs)
    top = _title_block(s, title, caption, accent)
    ncol, nrow = len(headers), len(rows) + 1
    total_w = SW - 2 * M
    scale = total_w / sum(colw)
    widths = [c * scale for c in colw]
    by = top + 0.18
    gtf = s.shapes.add_table(nrow, ncol, Inches(M), Inches(by),
                             Inches(total_w), Inches(row_h * nrow))
    tbl = gtf.table
    _strip_table_style(tbl)
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)
    tbl.rows[0].height = Inches(row_h)
    for i, htxt in enumerate(headers):
        _set_cell(tbl.cell(0, i), htxt, hdr_size, WHITE, bold=True, fill=accent)
    for r, row in enumerate(rows, start=1):
        tbl.rows[r].height = Inches(row_h)
        fill = ROWALT if (r % 2 == 0) else WHITE
        for c, val in enumerate(row):
            _set_cell(tbl.cell(r, c), val, body_size, INK, bold=(c == 0), fill=fill)
    if source:
        source_tag(s, source)
    return s


def add_references(prs, title, refs, accent=TEAL, cols=2, size=10.5):
    """refs: list of 'Author/Title — Publisher — url' strings, pre-numbered by caller
    or numbered here. Two-column small list."""
    s = _blank(prs)
    top = _title_block(s, title, None, accent)
    per = (len(refs) + cols - 1) // cols
    colw = (SW - 2 * M - 0.4) / cols
    for c in range(cols):
        chunk = refs[c * per:(c + 1) * per]
        x = M + c * (colw + 0.4)
        tb, tf = _txt(s, x, top + 0.05, colw, BODY_BOTTOM - top)
        firstp = True
        for idx, ref in enumerate(chunk):
            pp = tf.paragraphs[0] if firstp else tf.add_paragraph()
            firstp = False
            pp.space_after = Pt(6); pp.line_spacing = 1.02
            n = c * per + idx + 1
            _run(pp, f"{n}. ", size, accent, bold=True)
            # split "Title — Publisher — url": title bold-ish, rest gray
            _rich(pp, ref, size, GRAY)
    return s


def set_notes(slide, text):
    """Attach concise, factual speaker notes (definitions, figures, citations)."""
    if text:
        slide.notes_slide.notes_text_frame.text = text
    return slide


def caption_line(slide, text, y=None, accent=None):
    tb, tf = _txt(slide, M, y if y is not None else 4.6, SW - 2 * M, 0.45)
    _rich(tf.paragraphs[0], text, 11, GRAY, italic=True)
    return slide


def source_tag(slide, text):
    """A small gray source attribution, bottom-left, for a claim-bearing slide."""
    tb, tf = _txt(slide, M, BODY_BOTTOM + 0.12, SW - 2 * M - 0.8, 0.3, anchor=MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], text, 9, FAINT, italic=True)
    return slide


def finalize(prs, out_path, skip_numbers=()):
    for i, slide in enumerate(prs.slides, 1):
        if i in skip_numbers:
            continue
        _page_number(slide, i)
    prs.save(out_path)
