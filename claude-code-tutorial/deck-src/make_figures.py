#!/usr/bin/env python3
"""Regenerate the diagrams that must stay in sync with the deck's content.

These three are generated here so they can be re-edited reproducibly:
  - image3.png  — "two parts: the harness and the model" (title/subtitle placed
                  clearly above the box, so no border ever crosses text);
  - image4.png  — "everything around the model is yours to shape" (no stale
                  notebook section numbers);
  - image13.png — "slash commands: a saved prompt file" (bullets padded so no
                  text touches the box border).

The remaining figures in figures/ are static, hand-built assets (terminal
mockups and the other diagrams) and are reused unchanged.

Usage:  python make_figures.py [figures_dir]
Requires matplotlib. Palette matches deck_engine.py.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# palette (matches deck_engine)
BLUE = "#1155CC"
TEAL = "#007482"
ORANGE = "#E08000"
GRAY = "#59595F"
INK = "#1E2126"
DARK = "#20242B"
LIGHT = {"blue": "#EAF1FE", "teal": "#E4F3F4", "orange": "#FDF0DC", "gray": "#EEF0F3"}
EDGE = {"blue": BLUE, "teal": TEAL, "orange": ORANGE, "gray": "#8A9099"}
TXT = {"blue": BLUE, "teal": TEAL, "orange": ORANGE, "gray": GRAY}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "DejaVu Sans", "Arial"]


def _box(ax, x, y, w, h, kind, lw=2.4, radius=0.12, fill=None, edge=None):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         linewidth=lw, edgecolor=edge or EDGE[kind],
                         facecolor=fill or LIGHT[kind], zorder=3)
    ax.add_patch(box)


def _card(ax, cx, cy, w, h, kind, title, sub, tsize=15, ssize=12):
    _box(ax, cx - w / 2, cy - h / 2, w, h, kind)
    ax.text(cx, cy + 0.15, title, ha="center", va="center", color=TXT[kind],
            fontsize=tsize, fontweight="bold", zorder=4)
    ax.text(cx, cy - 0.18, sub, ha="center", va="center", color=GRAY,
            fontsize=ssize, zorder=4)


# --------------------------------------------------------------------------- image4
def build_map(out_path):
    fig, ax = plt.subplots(figsize=(11.2, 6.0), dpi=200)
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 6.0); ax.axis("off")
    cx, cy, R = 5.6, 3.0, 3.55
    spokes = [
        (0.00,  1.0, "blue",   "CLAUDE.md",      "project memory"),
        (0.72,  0.7, "blue",   "slash commands", "saved prompts"),
        (1.0,   0.0, "teal",   "skills",         "model-invoked"),
        (0.72, -0.7, "teal",   "subagents",      "scoped helpers"),
        (0.00, -1.0, "orange", "hooks",          "rules it can't skip"),
        (-0.72,-0.7, "orange", "permissions",    "what it may do"),
        (-1.0,  0.0, "gray",   "MCP",            "your own tools"),
        (-0.72, 0.7, "gray",   "sessions",       "resume / branch"),
    ]
    bw, bh = 2.35, 0.95
    for dx, dy, kind, title, sub in spokes:
        bx, by = cx + dx * R * 0.98, cy + dy * (R * 0.62)
        ax.plot([cx, bx], [cy, by], color="#B8BEC8", linewidth=1.6, zorder=1)
        _card(ax, bx, by, bw, bh, kind, title, sub, tsize=16, ssize=13)
    _box(ax, cx - 1.15, cy - 0.62, 2.3, 1.24, "gray", lw=0, radius=0.16, fill=DARK, edge=DARK)
    ax.text(cx, cy + 0.16, "claude", ha="center", va="center", color="white",
            fontsize=20, fontweight="bold", family="monospace", zorder=6)
    ax.text(cx, cy - 0.24, "the agent loop", ha="center", va="center",
            color="#C7CCD4", fontsize=12.5, zorder=6)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print("wrote", out_path)


# --------------------------------------------------------------------------- image3
def build_harness(out_path):
    fig, ax = plt.subplots(figsize=(11.4, 4.7), dpi=200)
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 4.7); ax.axis("off")

    # ----- left: the harness -----
    lx0, lx1, lb, lt = 0.3, 5.15, 0.55, 3.35
    # title + subtitle ABOVE the box (clear gap, so no border crosses text)
    ax.text((lx0 + lx1) / 2, 4.42, "Your machine — the harness", ha="center",
            va="center", color=BLUE, fontsize=17, fontweight="bold")
    ax.text((lx0 + lx1) / 2, 4.02, "the claude program — this is what you configure",
            ha="center", va="center", color=GRAY, fontsize=12, style="italic")
    _box(ax, lx0, lb, lx1 - lx0, lt - lb, "blue")
    # 3 x 2 grid of capability cards
    cols = [lx0 + 1.05, lx0 + 2.42, lx0 + 3.79]
    rows = [2.62, 1.35]
    cells = [
        ("gray", "your files", "& repo"),
        ("teal", "tools", "Read·Edit·Bash"),
        ("orange", "permissions", "& hooks"),
        ("blue", "CLAUDE.md", "memory"),
        ("teal", "skills &", "subagents"),
        ("gray", "MCP", "servers"),
    ]
    for i, (kind, t, s) in enumerate(cells):
        cx, cy = cols[i % 3], rows[i // 3]
        _box(ax, cx - 0.63, cy - 0.42, 1.26, 0.84, kind, lw=2.0, radius=0.1)
        ax.text(cx, cy + 0.13, t, ha="center", va="center", color=TXT[kind],
                fontsize=11.5, fontweight="bold")
        ax.text(cx, cy - 0.16, s, ha="center", va="center", color=INK, fontsize=10.5)

    # ----- right: the model -----
    rx0, rx1, rb, rt = 6.85, 11.0, 1.05, 3.05
    _box(ax, rx0, rb, rx1 - rx0, rt - rb, "teal")
    rcx = (rx0 + rx1) / 2
    ax.text(rcx, 2.78, "Anthropic", ha="center", va="center", color=GRAY, fontsize=12)
    ax.text(rcx, 2.36, "the model", ha="center", va="center", color=TEAL,
            fontsize=19, fontweight="bold")
    ax.text(rcx, 1.92, "Claude — Opus · Sonnet · Haiku", ha="center", va="center",
            color=INK, fontsize=12.5)
    ax.text(rcx, 1.45, "you don't touch this", ha="center", va="center",
            color=GRAY, fontsize=12, style="italic")

    # ----- arrows between (labels kept clear of the arrow lines) -----
    ax.annotate("", xy=(rx0 - 0.05, 2.45), xytext=(lx1 + 0.05, 2.45),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
    ax.text((lx1 + rx0) / 2, 2.92, "context +", ha="center", va="center", color=BLUE, fontsize=10.5)
    ax.text((lx1 + rx0) / 2, 2.70, "tool results", ha="center", va="center", color=BLUE, fontsize=10.5)
    ax.annotate("", xy=(lx1 + 0.05, 1.55), xytext=(rx0 - 0.05, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2))
    ax.text((lx1 + rx0) / 2, 1.30, "next action", ha="center", va="center", color=TEAL, fontsize=10.5)
    ax.text((lx1 + rx0) / 2, 1.08, "(a tool call)", ha="center", va="center", color=TEAL, fontsize=10.5)

    ax.text((rx0 + rx1) / 2, 0.55, "api.anthropic.com — needs network egress",
            ha="center", va="center", color=GRAY, fontsize=11.5)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print("wrote", out_path)


# --------------------------------------------------------------------------- image13
def build_slash(out_path):
    fig, ax = plt.subplots(figsize=(11.4, 4.6), dpi=200)
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 4.6); ax.axis("off")

    # box 1 — what you type
    _box(ax, 0.3, 1.9, 3.0, 1.5, "blue")
    ax.text(1.8, 2.9, "/add-test rolling", ha="center", va="center", color=BLUE,
            fontsize=15, fontweight="bold", family="monospace")
    ax.text(1.8, 2.35, "what you type", ha="center", va="center", color=GRAY, fontsize=12)

    # arrow
    ax.annotate("", xy=(4.05, 2.65), xytext=(3.45, 2.65),
                arrowprops=dict(arrowstyle="-|>", color="#8A9099", lw=2.2))

    # box 2 — the saved prompt file
    bx0, bx1 = 4.2, 8.05
    _box(ax, bx0, 0.85, bx1 - bx0, 3.35, "gray")
    ax.text((bx0 + bx1) / 2, 3.78, "a saved prompt file", ha="center", va="center",
            color=INK, fontsize=15, fontweight="bold")
    ax.text((bx0 + bx1) / 2, 3.4, ".claude/commands/add-test.md", ha="center",
            va="center", color=GRAY, fontsize=11.5, family="monospace")
    bullets = [
        "checked into git — shared by the lab",
        "frontmatter scopes its tools",
        "$ARGUMENTS, !cmd, @file expand inline",
    ]
    for i, b in enumerate(bullets):
        ax.text(bx0 + 0.28, 2.75 - i * 0.6, "•", ha="left", va="center",
                color="#8A9099", fontsize=13)
        ax.text(bx0 + 0.55, 2.75 - i * 0.6, b, ha="left", va="center",
                color=INK, fontsize=11.5)

    # arrow
    ax.annotate("", xy=(8.8, 2.65), xytext=(8.2, 2.65),
                arrowprops=dict(arrowstyle="-|>", color="#8A9099", lw=2.2))

    # box 3 — the agent runs it
    _box(ax, 8.95, 1.9, 2.15, 1.5, "teal")
    ax.text(10.02, 2.9, "the agent runs it", ha="center", va="center", color=TEAL,
            fontsize=14, fontweight="bold")
    ax.text(10.02, 2.35, "one reusable turn", ha="center", va="center", color=GRAY,
            fontsize=11.5, style="italic")

    ax.text(5.7, 0.42, "A slash command is a prompt you fire on demand — versioned, shareable, tool-scoped.",
            ha="center", va="center", color=GRAY, fontsize=12, style="italic")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print("wrote", out_path)


# --------------------------------------------------------------------------- image2
def build_loop(out_path):
    fig, ax = plt.subplots(figsize=(11.4, 5.3), dpi=200)
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 5.3); ax.axis("off")
    cx, cy = 5.7, 2.75
    bw, bh = 2.7, 1.12
    nodes = {  # (x, y, kind, title, sub)
        "top":    (5.7, 4.35, "blue",   "1 · Gather context", "read and search your files"),
        "right":  (9.05, 2.75, "gray",  "2 · Decide",         "the model plans the next step"),
        "bottom": (5.7, 1.15, "teal",   "3 · Act",            "edit a file, run a command"),
        "left":   (2.35, 2.75, "orange", "4 · Check",         "did it actually work?"),
    }
    order = ["top", "right", "bottom", "left"]
    # clockwise curved arrows between consecutive nodes
    for i in range(4):
        x0, y0 = nodes[order[i]][0], nodes[order[i]][1]
        x1, y1 = nodes[order[(i + 1) % 4]][0], nodes[order[(i + 1) % 4]][1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#9AA0A6", lw=2.2,
                                    shrinkA=52, shrinkB=52,
                                    connectionstyle="arc3,rad=-0.28"))
    for (x, y, kind, title, sub) in nodes.values():
        _box(ax, x - bw / 2, y - bh / 2, bw, bh, kind, lw=2.2, radius=0.12)
        ax.text(x, y + 0.18, title, ha="center", va="center", color=TXT[kind],
                fontsize=14.5, fontweight="bold")
        ax.text(x, y - 0.19, sub, ha="center", va="center", color=INK, fontsize=11.5)
    ax.text(cx, cy + 0.14, "the loop", ha="center", va="center", color=INK,
            fontsize=15, fontweight="bold")
    ax.text(cx, cy - 0.2, "until the goal is met", ha="center", va="center",
            color=GRAY, fontsize=12, style="italic")
    ax.text(cx, 0.22, "Every action in step 3 asks your permission first.",
            ha="center", va="center", color=TEAL, fontsize=12, style="italic")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    figures = sys.argv[1] if len(sys.argv) > 1 else "figures"
    build_loop(f"{figures}/image2.png")
    build_harness(f"{figures}/image3.png")
    build_map(f"{figures}/image4.png")
    build_slash(f"{figures}/image13.png")
