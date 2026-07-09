# Hands-on lab — everyday research with Claude Code

This is the interactive half of the workshop. There is **no notebook to run**. Instead you
open **Claude Code** inside a small, realistic research project and work through a series of
tasks by pasting the prompts on each task card. It is the same thing you will do on your own
work on Monday.

## The project: LakeWatch
You have inherited **LakeWatch**, a small water-quality monitoring project (`project/`). Like
most real projects, it is a little messy: three sensor exports with mismatched columns, units,
and date formats; a bug with a failing test; half-finished documentation; and a folder of
free-text field notes. Over six short tasks you will use Claude Code to clean it up, fix it,
document it, and mine the notes — then, in a bonus, scale one command into a batch.

## Before you start
- **Claude Code installed and logged in** — `claude --version` prints a 2.x build (see the
  top-level `../README.md` for the no-sudo install and login).
- **Python** available — `python -m pytest` should work (the RCC `AI` env has it).
- **On Midway3**, work on a node with internet egress (a login node, or the `test`/`caslake`
  partitions). The bonus batch task needs the network; the interactive tasks do too.

## The tasks

| # | Task | What you practice | Works on |
|---|------|-------------------|----------|
| 0 | [Set up](tasks/00-setup.md) | copy to scratch, start `claude` | — |
| 1 | [Get your bearings](tasks/01-orient.md) | project exploration · project memory (`CLAUDE.md`) | whole repo |
| 2 | [Tidy the messy data](tasks/02-clean-data.md) | Plan → Accept-edits · raw data stays read-only | `data/raw/*.csv` → `data/clean/` |
| 3 | [Fix the bug — safely](tasks/03-fix-the-bug.md) | guarded autonomous fix · verify it yourself | `src/waterquality.py`, `tests/` |
| 4 | [Document the project](tasks/04-document.md) | writing around research · you are the reviewer | `docs/data_dictionary.md`, `src/` |
| 5 | [Turn notes into data](tasks/05-extract-from-notes.md) | structured extraction from documents | `notes/*.md` → `data/clean/` |
| 6 | [Automate it](tasks/06-automate.md) | slash commands · Agent Skills | `.claude/commands/`, `.claude/skills/` |
| A | [Bonus: scale to a batch](tasks/07-bonus-batch.md) | headless `claude -p` · Slurm | `logs/`, `classify_logs.sh`, `run.sh` |
| B | [Bonus: your own tools (MCP)](tasks/08-bonus-mcp.md) | Model Context Protocol | `mcp_server.py` |
| ★ | [Take-home: your own project](tasks/09-your-project.md) | a safe first session on your code | your repo |

Work top to bottom — each task leaves the project in the state the next one expects. The core
(0–6) is about 45 minutes; the bonuses are optional.

## A word on safety
You will run **Accept-edits** mode, which lets the agent change files without asking each time.
That is why Task 0 has you work on a **git-clean scratch copy**. The project ships a
`.claude/settings.json` that keeps raw data and the tests read-only and blocks common secret
paths — a **deny rule always wins**, even if you (or the model) ask otherwise. Never use
`--dangerously-skip-permissions` on a shared filesystem.

Everything in `project/` is also a **template** to lift into your own repository:
`CLAUDE.md`, `.claude/settings.json`, the slash commands, and the skills.
