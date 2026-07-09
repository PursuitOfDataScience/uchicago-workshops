# Hands-on: analyze a real dataset with Claude Code

This is the interactive half of the workshop. There is **no notebook, no subfolders, and
nothing to download** — just this folder of Markdown task cards, a `CLAUDE.md`, and an
`answers.py` answer key. You open Claude Code here and ask it to analyze a dataset hosted
online: it writes and runs the analysis and reports the results back, while you review them.
This is what using Claude Code for everyday data work actually feels like.

Each task card (`01`–`09`) holds **only the prompt** to paste. The reference solution for every
task — the Python code and its expected output — is in **`answers.py`**, so you can check
Claude's numbers against a known-correct answer.

## The dataset
**Palmer Penguins** — 344 field measurements across 3 species and 3 Antarctic islands, read
live from a URL (see `CLAUDE.md`). Nothing to download.

## Before you start
- **Claude Code installed and logged in** — `claude --version` prints a 2.x build (see the
  top-level [`../README.md`](../README.md) for the no-sudo install and login).
- **A node with internet access** — the analysis reads the dataset over the network. On
  Midway3 that is a login node, or the `test`/`caslake` partitions.
- **The `AI` environment** (it has pandas + matplotlib):
  `source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI`.

## Start
```bash
cd hands-on        # (or copy this folder somewhere writable and cd there)
claude
```
Then work the task cards in order, pasting each prompt. When Claude asks to **run a command**
(to load the data or save a plot), approve it — that approval prompt is the permission system
from the lecture, live.

## The tasks
| # | Task | You ask Claude to… |
|---|------|--------------------|
| 1 | [First look](01-first-look.md) | load the data from the URL and describe it |
| 2 | [Summary by species](02-summary-by-species.md) | group and summarize body mass & flipper length |
| 3 | [The heaviest species](03-heaviest-species.md) | compare species and make a judgment call |
| 4 | [Flipper length vs body mass](04-flipper-vs-mass.md) | test a relationship — overall and within groups |
| 5 | [Data quality](05-data-quality.md) | find missing / implausible values (without dropping them) |
| 6 | [Make a figure](06-make-a-figure.md) | write and run a plotting script → a saved figure |
| 7 | [Write it up](07-write-it-up.md) | turn the numbers into a Results paragraph |
| B | [Bonus: run it headless](08-headless.md) | get the same analysis as one `claude -p` command |
| ★ | [Take-home: your own data](09-your-own-data.md) | point it at a CSV URL of your own |

Tasks 1–7 are about 40 minutes. Each builds on the last, so go in order.

## The one rule: check its work
Claude reports numbers by running code you can read. On at least one task, ask it to **show
the code** and spot-check a number yourself. An agent can be confidently wrong — verify a
result before you cite it. When in doubt, compare against **`answers.py`**, the reference key.
