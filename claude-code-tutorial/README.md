# Claude Code on Midway

A hands-on workshop that introduces **Claude Code** — Anthropic's AI *coding agent* — as
everyday **research infrastructure** on a shared HPC cluster. You will install it on Midway,
use it interactively on a real (messy) research project, keep it safe on shared hardware, and
scale one command into an unattended batch job.

**Tool:** [Claude Code](https://code.claude.com/docs) v2.x &nbsp;·&nbsp; **No GPU** — this is a
CPU + internet workshop &nbsp;·&nbsp; **Model used in the lab:** `haiku` (fast, ~1¢ a call).

## The workshop has two halves

1. **A lecture** — `claude-code-tutorial.pptx` (~41 slides). An accessible, plain-language
   introduction for a general research audience: what an agent in the terminal *is*, getting
   started on Midway, making it yours, staying in control, scaling up, and honest limits.
2. **A hands-on lab** — `hands-on/`. There is **no notebook to run**. Instead you open Claude
   Code inside a small research project and work through a set of **task cards**, pasting the
   prompt on each card. It is exactly what you will do on your own work afterward.

```
claude-code-tutorial/
├── claude-code-tutorial.pptx     # the lecture deck (~41 slides) — the presentation half
├── deck-src/                     # scripts + figures that generate the deck (edit + rebuild)
│   ├── build_deck.py             #   slide content (edit wording here)
│   ├── deck_engine.py            #   the design system (layout, palette, slide types)
│   ├── make_figures.py           #   regenerates the diagrams that must track the content
│   ├── render_preview.py         #   render the .pptx to PNGs without LibreOffice
│   ├── figures/                  #   the embedded diagrams and terminal mockups
│   └── BUILD.md
├── hands-on/                     # the interactive lab — the second half
│   ├── README.md                 #   how to start + the task index
│   ├── project/                  #   "LakeWatch": a small, messy research repo you work on
│   └── tasks/                    #   the guided task cards (paste the prompts)
└── README.md                     # this file
```

## Getting started

### 1. Install Claude Code (no sudo needed)
Native installer (recommended — installs the `claude` binary to `~/.local/bin`, no root, no Node):
```bash
curl -fsSL https://claude.ai/install.sh | bash
```
Or via npm with a user prefix (needs a recent Node.js; **do not** use `sudo`):
```bash
npm config set prefix ~/.npm-global && npm install -g @anthropic-ai/claude-code
export PATH="$HOME/.npm-global/bin:$PATH"
```
Check: `claude --version` should print a **2.x** build (`claude doctor` diagnoses a bad install).

### 2. Authenticate (choose one)
- **Claude.ai subscription (Pro/Max) via OAuth.** On a headless cluster you cannot open a browser
  mid-SSH, so mint a long-lived token once and reuse it:
  ```bash
  claude setup-token                    # do the paste-URL flow once; prints a token
  export CLAUDE_CODE_OAUTH_TOKEN=...     # export the token (e.g. in your job script)
  ```
  (Interactively, `claude` then `/login` stores credentials under `$CLAUDE_CONFIG_DIR` instead.)
- **Console API key (pay-per-token)** — simplest for scripts and the batch exercise:
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...   # source from a chmod-600 file; never hard-code it
  ```
Credentials live under `$CLAUDE_CONFIG_DIR` (default `~/.claude`). Run `chmod 700 ~/.claude` —
cluster home directories are sometimes group-readable.

### 3. Network requirement (read this)
Claude Code needs **egress to `api.anthropic.com`**. On **Midway3**, **login nodes and the
`test`/`caslake` partitions have egress**; many other clusters' compute nodes do **not**. Work on
a node with internet. See the RCC user guide:
[docs.rcc.uchicago.edu](https://docs.rcc.uchicago.edu/slurm/sinteractive/).

### 4. On Midway3
Request an interactive job (no GPU needed), then activate the shared environment used across this
workshop series:
```bash
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI
export CLAUDE_CONFIG_DIR=$HOME/.claude
export DISABLE_AUTOUPDATER=1              # pin CLI behaviour during the session
```

## Running the hands-on lab
Everything happens in `hands-on/`. In short:
```bash
mkdir -p ~/cc-lab && cp -r hands-on/project ~/cc-lab/lakewatch   # a git-clean scratch copy
cd ~/cc-lab/lakewatch && git init -q && git add -A && git commit -qm start
claude                                          # start the agent; then follow hands-on/tasks/
```
Full instructions and the task index are in [`hands-on/README.md`](hands-on/README.md). The core
is six short tasks (~45 min): get oriented, clean messy data, fix a bug under guardrails, document
the project, turn field notes into a table, and automate with a command and a skill. Two optional
bonuses cover a headless Slurm batch and a small MCP tool server.

## Rebuilding the lecture deck (optional)
The deck is generated from `deck-src/` so it can be re-edited reproducibly — edit the wording in
`build_deck.py` and rebuild:
```bash
cd deck-src
python make_figures.py figures                        # regenerate the generated diagrams
python build_deck.py figures ../claude-code-tutorial.pptx
python render_preview.py ../claude-code-tutorial.pptx preview   # optional PNG preview (no LibreOffice)
```

## Cost expectations (measured)
| Activity | Approx cost |
|---|---|
| one headless `haiku` call | ~$0.01 |
| the bonus batch (a handful of logs) | a few cents |
| a focused interactive hour on a stronger model | ~$2–6 |

For a live workshop, ask attendees to **install and authenticate before arriving** (~10 min). Note
that dozens of simultaneous `claude` calls from one org key will hit rate limits — provision
per-attendee keys or stagger the batch exercise.

## Safety on shared HPC (please teach this)
- **Permission ladder.** *Plan* mode to explore → *Accept-edits* only inside a **git-clean scratch
  repo** → **never `--dangerously-skip-permissions` on a shared filesystem.** A bad `rm`/`chmod`
  in `/project` hits your whole lab.
- **Claude can read anything you can read.** Homes hold `~/.ssh`, `~/.netrc`, tokens. Start it from
  the *project* dir, not `$HOME`, and add deny rules (see `hands-on/project/.claude/settings.json`):
  ```json
  { "permissions": { "deny": ["Read(~/.ssh/**)", "Read(**/.env)", "Bash(rm -rf:*)"] } }
  ```
- **Slurm secret leak.** `sbatch --export=ALL` copies `ANTHROPIC_API_KEY` into the job environment,
  visible via `scontrol`. Prefer the `CLAUDE_CONFIG_DIR` login token, or source a `chmod 600` key
  file at runtime.
- **Data governance.** By default Anthropic does not train on commercial API inputs, but IRB/PHI/
  export-controlled data must not be sent without institutional approval.

## Documentation
Claude Code: [`code.claude.com/docs`](https://code.claude.com/docs) &nbsp;·&nbsp; RCC user guide:
[`docs.rcc.uchicago.edu`](https://docs.rcc.uchicago.edu) &nbsp;·&nbsp; the lecture's References
slides list the primary sources for every claim.
