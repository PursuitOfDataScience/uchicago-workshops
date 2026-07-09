# Claude Code on Midway

A hands-on workshop that introduces **Claude Code** — Anthropic's AI *coding agent* — as
everyday **research infrastructure** on a shared HPC cluster. You will install it on Midway,
use it interactively to analyze a real dataset, keep it safe on shared hardware, and drive it
headlessly from a single command.

## The workshop has two halves

1. **A lecture** — `claude-code-tutorial.pptx` (~46 slides). An accessible, plain-language
   introduction for a general research audience: what an agent in the terminal *is*, getting
   started on Midway, making it yours, staying in control, scaling up, and honest limits.
2. **A hands-on lab** — `hands-on/`. There is **no notebook and nothing to download** — just a
   flat folder of Markdown task cards and a `CLAUDE.md`. You open Claude Code there and drive it in
   plain English, in two parts: **Part 1** navigates Midway3 itself (quota, jobs, allocations,
   partitions, software), and **Part 2** analyzes a dataset **hosted online**. It writes and runs
   the commands and reports back while you review — what using Claude Code as research infra feels like.

```
claude-code-tutorial/
├── claude-code-tutorial.pptx     # the lecture deck (~46 slides) — the presentation half
├── hands-on/                     # the interactive lab — the second half (flat: no subfolders)
│   ├── CLAUDE.md                 #   project memory: Midway3 facts + the dataset URL
│   ├── README.md                 #   how to start + the task index
│   ├── 01-…08-….md               #   Part 1: navigate Midway3 (prompts to paste)
│   ├── 09-…17-….md               #   Part 2: analyze a dataset (prompts to paste)
│   └── answers.py                #   reference answers for Part 2: code + expected output
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

### 2. Authenticate
Just start Claude Code and log in from inside it — no tokens to mint by hand:
```bash
claude          # start it, then type:
/login          # follow the prompt: open the URL it prints, authorize, paste the code back
```
That's it. Log in once with your Claude.ai account (Pro/Max) and the credentials are saved under
`$CLAUDE_CONFIG_DIR` (default `~/.claude`), so later `claude` and `claude -p` runs on that node
just work.

Run `chmod 700 ~/.claude` afterwards — cluster home directories are sometimes group-readable.

> **Scripts / batch jobs only:** if you'd rather not log in interactively, export a Console API
> key instead — `export ANTHROPIC_API_KEY=sk-ant-...` (source it from a `chmod 600` file; never
> hard-code it). For interactive use, `/login` is the easy path.

### 3. On Midway3
Request an interactive job (no GPU needed), then activate the shared environment used across this
workshop series:
```bash
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI
export CLAUDE_CONFIG_DIR=$HOME/.claude
export DISABLE_AUTOUPDATER=1              # pin CLI behaviour during the session
```

## Running the hands-on lab
Work on a node with internet, then:
```bash
cd hands-on          # (or copy the folder somewhere writable and cd there)
claude
```
Then follow the task cards, pasting each prompt and checking Claude's work. **Part 1 (Tasks 1–8)**
has Claude navigate Midway3 itself — your disk quota, your jobs, your allocation balance, cluster
and partition load, interactive sessions, batch scripts, and software modules — by running the
RCC/Slurm commands for you and explaining the output (read-only; nothing is submitted or deleted).
**Part 2 (Tasks 9–17, ~40 min)** points it at a **hosted dataset** (Palmer Penguins — from a URL,
nothing to download) for a short analysis: a first look, per-species summaries, a comparison, a
correlation, a data-quality audit, a saved figure, and a written-up Results paragraph — then a
headless bonus and a bring-your-own-data take-home. Full instructions and both task indexes are in
[`hands-on/README.md`](hands-on/README.md).

## Documentation
Claude Code: [`code.claude.com/docs`](https://code.claude.com/docs) &nbsp;·&nbsp; RCC user guide:
[`docs.rcc.uchicago.edu`](https://docs.rcc.uchicago.edu).
