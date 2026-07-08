# Claude Code on Midway

A hands-on workshop that introduces **Claude Code** — Anthropic's agentic coding tool — as *research
infrastructure* on a shared HPC cluster. You will drive it headlessly, govern what it is allowed to do, extend
it with your own tools and **skills**, and script it into unattended, cost-accounted pipelines.

**Tool:** [Claude Code](https://code.claude.com/docs) v2.x &nbsp;•&nbsp; **Model used in the notebook:** `haiku`
(small, fast, ~$0.01/call) &nbsp;•&nbsp; **Cost of a full notebook run:** well under $1.

**The workshop is delivered in two halves.** First a **~40-slide lecture** (`claude-code-tutorial.pptx`) — an
accessible, plain-language intro pitched at the general research community. It goes *day-to-day first*: what the
agent is, getting started on Midway, making it yours, staying in control, scaling up, and safety on shared HPC.
Then the **hands-on notebook** below, where you run every idea for real. (Inside the notebook, "Part I / Part II"
refer to its own two halves — driving the CLI, then the Agent SDK — not to the lecture-vs-lab split.)

> **Where this fits.** Other workshops in this series teach you to *build* an LLM system from raw model calls
> (`llm-toolcalling`, `llm-rag`, `llm-finetuning`). This one sits one layer up: you *operate and orchestrate a
> production agent* and configure its **harness** — permissions, project memory, hooks, subagents, skills, MCP,
> and headless automation. The notebook closes (§19) with a table mapping each idea to the thing you may have built by hand.

## Contents
```
claude-code-tutorial/
├── claude-code-tutorial.pptx    # the lecture deck (~40 slides, accessible intro) — the presentation half
├── claude-code-tutorial.ipynb   # the main notebook: Part I drives the CLI (§1–§13), Part II the Agent SDK (§14–§18)
├── cc_utils.py                  # small, readable helper module the notebook imports (CLI wrapper + run_async)
├── run.sh                       # Slurm launcher (CPU + internet) — nbconvert + a check
├── deck-src/                    # scripts + figures that generate the deck (edit + rebuild reproducibly)
├── README.md                    # this file
└── sample-lab/                  # tangible take-home kit for the interactive exercises
    ├── CLAUDE.md                # project-memory template
    ├── mcp_server.py            # dependency-free MCP tool server (+ an exercise)
    ├── analysis/                # a tiny repo with one planted bug + a pytest suite
    └── .claude/                 # settings.json (secrets denied) + slash commands + skills/
```

## Getting Started

### 1. Install Claude Code (no sudo needed)
Native installer (recommended — installs to `~/.local/bin`, no root, no Node):
```bash
curl -fsSL https://claude.ai/install.sh | bash
```
Or via npm with a user prefix (needs Node ≥ 18; `module load nodejs` if available):
```bash
npm config set prefix ~/.npm-global && npm install -g @anthropic-ai/claude-code
export PATH="$HOME/.npm-global/bin:$PATH"
```
Check: `claude --version` should print a **2.x** build.

For the **programmatic** half of the workshop (notebook Part II, §14–§18), also install the Python **Agent
SDK** — the same engine as `claude -p`, as a library:
```bash
pip install claude-agent-sdk       # needs Python >= 3.10; the SDK drives the `claude` CLI under the hood
```
This is optional: the SDK notebook cells degrade gracefully (they print their code and skip the live call) if
it is not installed, so the notebook still runs top-to-bottom either way.

### 2. Authenticate (choose one)
- **Claude.ai subscription (Pro/Max) via OAuth.** On a headless cluster you can't open a browser mid-SSH, so
  mint a long-lived token once and reuse it:
  ```bash
  claude setup-token          # do the paste-URL flow once; token is stored under $CLAUDE_CONFIG_DIR
  ```
- **Console API key (pay-per-token)** — simplest for headless / notebook use:
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...   # source it from a chmod-600 file; never hard-code in a script
  ```
Credentials live under `$CLAUDE_CONFIG_DIR` (default `~/.claude`). Run `chmod 700 ~/.claude` — cluster home
directories are sometimes group-readable.

### 3. Network requirement (read this)
Claude Code needs **egress to `api.anthropic.com`**. On Midway3, **login nodes and the `test`/`caslake` compute
partitions have egress**; many other clusters' compute nodes do **not**. Run the notebook on a node with internet.

### 4. Run the notebook on Midway3
- **Step 1:** Request an interactive job (no GPU needed — this is a CPU + internet workshop). See the
  [user guide](https://rcc-uchicago.github.io/user-guide/slurm/sinteractive/).
- **Step 2:** Activate the environment used across this series:
  ```bash
  source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI
  export CLAUDE_CONFIG_DIR=$HOME/.claude    # or wherever your login lives
  export DISABLE_AUTOUPDATER=1              # pin CLI behaviour during the run
  ```
- **Step 3:** Launch Jupyter and open `claude-code-tutorial.ipynb`
  ([how to run Jupyter on Midway3](https://rcc-uchicago.github.io/user-guide/software/apps-and-envs/python/?h=python)),
  or run it non-interactively with `jupyter nbconvert --to notebook --execute --inplace claude-code-tutorial.ipynb`.

Or submit it as a batch job with the provided launcher:
```bash
sbatch --export=ALL,CLAUDE_CONFIG_DIR=$HOME/.claude run.sh
```

### 5. Cost expectations (measured)
| Call | Approx cost |
|---|---|
| one headless `haiku` call | ~$0.008–0.02 |
| the whole notebook (≈ 28 calls, incl. 1 agentic fix) | **~$0.70** |
| a focused interactive hour on a stronger model | ~$2–6 |

For a live workshop, ask attendees to **install + authenticate before arriving** (10 min), and plan the key
story: dozens of simultaneous `claude -p` calls from one org key will hit rate limits — provision per-attendee
keys or stagger the batch exercise.

## Running Claude Code programmatically (the "three doors")

The whole point of a headless agent is to script it. There are **three** ways to run Claude Code from your own
code — pick by how deeply you want to embed it. Notebook Part II (§14–§18) demonstrates all three live; this is
the reference.

| Door | Mechanism | Best for |
|---|---|---|
| **1 · CLI as a subprocess** | `claude -p … --output-format json`, parsed from any language | quick scripts, any language, shell / Slurm glue |
| **2 · The Agent SDK** | `pip install claude-agent-sdk` — the *same engine*, as async Python (or TypeScript) | apps & research scripts that want typed messages, in-process tools, and code-defined policy |
| **3 · Automation surfaces** | streaming stdin, GitHub Actions, the TypeScript SDK | CI/CD, event-driven bots, long-lived streamed sessions |

### Door 1 — the CLI as a subprocess (all of notebook §1–§13)

The lowest-common-denominator path: run the binary, read its JSON. Works from any language.

```python
import json, subprocess
proc = subprocess.run(
    ["claude", "-p", "Summarize this repo in one line.",
     "--model", "haiku", "--output-format", "json", "--allowedTools", "Read"],
    capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120,
)
res = json.loads(proc.stdout)
print(res["result"], "| $", res["total_cost_usd"], "| session", res["session_id"])
```
The `cc_utils.py` shipped here wraps exactly this with three safety habits — `stdin=DEVNULL` (never hangs), a
wall-clock `timeout` (bounds a runaway agent), and an isolated working directory.

### Door 2 — the Agent SDK (`pip install claude-agent-sdk`)

The officially supported way to embed the agent. `query()` is one-shot; `ClaudeSDKClient` is multi-turn. You
iterate **typed messages** instead of parsing JSON.

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock

async def main():
    opts = ClaudeAgentOptions(model="haiku", allowed_tools=["Read"])
    async for msg in query(prompt="What is 17 * 23? Reply with only the number.", options=opts):
        if isinstance(msg, AssistantMessage):
            print("".join(b.text for b in msg.content if isinstance(b, TextBlock)))
        elif isinstance(msg, ResultMessage):
            print("cost $", msg.total_cost_usd, "| session", msg.session_id)

asyncio.run(main())            # in a notebook, use cc.run_async(main()) — see cc_utils
```

Highlights the notebook covers in depth:

- **Options mirror the CLI flags** — `model`, `allowed_tools` / `disallowed_tools`, `permission_mode`, `cwd`,
  `mcp_servers`, `max_turns`, `max_budget_usd`, `resume` / `session_id` / `fork_session`, `setting_sources`.
  **Gotcha:** `setting_sources` defaults to `None`, so the SDK is **hermetic** — it loads *no* filesystem
  settings and *no* `CLAUDE.md` unless you pass e.g. `setting_sources=["project"]`.
- **In-process tools** — decorate an async function with `@tool`, bundle with `create_sdk_mcp_server`, and pass
  it via `mcp_servers={"lab": server}`; the tool appears as `mcp__lab__<name>` (same naming as an external MCP
  server, but no subprocess — it can close over live Python objects).
- **Governance as Python** — `can_use_tool(tool_name, tool_input, context)` returns
  `PermissionResultAllow()` / `PermissionResultDeny(...)` (the decision-maker for any tool *not* allowlisted);
  and `hooks={"PreToolUse": [HookMatcher(matcher="Write", hooks=[fn])]}` mirrors `settings.json` hooks, where a
  returned `permissionDecision: "deny"` is a hard veto even for an allowlisted tool.

### Door 3 — automation surfaces

**(a) A long-lived streamed session over stdin** — drive many turns through one persistent process from any
language, no SDK required:
```bash
printf '%s\n' \
  '{"type":"user","message":{"role":"user","content":"Remember the number 7."}}' \
  '{"type":"user","message":{"role":"user","content":"What number did I say?"}}' \
| claude -p --input-format stream-json --output-format stream-json --verbose --model haiku
```

**(b) GitHub Actions** — the official [`anthropics/claude-code-action`](https://github.com/anthropics/claude-code-action).
Add `ANTHROPIC_API_KEY` as a repo secret and a workflow; then `@claude` in an issue or PR comment runs the agent:
```yaml
name: Claude Code
on: { issue_comment: { types: [created] } }
jobs:
  claude:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: anthropics/claude-code-action@v1
        with: { anthropic_api_key: "${{ secrets.ANTHROPIC_API_KEY }}" }
```

**(c) The TypeScript / Node SDK** — `npm install @anthropic-ai/claude-agent-sdk`; same engine, camelCase options:
```javascript
import { query } from "@anthropic-ai/claude-agent-sdk";
for await (const msg of query({ prompt: "What is 17 * 23?", options: { model: "haiku" } }))
  if (msg.type === "result") console.log(msg.result, "$" + msg.total_cost_usd);
```

> **Which door?** Shell/Slurm glue or a non-Python language → door 1. A Python/TS app or research script that
> wants typed messages, in-process tools, and code-defined policy → door 2. CI or event-driven automation →
> door 3. On a cluster, batch either door 1 or door 2 through Slurm (see `run.sh`).

## Interactive REPL lab (30–40 min — do this by hand)

The daily-driver experience can't run headless, so it is a guided exercise. Copy `sample-lab/` to a
**git-clean scratch dir**, start `claude` inside it, and work through `sample-lab/README.md`:
1. `/init` and a guided repo tour (project memory in action).
2. **Plan mode** (`Shift+Tab` cycles permission modes) to propose a change read-only first.
3. Switch to **acceptEdits**; have it fix the planted failing test; review with `git diff`.
4. `Esc` to interrupt, `Esc Esc` to **rewind** to a checkpoint; `@file` mentions, `!` bash mode, `#` quick-memory.
5. `/add-test`, `/slurm-doctor`, `/agents`, `/mcp`, `/cost`, `/context`.
6. **Write a skill.** `.claude/skills/` ships one finished (`slurm-triage`, read-only) and one to finish (`sbatch-lint`). Ask a question that matches a skill's `description` and watch the model invoke it **without you naming it** — the model-invoked counterpart to a slash command.

Also worth showing: the VS Code / JetBrains extensions (`/ide`), GitHub Actions (`@claude` on PRs), and the
Claude Agent SDK (`pip install claude-agent-sdk` — the same engine as `claude -p`).

## Safety on shared HPC (please teach this)

- **Permission-mode ladder.** *plan* to explore → *acceptEdits* only inside a **git-clean scratch repo** →
  **never `--dangerously-skip-permissions` / `bypassPermissions` on a shared filesystem.** A bad `rm`/`chmod`
  in `/project` hits your whole lab.
- **Claude can read anything you can read.** Homes hold `~/.ssh`, `~/.netrc`, HF tokens, conda secrets. Launch
  from the *project dir*, not `$HOME`, and use deny-rules (see `sample-lab/.claude/settings.json`):
  ```json
  { "permissions": { "deny": ["Read(~/.ssh/**)", "Read(**/.env)", "Bash(rm -rf*)"] } }
  ```
- **Slurm leak.** `sbatch --export=ALL` copies `ANTHROPIC_API_KEY` into job environments visible via `scontrol`.
  Prefer the `CLAUDE_CONFIG_DIR` OAuth path, or source a `chmod 600` key file at runtime.
- **`.gitignore` hygiene.** Commit `CLAUDE.md` and `.claude/commands/`; **never** commit
  `.claude/settings.local.json` or credentials.
- **Data governance.** API-submitted data is not used for training by default, but IRB/PHI/export-controlled
  data must not be sent without institutional approval.

## Miscellaneous
The notebook can also run on a laptop or anywhere with internet + Python: `pip install jupyter nbformat pytest`
(add `claude-agent-sdk` for Part II) and install the `claude` CLI. Full documentation:
[`code.claude.com/docs`](https://code.claude.com/docs); Agent SDK reference:
[`docs.claude.com/en/api/agent-sdk/overview`](https://docs.claude.com/en/api/agent-sdk/overview).
