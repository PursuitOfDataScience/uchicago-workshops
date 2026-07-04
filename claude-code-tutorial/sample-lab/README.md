# Sample Lab — hands-on materials

A tiny, self-contained "research repo" for the interactive part of the workshop. Copy it to a
scratch location, start Claude Code inside it, and work the exercises. Everything here is also a
**template** you can lift into your own project.

```
sample-lab/
├── CLAUDE.md                     # project memory template (conventions, data policy, env)
├── mcp_server.py                 # a dependency-free MCP stdio server (2 tools + an exercise)
├── analysis/
│   ├── stats.py                  # summary-stats helpers — contains ONE planted bug
│   └── test_stats.py             # ground-truth tests (one starts red)
└── .claude/
    ├── settings.json             # safe allow/deny permission rules (secrets denied)
    └── commands/
        ├── add-test.md           # /add-test <fn>  — parameterised slash command
        └── slurm-doctor.md       # /slurm-doctor   — injects live squeue/sinfo context
```

## Quick start

```bash
# 1. Copy to a git-clean scratch dir (never run acceptEdits on your only copy).
cp -r sample-lab ~/scratch/cc-lab && cd ~/scratch/cc-lab && git init -q && git add -A && git commit -qm init

# 2. Confirm the failing test.
cd analysis && python -m pytest -q      # -> 1 failed, 3 passed;  cd ..

# 3. Start Claude Code.
claude
```

## Exercises (in the REPL)

1. **Meet the project.** Type `/init`-style questions: *"Give me a tour of this repo. What does each function do?"* Notice it already knows your conventions — that is `CLAUDE.md` at work. Ask *"What is this project's codename?"* to prove it.
2. **Fix the bug under guardrails.** *"Run the tests, find the failing one, fix the bug in `analysis/`, and re-run until green. Don't touch the tests."* Watch it use `Read` → `Edit` → `Bash(pytest)`. Because `.claude/settings.json` **denies** `Edit(analysis/test_stats.py)`, it cannot "fix" the test by editing it. Review the change with `git diff` before you keep it.
3. **Author a command.** Run `/add-test standard_error` and see it write a new test. Then open `.claude/commands/add-test.md` and change the instructions.
4. **Plug in a tool.** Ask *"What's my storage quota?"* after starting Claude with the MCP server (see the header of `mcp_server.py`). Then do the `TODO` exercise: add a `gpu_free` tool and call it.
5. **Cluster copilot.** Create a deliberately broken `sbatch` script and an `.err` log, then run `/slurm-doctor broken.sbatch job.err`.

> Safety reminder: keep `--permission-mode acceptEdits` to a **git-clean scratch repo**, and never use
> `--dangerously-skip-permissions` on a shared filesystem. See the top-level README's safety section.
