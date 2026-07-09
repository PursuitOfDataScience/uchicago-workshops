# Task 0 · Set up (5 minutes)

**Goal:** get a working, throwaway copy of the project and start Claude Code inside it.

You will run `acceptEdits` mode later, which lets the agent change files on its own. So we
work on a **copy** under a git-clean scratch directory — never on your only copy of anything.

## Do this — in a terminal

```bash
# 1. Confirm Claude Code is installed and you are logged in.
claude --version          # expect a 2.x build
claude /status            # or run `claude` and type /status — shows your account + model

# 2. Copy the project to a fresh dir and make it a git repo.
#    (run this from the claude-code-tutorial directory)
mkdir -p ~/cc-lab
cp -r hands-on/project ~/cc-lab/lakewatch
cd ~/cc-lab/lakewatch
git init -q && git add -A && git commit -qm "start"

# 3. Confirm the failing test (this is the bug you will fix in Task 3).
python -m pytest -q       # expect: 2 failed, 6 passed

# 4. Start Claude Code in the project.
claude
```

Keep the task cards (`hands-on/tasks/`) open beside your terminal. Each task gives you a
prompt to paste into Claude Code.

## What to watch for
- On first launch, Claude reads `CLAUDE.md` automatically — that is your project's memory.
- `git init` matters: it means every change the agent makes is reviewable with `git diff`
  and reversible with `git restore`.

> **Next:** [Task 1 · Get your bearings](01-orient.md)
