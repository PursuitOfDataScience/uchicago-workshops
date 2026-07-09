# Take-home · Your own project

The real payoff is doing this on your own work. A safe first session:

1. **Copy first.** Work on a git-clean copy, or commit your current state so every change is
   reversible with `git diff` / `git restore`.
2. **Write a `CLAUDE.md`.** Run `/init` to generate a starting one, then add your test command,
   where data lives, and any "never touch this" rules. Keep it under ~200 lines.
3. **Set a guardrail.** Add a `.claude/settings.json` with a `deny` rule for anything
   irreplaceable — raw data, credentials (`Read(**/.env)`, `Read(~/.ssh/**)`), `Bash(rm -rf:*)`.
4. **Start in Plan mode.** Ask for a plan before any change; switch to Accept-edits only
   inside a clean repo.
5. **Always give it a check.** A failing test, an exit code, a file that should or should not
   exist. If you can't verify the output, don't ship it.

## A few good first prompts
- *"Give me a tour of this repo and flag anything that looks broken or undocumented."*
- *"Standardise the column names across the CSVs in `data/` — plan first."*
- *"Run the tests, fix the failing one, don't touch the tests, then show me the diff."*
- *"Read these result files and extract [fields] into a table."*

## Where to go next
- Full documentation: **code.claude.com/docs**
- The presentation deck (`../../claude-code-tutorial.pptx`) has the concepts behind every task.
- RCC support: user guide **docs.rcc.uchicago.edu**, help desk, and office hours.
