# Task 1 · Get your bearings

**Goal:** understand a project you have just inherited — without reading every file yourself.
**You'll practice:** letting the agent explore your codebase, and seeing project memory (`CLAUDE.md`) at work.

## Do this — paste into Claude Code

> Give me a tour of this project. What is it, what is in each folder, and what looks
> unfinished or broken? Keep it to a short bulleted summary — don't change anything yet.

Then, to prove it actually read the project's memory file, ask:

> What is this project's codename, and where did you find it?

## What to watch for
- The agent uses read-only tools (`Read`, `Grep`, `Glob`) to look around — nothing to
  index first; it reads your real files, the way you would.
- It should report the codename **HERON-7** and point to `CLAUDE.md`. That line lives only
  in `CLAUDE.md`, so a correct answer proves the file was loaded into context.
- Notice it already knows the conventions (test command, "raw data is read-only") — those
  came from `CLAUDE.md` too.

## Make it yours
- Ask: *"If you were onboarding a new lab member, what one thing would you add to `CLAUDE.md`?"*
  In your own repo, run `/init` to generate a first `CLAUDE.md` automatically.

> **Next:** [Task 2 · Tidy the messy data](02-clean-data.md)
