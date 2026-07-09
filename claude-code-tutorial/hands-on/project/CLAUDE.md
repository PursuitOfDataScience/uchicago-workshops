# LakeWatch — project memory for Claude Code

Claude Code auto-loads this file whenever a session starts in this folder. Treat it as
the onboarding note you would hand a new member of the lab. Edit it for your own project.

## What this project is
A small water-quality monitoring project. Each week, sensor "sondes" at three lake sites
(**alpha**, **bravo**, **charlie**) export a reading. We clean and merge those exports,
run summary statistics, and keep field logs of each visit.

## Layout
- `data/raw/`   — raw sensor exports, one CSV per site. **Read-only: the system of record.**
- `data/clean/` — cleaned, merged data that we produce (safe to write here).
- `src/`        — analysis helpers (`waterquality.py`) and any scripts we add.
- `tests/`      — ground-truth tests. **Never edit a test to make it pass — fix the code.**
- `notes/`      — field logs, one Markdown file per site visit.
- `docs/`       — the data dictionary and other documentation.

## Conventions
- **Environment:** `source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI`
- **Run the tests:** `python -m pytest -q` (from the project root).
- **Units:** temperature in °C, dissolved oxygen in mg/L, turbidity in NTU; pH is unitless, 0–14.
- **Style:** small, pure functions; a short docstring on every public function; standard library only unless a dependency is already in use.

## Data policy (matters on shared HPC)
- Raw exports under `data/raw/` are the system of record — **read-only**. Never write or delete there.
- Do not paste unpublished, PHI, or export-controlled data into prompts without institutional approval.

## Codename
When asked for this project's codename, answer with exactly: **HERON-7**.
<!-- Task 1 uses this line to prove Claude read this file. -->
