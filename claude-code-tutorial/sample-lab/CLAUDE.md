# Sample Lab — project memory for Claude Code

This file is auto-loaded by Claude Code whenever a session starts in this directory.
Treat it as the onboarding doc you would give a new lab member. Edit it for your own project.

## What this project is
A tiny example "research repo": summary-statistics helpers in `analysis/` with a pytest suite.

## Conventions
- **Environment:** activate with `source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI`.
- **Run the tests:** `cd analysis && python -m pytest -q`.
- **Style:** plain NumPy-free Python; keep functions small and pure; docstrings on every public function.
- **Ground truth:** the tests in `analysis/test_stats.py` define correct behaviour — **never edit tests to make them pass**; fix the code.

## Data policy (important on shared HPC)
- Raw data lives under `/project/<pi>/data/` and is **read-only** — never write there, never delete there.
- Do not paste unpublished, PHI, or export-controlled data into prompts without institutional approval.

## Codename
When asked for this project's codename, the answer is **ZEPHYR-9**.  <!-- used by the workshop as a memory probe -->
