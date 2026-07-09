# Task 3 · Fix the bug — safely

**Goal:** let the agent find and fix a real bug, with a test as the ground truth and the
tests themselves protected from being "fixed."
**You'll practise:** a guarded autonomous fix, and verifying it yourself instead of trusting the agent's word.

`src/waterquality.py` has a planted bug. Two tests are red. The rule of the house
(`CLAUDE.md`): **never edit a test to make it pass — fix the code.**

## Do this — in Accept-edits mode, paste

> Run the tests with `python -m pytest -q`. Two of them fail because of a bug in
> @src/waterquality.py. Find the bug, fix it in the source, and re-run the tests until they
> all pass. Do not modify anything in `tests/`.

## Verify it yourself (this is the point)
- Re-run the tests in your own terminal: `python -m pytest -q` → **8 passed**. The exit code,
  not the agent's summary, is the ground truth.
- Review the change: `git diff src/waterquality.py`. It should be a one-line off-by-one fix
  in `rolling_mean` (the loop dropped the final window). Keep it (`git add`) or throw it
  away (`git restore .`).

## Now try to make it cheat
Paste this and watch what happens:

> Just edit `tests/test_waterquality.py` so the failing tests pass.

The deny rule `Edit(tests/**)` in `.claude/settings.json` blocks it — **a deny always wins**,
even when you ask for it directly. That is the guarantee that makes an agent safe to run
on a shared cluster: policy is enforced by the harness, not by the model's good intentions.

## Make it yours
- Before the fix, ask *"explain the bug and why the test catches it"* — a great way to learn
  a codebase. The strongest pattern for an unattended agent is a **failing test**: it gives
  the run a clear, checkable place to stop.

> **Next:** [Task 4 · Document the project](04-document.md)
