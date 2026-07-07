"""
cc_utils.py  —  helper utilities for the "Claude Code on Midway" workshop.

This module keeps the notebook cells focused on *teaching*. It wraps the real `claude`
command-line tool in a small, safe Python API and adds a few pretty-printers.

Everything here is deliberately short and readable — skim it, you are meant to.

The one idea that makes it safe to call an autonomous agent from inside a notebook:
every invocation runs with
    * stdin = DEVNULL      (so an interactive prompt can never hang the kernel),
    * a wall-clock timeout  (so a runaway agent is bounded), and
    * an isolated working directory (so demos never touch your real files).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

# --------------------------------------------------------------------------- config
# Use the small, fast, cheap model for every teaching call (~3 s, ~$0.01 each).
DEFAULT_MODEL = "haiku"

# Auth and session transcripts live under CLAUDE_CONFIG_DIR (default ~/.claude).
# We only ever *read* from it — never repoint it, or you would orphan your login.
CONFIG_DIR = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
PROJECTS_DIR = os.path.join(CONFIG_DIR, "projects")

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# --------------------------------------------------------------------------- workspace
class Workspace:
    """A throwaway directory tree for demos, so nothing leaks into your real project.

    Create scenario sub-directories with `ws.dir("name")`, drop files in them with
    `ws.write("name/CLAUDE.md", "...")`, and call `ws.cleanup()` at the end.
    """

    def __init__(self, prefix: str = "ccws_"):
        self.root = tempfile.mkdtemp(prefix=prefix)

    def dir(self, name: str) -> str:
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        return d

    def write(self, relpath: str, content: str) -> str:
        path = os.path.join(self.root, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


# --------------------------------------------------------------------------- cost ledger
class Ledger:
    """Accumulates the dollar cost and call count across the whole notebook."""

    def __init__(self):
        self.usd = 0.0
        self.calls = 0

    def add(self, result_json: dict):
        self.usd += (result_json.get("total_cost_usd") or 0.0)
        self.calls += 1
        return result_json

    def line(self) -> str:
        s = "" if self.calls == 1 else "s"
        return f"cumulative: ${self.usd:.4f} over {self.calls} claude call{s}"


LEDGER = Ledger()


# --------------------------------------------------------------------------- the CLI wrapper
def argv(prompt, allowed=None, permission_mode=None, setting_sources=None,
         output="json", model=DEFAULT_MODEL, extra=None):
    """Build the exact `claude` command line (argv list) a call will run.

    Print it with `shlex.join(cc.argv(...))` to see the *real* command behind the
    `cc.ask` wrapper. Flags:
      allowed         -> --allowedTools   e.g. ["Write"] or ["mcp__demo__*"]
      permission_mode -> --permission-mode  e.g. "acceptEdits", "plan"
      setting_sources -> --setting-sources  e.g. "project"
      output          -> --output-format   "json" | "stream-json" | "text"
      extra           -> any additional raw flags, e.g. ["--json-schema", schema]
    """
    cmd = ["claude", "-p", prompt, "--model", model]
    if output:
        cmd += ["--output-format", output]
    if allowed:
        cmd += ["--allowedTools"] + list(allowed)
    if permission_mode:
        cmd += ["--permission-mode", permission_mode]
    if setting_sources:
        cmd += ["--setting-sources", setting_sources]
    if extra:
        cmd += extra
    return cmd


def run_claude(prompt, cwd, allowed=None, permission_mode=None, setting_sources=None,
               output="json", model=DEFAULT_MODEL, extra=None, timeout=120, echo=False):
    """Invoke `claude -p "<prompt>"` as a subprocess and return the CompletedProcess.

    Runs in `cwd` (its CLAUDE.md / .claude/ apply). `echo=True` prints the real
    command first. Three safety habits: stdin=DEVNULL (never hangs on a prompt), a
    wall-clock timeout (bounds a runaway agent), and an isolated working directory.
    See argv() for the flag meanings.
    """
    cmd = argv(prompt, allowed=allowed, permission_mode=permission_mode,
               setting_sources=setting_sources, output=output, model=model, extra=extra)
    if echo:
        import shlex
        print("→", shlex.join(cmd))
    env = dict(os.environ)
    env["DISABLE_AUTOUPDATER"] = "1"                   # pin behaviour: no version drift mid-run
    return subprocess.run(
        cmd, cwd=cwd, env=env,
        stdin=subprocess.DEVNULL,                      # <- the anti-hang rule
        capture_output=True, text=True, timeout=timeout,
    )


def ask(prompt, cwd, **kwargs):
    """Run `claude -p` with JSON output, record the cost, and return the parsed dict."""
    proc = run_claude(prompt, cwd, output="json", **kwargs)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"claude failed (exit {proc.returncode}):\n{proc.stderr[:500]}")
    return LEDGER.add(json.loads(proc.stdout))


# --------------------------------------------------------------------------- pretty printers
def show(result_json: dict, label: str = "reply"):
    """Print a claude JSON result compactly: the text plus the telemetry that matters."""
    text = str(result_json.get("result", "")).strip()
    print(f"┌─ {label} " + "─" * max(2, 60 - len(label)))
    for ln in text.splitlines() or [""]:
        print("│ " + ln)
    denials = [d.get("tool_name") for d in result_json.get("permission_denials", [])]
    usage = result_json.get("usage") or {}
    print("├─ telemetry " + "─" * 49)
    print(f"│ cost=${result_json.get('total_cost_usd') or 0:.5f}"
          f"  turns={result_json.get('num_turns')}"
          f"  out_tokens={usage.get('output_tokens')}"
          f"  is_error={result_json.get('is_error')}")
    if denials:
        print(f"│ permission_denials: {denials}")
    print("└" + "─" * 61)


def render_trace(stdout: str):
    """Turn a --output-format stream-json run into a readable tool-call audit trail.

    Returns (event_types, result_event) so callers can also inspect the final result.
    """
    events = [json.loads(l) for l in stdout.splitlines() if l.strip()]
    print("  step  tool     detail")
    print("  ----  -------  " + "-" * 46)
    n = 0
    for e in events:
        if e.get("type") == "assistant":
            for c in e["message"].get("content", []):
                if c.get("type") == "tool_use":
                    n += 1
                    inp = c.get("input", {}) or {}
                    fp = inp.get("file_path")
                    # show only the basename of any file path: cleaner to read, and it avoids
                    # printing machine-specific absolute paths into the notebook's output.
                    detail = inp.get("command") or (os.path.basename(fp) if fp else None) \
                        or inp.get("pattern") or ""
                    print(f"  {n:>4}  {c['name']:<7}  {str(detail)[:46]}")
    types = [e.get("type") for e in events]
    result_evt = next((e for e in reversed(events) if e.get("type") == "result"), {})
    return types, result_evt


# --------------------------------------------------------------------------- async helper
def run_async(coro):
    """Run an async coroutine from a notebook cell.

    The Claude Agent SDK (the *programmatic* half of this workshop) is async. A Jupyter
    kernel already has a running event loop, so a bare ``asyncio.run`` would raise
    "cannot be called from a running event loop." ``nest_asyncio`` patches the loop so a
    nested ``run_until_complete`` works; we fall back gracefully when it is absent (e.g.
    a plain script). This helper has no SDK dependency, so importing cc_utils never
    requires ``claude-agent-sdk`` to be installed.
    """
    import asyncio
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# --------------------------------------------------------------------------- misc helpers
def redact(text) -> str:
    """Mask machine-specific absolute paths so printed/committed output is safe to share.

    Replaces your home directory with ``~`` and collapses ``/project/<pi>/<user>`` and
    ``/home/<user>`` prefixes to a generic placeholder — so a notebook's output never
    leaks a username or private directory layout.
    """
    s = str(text)
    s = s.replace(os.path.expanduser("~"), "~")
    s = re.sub(r"/project/[^/\s]+/[^/\s]+", "/project/<you>", s)
    s = re.sub(r"/home/[^/\s]+", "/home/<you>", s)
    s = re.sub(r"/scratch/[^/\s]+", "/scratch/<you>", s)
    return s


def is_uuid(s: str) -> bool:
    return bool(_UUID_RE.match(str(s)))


def find_transcript(session_id: str):
    """Locate the on-disk transcript for a session: $CONFIG/projects/<cwd-encoded>/<id>.jsonl."""
    target = session_id + ".jsonl"
    if not os.path.isdir(PROJECTS_DIR):
        return None
    for root, _, files in os.walk(PROJECTS_DIR):
        if target in files:
            return os.path.join(root, target)
    return None
