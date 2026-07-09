# Bonus B · Give it your own tools (MCP)

**Goal:** let Claude call a tool *you* wrote — a database, a `squeue` wrapper, an internal API.
**You'll practise:** the Model Context Protocol (MCP), how Claude Code becomes cluster-native.

Some questions the data files can't answer — where a site is, which sensors are online. This
project ships a tiny MCP server (`mcp_server.py`, no dependencies) that exposes two such tools.

## Try it — in a terminal (in the project dir)

```bash
claude -p "Where is site charlie and how deep is it? Use the MCP tool." \
  --mcp-config '{"mcpServers":{"lakewatch":{"command":"python3","args":["mcp_server.py"]}}}' \
  --strict-mcp-config --allowedTools "mcp__lakewatch__*" --model haiku
```

Claude discovers the server's tools and calls `site_metadata` like any built-in. MCP tools
are named `mcp__<server>__<tool>` and are **deny-by-default** — you allow them explicitly.

## Your turn
Open `mcp_server.py` and do the `TODO`: add a third tool `site_list` (no inputs) that returns
all site names. Then re-run, asking *"list all monitoring sites using the MCP tool."*

## What to watch for
- Interactively you would register a server once with `claude mcp add` instead of passing it
  inline; the same `mcp__server__tool` naming and permissions apply.
- A server that fetches external content can carry hidden instructions (prompt injection) —
  only connect tools you trust.

## Make it yours
- Wrap something from your own workflow — a Slurm submitter, a dataset catalog, a SQL query —
  as an MCP tool, and Claude can drive it under the same permission system.
