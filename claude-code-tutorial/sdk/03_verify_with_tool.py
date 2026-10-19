#!/usr/bin/env python3
"""03 - Give the agent a tool to check its own work.

This mirrors the exercise's signature theme: "an agent can be confidently wrong, so
verify it." Here we hand the agent one of our own Python functions as a tool
(it computes the ground-truth mean body mass per species straight from the CSV)
and ask the agent to check its answer against it. Shows the `@tool` decorator
and an in-process SDK MCP server (`create_sdk_mcp_server`).

The agent is given ONLY this tool (no bash/read), so it must answer from what it
knows and then confirm against the tool: it can't quietly recompute another way.

Run it:   python 03_verify_with_tool.py
"""
import asyncio
import csv
import io
import urllib.request

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
)

from _transcript import show, ask

MODEL = "claude-sonnet-5-5"   # the exercise's model: Sonnet 5.5, at medium effort
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"


@tool(
    "species_mean_mass",
    "Ground-truth mean body_mass_g per penguin species, computed directly from "
    "the source CSV. Returns each species with its mean mass and the number of "
    "rows used.",
    {},
)
async def species_mean_mass(args: dict) -> dict:
    data = urllib.request.urlopen(URL, timeout=30).read().decode()
    rows = list(csv.DictReader(io.StringIO(data)))
    groups: dict[str, list[float]] = {}
    for r in rows:
        mass = (r.get("body_mass_g") or "").strip()
        if mass not in ("", "NA"):
            groups.setdefault(r["species"], []).append(float(mass))
    lines = [
        f"{sp}: mean = {sum(v) / len(v):.1f} g  (n = {len(v)})"
        for sp, v in sorted(groups.items())
    ]
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


async def main() -> None:
    server = create_sdk_mcp_server(
        name="truth", version="1.0.0", tools=[species_mean_mass]
    )
    options = ClaudeAgentOptions(
        model=MODEL,
        effort="medium",
        mcp_servers={"truth": server},
        allowed_tools=["mcp__truth__species_mean_mass"],  # the only tool it may use
        permission_mode="bypassPermissions",
        max_turns=6,
        max_budget_usd=1.0,
    )
    prompt = ask(
        "Which penguin species do you expect is heaviest, and roughly what is "
        "each species' mean body mass? Then call the species_mean_mass tool to "
        "check yourself, and tell me whether your answer held up, and exactly "
        "how many penguins went into each number."
    )
    async for msg in query(prompt=prompt, options=options):
        show(msg)


if __name__ == "__main__":
    asyncio.run(main())
