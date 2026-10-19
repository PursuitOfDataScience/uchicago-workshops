#!/usr/bin/env python3
"""01 - A one-shot analysis with query().

The simplest use of the Agent SDK: hand the agent one goal and let it run to
completion, headless. This is the same "first look" as the exercise's `2-explore.md` prompt, but
driven by a script instead of a person typing at the terminal.

Run it (from this folder, with claude-agent-sdk installed):
    python 01_one_shot.py
"""
import asyncio

from claude_agent_sdk import query, ClaudeAgentOptions

from _transcript import show, ask

MODEL = "claude-sonnet-5-5"   # the exercise's model: Sonnet 5.5, at medium effort
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"


async def main() -> None:
    options = ClaudeAgentOptions(
        model=MODEL,
        effort="medium",
        # Headless: there is no one here to approve tool calls, so let the agent
        # run code on its own. Only do this for work you trust; here it just
        # reads a public CSV. (The interactive exercise reviews each step instead;
        # this is the SDK analogue of `claude -p --dangerously-skip-permissions`.)
        permission_mode="bypassPermissions",
        max_turns=20,
        max_budget_usd=1.0,   # stop if the run would cost more than $1
    )
    prompt = ask(
        f"Load the penguins dataset from {URL} and give me a first look: "
        "how many rows and columns, the species and their counts, and anything "
        "missing or odd in the data. Show the code you run."
    )
    async for msg in query(prompt=prompt, options=options):
        show(msg)


if __name__ == "__main__":
    asyncio.run(main())
