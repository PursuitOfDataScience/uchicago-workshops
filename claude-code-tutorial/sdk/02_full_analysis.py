#!/usr/bin/env python3
"""02 - A full analysis from a single goal.

One call, many steps: the agent plans and carries out the whole penguin
analysis and saves a figure: the same work as the exercise's steps 2 and 3, done from
code. The point is that a single `query()` produces a real deliverable while you
just watch the loop scroll past.

Run it:   python 02_full_analysis.py
Output:   writes body_mass_by_species.png into ./out/
"""
import asyncio
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions

from _transcript import show, ask

MODEL = "claude-sonnet-5-5"   # the exercise's model: Sonnet 5.5, at medium effort
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
OUT = Path(__file__).parent / "out"


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    options = ClaudeAgentOptions(
        model=MODEL,
        cwd=str(OUT),                       # the agent reads/writes files here
        permission_mode="bypassPermissions",  # headless + trusted; see 01_one_shot.py
        effort="medium",
        max_turns=40,
        max_budget_usd=2.0,
    )
    prompt = ask(
        f"Analyze the penguins dataset at {URL}. This runs with no "
        "display, so save any figure to a PNG rather than trying to show it. "
        "Specifically: (1) load it and describe what's there; (2) find which "
        "species is heaviest and test whether flipper length predicts body mass "
        "both across all penguins and within each species; (3) save a labeled "
        "boxplot of body mass by species as body_mass_by_species.png; and (4) "
        "finish with a short Results paragraph. Show the code you run, and don't "
        "drop any rows silently."
    )
    async for msg in query(prompt=prompt, options=options):
        show(msg)

    print("\nFiles now in ./out/:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    asyncio.run(main())
