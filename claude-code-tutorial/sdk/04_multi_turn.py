#!/usr/bin/env python3
"""04 - A multi-turn session with ClaudeSDKClient.

`query()` is one-shot. `ClaudeSDKClient` keeps the conversation open, so later
turns build on earlier ones: the same "one running session, in order" that
the hands-on exercise asks for. Here the four turns walk the same arc as its tasks.

Run it:   python 04_multi_turn.py
Output:   writes body_mass_by_species.png into ./out/
"""
import asyncio
from pathlib import Path

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

from _transcript import show, ask

MODEL = "claude-sonnet-5-5"   # the exercise's model: Sonnet 5.5, at medium effort
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
OUT = Path(__file__).parent / "out"


async def turn(client: ClaudeSDKClient, prompt: str) -> None:
    await client.query(ask(prompt))
    async for msg in client.receive_response():
        show(msg)
    print()


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    options = ClaudeAgentOptions(
        model=MODEL,
        cwd=str(OUT),
        permission_mode="bypassPermissions",   # headless + trusted; see 01_one_shot.py
        effort="medium",
        max_turns=40,
        max_budget_usd=2.0,
    )
    async with ClaudeSDKClient(options=options) as client:
        await turn(client, f"Load the penguins dataset from {URL} and give me a "
                           "one-paragraph overview. Show the code.")
        await turn(client, "Now: which species is heaviest, and by how much "
                           "compared with the lightest?")
        await turn(client, "Save a labeled boxplot of body mass by species to "
                           "body_mass_by_species.png. There is no display, so save "
                           "the file rather than showing it.")
        await turn(client, "Finally, double-check the per-species means a second, "
                           "independent way and tell me exactly how many rows "
                           "went into each one.")

    print("Files now in ./out/:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    asyncio.run(main())
