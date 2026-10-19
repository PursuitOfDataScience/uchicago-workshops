"""Shared transcript printer for the Agent SDK examples.

Each SDK call streams back a sequence of messages: the prompt you sent, any
tool the agent calls plus that tool's result, and the agent's replies. `show()`
turns each one into a readable line so you can watch the whole loop, not just
the final answer. Imported by the numbered example scripts in this folder.
"""
from claude_agent_sdk import (
    AssistantMessage,
    UserMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)


def _text_of(content) -> str:
    """Flatten a block's content (a str, or a list of blocks/dicts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(getattr(item, "text", str(item)))
        return " ".join(parts)
    return str(content)


def show(msg) -> None:
    """Print one streamed message as a single readable line."""
    if isinstance(msg, AssistantMessage):
        for b in msg.content:
            if isinstance(b, TextBlock) and b.text.strip():
                print("  claude   :", b.text.strip())
            elif isinstance(b, ToolUseBlock):
                # Tool inputs can be long (e.g. a whole script), so trim them for display.
                shown = str(b.input)
                if len(shown) > 300:
                    shown = shown[:300] + " …"
                print(f"  claude   ▸ calls {b.name}({shown})")
    elif isinstance(msg, UserMessage):
        # A tool's result is fed back to the model as a UserMessage.
        blocks = msg.content if isinstance(msg.content, list) else []
        for b in blocks:
            if isinstance(b, ToolResultBlock):
                tag = "tool ERR " if b.is_error else "tool     "
                text = _text_of(b.content)
                if len(text) > 500:
                    text = text[:500] + " …"
                print(f"  {tag}-> {text}")
    elif isinstance(msg, ResultMessage):
        print(f"  |__ {msg.num_turns} turn(s), ${msg.total_cost_usd or 0:.4f}")


def ask(prompt: str) -> str:
    """Echo the prompt we're sending, then return it to pass into the SDK."""
    print(f"  you      : {prompt}")
    return prompt
