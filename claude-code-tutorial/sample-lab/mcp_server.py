#!/usr/bin/env python3
"""A minimal Model Context Protocol (MCP) stdio server for the workshop.

It speaks newline-delimited JSON-RPC 2.0 over stdin/stdout — the same protocol Claude Code
uses to talk to any MCP server — and exposes two tiny "cluster" tools. It has NO third-party
dependencies so it runs anywhere the AI env does.

Try it:
    claude -p "What is my cluster storage quota? Use the MCP tool." \
        --mcp-config '{"mcpServers":{"lab":{"command":"python3","args":["mcp_server.py"]}}}' \
        --strict-mcp-config --allowedTools "mcp__lab__*" --model haiku

EXERCISE: add a third tool (see the TODO near the bottom), restart, and ask Claude to call it.
"""
import json
import sys

TOOLS = [
    {
        "name": "storage_quota",
        "description": "Return the caller's cluster storage quota usage as a human string.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "partition_info",
        "description": "Return a short description of a Slurm partition.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "partition name"}},
            "required": ["name"],
        },
    },
    # TODO (exercise): append a third tool spec here, e.g. "gpu_free" with no inputs,
    # then handle its name in call_tool() below.
]


def call_tool(name, arguments):
    """Return the text payload for a tools/call. In a real server this would hit a DB or API."""
    if name == "storage_quota":
        return "You are using 812 GB of your 2 TB quota on /project (41%)."
    if name == "partition_info":
        p = (arguments or {}).get("name", "unknown")
        table = {
            "test": "test: short debug jobs, has internet egress, max a few nodes.",
            "caslake": "caslake: default CPU partition, has internet egress.",
            "gpu": "gpu: A100/H100 nodes, no default internet egress.",
        }
        return table.get(p, f"No info for partition '{p}'.")
    # TODO (exercise): handle your third tool here.
    return f"Unknown tool: {name}"


def reply(msg_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, msg_id = msg.get("method"), msg.get("id")
        if method == "initialize":
            reply(msg_id, {
                "protocolVersion": msg["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "lab", "version": "1.0.0"},
            })
        elif method == "tools/list":
            reply(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = msg.get("params", {})
            text = call_tool(params.get("name"), params.get("arguments"))
            reply(msg_id, {"content": [{"type": "text", "text": text}]})
        elif msg_id is not None:
            reply(msg_id, {})  # ack anything else that expects a response


if __name__ == "__main__":
    main()
