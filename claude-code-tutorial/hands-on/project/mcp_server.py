#!/usr/bin/env python3
"""A minimal Model Context Protocol (MCP) server for the LakeWatch project.

It speaks newline-delimited JSON-RPC 2.0 over stdin/stdout — the same protocol Claude
Code uses to talk to any MCP server — and exposes two small "lab" tools that answer
questions the data files alone cannot. It has NO third-party dependencies.

Try it (Bonus B):
    claude -p "Where is site charlie and how deep is it? Use the MCP tool." \
        --mcp-config '{"mcpServers":{"lakewatch":{"command":"python3","args":["mcp_server.py"]}}}' \
        --strict-mcp-config --allowedTools "mcp__lakewatch__*" --model haiku

EXERCISE: add a third tool (see the TODO near the bottom), restart, and ask Claude to use it.
"""
import json
import sys

SITES = {
    "alpha":   {"name": "Alpha",   "lat": 41.79, "lon": -87.59, "max_depth_m": 4.2},
    "bravo":   {"name": "Bravo",   "lat": 41.88, "lon": -87.63, "max_depth_m": 7.8},
    "charlie": {"name": "Charlie", "lat": 41.85, "lon": -87.65, "max_depth_m": 3.1},
}

TOOLS = [
    {
        "name": "site_metadata",
        "description": "Return location and depth metadata for a monitoring site.",
        "inputSchema": {
            "type": "object",
            "properties": {"site": {"type": "string", "description": "alpha, bravo, or charlie"}},
            "required": ["site"],
        },
    },
    {
        "name": "sensor_status",
        "description": "Return which sensors are currently reporting as online.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    # TODO (Bonus B exercise): append a third tool spec here — e.g. "site_list" with no
    # inputs that returns all site names — then handle its name in call_tool() below.
]


def call_tool(name, arguments):
    """Return the text payload for a tools/call. A real server would hit a DB or API."""
    if name == "site_metadata":
        site = (arguments or {}).get("site", "").strip().lower()
        meta = SITES.get(site)
        if not meta:
            return f"No metadata for site '{site}'. Known sites: {', '.join(SITES)}."
        return (f"Site {meta['name']}: lat {meta['lat']}, lon {meta['lon']}, "
                f"maximum depth {meta['max_depth_m']} m.")
    if name == "sensor_status":
        return "Online: temperature, dissolved-oxygen, pH. Offline for calibration: turbidity (site charlie)."
    # TODO (Bonus B exercise): handle your third tool here.
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
                "serverInfo": {"name": "lakewatch", "version": "1.0.0"},
            })
        elif method == "tools/list":
            reply(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = msg.get("params", {})
            text = call_tool(params.get("name"), params.get("arguments"))
            reply(msg_id, {"content": [{"type": "text", "text": text}]})
        elif msg_id is not None:
            reply(msg_id, {})  # acknowledge anything else that expects a response


if __name__ == "__main__":
    main()
