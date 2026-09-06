"""
MCP Server for Graph Random Walk Personalized PageRank Skill.
"""

import json
import sys
from client import PersonalizedPageRank

PPR = PersonalizedPageRank()


def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "add_edge",
                    "description": "Add an edge between entities in the memory graph",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "weight": {"type": "number", "default": 1.0},
                            "bidirectional": {"type": "boolean", "default": False}
                        },
                        "required": ["source", "target"]
                    }
                },
                {
                    "name": "top_k_associates",
                    "description": "Compute personalized PageRank and return top-k associated entities",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "seed_nodes": {"type": "object"},
                            "k": {"type": "integer", "default": 5}
                        },
                        "required": ["seed_nodes"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "add_edge":
            PPR.add_edge(
                args["source"],
                args["target"],
                args.get("weight", 1.0),
                args.get("bidirectional", False)
            )
            return {"content": [{"type": "text", "text": json.dumps({"status": "edge_added"})}]}

        elif tool_name == "top_k_associates":
            res = PPR.top_k_associates(args["seed_nodes"], args.get("k", 5))
            return {"content": [{"type": "text", "text": json.dumps([{"node": n, "score": s} for n, s in res])}]}

        return {"error": f"Unknown tool: {tool_name}"}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
