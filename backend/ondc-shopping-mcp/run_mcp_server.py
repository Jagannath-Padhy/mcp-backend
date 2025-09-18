#!/usr/bin/env python3
"""Run the ONDC Shopping MCP Server - FastMCP Implementation"""

import sys
from src.mcp_server_fastmcp import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)