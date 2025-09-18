#!/bin/bash
# Silent MCP wrapper - ensures clean JSON-RPC output
# Suppresses all stderr output to prevent log contamination

# Set environment to minimize Python output
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=0
export PYTHONWARNINGS=ignore

# Run MCP server with stderr redirected to /dev/null
exec python -m src.mcp_server 2>/dev/null