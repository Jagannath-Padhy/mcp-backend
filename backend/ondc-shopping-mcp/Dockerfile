FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mcp && \
    mkdir -p /app /home/mcp/.ondc-mcp/sessions && \
    chown -R mcp:mcp /app /home/mcp/.ondc-mcp

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=mcp:mcp . .

# Make the silent wrapper script executable and dual mode runner
RUN chmod +x /app/silent_mcp.sh && \
    chmod +x /app/run_mcp_dual.py

# Switch to non-root user
USER mcp

# Set Python path
ENV PYTHONPATH=/app

# MCP servers communicate via stdio, not HTTP
# Port 8080 kept for potential future HTTP endpoint
EXPOSE 8080

# Health check - simple Python check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Keep container running - MCP server will be spawned by Claude via docker exec
CMD ["tail", "-f", "/dev/null"]