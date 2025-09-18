FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl for healthchecks, supervisor for process management)
RUN apt-get update && apt-get install -y \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/sessions /app/data /app/logs

# Copy supervisor configuration
COPY supervisord.conf /app/supervisord.conf
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose API port
EXPOSE 8001

# Run supervisor to manage both MCP server and API server
CMD ["/app/start.sh"]