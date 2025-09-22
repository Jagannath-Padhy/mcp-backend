#!/bin/bash

echo "Starting ONDC MCP Backend Services..."

# Create necessary directories
mkdir -p /app/logs /app/sessions

# Validate environment
echo "Validating environment configuration..."
python /app/validate_env.py
if [ $? -ne 0 ]; then
    echo "Environment validation failed! Please check your .env configuration."
    exit 1
fi

# Generate MCP agent config from environment variables
echo "Generating MCP agent configuration..."
python /app/generate_config.py
if [ $? -ne 0 ]; then
    echo "Config generation failed! Please check your environment variables."
    exit 1
fi

# Generate supervisor configuration from template
echo "Generating supervisor configuration..."
envsubst < /app/supervisord.conf.template > /app/supervisord.conf
if [ $? -ne 0 ]; then
    echo "Supervisor config generation failed! Please check your environment variables."
    exit 1
fi

# Start supervisor
echo "Starting supervisor to manage services..."
exec /usr/bin/supervisord -c /app/supervisord.conf