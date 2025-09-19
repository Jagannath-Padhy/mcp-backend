#!/bin/bash

# ONDC MCP Backend Setup Script
# This script helps you set up the environment for first-time users

set -e

echo "======================================"
echo "  ONDC MCP Backend Setup Assistant   "
echo "======================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Check if .env file exists
if [ -f .env ]; then
    echo "📄 .env file already exists"
    read -p "Do you want to reconfigure it? (y/N): " reconfigure
    if [[ ! "$reconfigure" =~ ^[Yy]$ ]]; then
        echo "Using existing .env file"
    else
        mv .env .env.backup
        echo "Backed up existing .env to .env.backup"
    fi
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    
    echo ""
    echo "===== API Key Configuration ====="
    echo ""
    echo "The system requires two API keys to function:"
    echo ""
    echo "1. GEMINI API KEY (for AI and vector search)"
    echo "   Get your key at: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Enter your Gemini API Key: " gemini_key
    
    if [ -z "$gemini_key" ]; then
        echo "⚠️  No Gemini API key provided. Vector search will not work."
        echo "   You can add it later by editing the .env file"
    else
        # Update both GEMINI_API_KEY and GOOGLE_API_KEY
        sed -i.bak "s/GEMINI_API_KEY=.*/GEMINI_API_KEY=$gemini_key/" .env
        sed -i.bak "s/GOOGLE_API_KEY=.*/GOOGLE_API_KEY=$gemini_key/" .env
        echo "✅ Gemini API key configured"
    fi
    
    echo ""
    echo "2. HIMIRA BACKEND API KEY (for ONDC operations)"
    echo "   Contact Himira for access or use test credentials"
    echo ""
    read -p "Enter your Himira API Key (WIL_API_KEY): " himira_key
    
    if [ -z "$himira_key" ]; then
        echo "⚠️  No Himira API key provided. ONDC operations will not work."
        echo "   You can add it later by editing the .env file"
    else
        sed -i.bak "s/WIL_API_KEY=.*/WIL_API_KEY=$himira_key/" .env
        echo "✅ Himira API key configured"
    fi
    
    # Clean up backup files
    rm -f .env.bak
    
    echo ""
    echo "✅ Configuration complete!"
fi

# Verify critical keys are set
echo ""
echo "Verifying configuration..."

gemini_configured=$(grep -E "^GEMINI_API_KEY=.+" .env | grep -v "your-gemini-api-key-here" || echo "")
himira_configured=$(grep -E "^WIL_API_KEY=.+" .env | grep -v "your-himira-api-key-here" || echo "")

if [ -z "$gemini_configured" ]; then
    echo "⚠️  Warning: Gemini API key not configured - vector search will not work"
fi

if [ -z "$himira_configured" ]; then
    echo "⚠️  Warning: Himira API key not configured - ONDC operations will not work"
fi

echo ""
echo "===== Starting Services ====="
echo ""

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Show service status
docker-compose ps

# Run ETL to load initial data
echo ""
echo "Loading initial product data..."
docker-compose up -d etl

echo ""
echo "======================================"
echo "    🎉 Setup Complete! 🎉            "
echo "======================================"
echo ""
echo "Services are running at:"
echo "  • API Server: http://localhost:8001"
echo "  • Qdrant UI: http://localhost:6333/dashboard"
echo ""
echo "Test the setup:"
echo "  curl http://localhost:8001/health"
echo ""
echo "Or try the Chat API:"
echo '  curl -X POST http://localhost:8001/api/v1/chat \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"message": "Show me organic products"}'"'"
echo ""
echo "To stop services: make down"
echo "To view logs: make logs"
echo ""