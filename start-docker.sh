#!/bin/bash

# ONDC MCP Backend - Docker Compose Startup Script
# This script provides an out-of-the-box experience for starting the system

set -e  # Exit on error

echo "========================================="
echo "ONDC MCP Backend - Docker Compose Setup"
echo "========================================="

# Check if .env exists in parent directory
if [ ! -f "../.env" ]; then
    echo "❌ Error: ../.env file not found!"
    echo "Please ensure the .env file exists in the parent directory with required API keys."
    exit 1
fi

echo "✅ Found .env file in parent directory"

# Validate environment variables
echo "Validating environment configuration..."
source ../.env
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY is not set in .env"
    exit 1
fi
if [ -z "$WIL_API_KEY" ]; then
    echo "❌ Error: WIL_API_KEY is not set in .env"
    exit 1
fi
echo "✅ Environment variables validated"

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose down 2>/dev/null || true

# Build images
echo "Building Docker images..."
docker-compose build

# Start MongoDB and Qdrant first
echo "Starting database services..."
docker-compose up -d mongodb qdrant

# Wait for databases to be ready
echo "Waiting for databases to be ready..."
sleep 10

# Run ETL initialization (optional)
read -p "Do you want to run ETL data initialization? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running ETL data initialization..."
    docker-compose --profile init up etl
fi

# Start backend service
echo "Starting backend service..."
docker-compose up -d backend

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
sleep 15

# Check service health
echo "Checking service health..."
docker-compose ps

# Test API endpoint
echo "Testing API endpoint..."
curl -f http://localhost:8001/health || echo "⚠️  API might still be initializing..."

echo ""
echo "========================================="
echo "✅ ONDC MCP Backend is starting up!"
echo "========================================="
echo ""
echo "Services:"
echo "  - MongoDB:     http://localhost:27017"
echo "  - Qdrant:      http://localhost:6333"
echo "  - Backend API: http://localhost:8001"
echo ""
echo "Useful commands:"
echo "  - View logs:        docker-compose logs -f backend"
echo "  - Stop services:    docker-compose down"
echo "  - Restart backend:  docker-compose restart backend"
echo ""
echo "API Documentation: http://localhost:8001/docs"
echo ""