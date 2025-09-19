.PHONY: help build up down restart logs init clean test setup quickstart

help:
	@echo "ONDC MCP Backend - Make Commands"
	@echo ""
	@echo "  make quickstart - One-command setup for new users"
	@echo "  make setup     - Interactive setup wizard"
	@echo "  make build     - Build all Docker images"
	@echo "  make up        - Start all services"
	@echo "  make down      - Stop all services"
	@echo "  make restart   - Restart all services"
	@echo "  make logs      - Show logs from all services"
	@echo "  make init      - Initialize with sample data"
	@echo "  make clean     - Clean up volumes and data"
	@echo "  make test      - Test the API endpoints"

build:
	@echo "Building Docker images..."
	docker-compose build

up:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@docker-compose ps
	@echo ""
	@echo "✅ Services are running!"
	@echo "API endpoint: http://localhost:8001"
	@echo "MongoDB: mongodb://localhost:27017"
	@echo "Qdrant: http://localhost:6333"

down:
	@echo "Stopping services..."
	docker-compose down

restart:
	@echo "Restarting services..."
	docker-compose restart

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-mongodb:
	docker-compose logs -f mongodb

logs-qdrant:
	docker-compose logs -f qdrant

init:
	@echo "Initializing data..."
	docker-compose --profile init up etl
	@echo "✅ Data initialization complete"

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	rm -rf data/mongodb data/qdrant data/sessions
	@echo "✅ Cleanup complete"

test:
	@echo "Testing API endpoints..."
	@echo ""
	@echo "1. Health check:"
	curl -s http://localhost:8001/health | python -m json.tool
	@echo ""
	@echo "2. Create session:"
	curl -s -X POST http://localhost:8001/api/v1/sessions \
		-H "Content-Type: application/json" \
		-d '{"device_id": "test-device"}' | python -m json.tool
	@echo ""
	@echo "3. Chat test:"
	curl -s -X POST http://localhost:8001/api/v1/chat \
		-H "Content-Type: application/json" \
		-d '{"message": "I want to buy organic rice"}' | python -m json.tool

status:
	@echo "Service Status:"
	@docker-compose ps
	@echo ""
	@echo "Port Status:"
	@netstat -an | grep -E "(8001|27017|6333)" | grep LISTEN || echo "Ports not listening"

shell-backend:
	docker-compose exec backend bash

shell-mongodb:
	docker-compose exec mongodb mongosh

rebuild:
	@echo "Rebuilding and restarting services..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Rebuild complete"

setup:
	@echo "Running interactive setup..."
	@bash setup.sh

quickstart:
	@echo "🚀 Quick Start - Setting up ONDC MCP Backend"
	@echo ""
	@if [ ! -f .env ]; then \
		echo "Creating .env file from template..."; \
		cp .env.example .env; \
		echo "⚠️  Please edit .env file and add your API keys:"; \
		echo "   - GEMINI_API_KEY (get from https://makersuite.google.com/app/apikey)"; \
		echo "   - WIL_API_KEY (contact Himira for access)"; \
		echo ""; \
		echo "Then run 'make quickstart' again."; \
		exit 1; \
	fi
	@echo "✅ Configuration found"
	@echo "📦 Building Docker images..."
	@docker-compose build
	@echo "🚀 Starting services..."
	@docker-compose up -d
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 15
	@echo "📊 Loading initial data..."
	@docker-compose up -d etl
	@sleep 5
	@echo ""
	@echo "✅ Setup complete! Services running at:"
	@echo "   • API: http://localhost:8001"
	@echo "   • Qdrant: http://localhost:6333"
	@echo ""
	@echo "Test with: curl http://localhost:8001/health"