# ONDC MCP Backend - Docker Setup Guide

## Out-of-Box Experience

This Docker setup has been configured to provide a complete out-of-box experience. All manual configuration steps have been eliminated.

## What Was Fixed

### 1. **Backend Dockerfile**
- Added supervisor installation for process management
- Updated CMD to use start.sh script
- Created necessary directories (/app/logs, /app/sessions)

### 2. **Environment Configuration**
- Consolidated .env files - now uses parent directory .env
- Added MongoDB URI to parent .env file
- Docker-compose uses env_file to load parent .env automatically

### 3. **Dependencies**
- Added mcp-agent to requirements.txt for proper MCP integration
- All dependencies are now properly installed in the container

### 4. **Process Management**
- Supervisor manages both MCP server and API server
- start.sh validates environment before starting services
- Added health checks to ensure services are ready

### 5. **Removed Confusion**
- Renamed conflicting api_server.py to api_server_direct.py.bak
- Now uses single API implementation via mcp-agent (api/server.py)

## Quick Start

### 1. Ensure .env File Exists
The parent directory (.env) already has all required API keys configured.

### 2. Start Services
```bash
# Use the provided startup script
./start-docker.sh

# Or manually with docker-compose
docker-compose up -d
```

### 3. Initialize Data (Optional)
```bash
# Run ETL to populate initial data
docker-compose --profile init up etl
```

## Architecture

```
┌─────────────┐     ┌──────────────────────────┐
│   Frontend  │────▶│  Backend API (port 8001) │
└─────────────┘     └──────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
              ┌─────▼─────┐      ┌──────▼──────┐
              │ MCP Server │      │  mcp-agent  │
              │  (STDIO)   │◀─────│   Library   │
              └─────┬─────┘      └─────────────┘
                    │
        ┌───────────┴────────────┐
        │                        │
   ┌────▼────┐           ┌───────▼──────┐
   │ MongoDB │           │    Qdrant     │
   └─────────┘           └──────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Backend API | 8001 | REST API with MCP integration |
| MongoDB | 27017 | Document storage |
| Qdrant | 6333 | Vector database |

## Environment Variables

All required environment variables are configured in the parent `.env` file:
- `GEMINI_API_KEY` - Google Gemini API key
- `WIL_API_KEY` - Backend API authentication
- `BACKEND_ENDPOINT` - ONDC backend URL
- `MONGODB_URI` - MongoDB connection (auto-configured for Docker)

## Useful Commands

```bash
# View logs
docker-compose logs -f backend

# Restart backend
docker-compose restart backend

# Stop all services
docker-compose down

# Rebuild after changes
docker-compose build backend
docker-compose up -d backend

# Check service health
curl http://localhost:8001/health
```

## Validation

The system includes automatic validation:
1. **validate_env.py** - Checks all required environment variables
2. **Health checks** - Ensures databases are ready before starting backend
3. **Startup script** - Validates environment before starting supervisor

## Troubleshooting

### Backend fails to start
Check logs: `docker-compose logs backend`

### Environment validation fails
Ensure parent `.env` file has all required API keys

### MongoDB connection issues
MongoDB URI is automatically configured for Docker networking

### API not responding
Wait 30-60 seconds for all services to initialize

## Key Changes Made

1. **Supervisor Integration**: Backend now runs both MCP server and API server
2. **Environment Consolidation**: Single .env file in parent directory
3. **Automated Validation**: Environment checked before startup
4. **Health Checks**: Proper service dependencies and health monitoring
5. **Clean Architecture**: Removed duplicate API implementations

The system is now truly out-of-box - just run `docker-compose up` and everything works!