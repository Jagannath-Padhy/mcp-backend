# ONDC Shopping MCP - API Integration Guide

## Overview

This guide provides comprehensive instructions for integrating with the ONDC Shopping MCP server, including setup, configuration, deployment, and practical usage examples for different integration scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Integration Options](#integration-options)
3. [Environment Setup](#environment-setup)
4. [MCP Protocol Integration](#mcp-protocol-integration)
5. [Configuration Management](#configuration-management)
6. [Authentication & Security](#authentication--security)
7. [Deployment Strategies](#deployment-strategies)
8. [Error Handling](#error-handling)
9. [Performance Optimization](#performance-optimization)
10. [Testing & Validation](#testing--validation)
11. [Production Considerations](#production-considerations)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended)
- **Python 3.11+** (for direct installation)
- **Desktop Client Pro** or **MCP Inspector** (for testing)
- **Himira Backend Access** (WIL_API_KEY required)

### 30-Second Setup

```bash
# Clone and start with Docker
git clone <repository-url>
cd ondc-shopping-mcp/

# Copy environment configuration
cp .env.example .env
# Edit .env with your WIL_API_KEY

# Start all services
docker-compose -f docker-compose.unified.yml up -d --build

# Test MCP server
docker exec ondc-mcp-server python -c "
from src.mcp_adapters import initialize_shopping
import asyncio
result = asyncio.run(initialize_shopping())
print(' System operational:', result.get('success'))
"
```

### Verify Installation

```bash
# Check service health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Expected output:
# NAMES                STATUS
# ondc-mcp-server      Up 30 seconds (healthy)
# himira-qdrant        Up 31 seconds
# himira-etl           Up 31 seconds
```

---

## Integration Options

### 1. Desktop Client Integration (Recommended)

**Use Case**: Conversational commerce with AI assistant interface

**Setup**: Configure Desktop Client with MCP server connection

```json
// ~/.config/assistant_desktop_config.json
{
  "mcpServers": {
    "ondc-shopping": {
      "command": "docker",
      "args": ["exec", "-i", "ondc-mcp-server", "python", "-m", "src.mcp_server"],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Benefits**:
- Natural language interaction
- Rich conversation context
- Automatic session management
- Error handling with user guidance

---

### 2. MCP Inspector Integration (Free Testing)

**Use Case**: Development and testing without Desktop Client Pro

**Setup**: Use MCP Inspector web interface

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run MCP Inspector (port 6274)
mcp-inspector

# Connect to: stdio transport
# Command: docker exec -i ondc-mcp-server python -m src.mcp_server
```

**Access**: http://localhost:6274

**Benefits**:
- Free testing environment
- Tool-by-tool validation
- JSON response inspection
- No subscription required

---

### 3. Direct Python Integration

**Use Case**: Embedded shopping functionality in Python applications

**Setup**: Direct import and usage

```python
import asyncio
from src.mcp_adapters import (
    initialize_shopping,
    search_products,
    add_to_cart,
    confirm_order
)

async def shopping_flow():
    # Initialize session
    session = await initialize_shopping()
    session_id = session['session_id']
    
    # Search for products
    products = await search_products(
        session=session_id,
        query="organic honey",
        limit=5
    )
    
    # Add to cart
    if products['success'] and products['products']:
        cart_result = await add_to_cart(
            session=session_id,
            item=products['products'][0],
            quantity=2
        )
        
    return cart_result

# Run the flow
result = asyncio.run(shopping_flow())
print(f"Cart operation: {result['success']}")
```

---

### 4. REST API Wrapper (Custom Implementation)

**Use Case**: HTTP REST API for web applications

**Implementation**: Create FastAPI wrapper around MCP tools

```python
from fastapi import FastAPI, HTTPException
from src.mcp_adapters import *

app = FastAPI(title="ONDC Shopping API")

@app.post("/api/v1/session/initialize")
async def create_session():
    result = await initialize_shopping()
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    return result

@app.post("/api/v1/products/search")
async def search_products_api(query: str, session_id: str, limit: int = 10):
    result = await search_products(
        session=session_id,
        query=query,
        limit=limit
    )
    return result

@app.post("/api/v1/cart/add")
async def add_to_cart_api(session_id: str, item: dict, quantity: int = 1):
    result = await add_to_cart(
        session=session_id,
        item=item,
        quantity=quantity
    )
    return result

# Run with: uvicorn api_wrapper:app --host 0.0.0.0 --port 8000
```

---

## Environment Setup

### Docker Environment (Recommended)

```yaml
# docker-compose.unified.yml
version: '3.8'
services:
  ondc-mcp-server:
    build: .
    environment:
      - WIL_API_KEY=${WIL_API_KEY}
      - BACKEND_ENDPOINT=${BACKEND_ENDPOINT}
      - VECTOR_SEARCH_ENABLED=${VECTOR_SEARCH_ENABLED}
      - LOG_LEVEL=${LOG_LEVEL}
    volumes:
      - ~/.ondc-mcp:/app/.ondc-mcp
    depends_on:
      - himira-qdrant
    healthcheck:
      test: ["CMD", "python", "-c", "from src.config import config; print(config.validate())"]
      interval: 30s
      timeout: 10s
      retries: 3

  himira-qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
```

### Python Virtual Environment

```bash
# Create isolated environment
python -m venv ondc-shopping-env
source ondc-shopping-env/bin/activate  # Linux/Mac
# ondc-shopping-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export WIL_API_KEY="your_api_key_here"
export BACKEND_ENDPOINT="https://hp-buyer-backend-preprod.himira.co.in"
export VECTOR_SEARCH_ENABLED="true"

# Run MCP server
python run_mcp_server.py
```

### Configuration File Setup

```bash
# Create .env file
cp .env.example .env

# Required environment variables
cat > .env << EOF
# Backend Configuration
WIL_API_KEY=your_wil_api_key_here
BACKEND_ENDPOINT=https://hp-buyer-backend-preprod.himira.co.in

# Feature Flags
VECTOR_SEARCH_ENABLED=true
MOCK_PAYMENTS_ENABLED=true
DEBUG_MODE=false

# Performance Settings
MAX_CONCURRENT_REQUESTS=10
CACHE_TTL_SECONDS=300
SESSION_CLEANUP_HOURS=24

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
EOF
```

---

## MCP Protocol Integration

### JSON-RPC Communication

The MCP server communicates using JSON-RPC 2.0 over stdio:

```json
// Request format
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_products",
    "arguments": {
      "query": "organic honey",
      "limit": 5
    }
  }
}

// Response format
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "success": true,
    "message": " Found 5 products for 'organic honey'",
    "products": [...],
    "session_id": "shop_20250116_143052_abc123"
  }
}
```

### Tool Schema Discovery

```python
# Get available tools
tools_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
}

# Response includes all 21 available tools
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "initialize_shopping",
        "description": "Create new shopping session",
        "inputSchema": {
          "type": "object",
          "properties": {},
          "required": []
        }
      },
      {
        "name": "search_products", 
        "description": "Search for products",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {"type": "string"},
            "limit": {"type": "number"}
          },
          "required": ["query"]
        }
      }
    ]
  }
}
```

### Session Management

```python
# Session lifecycle management
class MCPSessionManager:
    def __init__(self):
        self.active_sessions = {}
    
    async def create_session(self) -> str:
        result = await initialize_shopping()
        session_id = result['session_id']
        self.active_sessions[session_id] = {
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'tool_calls': 0
        }
        return session_id
    
    async def call_tool(self, session_id: str, tool_name: str, **params):
        if session_id not in self.active_sessions:
            raise ValueError("Invalid session ID")
        
        # Import tool function dynamically
        tool_func = getattr(mcp_adapters, tool_name)
        result = await tool_func(session=session_id, **params)
        
        # Update session activity
        self.active_sessions[session_id]['last_activity'] = datetime.now()
        self.active_sessions[session_id]['tool_calls'] += 1
        
        return result
```

---

## Configuration Management

### Environment-Based Configuration

```python
# src/config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Backend settings
    wil_api_key: str = os.getenv("WIL_API_KEY", "")
    backend_endpoint: str = os.getenv("BACKEND_ENDPOINT", 
        "https://hp-buyer-backend-preprod.himira.co.in")
    
    # Feature flags
    vector_search_enabled: bool = os.getenv("VECTOR_SEARCH_ENABLED", "true").lower() == "true"
    mock_payments_enabled: bool = os.getenv("MOCK_PAYMENTS_ENABLED", "true").lower() == "true"
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # Performance settings
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    session_cleanup_hours: int = int(os.getenv("SESSION_CLEANUP_HOURS", "24"))
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.wil_api_key:
            raise ValueError("WIL_API_KEY is required")
        if not self.backend_endpoint:
            raise ValueError("BACKEND_ENDPOINT is required")
        return True

# Global config instance
config = Config()
```

### Runtime Configuration Updates

```python
# Dynamic configuration updates
class ConfigManager:
    def __init__(self):
        self.config = Config()
        self.watchers = []
    
    def update_config(self, **kwargs):
        """Update configuration at runtime"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self._notify_watchers(key, value)
    
    def register_watcher(self, callback):
        """Register configuration change callback"""
        self.watchers.append(callback)
    
    def _notify_watchers(self, key, value):
        for watcher in self.watchers:
            watcher(key, value)

# Usage
config_manager = ConfigManager()
config_manager.update_config(debug_mode=True, log_level="DEBUG")
```

---

## Authentication & Security

### API Key Management

```python
# Secure API key handling
class APIKeyManager:
    def __init__(self):
        self.api_key = self._load_api_key()
    
    def _load_api_key(self) -> str:
        # Priority order: env var > file > user input
        api_key = os.getenv("WIL_API_KEY")
        if api_key:
            return api_key
        
        # Try loading from secure file
        key_file = os.path.expanduser("~/.ondc-mcp/api_key")
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                return f.read().strip()
        
        raise ValueError("API key not found. Set WIL_API_KEY environment variable.")
    
    def get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
```

### Session Security

```python
# Secure session management
import jwt
import secrets
from datetime import datetime, timedelta

class SecureSessionManager:
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
    
    def create_session_token(self, session_id: str) -> str:
        payload = {
            'session_id': session_id,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_session_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            expires_at = datetime.fromisoformat(payload['expires_at'])
            if datetime.utcnow() > expires_at:
                raise ValueError("Session token expired")
            return payload
        except jwt.InvalidTokenError:
            raise ValueError("Invalid session token")
```

### Rate Limiting

```python
# Rate limiting implementation
from collections import defaultdict
from time import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time()
        client_requests = self.requests[client_id]
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests 
                             if now - req_time < self.window_seconds]
        
        # Check if under the limit
        if len(client_requests) < self.max_requests:
            client_requests.append(now)
            return True
        
        return False
    
    def get_reset_time(self, client_id: str) -> int:
        if not self.requests[client_id]:
            return 0
        oldest_request = min(self.requests[client_id])
        return int(oldest_request + self.window_seconds)
```

---

## Deployment Strategies

### 1. Single Container Deployment

```dockerfile
# Dockerfile.standalone
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY run_mcp_server.py .

EXPOSE 8080
CMD ["python", "run_mcp_server.py"]
```

```bash
# Build and run
docker build -f Dockerfile.standalone -t ondc-mcp:latest .
docker run -d \
  --name ondc-mcp-server \
  -p 8080:8080 \
  -e WIL_API_KEY="your_key" \
  -v ~/.ondc-mcp:/app/.ondc-mcp \
  ondc-mcp:latest
```

### 2. Multi-Service Deployment

```yaml
# docker-compose.production.yml
version: '3.8'
services:
  ondc-mcp-server:
    image: ondc-mcp:latest
    restart: unless-stopped
    environment:
      - WIL_API_KEY=${WIL_API_KEY}
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - redis
      - qdrant
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  qdrant:
    image: qdrant/qdrant:latest
    restart: unless-stopped
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ondc-mcp-server

volumes:
  redis_data:
  qdrant_data:
```

### 3. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ondc-mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ondc-mcp-server
  template:
    metadata:
      labels:
        app: ondc-mcp-server
    spec:
      containers:
      - name: ondc-mcp-server
        image: ondc-mcp:latest
        ports:
        - containerPort: 8080
        env:
        - name: WIL_API_KEY
          valueFrom:
            secretKeyRef:
              name: ondc-secrets
              key: wil-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ondc-mcp-service
spec:
  selector:
    app: ondc-mcp-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### 4. Serverless Deployment (AWS Lambda)

```python
# lambda_handler.py
import json
import asyncio
from src.mcp_adapters import *

def lambda_handler(event, context):
    """AWS Lambda handler for ONDC MCP tools"""
    
    tool_name = event.get('tool_name')
    parameters = event.get('parameters', {})
    
    # Get tool function
    tool_func = globals().get(tool_name)
    if not tool_func:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'success': False,
                'message': f'Unknown tool: {tool_name}'
            })
        }
    
    # Execute tool asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(tool_func(**parameters))
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'message': f'Error: {str(e)}'
            })
        }
    finally:
        loop.close()
```

---

## Error Handling

### Comprehensive Error Strategy

```python
# Error handling framework
from enum import Enum
from typing import Optional, Dict, Any

class ErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    BACKEND_ERROR = "backend_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SESSION_ERROR = "session_error"
    NETWORK_ERROR = "network_error"
    INTERNAL_ERROR = "internal_error"

class MCPError(Exception):
    def __init__(self, error_type: ErrorType, message: str, 
                 session_id: Optional[str] = None, 
                 details: Optional[Dict] = None):
        self.error_type = error_type
        self.message = message
        self.session_id = session_id
        self.details = details or {}
        super().__init__(message)

def handle_mcp_error(error: Exception, session_id: str = None) -> Dict[str, Any]:
    """Convert exceptions to standardized error responses"""
    
    if isinstance(error, MCPError):
        return {
            'success': False,
            'message': f" {error.message}",
            'session_id': error.session_id or session_id,
            'error_type': error.error_type.value,
            'details': error.details
        }
    
    elif isinstance(error, ValueError):
        return {
            'success': False,
            'message': f" Validation error: {str(error)}",
            'session_id': session_id,
            'error_type': ErrorType.VALIDATION_ERROR.value
        }
    
    elif isinstance(error, ConnectionError):
        return {
            'success': False,
            'message': " Network connectivity issue. Please try again.",
            'session_id': session_id,
            'error_type': ErrorType.NETWORK_ERROR.value,
            'retry_suggested': True
        }
    
    else:
        return {
            'success': False,
            'message': " An unexpected error occurred. Please contact support.",
            'session_id': session_id,
            'error_type': ErrorType.INTERNAL_ERROR.value
        }
```

### Retry Logic with Exponential Backoff

```python
import asyncio
import random
from functools import wraps

def async_retry(max_attempts: int = 3, backoff_factor: float = 2.0, 
                max_delay: float = 60.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        break
                    
                    # Calculate backoff delay with jitter
                    delay = min(backoff_factor ** attempt, max_delay)
                    jitter = random.uniform(0, 0.1) * delay
                    await asyncio.sleep(delay + jitter)
            
            raise last_exception
        return wrapper
    return decorator

# Usage
@async_retry(max_attempts=3, backoff_factor=2.0)
async def make_backend_request(endpoint: str, data: dict):
    # Implementation with automatic retry
    pass
```

### Circuit Breaker Pattern

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise MCPError(ErrorType.BACKEND_ERROR, 
                             "Service temporarily unavailable (circuit breaker open)")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

---

## Performance Optimization

### Caching Strategy

```python
import asyncio
from typing import Any, Optional
from datetime import datetime, timedelta

class AsyncCache:
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, expires_at = self.cache[key]
            if datetime.now() < expires_at:
                return value
            else:
                del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expires_at)
    
    async def get_or_set(self, key: str, func, ttl: Optional[int] = None) -> Any:
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        value = await func() if asyncio.iscoroutinefunction(func) else func()
        await self.set(key, value, ttl)
        return value

# Global cache instance
cache = AsyncCache(default_ttl=300)

# Usage in tools
async def search_products_cached(query: str, **kwargs):
    cache_key = f"search:{hash(query)}:{kwargs}"
    return await cache.get_or_set(
        cache_key,
        lambda: search_products_uncached(query, **kwargs),
        ttl=600  # 10 minutes
    )
```

### Connection Pooling

```python
import aiohttp
from typing import Optional

class HTTPClientManager:
    def __init__(self, max_connections: int = 100, 
                 max_connections_per_host: int = 30):
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections_per_host,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'ONDC-MCP-Server/1.0',
                'Accept': 'application/json'
            }
        )
    
    async def make_request(self, method: str, url: str, **kwargs) -> dict:
        async with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
    
    async def close(self):
        await self.session.close()

# Global HTTP client
http_client = HTTPClientManager()
```

### Async Task Management

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Any

class TaskManager:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def run_parallel(self, tasks: List[Callable]) -> List[Any]:
        """Run multiple async tasks in parallel"""
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def run_cpu_bound(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-bound tasks in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def run_with_timeout(self, coro, timeout: float) -> Any:
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise MCPError(ErrorType.NETWORK_ERROR, 
                          f"Operation timed out after {timeout} seconds")

# Global task manager
task_manager = TaskManager(max_workers=10)
```

---

## Testing & Validation

### Unit Testing Framework

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.mcp_adapters import *

@pytest.fixture
def mock_session():
    return {
        'session_id': 'test_session_123',
        'user_authenticated': True,
        'auth_token': 'test_token'
    }

@pytest.mark.asyncio
async def test_initialize_shopping():
    """Test session initialization"""
    result = await initialize_shopping()
    
    assert result['success'] is True
    assert 'session_id' in result
    assert result['session_id'].startswith('shop_')

@pytest.mark.asyncio
async def test_search_products(mock_session):
    """Test product search functionality"""
    with patch('src.services.search_service.SearchService.search_products') as mock_search:
        mock_search.return_value = {
            'success': True,
            'products': [
                {'id': 'test_product', 'name': 'Test Product', 'price': 100.0}
            ]
        }
        
        result = await search_products(
            session=mock_session['session_id'],
            query='test product'
        )
        
        assert result['success'] is True
        assert len(result['products']) == 1
        assert result['products'][0]['name'] == 'Test Product'

@pytest.mark.asyncio
async def test_add_to_cart(mock_session):
    """Test cart functionality"""
    test_item = {
        'id': 'test_product',
        'name': 'Test Product',
        'price': 100.0,
        'provider_id': 'test_provider'
    }
    
    result = await add_to_cart(
        session=mock_session['session_id'],
        item=test_item,
        quantity=2
    )
    
    assert result['success'] is True
    assert result['cart_summary']['total_items'] >= 2

# Run tests
# pytest tests/ -v --asyncio-mode=auto
```

### Integration Testing

```python
import pytest
import asyncio
from src.mcp_adapters import *

@pytest.mark.integration
class TestCompleteShoppingFlow:
    """Test complete shopping workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_journey(self):
        """Test end-to-end shopping flow"""
        
        # 1. Initialize session
        session_result = await initialize_shopping()
        assert session_result['success']
        session_id = session_result['session_id']
        
        # 2. Search for products
        search_result = await search_products(
            session=session_id,
            query='organic honey',
            limit=3
        )
        assert search_result['success']
        assert len(search_result['products']) > 0
        
        # 3. Add product to cart
        product = search_result['products'][0]
        cart_result = await add_to_cart(
            session=session_id,
            item=product,
            quantity=1
        )
        assert cart_result['success']
        
        # 4. Authenticate user
        auth_result = await phone_login(
            phone='9876543210',
            session=session_id
        )
        assert auth_result['success']
        
        # 5. Select items for order
        select_result = await select_items_for_order(
            session=session_id,
            delivery_city='Bangalore',
            delivery_state='Karnataka',
            delivery_pincode='560001'
        )
        assert select_result['success']
        
        # 6. Initialize order
        init_result = await initialize_order(
            session=session_id,
            customer_name='Test User',
            delivery_address='Test Address',
            phone='9876543210',
            email='test@example.com'
        )
        assert init_result['success']
        
        # 7. Create payment
        payment_result = await create_payment(
            session=session_id,
            payment_method='razorpay'
        )
        assert payment_result['success']
        
        # 8. Confirm order
        confirm_result = await confirm_order(
            session=session_id,
            payment_status='PAID'
        )
        assert confirm_result['success']
        assert 'order_id' in confirm_result
```

### Load Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

async def load_test_search(concurrent_users: int = 10, 
                          requests_per_user: int = 5):
    """Load test search functionality"""
    
    async def user_session():
        session_times = []
        
        for _ in range(requests_per_user):
            start_time = time.time()
            
            try:
                result = await search_products(
                    query='test product',
                    limit=10
                )
                success = result['success']
            except Exception:
                success = False
            
            end_time = time.time()
            session_times.append({
                'duration': end_time - start_time,
                'success': success
            })
        
        return session_times
    
    # Run concurrent user sessions
    tasks = [user_session() for _ in range(concurrent_users)]
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    all_times = []
    success_count = 0
    total_requests = 0
    
    for user_results in results:
        for request in user_results:
            all_times.append(request['duration'])
            if request['success']:
                success_count += 1
            total_requests += 1
    
    return {
        'total_requests': total_requests,
        'successful_requests': success_count,
        'success_rate': success_count / total_requests,
        'avg_response_time': statistics.mean(all_times),
        'p95_response_time': statistics.quantiles(all_times, n=20)[18],
        'max_response_time': max(all_times),
        'min_response_time': min(all_times)
    }

# Run load test
if __name__ == "__main__":
    results = asyncio.run(load_test_search(concurrent_users=20, requests_per_user=10))
    print(f"Load test results: {results}")
```

---

## Production Considerations

### Monitoring & Observability

```python
import time
import logging
from functools import wraps
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]

class MetricsCollector:
    def __init__(self):
        self.metrics = []
    
    def record_counter(self, name: str, value: float = 1, tags: Dict = None):
        self.metrics.append(MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        ))
    
    def record_timing(self, name: str, duration: float, tags: Dict = None):
        self.metrics.append(MetricPoint(
            name=f"{name}.duration",
            value=duration,
            timestamp=time.time(),
            tags=tags or {}
        ))
    
    def get_metrics(self) -> list:
        return self.metrics.copy()

# Global metrics collector
metrics = MetricsCollector()

def monitor_performance(func_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                metrics.record_counter(f"{function_name}.success")
                return result
            except Exception as e:
                metrics.record_counter(f"{function_name}.error", 
                                     tags={'error_type': type(e).__name__})
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_timing(function_name, duration)
        
        return wrapper
    return decorator

# Usage
@monitor_performance("search_products")
async def search_products_monitored(*args, **kwargs):
    return await search_products(*args, **kwargs)
```

### Health Checks

```python
from typing import Dict, List
import asyncio

class HealthChecker:
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks[name] = check_func
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'details': result if isinstance(result, dict) else {}
                }
                if not result:
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
                overall_healthy = False
        
        return {
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': results,
            'timestamp': time.time()
        }

# Health check implementations
async def check_backend_connectivity():
    """Check if Himira backend is accessible"""
    try:
        # Simple connectivity test
        response = await http_client.make_request(
            'GET', 
            f"{config.backend_endpoint}/health"
        )
        return True
    except Exception:
        return False

async def check_session_storage():
    """Check if session storage is working"""
    try:
        test_session = await initialize_shopping()
        return test_session['success']
    except Exception:
        return False

async def check_vector_search():
    """Check if vector search is working"""
    if not config.vector_search_enabled:
        return True
    
    try:
        # Test vector search connectivity
        # Implementation depends on vector search setup
        return True
    except Exception:
        return False

# Register health checks
health_checker = HealthChecker()
health_checker.register_check('backend', check_backend_connectivity)
health_checker.register_check('sessions', check_session_storage)
health_checker.register_check('vector_search', check_vector_search)
```

### Logging & Audit Trail

```python
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

class AuditLogger:
    def __init__(self, logger_name: str = "ondc_mcp_audit"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Create structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler for audit logs
        file_handler = logging.FileHandler('audit.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_tool_call(self, tool_name: str, session_id: str, 
                     parameters: Dict, result: Dict, 
                     user_info: Optional[Dict] = None):
        """Log MCP tool call for audit purposes"""
        audit_entry = {
            'event_type': 'tool_call',
            'timestamp': datetime.utcnow().isoformat(),
            'tool_name': tool_name,
            'session_id': session_id,
            'parameters': self._sanitize_parameters(parameters),
            'success': result.get('success', False),
            'user_info': user_info or {},
            'execution_time_ms': result.get('execution_time_ms')
        }
        
        self.logger.info(json.dumps(audit_entry))
    
    def log_authentication(self, session_id: str, phone: str, 
                          success: bool, method: str = 'phone'):
        """Log authentication attempts"""
        audit_entry = {
            'event_type': 'authentication',
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'phone': self._mask_phone(phone),
            'method': method,
            'success': success
        }
        
        self.logger.info(json.dumps(audit_entry))
    
    def _sanitize_parameters(self, params: Dict) -> Dict:
        """Remove sensitive information from parameters"""
        sanitized = params.copy()
        sensitive_fields = ['password', 'token', 'api_key', 'phone']
        
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = '[REDACTED]'
        
        return sanitized
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number for privacy"""
        if len(phone) >= 10:
            return f"***-***-{phone[-4:]}"
        return "[REDACTED]"

# Global audit logger
audit_logger = AuditLogger()
```

### Backup & Recovery

```python
import os
import json
import shutil
import zipfile
from datetime import datetime
from typing import Dict, List
import asyncio

class BackupManager:
    def __init__(self, backup_dir: str = None):
        self.backup_dir = backup_dir or os.path.expanduser("~/.ondc-mcp/backups")
        os.makedirs(self.backup_dir, exist_ok=True)
    
    async def create_backup(self, include_sessions: bool = True,
                           include_logs: bool = True) -> str:
        """Create comprehensive system backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"ondc_mcp_backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.zip")
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Backup configuration
            if os.path.exists('.env'):
                zipf.write('.env', 'config/.env')
            
            # Backup sessions
            if include_sessions:
                sessions_dir = os.path.expanduser("~/.ondc-mcp/sessions")
                if os.path.exists(sessions_dir):
                    for root, dirs, files in os.walk(sessions_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_path = os.path.relpath(file_path, 
                                                     os.path.dirname(sessions_dir))
                            zipf.write(file_path, f"sessions/{arc_path}")
            
            # Backup logs
            if include_logs:
                log_files = ['audit.log', 'mcp_server.log', 'error.log']
                for log_file in log_files:
                    if os.path.exists(log_file):
                        zipf.write(log_file, f"logs/{log_file}")
            
            # Create backup metadata
            metadata = {
                'backup_timestamp': timestamp,
                'backup_type': 'full',
                'includes_sessions': include_sessions,
                'includes_logs': include_logs,
                'system_info': {
                    'python_version': '3.11+',
                    'mcp_version': '1.0.0'
                }
            }
            
            zipf.writestr('backup_metadata.json', json.dumps(metadata, indent=2))
        
        return backup_path
    
    async def restore_backup(self, backup_path: str) -> bool:
        """Restore system from backup"""
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Read metadata
                metadata = json.loads(zipf.read('backup_metadata.json'))
                
                # Restore configuration
                if 'config/.env' in zipf.namelist():
                    zipf.extract('config/.env', '.')
                    shutil.move('config/.env', '.env')
                
                # Restore sessions
                sessions_files = [f for f in zipf.namelist() 
                                if f.startswith('sessions/')]
                for session_file in sessions_files:
                    zipf.extract(session_file, 
                               os.path.expanduser("~/.ondc-mcp/"))
                
                return True
        except Exception as e:
            logging.error(f"Backup restoration failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List available backups"""
        backups = []
        for file in os.listdir(self.backup_dir):
            if file.endswith('.zip') and file.startswith('ondc_mcp_backup_'):
                backup_path = os.path.join(self.backup_dir, file)
                stat = os.stat(backup_path)
                
                backups.append({
                    'filename': file,
                    'path': backup_path,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)

# Global backup manager
backup_manager = BackupManager()
```

This comprehensive API integration guide provides everything needed to successfully integrate with the ONDC Shopping MCP system, from basic setup to production deployment strategies.