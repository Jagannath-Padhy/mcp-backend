# ONDC MCP Backend

🚀 Production-ready backend for ONDC shopping with AI-powered assistance using MCP (Model Context Protocol) and Google Gemini.

## Architecture

This backend provides a unified solution combining:
- **MCP-Agent API Server**: REST API endpoints for frontend applications
- **MCP Server**: STDIO-based server with ONDC shopping tools
- **MongoDB**: Document storage for sessions, orders, and products
- **Qdrant**: Vector database for semantic product search
- **ETL Pipeline**: Data initialization and indexing

### Key Design Decisions

1. **Single Container Architecture**: The MCP-Agent and MCP Server run in the same container using `supervisord` because they communicate via STDIO transport.

2. **Hybrid Search**: Combines MongoDB text search with Qdrant vector similarity for comprehensive product discovery.

3. **Session Management**: Persistent session storage in MongoDB with 24-hour TTL.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Google Gemini API key
- (Optional) ONDC Backend API credentials

### Setup

1. Clone the repository:
```bash
cd mcp-backend
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Start services:
```bash
make up        # Start all services
make init      # Initialize sample data
make test      # Test API endpoints
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Session Management
```bash
# Create session
POST /api/v1/sessions
{
  "device_id": "optional-device-id",
  "metadata": {}
}

# Get session
GET /api/v1/sessions/{session_id}

# Delete session
DELETE /api/v1/sessions/{session_id}
```

### Chat Interface
```bash
POST /api/v1/chat
{
  "message": "I want to buy organic rice",
  "session_id": "optional-session-id",
  "device_id": "optional-device-id"
}
```

### Product Search
```bash
POST /api/v1/search
{
  "query": "organic vegetables",
  "filters": {
    "category": "Groceries",
    "price_max": 500
  },
  "limit": 20
}
```

### Cart Management
```bash
POST /api/v1/cart/{device_id}
{
  "action": "add",  # add, remove, update, view
  "item": {
    "product_id": "prod_123",
    "name": "Organic Rice",
    "price": 250
  },
  "quantity": 2
}
```

## MCP Tools Available

The MCP Server provides these tools to the AI agent:

- `start_session`: Initialize shopping session
- `get_session`: Retrieve session information
- `search_products`: Hybrid search with MongoDB + Qdrant
- `add_to_cart`: Add products to cart
- `view_cart`: View cart contents
- `remove_from_cart`: Remove items from cart
- `checkout`: Process order checkout
- `get_order_status`: Track order status

## Development

### Useful Commands

```bash
# View logs
make logs              # All services
make logs-backend      # Backend only
make logs-mongodb      # MongoDB only
make logs-qdrant       # Qdrant only

# Service management
make restart           # Restart all services
make down              # Stop all services
make rebuild           # Rebuild and restart
make clean             # Remove all data

# Debugging
make shell-backend     # Shell into backend container
make shell-mongodb     # MongoDB shell
make status            # Check service status
```

### Project Structure

```
mcp-backend/
├── backend/
│   ├── api/
│   │   └── server.py          # FastAPI server with MCP-Agent
│   ├── mcp_server/
│   │   └── run_server.py      # MCP Server with ONDC tools
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── supervisord.conf      # Process management
│   └── start.sh
├── etl/
│   ├── etl_pipeline.py        # Data initialization
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
├── Makefile
├── .env.example
└── README.md
```

## Environment Variables

```bash
# AI Configuration (Required)
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_API_KEY=same-as-gemini-key

# Backend Integration (Optional)
BACKEND_ENDPOINT=https://your-ondc-backend.com
WIL_API_KEY=your-backend-api-key

# Database Configuration
MONGODB_URI=mongodb://admin:admin123@mongodb:27017/ondc_shopping?authSource=admin
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Feature Flags
HYBRID_SEARCH_ENABLED=true
VECTOR_SEARCH_ENABLED=true

# API Settings
CORS_ORIGINS=*
RATE_LIMIT_PER_MIN=20
SESSION_TTL_HOURS=24
LOG_LEVEL=INFO
```

## Testing

### Manual Testing

```bash
# Test complete flow
make test

# Custom test
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me organic products under 200 rupees",
    "session_id": "test-session-123"
  }'
```

### Postman Collection

Import the included Postman collection for comprehensive API testing:
- Session management flows
- Product search scenarios
- Cart operations
- Order processing

## Troubleshooting

### Common Issues

1. **Services not starting**: Check logs with `make logs`
2. **Connection refused**: Ensure all services are healthy with `make status`
3. **MCP tools not responding**: Check STDIO connection in supervisor logs
4. **Search not working**: Verify Qdrant is running and ETL has completed

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG docker-compose up
```

## Production Deployment

### Recommendations

1. **Session Storage**: Replace in-memory sessions with Redis
2. **Database Security**: Use strong passwords and network isolation
3. **API Gateway**: Add nginx or similar for load balancing
4. **Monitoring**: Integrate Prometheus/Grafana for metrics
5. **Backup**: Regular MongoDB and Qdrant backups

### Scaling

- The API server can be horizontally scaled
- MongoDB and Qdrant support replication
- Use external session storage for stateless API servers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT

## Support

For issues or questions, please open a GitHub issue.