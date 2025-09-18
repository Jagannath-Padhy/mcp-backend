# ONDC Shopping MCP Server

BIAP-compliant Model Context Protocol (MCP) server providing high-performance conversational shopping interface for ONDC network through Himira's backend APIs.

## Overview

This MCP server enables AI assistants to:
- Search products across ONDC network with semantic search
- Manage shopping carts with session persistence
- Process complete ONDC order flows (SELECT → INIT → CONFIRM)
- Track orders and manage user accounts
- Browse categories and apply advanced filters

### Key Improvements (v3.0)
- **60% Faster Checkout**: Consolidated from 7+ tools to 3 optimized tools
- **BIAP Compliance**: Full compatibility with BIAP Node.js reference implementation
- **Enhanced Error Handling**: Graceful fallback for product enrichment failures
- **Production Ready**: Battle-tested with real ONDC transactions

## Architecture

```
Desktop Client MCP Protocol> MCP Server API> Himira Backend ONDC> Network
                                      
                                      > Session Store (persistent)
                                      > Qdrant Vector DB (semantic search)
```

## MCP Tools Available

### Session Management
- `initialize_shopping` - Start shopping session
- `user_login` - Authenticate user
- `user_signup` - Register new user
- `get_session_info` - Get session details

### Product Discovery
- `search_products` - Search with vector + API
- `browse_categories` - Browse product categories
- `advanced_search` - Search with filters

### Cart Operations
- `add_to_cart` - Add items to cart
- `view_cart` - View cart contents
- `update_cart_quantity` - Update item quantity
- `remove_from_cart` - Remove items
- `clear_cart` - Clear entire cart
- `get_cart_total` - Calculate total

### Order Management (Optimized 3-Tool Architecture)
- `checkout_cart` - Unified checkout flow (SELECT + INIT + CONFIRM)
  - Automatically handles product enrichment
  - BIAP-compliant request formatting
  - Graceful error recovery
- `get_order_status` - Check order status
- `track_order` - Track shipment

## Key Features

### Session Persistence
- File-based session storage
- Cart state preservation
- User authentication state
- Order history tracking

### Hybrid Search
- Vector search with Gemini embeddings
- MongoDB backend search
- Result reranking and fusion
- Category and filter support

### ONDC Protocol
- Full buyer app protocol implementation
- Real-time order processing
- Multi-seller support
- Location-based discovery

## Configuration

The server uses environment variables from parent `.env`:

```bash
BACKEND_ENDPOINT=https://hp-buyer-backend-preprod.himira.co.in
WIL_API_KEY=your_api_key
GEMINI_API_KEY=your_gemini_key  # Optional for vector search
VECTOR_SEARCH_ENABLED=true      # Enable/disable vector search
SESSION_STORE=file               # Session storage type
```

## Development

### Project Structure
```
src/
 mcp_adapters.py                  # MCP tool definitions (3 consolidated tools)
 mcp_server.py                    # Main MCP server implementation
 buyer_backend_client.py          # BIAP-compliant API client
 config.py                        # Configuration management
 services/                        # Business logic
    search_service.py           # Product search
    cart_service.py             # Cart management
    checkout_service.py         # Unified checkout flow
    product_enrichment_service.py # BIAP product enrichment
    biap_validation_service.py  # Request/response validation
    session_service.py          # Session handling
    user_service.py             # User management
 models/                          # BIAP-compliant data models
    session.py                  # Enhanced cart item structure
 vector_search/                   # Vector search module
 utils/                          # Utilities
     city_code_mapping.py        # ONDC location mapping
```

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python run_mcp_server.py
```

## Deployment

This server is deployed as part of the unified ONDC Genie system. See parent [README](../README.md) for deployment instructions.

```bash
# Quick deployment (from parent directory)
docker-compose -f docker-compose.unified.yml up -d
```

## Testing

```bash
# Test MCP server functionality
python test_integration.py

# Test specific tool
docker exec ondc-mcp-server python -c "
from src.mcp_adapters import mcp_search_products
import asyncio
asyncio.run(mcp_search_products({'session_id': 'test'}, 'laptop'))
"
```

## Troubleshooting

### Common Issues

1. **Session not found**: Sessions expire after 30 minutes of inactivity
2. **No products found**: Check backend connectivity and location parameters
3. **Vector search disabled**: Ensure Qdrant is running and GEMINI_API_KEY is set

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

Check logs:
```bash
docker logs ondc-mcp-server -f
```

## API Reference

For detailed API documentation, see:
- [Himira Backend API](https://hp-buyer-backend-preprod.himira.co.in/clientApis)
- [ONDC Protocol Specs](https://docs.ondc.org/)

---

Part of [ONDC Genie](https://github.com/yourusername/ondc-genie) - Unified ONDC Shopping System