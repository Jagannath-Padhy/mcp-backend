# ONDC Shopping MCP Backend

🛍️ Production-ready ONDC shopping backend with Model Context Protocol (MCP) integration for AI-powered conversational commerce.

## 🏗️ Architecture Overview

This backend implements a complete ONDC (Open Network for Digital Commerce) shopping solution with AI assistance capabilities through MCP tools. It enables conversational shopping experiences where users can search, browse, add to cart, and complete purchases through natural language interactions.

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Application                      │
│                  (Web/Mobile/Chat Interface)                 │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP/REST API
┌─────────────────▼───────────────────────────────────────────┐
│                    MCP Backend (Port 8001)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               MCP Server (STDIO Transport)            │  │
│  │                    40+ Shopping Tools                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Himira BIAP Backend                   │  │
│  │              (ONDC Protocol Implementation)           │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                         ONDC Network                         │
│                    (Sellers, Logistics, etc.)                │
└─────────────────────────────────────────────────────────────┘
```

### Core Services

- **MCP Server**: Provides 40+ shopping tools via STDIO protocol for AI agents
- **Guest Authentication**: Device-based authentication for seamless guest checkout
- **Session Management**: Persistent sessions with auth token management
- **ONDC Protocol**: Full implementation of SELECT, INIT, CONFIRM with async polling
- **Vector Search**: Semantic product search using Qdrant (optional)
- **Cart Management**: Comprehensive cart operations with persistence

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- WIL API Key (for Himira backend access)
- (Optional) Gemini API key for vector search

### Installation

1. **Clone and setup**:
```bash
git clone <repository>
cd mcp-backend
cp .env.example .env
```

2. **Configure environment** (.env):
```env
# Required: Himira Backend Access
WIL_API_KEY=your_api_key_here
BACKEND_ENDPOINT=https://hp-buyer-backend-preprod.himira.co.in

# Guest User Configuration
GUEST_USER_ID=guestUser
GUEST_DEVICE_ID=d58dc5e2119ae5430b9321602618c878

# Optional: Vector Search
GEMINI_API_KEY=your_gemini_key_here
VECTOR_SEARCH_ENABLED=false
```

3. **Start services**:
```bash
make up        # Start all services
make logs      # View logs
make status    # Check health
```

## 📚 API Documentation

### Base URL
```
http://localhost:8001
```

### Authentication Flow

The system uses guest authentication with device-based tokens:

1. **Initialize Shopping Session**:
```bash
curl -X POST http://localhost:8001/api/v1/initialize_shopping \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response:
```json
{
  "success": true,
  "session_id": "session_2b209e24bbea4359",
  "auth_token": "eyJhbGciOiJIUzI1NiIs...",
  "device_id": "d58dc5e2119ae5430b9321602618c878",
  "message": "Guest session ready with authentication"
}
```

### Core Shopping APIs

#### 1. Search Products
```bash
curl -X POST http://localhost:8001/api/v1/search_products \
  -H "Content-Type: application/json" \
  -d '{
    "query": "organic rice",
    "session_id": "session_2b209e24bbea4359",
    "limit": 10
  }'
```

#### 2. Add to Cart
```bash
curl -X POST http://localhost:8001/api/v1/add_to_cart \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "ab0afa97-ee02-4e22-b446-b826507b2223",
    "quantity": 2,
    "session_id": "session_2b209e24bbea4359"
  }'
```

#### 3. View Cart
```bash
curl -X POST http://localhost:8001/api/v1/view_cart \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_2b209e24bbea4359"
  }'
```

#### 4. Checkout (SELECT - Get Delivery Quotes)
```bash
curl -X POST http://localhost:8001/api/v1/select_items_for_order \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_2b209e24bbea4359",
    "delivery_city": "Bangalore",
    "delivery_state": "Karnataka",
    "delivery_pincode": "560001"
  }'
```

#### 5. Initialize Order
```bash
curl -X POST http://localhost:8001/api/v1/initialize_order \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_2b209e24bbea4359",
    "customer_name": "John Doe",
    "delivery_address": "123 Main St, Koramangala",
    "phone": "9999999999",
    "email": "john@example.com",
    "payment_method": "razorpay"
  }'
```

#### 6. Confirm Order
```bash
curl -X POST http://localhost:8001/api/v1/confirm_order \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_2b209e24bbea4359",
    "payment_status": "PAID"
  }'
```

## 🛠️ MCP Tools Available

The MCP server provides 40+ tools for comprehensive shopping operations:

### Session Management
- `initialize_shopping` - Create guest session with authentication
- `get_session_info` - Get current session state

### Product Discovery
- `search_products` - Text and semantic search
- `advanced_search` - Multi-criteria filtering
- `browse_categories` - Category navigation

### Cart Operations
- `add_to_cart` - Add items to cart
- `view_cart` - View cart contents
- `update_cart_quantity` - Modify quantities
- `remove_from_cart` - Remove items
- `clear_cart` - Empty cart
- `get_cart_total` - Calculate totals

### ONDC Checkout Flow
- `select_items_for_order` - Get delivery quotes (ONDC SELECT)
- `initialize_order` - Set billing/shipping (ONDC INIT)
- `confirm_order` - Complete purchase (ONDC CONFIRM)

### Order Management
- `get_order_status` - Track order status
- `track_order` - Detailed tracking
- `initiate_payment` - Payment processing
- `confirm_order_simple` - Alternative confirmation

### User Features
- `get_delivery_addresses` - Saved addresses
- `add_delivery_address` - Add new address
- `update_delivery_address` - Update address
- `delete_delivery_address` - Remove address
- `get_active_offers` - Available offers
- `apply_offer` - Apply discount
- `get_user_profile` - Profile information
- `update_user_profile` - Update profile

## 🔄 ONDC Journey Flow

The system implements the complete ONDC protocol flow:

```
1. SEARCH → 2. SELECT → 3. INIT → 4. PAYMENT → 5. CONFIRM → 6. TRACK
     ↓          ↓         ↓          ↓            ↓           ↓
   Find      Delivery   Order    Process      Complete    Monitor
  Products    Quote     Setup     Payment       Order       Status
```

### Asynchronous Operations

SELECT, INIT, and CONFIRM operations use async polling:

1. **Request**: Send action request (e.g., SELECT)
2. **Response**: Receive messageId immediately
3. **Poll**: Check for on_action response (e.g., on_select)
4. **Result**: Get final response after 2-10 seconds

## 🏗️ Frontend Integration Guide

### Session Management

1. **Initialize on app start**:
```javascript
const initSession = async () => {
  const response = await fetch('/api/v1/initialize_shopping', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({})
  });
  const data = await response.json();
  
  // Store session_id for all subsequent calls
  localStorage.setItem('session_id', data.session_id);
  localStorage.setItem('auth_token', data.auth_token);
  
  return data;
};
```

2. **Use session in all requests**:
```javascript
const searchProducts = async (query) => {
  const session_id = localStorage.getItem('session_id');
  
  const response = await fetch('/api/v1/search_products', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      query: query,
      session_id: session_id
    })
  });
  
  return response.json();
};
```

### Cart Management Pattern

```javascript
// Add to cart
const addToCart = async (product, quantity = 1) => {
  const session_id = localStorage.getItem('session_id');
  
  const response = await fetch('/api/v1/add_to_cart', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      item: product,  // Pass entire product object from search
      quantity: quantity,
      session_id: session_id
    })
  });
  
  return response.json();
};

// View cart
const viewCart = async () => {
  const session_id = localStorage.getItem('session_id');
  
  const response = await fetch('/api/v1/view_cart', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: session_id
    })
  });
  
  return response.json();
};
```

### Checkout Flow

```javascript
// Complete checkout flow
const checkout = async (deliveryInfo, customerInfo) => {
  const session_id = localStorage.getItem('session_id');
  
  // Step 1: Get delivery quotes
  const selectResponse = await fetch('/api/v1/select_items_for_order', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: session_id,
      delivery_city: deliveryInfo.city,
      delivery_state: deliveryInfo.state,
      delivery_pincode: deliveryInfo.pincode
    })
  });
  
  if (!selectResponse.ok) {
    throw new Error('Failed to get delivery quotes');
  }
  
  // Step 2: Initialize order
  const initResponse = await fetch('/api/v1/initialize_order', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: session_id,
      customer_name: customerInfo.name,
      delivery_address: customerInfo.address,
      phone: customerInfo.phone,
      email: customerInfo.email,
      payment_method: 'razorpay'
    })
  });
  
  if (!initResponse.ok) {
    throw new Error('Failed to initialize order');
  }
  
  // Step 3: Process payment (implement Razorpay/payment gateway)
  const paymentStatus = await processPayment();
  
  // Step 4: Confirm order
  const confirmResponse = await fetch('/api/v1/confirm_order', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: session_id,
      payment_status: paymentStatus
    })
  });
  
  return confirmResponse.json();
};
```

## 📁 Project Structure

```
mcp-backend/
├── backend/
│   ├── ondc-shopping-mcp/
│   │   ├── src/
│   │   │   ├── adapters/          # MCP tool implementations
│   │   │   ├── services/          # Business logic
│   │   │   ├── models/            # Data models
│   │   │   ├── utils/             # Utilities
│   │   │   ├── buyer_backend_client.py  # Himira API client
│   │   │   ├── mcp_server.py      # Main MCP server
│   │   │   └── config.py          # Configuration
│   │   └── requirements.txt
│   ├── api/
│   │   └── server.py              # REST API server
│   ├── Dockerfile
│   └── supervisord.conf           # Process management
├── etl/
│   └── etl_pipeline.py            # Data initialization
├── docker-compose.yml
├── Makefile
├── .env.example
└── README.md
```

## 🔧 Development

### Local Development

```bash
# Install dependencies
cd backend/ondc-shopping-mcp
pip install -r requirements.txt

# Run MCP server
python src/mcp_server.py

# Run API server
cd ../api
python server.py
```

### Docker Commands

```bash
make up          # Start all services
make down        # Stop all services
make restart     # Restart services
make logs        # View logs
make shell-backend  # Shell into backend
make status      # Check service health
```

### Environment Variables

```env
# Backend Configuration (Required)
BACKEND_ENDPOINT=https://hp-buyer-backend-preprod.himira.co.in
WIL_API_KEY=your_api_key_here

# Guest Configuration
GUEST_USER_ID=guestUser
GUEST_DEVICE_ID=d58dc5e2119ae5430b9321602618c878

# Database Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=himira_products

# Optional Features
VECTOR_SEARCH_ENABLED=false
GEMINI_API_KEY=your_key_here

# Session Configuration
SESSION_TIMEOUT_MINUTES=30
SESSION_STORE=file
SESSION_STORE_PATH=~/.ondc-mcp/sessions

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/mcp_operations.log
```

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Errors (404s)**:
   - Ensure WIL_API_KEY is set correctly
   - Check guest device ID configuration
   - Verify session has auth token

2. **Cart Empty After Adding Items**:
   - Check session persistence
   - Ensure same session_id is used across calls
   - Verify session storage path is writable

3. **SELECT API Timeouts**:
   - Backend may be slow, increase timeout
   - Check network connectivity
   - Verify product enrichment APIs are accessible

4. **Vector Search Not Working**:
   - Run ETL pipeline: `make init`
   - Check Qdrant is running: `docker ps`
   - Verify GEMINI_API_KEY is valid

### Debug Mode

Enable detailed logging:
```bash
LOG_LEVEL=DEBUG docker-compose up
```

View backend logs:
```bash
docker logs mcp-backend -f
```

## 🚀 Production Deployment

### Recommendations

1. **Security**:
   - Use HTTPS with TLS certificates
   - Implement rate limiting
   - Add API authentication
   - Secure database credentials

2. **Scaling**:
   - Use Redis for session storage
   - Implement load balancing
   - Add caching layer
   - Use connection pooling

3. **Monitoring**:
   - Add health check endpoints
   - Implement logging aggregation
   - Set up alerting
   - Track API metrics

4. **Backup**:
   - Regular database backups
   - Session state persistence
   - Configuration management

## 📝 License

MIT

## 🤝 Support

For issues or questions, please open a GitHub issue or contact the development team.