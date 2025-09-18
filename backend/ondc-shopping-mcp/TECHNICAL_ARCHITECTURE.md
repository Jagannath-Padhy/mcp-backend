# ONDC Shopping MCP - Technical Architecture Documentation

## Executive Summary

The ONDC Shopping MCP Server is a production-ready, BIAP-compliant shopping system implementing the Model Context Protocol (MCP) for conversational commerce. The system provides 21 MCP tools organized across 6 functional modules, supporting the complete ONDC shopping journey from product discovery to order fulfillment.

## System Overview

### Core Components

```

                   MCP CLIENT                            
              (Desktop Client Pro)                      

                       JSON-RPC over stdio

                MCP SERVER                               
            (ondc-shopping-mcp)                         
          
     Adapter       Service       Data            
      Layer         Layer        Models          
          

                       HTTP REST API

               HIMIRA BACKEND                            
          (BIAP Reference Implementation)                
                hp-buyer-backend-preprod                 

```

### Architecture Layers

1. **MCP Adapter Layer** (`src/adapters/`) - 21 MCP tools organized into 6 modules
2. **Service Layer** (`src/services/`) - Business logic and ONDC protocol handling
3. **Data Layer** (`src/models/`, `src/data_models/`) - Session management and ONDC data structures
4. **Integration Layer** (`src/buyer_backend_client.py`) - Himira backend communication
5. **Utility Layer** (`src/utils/`) - Shared utilities, logging, and validation

## MCP Tools Architecture

### 1. Tool Organization (6 Modules)

```
src/adapters/
 __init__.py          # Package documentation
 utils.py             # Shared utilities (6 functions)
 cart.py              # Cart operations (6 functions)
 search.py            # Search operations (3 functions)
 checkout.py          # ONDC checkout flow (4 functions)
 auth.py              # Authentication (1 function)
 session.py           # Session management (2 functions)
 orders.py            # Order management (4 functions)
```

### 2. MCP Tool Categories

#### Session Management (2 tools)
- `initialize_shopping` - Create new shopping session with enhanced persistence
- `get_session_info` - Retrieve current session state and statistics

#### Search & Discovery (3 tools)
- `search_products` - Text-based product search with vector enhancement
- `advanced_search` - Multi-criteria product filtering
- `browse_categories` - Category-based product browsing

#### Cart Management (6 tools)
- `add_to_cart` - Add products with quantity validation
- `view_cart` - Display cart contents with pricing
- `update_cart_quantity` - Modify item quantities
- `remove_from_cart` - Remove specific items
- `clear_cart` - Empty entire cart
- `get_cart_total` - Calculate cart totals and summaries

#### Authentication (1 tool)
- `phone_login` - Quick phone-based authentication (OTP bypass for development)

#### ONDC Checkout Flow (4 tools)
- `select_items_for_order` - ONDC SELECT stage with delivery quotes
- `initialize_order` - ONDC INIT stage with customer details
- `create_payment` - Mock payment creation between INIT and CONFIRM
- `confirm_order` - ONDC CONFIRM stage with order finalization

#### Order Management (4 tools)
- `initiate_payment` - Production payment processing
- `confirm_order_simple` - Simplified order confirmation (deprecated)
- `get_order_status` - Check order status and tracking
- `track_order` - Detailed order tracking information

### 3. Enhanced Session Management

#### ConversationSessionManager
```python
# Location: src/services/conversation_session_manager.py
# Purpose: Solve Langflow session_id propagation issues

Features:
- File-based session persistence (~/.ondc-mcp/sessions/)
- Conversation-level tracking across tool calls
- Session lifecycle management
- Enhanced error handling and recovery
```

#### Session Data Structure
```python
class ShoppingSession:
    session_id: str
    user_authenticated: bool
    auth_token: Optional[str]
    cart: Cart
    checkout_state: CheckoutState
    delivery_location: Optional[Dict]
    conversation_history: List[ConversationTurn]
    created_at: datetime
    updated_at: datetime
```

## ONDC Protocol Implementation

### 1. BIAP Compliance Architecture

The system implements the full BIAP (Buyer App Integration Platform) specification:

```
ONDC Journey Flow:
SEARCH → SELECT → INIT → PAYMENT → CONFIRM → TRACK
   ↓        ↓       ↓        ↓         ↓        ↓
search   select   init   payment   confirm   track
tools    quotes  order   creation   order    order
```

### 2. ONDC Data Models

#### Core Models (`src/data_models/ondc_models.py`)
- `BiapContext` - BIAP-compliant context generation
- `ONDCItem` - Product representation with ONDC fields
- `ONDCOrder` - Order structure matching ONDC schema
- `ONDCPayment` - Payment details with gateway support
- `ONDCFulfillment` - Delivery and logistics information

#### Context Factory (`src/data_models/biap_context_factory.py`)
```python
class BiapContextFactory:
    @staticmethod
    def create_search_context(query: str, location: Dict) -> Dict
    @staticmethod
    def create_select_context(items: List, delivery_info: Dict) -> Dict
    @staticmethod
    def create_init_context(order_details: Dict, customer: Dict) -> Dict
    @staticmethod
    def create_confirm_context(payment: Dict, order: Dict) -> Dict
```

### 3. Payment Gateway Integration

#### Multi-Gateway Support
- **RazorPay** - Primary payment gateway with mock implementation
- **JusPay** - Alternative payment gateway
- **COD** - Cash on Delivery support
- **Mock Payments** - Development and testing support

#### Payment Flow
```python
# Payment Service Architecture
PaymentService:
   get_available_payment_methods()
   initiate_payment()
   verify_payment_status()
   handle_payment_callback()
```

## Service Layer Architecture

### 1. Service Dependencies

```
CheckoutService
     depends_on: OrderService, PaymentService, CartService
     provides: Consolidated 3-tool checkout flow

OrderService
     depends_on: BiapContextFactory, BuyerBackendClient
     provides: ONDC protocol operations

CartService
     depends_on: SessionService, ProductValidation
     provides: Cart lifecycle management

SearchService
     depends_on: VectorSearchClient (optional), BuyerBackendClient
     provides: Product discovery with semantic enhancement

PaymentService
     depends_on: RazorPay, JusPay APIs
     provides: Multi-gateway payment processing

SessionService
     depends_on: File system persistence
     provides: Session lifecycle and conversation tracking
```

### 2. Service Initialization Pattern

```python
# Centralized service factory pattern
def get_services() -> Dict[str, Any]:
    return {
        'session_service': get_session_service(),
        'search_service': get_search_service(),
        'cart_service': get_cart_service(),
        'checkout_service': get_checkout_service(),
        'order_service': get_order_service(),
        'payment_service': get_payment_service()
    }
```

## Backend Integration Architecture

### 1. Himira Backend Client

#### Connection Management
```python
class BuyerBackendClient:
    base_url: str = "https://hp-buyer-backend-preprod.himira.co.in"
    api_key: str = config.wil_api_key
    timeout: int = 30
    
    async def make_request(method: str, endpoint: str, data: Dict) -> Dict
    async def search_products(query: str, location: Dict) -> Dict
    async def select_items(cart_items: List, delivery: Dict) -> Dict
    async def initialize_order(order_details: Dict) -> Dict
    async def confirm_order(order_data: Dict) -> Dict
```

#### API Endpoints Integration
- `/search` - Product search with ONDC catalog integration
- `/select` - Get delivery quotes and provider options
- `/init` - Initialize order with customer details
- `/confirm` - Finalize order with payment confirmation
- `/status` - Check order status and updates
- `/track` - Get tracking information

### 2. Error Handling & Resilience

#### Retry Logic
```python
@retry(max_attempts=3, backoff_factor=2.0)
async def make_backend_request(endpoint: str, data: Dict) -> Dict:
    # Exponential backoff with jitter
    # Circuit breaker pattern for degraded service
    # Fallback to cached data when available
```

#### Graceful Degradation
- Vector search fallback to API search
- Session recovery from persistent storage
- Cached product data for offline scenarios

## Data Flow Architecture

### 1. Request Processing Flow

```
MCP Client Request
       ↓
JSON-RPC Validation
       ↓
Tool Schema Validation
       ↓
Adapter Function Call
       ↓
Session Enhancement & Retrieval
       ↓
Service Layer Processing
       ↓
Backend API Communication
       ↓
Response Formatting
       ↓
Session Persistence
       ↓
MCP Response to Client
```

### 2. Session Data Flow

```
Tool Call → get_persistent_session() → ConversationSessionManager
                                            ↓
                                    Load from file system
                                            ↓
                                    Create session object
                                            ↓
                              Enhanced conversation tracking
                                            ↓
Business Logic Processing → Service Layer → Backend APIs
                                            ↓
Response Generation → save_persistent_session() → File persistence
```

### 3. Error Propagation

```python
try:
    # Service layer operation
    result = await service.operation(data)
    return format_mcp_response(True, result, session_id)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    return format_mcp_response(False, f"Validation error: {e}", session_id)
except BackendError as e:
    logger.error(f"Backend communication failed: {e}")
    return format_mcp_response(False, f"Service temporarily unavailable", session_id)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return format_mcp_response(False, f"Internal error occurred", session_id)
```

## Configuration Architecture

### 1. Environment Configuration

```python
# src/config.py - Centralized configuration management
class Config:
    class Logging:
        level: str = "INFO"
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Backend:
        endpoint: str = "https://hp-buyer-backend-preprod.himira.co.in"
        api_key: str = os.getenv("WIL_API_KEY")
        timeout: int = 30
    
    class Performance:
        max_concurrent_requests: int = 10
        cache_ttl_seconds: int = 300
        max_image_size_mb: int = 5
    
    class Features:
        vector_search_enabled: bool = True
        mock_payments_enabled: bool = True
        debug_mode: bool = False
```

### 2. Feature Flags

- `VECTOR_SEARCH_ENABLED` - Enable/disable semantic search enhancement
- `MOCK_PAYMENTS_ENABLED` - Use mock payments for development
- `DEBUG_MODE` - Enable detailed logging and validation
- `RATE_LIMITING_ENABLED` - Apply rate limits to API calls

## Security Architecture

### 1. Authentication Flow

```python
# Phone-based authentication with backend validation
async def phone_login(phone: str) -> Dict:
    # Format phone number (+91 prefix)
    formatted_phone = f"+91{phone.strip()}"
    
    # Call Himira backend loginWithPhone endpoint
    response = await backend_client.login_with_phone(formatted_phone)
    
    # Store auth token in session
    session.auth_token = response.get('token')
    session.user_authenticated = True
    
    return format_mcp_response(True, "Authentication successful", session_id)
```

### 2. API Security

- **API Key Authentication** - WIL_API_KEY for Himira backend
- **Session Token Management** - Secure token storage and validation
- **Input Validation** - All user inputs validated and sanitized
- **Rate Limiting** - Prevent API abuse and ensure fair usage

### 3. Data Privacy

- **Session Isolation** - Each session has isolated data
- **Secure Storage** - Session files stored in user home directory
- **No Credential Logging** - API keys and tokens excluded from logs
- **Data Retention** - Configurable session cleanup policies

## Performance Architecture

### 1. Caching Strategy

```python
# Multi-level caching approach
class CacheManager:
    # L1: In-memory cache for frequently accessed data
    memory_cache: Dict[str, Any] = {}
    
    # L2: File-based cache for session persistence
    file_cache_dir: str = "~/.ondc-mcp/cache/"
    
    # L3: Backend response caching
    api_cache_ttl: int = 300  # 5 minutes
```

### 2. Async Operations

- **Concurrent API Calls** - Multiple backend requests in parallel
- **Non-blocking I/O** - Async file operations for session management
- **Stream Processing** - Large response handling with streaming
- **Background Tasks** - Session cleanup and cache maintenance

### 3. Resource Optimization

- **Memory Management** - Efficient session object lifecycle
- **Connection Pooling** - Reuse HTTP connections to backend
- **Request Batching** - Combine multiple operations when possible
- **Lazy Loading** - Load session data only when needed

## Monitoring & Observability

### 1. Logging Architecture

```python
# Structured logging with correlation IDs
logger = get_logger(__name__)

# Log levels and purposes:
# DEBUG - Detailed execution flow for development
# INFO - Normal operation events and state changes
# WARNING - Recoverable errors and fallback scenarios
# ERROR - Unrecoverable errors requiring attention
# CRITICAL - System-level failures
```

### 2. Metrics Collection

- **Request Latency** - MCP tool execution times
- **Backend Response Times** - API call performance
- **Error Rates** - Failure rates by tool and operation
- **Session Statistics** - Active sessions and conversation length
- **Resource Usage** - Memory and CPU utilization

### 3. Health Checks

```python
# Health check endpoints for monitoring
class HealthService:
    async def check_backend_connectivity() -> bool
    async def check_session_storage() -> bool
    async def check_vector_search() -> bool
    async def get_system_metrics() -> Dict
```

## Deployment Architecture

### 1. Container Strategy

```dockerfile
# Multi-stage Docker build
FROM python:3.11-slim as base
# Dependencies installation
FROM base as dependencies
# Application code
FROM dependencies as runtime
```

### 2. Environment Management

- **Development** - Local development with mock services
- **Staging** - Pre-production testing with real backend
- **Production** - Full production deployment with monitoring

### 3. Scaling Considerations

- **Horizontal Scaling** - Multiple MCP server instances
- **Load Balancing** - Distribute requests across instances
- **Session Affinity** - Route user sessions to consistent instances
- **Resource Limits** - Memory and CPU constraints per instance

## Future Architecture Considerations

### 1. Microservices Migration

```
Current Monolithic MCP Server
               ↓
Planned Microservices Architecture:
 Authentication Service
 Session Management Service  
 Product Search Service
 Cart Management Service
 Order Processing Service
 Payment Gateway Service
```

### 2. Event-Driven Architecture

- **Event Sourcing** - Capture all state changes as events
- **Message Queues** - Async communication between services
- **Event Store** - Persistent event history for debugging
- **CQRS** - Separate read and write models

### 3. Enhanced Vector Search

- **Semantic Product Discovery** - Advanced embedding models
- **Personalization** - User preference-based recommendations
- **Real-time Indexing** - Dynamic product catalog updates
- **Multi-modal Search** - Image and voice search capabilities

## Technical Debt & Maintenance

### 1. Known Technical Debt

- **Legacy Tool Implementations** - Gradual migration to adapter pattern
- **Mock Payment Implementation** - Replace with production payment flows
- **Session File Storage** - Consider database persistence for scalability
- **Error Message Standardization** - Consistent error response format

### 2. Maintenance Tasks

- **Dependency Updates** - Regular security and feature updates
- **Performance Optimization** - Continuous performance monitoring and tuning
- **Code Quality** - Regular code reviews and refactoring
- **Documentation Updates** - Keep documentation synchronized with code changes

This architecture documentation provides a comprehensive overview of the ONDC Shopping MCP system, covering all major technical aspects and implementation details.