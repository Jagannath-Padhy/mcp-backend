# ONDC Shopping MCP - Features Documentation

## Overview

The ONDC Shopping MCP server provides 21 specialized tools organized across 6 functional modules, enabling complete conversational commerce experiences. Each tool is designed to handle specific aspects of the ONDC shopping journey while maintaining session state and providing rich user feedback.

## Tool Categories & Complete Shopping Flow

```
Shopping Journey: DISCOVER → AUTHENTICATE → SHOP → CHECKOUT → FULFILL
                     ↓            ↓          ↓         ↓         ↓
Tools Used:      [Search]    [Authentication] [Cart]  [Checkout] [Orders]
                     +            +          +         +         +
                [Session]    [Session]    [Session] [Session] [Session]
```

---

## 1. Session Management Tools (2 tools)

### 1.1 `initialize_shopping`

**Purpose**: Create a new shopping session with enhanced persistence and conversation tracking.

**Parameters**: None required

**Returns**:
- `session_id`: Unique session identifier
- `success`: Boolean operation status
- `message`: User-friendly status message
- `session_info`: Basic session details

**Usage Example**:
```json
{
  "tool": "initialize_shopping",
  "parameters": {}
}
```

**Response**:
```json
{
  "success": true,
  "message": " Shopping session initialized successfully!\n\nSession ID: shop_20250116_143052_abc123\n\n Ready to start shopping! You can now:\n• Search for products\n• Browse categories\n• Add items to cart",
  "session_id": "shop_20250116_143052_abc123",
  "session_info": {
    "created_at": "2025-01-16T14:30:52.123Z",
    "user_authenticated": false,
    "cart_items": 0
  }
}
```

**Key Features**:
- Creates persistent session with file-based storage
- Generates conversation-level tracking for Langflow integration
- Initializes empty cart and checkout state
- Sets up enhanced session management infrastructure

---

### 1.2 `get_session_info`

**Purpose**: Retrieve current session state, statistics, and debugging information.

**Parameters**:
- `session` (optional): Session object or ID

**Returns**:
- Session details and statistics
- Cart summary
- Authentication status
- Conversation history summary

**Usage Example**:
```json
{
  "tool": "get_session_info",
  "parameters": {
    "session": "shop_20250116_143052_abc123"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " Session Information\n\nSession ID: shop_20250116_143052_abc123\nStatus: Active\nAuthenticated: Yes\nCart Items: 3\nTotal Value: ₹850.00\nLast Activity: 2 minutes ago",
  "session_details": {
    "session_id": "shop_20250116_143052_abc123",
    "user_authenticated": true,
    "cart_summary": {
      "total_items": 3,
      "total_value": 850.00,
      "currency": "INR"
    },
    "conversation_turns": 12,
    "created_at": "2025-01-16T14:30:52.123Z",
    "updated_at": "2025-01-16T14:45:30.456Z"
  }
}
```

---

## 2. Search & Discovery Tools (3 tools)

### 2.1 `search_products`

**Purpose**: Search for products using text queries with optional vector search enhancement.

**Parameters**:
- `session` (optional): Session object or ID
- `query` (required): Search query text
- `limit` (optional): Number of results (default: 10)
- `location` (optional): Geographic location for local results

**Returns**:
- Array of matching products
- Search metadata
- Vector search indicators (if enabled)

**Usage Example**:
```json
{
  "tool": "search_products",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "query": "organic honey",
    "limit": 5
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " Found 5 products for 'organic honey'\n\n1. **Pure Organic Wild Honey** - ₹450.00\n   Brand: Nature's Best | 500g\n    4.5/5 (230 reviews)\n\n2. **Himalayan Raw Honey** - ₹320.00\n   Brand: Organic Valley | 350g\n    4.3/5 (156 reviews)",
  "products": [
    {
      "id": "prod_honey_001",
      "name": "Pure Organic Wild Honey",
      "price": 450.00,
      "currency": "INR",
      "brand": "Nature's Best",
      "weight": "500g",
      "rating": 4.5,
      "reviews_count": 230,
      "in_stock": true,
      "provider_id": "provider_123"
    }
  ],
  "search_metadata": {
    "query": "organic honey",
    "total_results": 5,
    "vector_search_used": true,
    "search_time_ms": 245
  }
}
```

**Key Features**:
- Semantic search with vector embeddings (when enabled)
- Fallback to traditional API search
- Rich product information with ratings and availability
- Geographic filtering support

---

### 2.2 `advanced_search`

**Purpose**: Advanced product search with multiple filters and criteria.

**Parameters**:
- `session` (optional): Session object or ID
- `query` (optional): Base search query
- `category` (optional): Product category filter
- `price_min` (optional): Minimum price filter
- `price_max` (optional): Maximum price filter
- `brand` (optional): Brand name filter
- `rating_min` (optional): Minimum rating filter
- `in_stock_only` (optional): Show only available products
- `sort_by` (optional): Sort criteria (price, rating, relevance)
- `limit` (optional): Number of results

**Usage Example**:
```json
{
  "tool": "advanced_search",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "category": "food_and_beverages",
    "price_min": 100,
    "price_max": 500,
    "rating_min": 4.0,
    "in_stock_only": true,
    "sort_by": "rating",
    "limit": 10
  }
}
```

**Key Features**:
- Multi-criteria filtering
- Price range filtering
- Brand and category filters
- Rating-based filtering
- Multiple sorting options

---

### 2.3 `browse_categories`

**Purpose**: Browse products by category hierarchy with navigation support.

**Parameters**:
- `session` (optional): Session object or ID
- `category` (optional): Category to browse (shows all if not specified)
- `subcategory` (optional): Subcategory filter
- `limit` (optional): Number of results per category

**Usage Example**:
```json
{
  "tool": "browse_categories",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "category": "electronics"
  }
}
```

**Key Features**:
- Category hierarchy navigation
- Product counts per category
- Featured products in each category
- Category-specific filtering

---

## 3. Cart Management Tools (6 tools)

### 3.1 `add_to_cart`

**Purpose**: Add products to the shopping cart with quantity validation.

**Parameters**:
- `session` (optional): Session object or ID
- `item` (required): Product object or ID to add
- `quantity` (optional): Quantity to add (default: 1)

**Returns**:
- Updated cart summary
- Item addition confirmation
- Cart totals and item count

**Usage Example**:
```json
{
  "tool": "add_to_cart",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "item": {
      "id": "prod_honey_001",
      "name": "Pure Organic Wild Honey",
      "price": 450.00,
      "provider_id": "provider_123"
    },
    "quantity": 2
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " Added to cart successfully!\n\n**Pure Organic Wild Honey** × 2\nPrice: ₹450.00 each\nSubtotal: ₹900.00\n\n **Cart Summary:**\nTotal Items: 3\nTotal Value: ₹1,350.00",
  "cart_summary": {
    "total_items": 3,
    "total_value": 1350.00,
    "currency": "INR",
    "item_count_by_provider": {
      "provider_123": 2,
      "provider_456": 1
    }
  }
}
```

**Key Features**:
- Automatic quantity validation
- Price calculation and subtotals
- Provider grouping for ONDC compliance
- Inventory availability checking

---

### 3.2 `view_cart`

**Purpose**: Display current cart contents with detailed item information and totals.

**Parameters**:
- `session` (optional): Session object or ID

**Returns**:
- Complete cart contents
- Item details with pricing
- Cart totals and summaries
- Provider breakdown for checkout

**Usage Example**:
```json
{
  "tool": "view_cart",
  "parameters": {
    "session": "shop_20250116_143052_abc123"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " **Your Shopping Cart**\n\n**Provider: Nature's Best Store**\n1. Pure Organic Wild Honey × 2\n   ₹450.00 each = ₹900.00\n\n**Provider: Organic Valley**\n2. Raw Almonds 250g × 1\n   ₹450.00\n\n**Cart Total:**\nItems: 3\nSubtotal: ₹1,350.00\nTax: ₹135.00\n**Total: ₹1,485.00**",
  "cart_contents": {
    "items": [
      {
        "id": "prod_honey_001",
        "name": "Pure Organic Wild Honey",
        "quantity": 2,
        "unit_price": 450.00,
        "subtotal": 900.00,
        "provider_name": "Nature's Best Store",
        "provider_id": "provider_123"
      }
    ],
    "totals": {
      "subtotal": 1350.00,
      "tax": 135.00,
      "total": 1485.00,
      "currency": "INR"
    }
  }
}
```

**Key Features**:
- Detailed item breakdown
- Provider grouping for multi-vendor support
- Tax calculations
- Real-time price updates

---

### 3.3 `update_cart_quantity`

**Purpose**: Modify the quantity of items already in the cart.

**Parameters**:
- `session` (optional): Session object or ID
- `item_id` (required): Product ID to update
- `quantity` (required): New quantity (0 to remove)

**Usage Example**:
```json
{
  "tool": "update_cart_quantity",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "item_id": "prod_honey_001",
    "quantity": 3
  }
}
```

**Key Features**:
- Quantity validation and limits
- Automatic cart total recalculation
- Inventory availability checking
- Support for quantity = 0 to remove items

---

### 3.4 `remove_from_cart`

**Purpose**: Remove specific items from the shopping cart.

**Parameters**:
- `session` (optional): Session object or ID
- `item_id` (required): Product ID to remove

**Usage Example**:
```json
{
  "tool": "remove_from_cart",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "item_id": "prod_honey_001"
  }
}
```

**Key Features**:
- Complete item removal
- Cart total recalculation
- Provider grouping updates
- Undo support through session history

---

### 3.5 `clear_cart`

**Purpose**: Empty the entire shopping cart and reset cart state.

**Parameters**:
- `session` (optional): Session object or ID
- `confirm` (optional): Confirmation flag to prevent accidental clearing

**Usage Example**:
```json
{
  "tool": "clear_cart",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "confirm": true
  }
}
```

**Key Features**:
- Complete cart reset
- Confirmation requirement
- Session state preservation
- Backup creation for recovery

---

### 3.6 `get_cart_total`

**Purpose**: Calculate and return detailed cart totals and pricing breakdown.

**Parameters**:
- `session` (optional): Session object or ID

**Returns**:
- Detailed pricing breakdown
- Tax calculations
- Discount applications
- Final totals by currency

**Usage Example**:
```json
{
  "tool": "get_cart_total",
  "parameters": {
    "session": "shop_20250116_143052_abc123"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " **Cart Total Breakdown**\n\nSubtotal: ₹1,350.00\nShipping: ₹50.00\nTax (10%): ₹135.00\nDiscount: -₹50.00\n**Final Total: ₹1,485.00**",
  "totals": {
    "subtotal": 1350.00,
    "shipping": 50.00,
    "tax": 135.00,
    "discount": -50.00,
    "final_total": 1485.00,
    "currency": "INR",
    "item_count": 3
  }
}
```

---

## 4. Authentication Tools (1 tool)

### 4.1 `phone_login`

**Purpose**: Quick phone-based authentication with the Himira backend system.

**Parameters**:
- `phone` (required): 10-digit phone number
- `session` (optional): Session object or ID

**Returns**:
- Authentication status
- User profile information
- Session authentication token
- Login success confirmation

**Usage Example**:
```json
{
  "tool": "phone_login",
  "parameters": {
    "phone": "9876543210",
    "session": "shop_20250116_143052_abc123"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " **Login Successful!**\n\nWelcome back, John Doe!\nPhone: +91-9876543210\nProfile: Verified Customer\n\n You're now authenticated and can:\n• Place orders\n• View order history\n• Access exclusive deals",
  "user_profile": {
    "name": "John Doe",
    "phone": "+91-9876543210",
    "email": "john.doe@example.com",
    "verified": true,
    "customer_type": "premium"
  },
  "authentication": {
    "authenticated": true,
    "auth_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_expires": "2025-01-17T14:30:52.123Z"
  }
}
```

**Key Features**:
- Direct integration with Himira backend
- No OTP required (simplified for development)
- JWT token-based session management
- User profile retrieval and caching

---

## 5. ONDC Checkout Flow Tools (4 tools)

### 5.1 `select_items_for_order`

**Purpose**: ONDC SELECT stage - Get delivery quotes and fulfillment options for cart items.

**Parameters**:
- `session` (optional): Session object or ID
- `delivery_city` (optional): Delivery city
- `delivery_state` (optional): Delivery state  
- `delivery_pincode` (optional): Delivery postal code

**Returns**:
- Available delivery options
- Provider quotes and pricing
- Estimated delivery times
- Service availability by location

**Usage Example**:
```json
{
  "tool": "select_items_for_order",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "delivery_city": "Bangalore",
    "delivery_state": "Karnataka", 
    "delivery_pincode": "560001"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " **Delivery Quotes Available**\n\n**Provider: Nature's Best Store**\n Standard Delivery: ₹50.00 (2-3 days)\n Express Delivery: ₹100.00 (1 day)\n\n**Provider: Organic Valley**\n Standard Delivery: ₹40.00 (3-4 days)\n\n All items available for delivery to Bangalore 560001",
  "quotes": [
    {
      "provider_id": "provider_123",
      "provider_name": "Nature's Best Store",
      "delivery_options": [
        {
          "type": "standard",
          "cost": 50.00,
          "duration": "2-3 days",
          "available": true
        },
        {
          "type": "express", 
          "cost": 100.00,
          "duration": "1 day",
          "available": true
        }
      ]
    }
  ]
}
```

**Key Features**:
- BIAP-compliant SELECT API integration
- Multi-provider quote aggregation
- Delivery option comparison
- Location-based service availability

---

### 5.2 `initialize_order`

**Purpose**: ONDC INIT stage - Initialize order with customer details and delivery information.

**Parameters**:
- `session` (optional): Session object or ID
- `customer_name` (required): Full customer name
- `delivery_address` (required): Complete delivery address
- `phone` (required): Contact phone number
- `email` (required): Email address
- `payment_method` (optional): Preferred payment method
- `city` (optional): Delivery city
- `state` (optional): Delivery state
- `pincode` (optional): Postal code

**Returns**:
- Order initialization confirmation
- BIAP-compliant order structure
- Payment preparation details
- Next step instructions

**Usage Example**:
```json
{
  "tool": "initialize_order",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "customer_name": "John Doe",
    "delivery_address": "123 MG Road, Apartment 4B, Near Metro Station",
    "phone": "9876543210",
    "email": "john.doe@example.com",
    "city": "Bangalore",
    "state": "Karnataka",
    "pincode": "560001"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " **Order Initialized Successfully!**\n\n**Customer Details:**\nName: John Doe\nPhone: +91-9876543210\nEmail: john.doe@example.com\n\n**Delivery Address:**\n123 MG Road, Apartment 4B\nBangalore, Karnataka 560001\n\n**Order Summary:**\nItems: 3\nSubtotal: ₹1,350.00\nDelivery: ₹50.00\nTotal: ₹1,400.00\n\n Ready for payment. Use 'create_payment' next.",
  "order_details": {
    "order_id": "order_20250116_143052",
    "customer": {
      "name": "John Doe",
      "phone": "+91-9876543210",
      "email": "john.doe@example.com"
    },
    "delivery": {
      "address": "123 MG Road, Apartment 4B, Near Metro Station",
      "city": "Bangalore",
      "state": "Karnataka",
      "pincode": "560001"
    },
    "totals": {
      "subtotal": 1350.00,
      "delivery": 50.00,
      "total": 1400.00
    }
  }
}
```

**Key Features**:
- BIAP-compliant order structure generation
- Customer information validation
- Address standardization
- Payment method preparation

---

### 5.3 `create_payment`

**Purpose**: Create mock payment between INIT and CONFIRM stages for development/testing.

**Parameters**:
- `session` (optional): Session object or ID
- `payment_method` (optional): Payment method (default: razorpay)

**Returns**:
- Mock payment creation confirmation
- Payment ID and transaction details
- Payment status information
- Next step instructions

**Usage Example**:
```json
{
  "tool": "create_payment",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "payment_method": "razorpay"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " [MOCK] Payment created successfully!\nPayment ID: pay_RFWPuAV50T2Qnj\nAmount: ₹1400 INR\nStatus: created\n\n Ready for order confirmation. Use 'confirm_order' next.",
  "payment_data": {
    "payment_id": "pay_RFWPuAV50T2Qnj", 
    "amount": 1400,
    "currency": "INR",
    "status": "created",
    "method": "razorpay",
    "_mock_indicators": {
      "is_mock": true,
      "source": "Himira Order Postman collection values"
    }
  }
}
```

**Key Features**:
- Mock payment implementation for development
- Real payment gateway structure simulation
- ONDC-compliant payment data format
- Clear mock indicators for debugging

---

### 5.4 `confirm_order`

**Purpose**: ONDC CONFIRM stage - Finalize order with payment validation and order placement.

**Parameters**:
- `session` (optional): Session object or ID
- `payment_status` (optional): Payment status (default: PENDING)

**Returns**:
- Order confirmation details
- Final order ID and tracking information
- Order status and next actions
- Receipt and summary

**Usage Example**:
```json
{
  "tool": "confirm_order",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "payment_status": "PAID"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " **ORDER CONFIRMED SUCCESSFULLY!**\n\n **Order Details:**\nOrder ID: ORD-20250116-143052\nStatus: Confirmed\nPayment: ₹1,400.00 (PAID)\n\n **Delivery Information:**\nExpected: 2-3 business days\nTracking: Available after dispatch\n\n**Next Steps:**\n• Use 'track_order' to monitor progress\n• Use 'get_order_status' for updates",
  "order_confirmation": {
    "order_id": "ORD-20250116-143052",
    "status": "confirmed",
    "payment_status": "PAID",
    "total_amount": 1400.00,
    "estimated_delivery": "2-3 business days",
    "tracking_available": false,
    "confirmation_time": "2025-01-16T14:35:52.123Z"
  }
}
```

**Key Features**:
- BIAP-compliant order confirmation
- Payment status validation
- Order ID generation and tracking setup
- Post-order action guidance

---

## 6. Order Management Tools (4 tools)

### 6.1 `initiate_payment`

**Purpose**: Production payment processing with multiple gateway support.

**Parameters**:
- `session` (optional): Session object or ID
- `payment_method` (optional): Payment method selection
- `amount` (optional): Payment amount (calculated from cart if not provided)

**Returns**:
- Available payment methods
- Payment processing status
- Gateway-specific payment details
- Next step instructions

**Usage Example**:
```json
{
  "tool": "initiate_payment",
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "payment_method": "razorpay"
  }
}
```

**Key Features**:
- Multi-gateway support (RazorPay, JusPay, COD)
- Real payment processing integration
- Payment method selection interface
- Gateway fee calculation

---

### 6.2 `confirm_order_simple`

**Purpose**: Simplified order confirmation (deprecated - use `confirm_order` instead).

**Parameters**:
- `session` (optional): Session object or ID

**Returns**:
- Basic order confirmation
- Order placement status
- Simple order details

**Note**: This tool is deprecated in favor of the enhanced `confirm_order` tool that provides BIAP compliance and better error handling.

---

### 6.3 `get_order_status`

**Purpose**: Check the current status of placed orders with detailed information.

**Parameters**:
- `session` (optional): Session object or ID
- `order_id` (optional): Specific order ID to check

**Returns**:
- Order status and progress
- Payment status information
- Delivery status and tracking
- Order timeline and updates

**Usage Example**:
```json
{
  "tool": "get_order_status", 
  "parameters": {
    "session": "shop_20250116_143052_abc123",
    "order_id": "ORD-20250116-143052"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " **Order Status Update**\n\n**Order ID:** ORD-20250116-143052\n**Status:** In Transit\n**Payment:** Completed (₹1,400.00)\n\n **Delivery Progress:**\n Order Confirmed (16 Jan, 2:35 PM)\n Payment Processed (16 Jan, 2:36 PM)\n Dispatched (17 Jan, 10:00 AM)\n In Transit (Expected: 18 Jan)\n\n **Tracking:** Use 'track_order' for live updates",
  "order_status": {
    "order_id": "ORD-20250116-143052",
    "current_status": "in_transit",
    "payment_status": "completed",
    "delivery_progress": [
      {
        "stage": "confirmed",
        "timestamp": "2025-01-16T14:35:52.123Z",
        "completed": true
      },
      {
        "stage": "dispatched", 
        "timestamp": "2025-01-17T10:00:00.000Z",
        "completed": true
      },
      {
        "stage": "in_transit",
        "timestamp": "2025-01-17T14:30:00.000Z",
        "completed": true
      }
    ]
  }
}
```

**Key Features**:
- Real-time status updates from backend
- Detailed order timeline
- Payment status tracking
- Delivery progress visualization

---

### 6.4 `track_order`

**Purpose**: Get detailed tracking information and live updates for orders in progress.

**Parameters**:
- `session` (optional): Session object or ID  
- `order_id` (optional): Specific order ID to track

**Returns**:
- Live tracking information
- GPS coordinates (if available)
- Estimated delivery time
- Delivery partner details
- Real-time status updates

**Usage Example**:
```json
{
  "tool": "track_order",
  "parameters": {
    "session": "shop_20250116_143052_abc123", 
    "order_id": "ORD-20250116-143052"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": " **Live Order Tracking**\n\n**Order ID:** ORD-20250116-143052\n\n **Current Location:**\nNear Hebbal Flyover, Bangalore\nLast Update: 5 minutes ago\n\n **Delivery Partner:**\nName: Raj Kumar\nPhone: +91-9876543210\nVehicle: KA-03-AB-1234\n\n⏰ **Estimated Delivery:**\nToday, 6:30 PM - 7:00 PM\n\n **Live Updates:**\n• 2:45 PM - Out for delivery\n• 2:30 PM - Reached local hub\n• 10:00 AM - Dispatched from warehouse",
  "tracking_info": {
    "order_id": "ORD-20250116-143052",
    "current_location": {
      "description": "Near Hebbal Flyover, Bangalore",
      "latitude": 13.0359,
      "longitude": 77.5908,
      "last_update": "2025-01-17T16:25:00.000Z"
    },
    "delivery_partner": {
      "name": "Raj Kumar", 
      "phone": "+91-9876543210",
      "vehicle_number": "KA-03-AB-1234"
    },
    "estimated_delivery": {
      "date": "2025-01-17",
      "time_window": "18:30-19:00"
    }
  }
}
```

**Key Features**:
- Real-time GPS tracking integration
- Delivery partner information
- Live status updates
- Estimated delivery windows
- Historical tracking timeline

---

## Tool Integration & Workflows

### Complete Shopping Workflow

```
1. Session Setup:
   initialize_shopping() → get_session_info()

2. Product Discovery:
   search_products("query") → browse_categories() → advanced_search()

3. Cart Management:
   add_to_cart(item) → view_cart() → update_cart_quantity() → get_cart_total()

4. Authentication:
   phone_login("9876543210")

5. Checkout Process:
   select_items_for_order(location) → initialize_order(details) → 
   create_payment() → confirm_order()

6. Order Management:
   get_order_status() → track_order()
```

### Error Handling Patterns

All tools follow consistent error handling:

```json
{
  "success": false,
  "message": " Error description with user-friendly guidance",
  "session_id": "session_identifier",
  "error_type": "validation_error",
  "required_action": "phone_login",
  "suggestions": ["Try authenticating first", "Check cart contents"]
}
```

### Session State Management

Each tool call:
1. Retrieves/creates persistent session
2. Validates required permissions/state
3. Performs business logic operation
4. Updates session state
5. Saves session to persistent storage
6. Returns formatted response

This comprehensive features documentation covers all 21 MCP tools with detailed usage examples, parameters, responses, and integration patterns for building complete conversational commerce experiences.