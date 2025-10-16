#!/usr/bin/env python3
"""
ONDC Shopping MCP Server - Official FastMCP Implementation

A fully compliant MCP server implementation using the official MCP SDK.
This implementation uses FastMCP with decorator patterns for maximum compatibility
with MCP clients and AI agents.

Key Features:
- Official MCP SDK compliance
- Decorator-based tool registration
- Session continuity with biap-client-node-js patterns  
- Full cart operations support
- Vector search integration
- Comprehensive error handling

=== AGENT INSTRUCTIONS ===
You are an ONDC shopping assistant that works with authenticated user sessions.
User credentials are managed through session persistence:
- userId: Retrieved from session storage
- deviceId: Retrieved from session storage

Order Journey Flow:
1. initialize_shopping → Create authenticated session
2. search_products → Find products  
3. add_to_cart → Add to cart (auto-detects from search)
4. view_cart → Show cart contents

AUTOMATED CHECKOUT WORKFLOW (Phase 1 - Ends at Payment Creation):
5. "proceed to checkout" → select_items_for_order → initialize_order → create_payment → STOP
   a. select_items_for_order → Auto-fetches saved address, gets delivery quotes
   b. initialize_order → Auto-fills customer details from saved data
   c. create_payment → Create payment order and STOP AUTOMATION

MANUAL PAYMENT WORKFLOW (Phase 2 - User/Frontend Driven):
6. [USER COMPLETES PAYMENT] → User pays via frontend Razorpay/UPI
7. verify_payment → Frontend calls this after successful payment
8. confirm_order → Automatic ONLY after verify_payment succeeds

AUTOMATED BEHAVIOR:
PHASE 1 - Address & Order Setup (FULLY AUTOMATED):
- When user says "checkout", "proceed with address", "start order" → ALWAYS start with select_items_for_order
- When select_items_for_order succeeds with saved address → IMMEDIATELY call initialize_order
- When initialize_order succeeds → IMMEDIATELY call create_payment
- When create_payment succeeds → STOP ALL AUTOMATION IMMEDIATELY

PHASE 2 - Payment Processing (MANUAL/FRONTEND ONLY):
- After create_payment → DO NOT call any more tools automatically
- verify_payment → ONLY called manually by user/frontend after payment completion
- confirm_order → ONLY called automatically AFTER manual verify_payment succeeds
- USE SAVED ADDRESS DATA automatically for authenticated users in Phase 1 only

CRITICAL PHRASE MAPPING:
- "proceed with default address" = select_items_for_order (NOT initialize_order)
- "checkout" = select_items_for_order (NOT initialize_order)
- "start order process" = select_items_for_order (NOT initialize_order)

IMPORTANT - AUTOMATION BOUNDARIES:
- NEVER call initialize_order first, even if user mentions address/customer details
- ALWAYS start checkout flow with select_items_for_order
- AUTOMATE ONLY: select_items_for_order → initialize_order → create_payment
- NEVER AUTOMATE: verify_payment or confirm_order (these are manual/frontend only)
- STOP ALL AUTOMATION: immediately after create_payment succeeds
- DO NOT CALL: verify_payment automatically - wait for manual user/frontend call
- MANUAL STEPS: Payment completion, payment verification, order confirmation
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
import json

# Official MCP SDK imports
from mcp.server.fastmcp import FastMCP, Context

# Import existing functionality
from .adapters.utils import (
    get_persistent_session,
    save_persistent_session,
    extract_session_id,
    format_mcp_response,
    format_products_for_display,
    get_services,
    send_raw_data_to_frontend
)

# Import all tool adapters
from .adapters.cart import (
    add_to_cart as cart_add_adapter,
    view_cart as cart_view_adapter,
    remove_from_cart as cart_remove_adapter,
    update_cart_quantity as cart_update_adapter,
    clear_cart as cart_clear_adapter,
    get_cart_total as cart_total_adapter
)

from .adapters.search import (
    search_products as search_adapter,
    advanced_search as advanced_search_adapter,
    browse_categories as categories_adapter
)

from .adapters.checkout import (
    select_items_for_order as select_items_adapter,
    initialize_order as init_order_adapter,
    create_payment as payment_adapter,
    confirm_order as confirm_adapter
)
from .adapters.payment import (
    verify_payment as verify_payment_adapter,
    get_payment_status as payment_status_adapter
)

# GUEST MODE ONLY - Authentication disabled
# from .adapters.auth import phone_login as auth_adapter
from .adapters.session import (
    initialize_shopping as init_session_adapter,
    get_session_info as session_info_adapter
)

from .adapters.orders import (
    initiate_payment as payment_init_adapter,
    confirm_order_simple as confirm_simple_adapter,
    get_order_status as order_status_adapter,
    track_order as track_adapter
)

from .adapters.address import (
    get_delivery_addresses as address_get_adapter,
    add_delivery_address as address_add_adapter,
    update_delivery_address as address_update_adapter,
    delete_delivery_address as address_delete_adapter
)

from .adapters.offer import (
    get_active_offers as offer_get_active_adapter,
    get_applied_offers as offer_get_applied_adapter,
    apply_offer as offer_apply_adapter,
    clear_offers as offer_clear_adapter,
    delete_offer as offer_delete_adapter
)

from .adapters.profile import (
    get_user_profile as profile_get_adapter,
    update_user_profile as profile_update_adapter
)

# Import existing configuration and logging
from .config import config
from .utils import setup_mcp_logging, get_logger
from .utils.logger import get_mcp_operations_logger

logger = get_logger(__name__)
mcp_ops_logger = get_mcp_operations_logger()

# Universal SSE Data Transmission Helper Functions
def extract_raw_data(result, tool_name):
    """Extract raw data based on tool type for SSE transmission"""
    if tool_name == 'search_products':
        return {
            'products': result.get('products', []),
            'total_results': result.get('total_results', 0),
            'search_type': result.get('search_type', 'hybrid'),
            'page': result.get('page', 1),
            'page_size': result.get('page_size', 10)
        }
    elif tool_name in ['add_to_cart', 'view_cart', 'update_cart_quantity', 'remove_from_cart', 'clear_cart', 'get_cart_total']:
        return {
            'cart_items': result.get('raw_backend_data', []),
            'cart_summary': result.get('cart', {}),
            'biap_specifications': True
        }
    # Future tools: just return raw_backend_data if available
    return result.get('raw_backend_data', result)

def has_raw_data(result):
    """Check if result contains any raw data worth transmitting"""
    if not isinstance(result, dict) or not result.get('success'):
        return False
    
    # Check for products data
    if result.get('products'):
        return True
        
    # Check for raw_backend_data (cart, orders, etc.)
    if result.get('raw_backend_data'):
        return True
        
    return False


# Initialize FastMCP server with official SDK
mcp = FastMCP("ondc-shopping")

# ============================================================================
# SESSION HELPER FUNCTIONS
# ============================================================================

def extract_session_from_context(ctx: Context, **kwargs) -> Optional[str]:
    """Extract session ID from MCP context and kwargs using biap-client patterns
    
    Enhanced to prefer existing sessions and maintain continuity between tool calls.
    Session precedence order:
    1. Explicit session_id parameter
    2. Existing session from userId/deviceId combination
    3. MCP transport context session
    4. Default consistent session for testing
    """
    from .services.session_service import get_session_service
    session_service = get_session_service()
    
    session_id = None
    
    # Method 1: Direct session_id parameter (highest priority)
    if kwargs.get('session_id'):
        session_id = kwargs['session_id']
        logger.info(f"[Session] Found explicit session_id: {session_id}")
        return session_id
    
    # Method 2: Extract from userId/deviceId (biap-client pattern)
    user_id = kwargs.get('userId') or kwargs.get('user_id')
    device_id = kwargs.get('deviceId') or kwargs.get('device_id')
    
    # Create consistent session ID patterns
    if user_id and device_id:
        candidate_session_id = f"{user_id}_{device_id}"
        # Check if this session already exists
        existing_session = session_service.get(candidate_session_id)
        if existing_session:
            logger.info(f"[Session] Reusing existing session: {candidate_session_id}")
            return candidate_session_id
        else:
            logger.info(f"[Session] Creating new session from userId/deviceId: {candidate_session_id}")
            return candidate_session_id
            
    elif device_id and (not user_id or user_id == "guestUser"):
        candidate_session_id = f"guest_{device_id}"
        existing_session = session_service.get(candidate_session_id)
        if existing_session:
            logger.info(f"[Session] Reusing existing guest session: {candidate_session_id}")
            return candidate_session_id
        else:
            logger.info(f"[Session] Creating new guest session: {candidate_session_id}")
            return candidate_session_id
            
    elif user_id and user_id != "guestUser":
        candidate_session_id = f"user_{user_id}"
        existing_session = session_service.get(candidate_session_id)
        if existing_session:
            logger.info(f"[Session] Reusing existing user session: {candidate_session_id}")
            return candidate_session_id
        else:
            logger.info(f"[Session] Creating new user session: {candidate_session_id}")
            return candidate_session_id
    
    # Method 3: Extract from MCP context (transport session)
    if hasattr(ctx, 'session') and ctx.session:
        mcp_session_id = getattr(ctx.session, 'id', None)
        if mcp_session_id:
            logger.info(f"[Session] Found session in MCP context: {mcp_session_id}")
            return mcp_session_id
    
    # Method 4: Check for existing default session (for test continuity)
    default_session_id = "default_mcp_session"
    existing_default = session_service.get(default_session_id)
    if existing_default:
        logger.info(f"[Session] Reusing existing default session: {default_session_id}")
        return default_session_id
    else:
        logger.warning(f"[Session] Creating new default session for testing: {default_session_id}")
        return default_session_id

async def handle_tool_execution(tool_name: str, adapter_func, ctx: Context, **kwargs):
    """Generic handler for tool execution with comprehensive request/response logging"""
    logger.info(f"[ENTRY-POINT] handle_tool_execution called for {tool_name}")
    logger.info(f"[ENTRY-POINT] kwargs: {kwargs}")
    
    session_id = None
    start_time = time.time()
    backend_calls = []
    
    try:
        # Extract session ID using biap-client patterns
        session_id = extract_session_from_context(ctx, **kwargs)
        
        # Log comprehensive request information
        request_data = {
            "tool": tool_name,
            "session_id": session_id,
            "parameters": kwargs.copy()
        }
        mcp_ops_logger.log_tool_request(tool_name, session_id, request_data)
        
        # Send real-time tool execution start event to SSE stream
        if session_id:
            try:
                import requests
                import os
                from datetime import datetime
                api_url = os.getenv('API_URL', 'http://localhost:8001')
                tool_start_data = {
                    'session_id': session_id,
                    'event_type': 'tool_start',
                    'tool_name': tool_name,
                    'message': f"Executing {tool_name}...",
                    'timestamp': datetime.now().isoformat()
                }
                requests.post(f"{api_url}/internal/tool-event", json=tool_start_data, timeout=0.5)
                logger.debug(f"[SSE] Sent tool_start event for {tool_name}")
            except Exception as e:
                logger.debug(f"[SSE] Failed to send tool_start event: {e}")
        
        # Basic logging (keep existing for backward compatibility)
        logger.info(f"[{tool_name}] Executing with session: {session_id}")
        logger.debug(f"[{tool_name}] Parameters: {json.dumps(kwargs, indent=2, default=str)}")
        
        # Add session_id to kwargs for adapter
        kwargs['session_id'] = session_id
        
        # Execute the adapter function
        result = await adapter_func(**kwargs)
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Extract result data for logging
        if isinstance(result, dict):
            result_data = result
        else:
            try:
                result_data = json.loads(result) if isinstance(result, str) else {"result": str(result)}
            except:
                result_data = {"result": str(result)}
        
        # Log comprehensive response
        mcp_ops_logger.log_tool_response(
            tool_name, session_id, result_data, execution_time_ms, backend_calls, "success"
        )
        
        # Send real-time tool execution completion event to SSE stream
        if session_id:
            try:
                import requests
                import os
                from datetime import datetime
                api_url = os.getenv('API_URL', 'http://localhost:8001')
                tool_complete_data = {
                    'session_id': session_id,
                    'event_type': 'tool_complete',
                    'tool_name': tool_name,
                    'message': f"Completed {tool_name} in {execution_time_ms:.0f}ms",
                    'execution_time_ms': execution_time_ms,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
                requests.post(f"{api_url}/internal/tool-event", json=tool_complete_data, timeout=0.5)
                logger.debug(f"[SSE] Sent tool_complete event for {tool_name}")
            except Exception as e:
                logger.debug(f"[SSE] Failed to send tool_complete event: {e}")
        
        # Basic logging (keep existing for backward compatibility)
        logger.info(f"[{tool_name}] Execution successful in {execution_time_ms:.2f}ms")
        
        # Log result structure for debugging
        logger.info(f"[Universal Pattern] Tool: {tool_name}, checking result structure")
        logger.info(f"[Universal Pattern] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        if isinstance(result, dict):
            logger.info(f"[Universal Pattern] Has success: {result.get('success')}")
            logger.info(f"[Universal Pattern] Has products: {bool(result.get('products'))}")
            logger.info(f"[Universal Pattern] Has raw_backend_data: {bool(result.get('raw_backend_data'))}")
        
        # Universal SSE data transmission for all tools with raw data
        logger.info(f"[Universal Pattern] About to call has_raw_data() for {tool_name}")
        if has_raw_data(result):
            logger.info(f"[Universal Pattern] has_raw_data() returned True, calling send_raw_data_to_frontend()")
            send_raw_data_to_frontend(session_id, tool_name, result)
        else:
            logger.info(f"[Universal Pattern] has_raw_data() returned False, no SSE transmission")
        
        # Return formatted result (JSON strings for mcp-agent compatibility)
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        else:
            return str(result)
            
    except Exception as e:
        # Calculate execution time for error case
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Log comprehensive error
        if session_id:
            mcp_ops_logger.log_tool_error(tool_name, session_id, e, execution_time_ms)
        
        # Basic error logging (keep existing for backward compatibility)
        error_msg = f"Error in {tool_name}: {str(e)}"
        logger.error(f"[{tool_name}] {error_msg} (after {execution_time_ms:.2f}ms)", exc_info=True)
        
        return json.dumps({
            "success": False,
            "error": error_msg,
            "session_id": session_id,
            "execution_time_ms": execution_time_ms
        }, indent=2)

# ============================================================================
# DRY PRINCIPLE: Tool Factory Pattern 
# ============================================================================

def create_standard_tool(name: str, adapter_func, docstring: str, extra_params: dict = None):
    """Create a standard MCP tool with common parameters.
    
    This factory function reduces code duplication by generating tools
    with the standard parameters that almost all tools share:
    - userId (optional, from session)
    - deviceId (optional, from session)
    - session_id (optional)
    
    Args:
        name: Tool name
        adapter_func: Adapter function to call
        docstring: Tool documentation
        extra_params: Additional parameters beyond the standard ones
    """
    # Define the standard parameters that all tools have
    standard_params = {
        'ctx': 'Context',
        'userId': 'Optional[str] = None',
        'deviceId': 'Optional[str] = None',
        'session_id': 'Optional[str] = None'
    }
    
    # Merge with extra params if provided
    all_params = {**(extra_params or {}), **standard_params}
    
    # Create the async function dynamically
    async def tool_func(**kwargs):
        ctx = kwargs.pop('ctx')
        return await handle_tool_execution(name, adapter_func, ctx, **kwargs)
    
    # Set function metadata
    tool_func.__name__ = name
    tool_func.__doc__ = docstring
    
    # Register with MCP
    return mcp.tool()(tool_func)

# ============================================================================
# CART OPERATIONS - FastMCP Tools
# ============================================================================

@mcp.tool()
async def add_to_cart(
    ctx: Context,
    item: Dict[str, Any],
    quantity: int = 1,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Add an item to the shopping cart.
    
    Compatible with biap-client-node-js cart patterns.
    Supports both guest and authenticated user sessions.
    """
    return await handle_tool_execution("add_to_cart", cart_add_adapter, ctx, 
                                     item=item, quantity=quantity, userId=userId, 
                                     deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def view_cart(
    ctx: Context,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """View all items in the shopping cart.
    
    Returns cart contents with product details and totals.
    """
    return await handle_tool_execution("view_cart", cart_view_adapter, ctx,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool() 
async def update_cart_quantity(
    ctx: Context,
    item_id: str,
    quantity: int,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Update the quantity of an item in the cart."""
    return await handle_tool_execution("update_cart_quantity", cart_update_adapter, ctx,
                                     item_id=item_id, quantity=quantity, userId=userId,
                                     deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def remove_from_cart(
    ctx: Context,
    item_id: str,
    userId: Optional[str] = None, 
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Remove an item from the shopping cart."""
    return await handle_tool_execution("remove_from_cart", cart_remove_adapter, ctx,
                                     item_id=item_id, userId=userId, deviceId=deviceId,
                                     session_id=session_id)

@mcp.tool()
async def clear_cart(
    ctx: Context,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Clear all items from the shopping cart."""
    return await handle_tool_execution("clear_cart", cart_clear_adapter, ctx,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def get_cart_total(
    ctx: Context,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Get the total price of items in the cart."""
    return await handle_tool_execution("get_cart_total", cart_total_adapter, ctx,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

# ============================================================================
# SEARCH OPERATIONS - FastMCP Tools  
# ============================================================================

@mcp.tool()
async def search_products(
    ctx: Context,
    query: str,
    category: Optional[str] = None,
    location: Optional[str] = None,
    max_results: Optional[int] = None,
    relevance_threshold: Optional[float] = None,
    adaptive_results: bool = True,
    context_aware: bool = True,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Search for products with intelligent result sizing and relevance filtering.
    
    Uses AI-driven query analysis to determine optimal result count and relevance filtering.
    Only returns products that are highly relevant to the search query.
    
    Args:
        query: Search query for products
        category: Optional category filter
        location: Optional location preference
        max_results: Maximum results (optional - auto-determined if not specified)
        relevance_threshold: Minimum relevance score (optional - auto-determined if not specified) 
        adaptive_results: Enable intelligent result sizing based on query intent
        context_aware: Use shopping context for better results
        userId: User identifier
        deviceId: Device identifier  
        session_id: Session identifier
        
    Returns:
        Relevant products with search metadata for optimal user experience
    """
    return await handle_tool_execution("search_products", search_adapter, ctx,
                                     query=query, category=category, location=location,
                                     max_results=max_results, relevance_threshold=relevance_threshold,
                                     adaptive_results=adaptive_results, context_aware=context_aware,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def advanced_search(
    ctx: Context,
    filters: Dict[str, Any],
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Perform advanced product search with multiple filters."""
    return await handle_tool_execution("advanced_search", advanced_search_adapter, ctx,
                                     filters=filters, userId=userId, deviceId=deviceId,
                                     session_id=session_id)

@mcp.tool()
async def browse_categories(
    ctx: Context,
    parent_category: Optional[str] = None,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Browse available product categories in the ONDC network."""
    return await handle_tool_execution("browse_categories", categories_adapter, ctx,
                                     parent_category=parent_category, userId=userId,
                                     deviceId=deviceId, session_id=session_id)

# ============================================================================
# ORDER & CHECKOUT OPERATIONS - FastMCP Tools
# ============================================================================

@mcp.tool()
async def select_items_for_order(
    ctx: Context,
    delivery_city: Optional[str] = None,
    delivery_state: Optional[str] = None,
    delivery_pincode: Optional[str] = None,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """🚀 CHECKOUT ENTRY POINT: Always start checkout here! (ONDC SELECT stage)
    
    Use this when user wants to 'checkout', 'proceed with address', or 'start order process'.
    This initiates the checkout process by checking delivery availability and getting quotes.
    
    SMART AUTOMATION: Can be called without parameters to auto-fetch saved addresses.
    MANUAL MODE: Provide delivery_city, delivery_state, delivery_pincode manually.
    
    Args:
        delivery_city: City name (e.g., "Bangalore") - Optional, will auto-fetch if available
        delivery_state: State name (e.g., "Karnataka") - Optional, will auto-fetch if available  
        delivery_pincode: PIN code (e.g., "560001") - Optional, will auto-fetch if available
        userId: User ID (from session)
        deviceId: Device identifier
        session_id: Session identifier
        
    Returns:
        Delivery quotes and availability information
    """
    return await handle_tool_execution("select_items_for_order", select_items_adapter, ctx,
                                     delivery_city=delivery_city, delivery_state=delivery_state,
                                     delivery_pincode=delivery_pincode, userId=userId, 
                                     deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def initialize_order(
    ctx: Context,
    customer_name: Optional[str] = None,
    delivery_address: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    payment_method: str = "razorpay",
    city: Optional[str] = None,
    state: Optional[str] = None,
    pincode: Optional[str] = None,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """⚡ STEP 2: Initialize order with customer details (ONDC INIT stage).
    
    ⚠️ REQUIRES: Must call select_items_for_order first!
    Prepares the order with complete billing and shipping information after getting delivery quotes.
    
    SMART AUTOMATION: Can be called without parameters to auto-extract customer details from saved addresses.
    MANUAL MODE: Provide customer_name, delivery_address, phone, email manually.
    
    Args:
        customer_name: Full name of the customer - Optional, will auto-extract if available
        delivery_address: Complete street address (e.g., "123 Main St, Apt 4B") - Optional, will auto-extract if available
        phone: Contact phone number (10 digits) - Optional, will auto-extract if available
        email: Email address - Optional, will auto-extract if available
        payment_method: Payment type - "razorpay", "upi", "card", "netbanking" (COD not supported)
        city: Delivery city (optional - uses SELECT stage data if not provided)
        state: Delivery state (optional - uses SELECT stage data if not provided)
        pincode: Delivery PIN code (optional - uses SELECT stage data if not provided)
        userId: User ID (from session)
        deviceId: Device identifier
        session_id: Session identifier
        
    Returns:
        Order initialization confirmation with order draft details
    """
    return await handle_tool_execution("initialize_order", init_order_adapter, ctx,
                                     customer_name=customer_name, delivery_address=delivery_address,
                                     phone=phone, email=email, payment_method=payment_method,
                                     city=city, state=state, pincode=pincode,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def create_payment(
    ctx: Context,
    payment_method: str,
    amount: float,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Create payment for the order."""
    return await handle_tool_execution("create_payment", payment_adapter, ctx,
                                     payment_method=payment_method, amount=amount,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def verify_payment(
    ctx: Context,
    payment_status: str,
    payment_id: Optional[str] = None,
    razorpay_payment_id: Optional[str] = None,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Verify payment status after user completes payment via frontend Razorpay SDK.
    
    Call this after the user completes payment on the frontend to update the session
    with the payment status. Required before order confirmation.
    
    Args:
        payment_status: Payment status ('PAID', 'SUCCESS', 'FAILED', 'PENDING')
        payment_id: Payment ID from create_payment response (optional)
        razorpay_payment_id: Actual Razorpay payment ID from frontend (optional)
    """
    return await handle_tool_execution("verify_payment", verify_payment_adapter, ctx,
                                     payment_status=payment_status, payment_id=payment_id,
                                     razorpay_payment_id=razorpay_payment_id,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def get_payment_status(
    ctx: Context,
    userId: Optional[str] = None, 
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Get current payment status for the session.
    
    Returns the current payment status, payment ID, and next action required.
    Useful for checking payment state during the checkout flow.
    """
    return await handle_tool_execution("get_payment_status", payment_status_adapter, ctx,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def confirm_order(
    ctx: Context,
    payment_status: str = "PAID",
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Confirm and finalize the order (ONDC CONFIRM stage).
    
    Finalizes the order after payment processing. In production mode,
    this should be called after successful payment. In mock mode,
    it can be called directly with payment_status="PAID".
    
    Args:
        payment_status: Payment status - "PAID", "PENDING", or "FAILED" (default: "PAID" for mock mode)
        userId: User ID (from session)
        deviceId: Device identifier
        session_id: Session identifier
        
    Returns:
        Order confirmation with order ID and tracking information
    """
    return await handle_tool_execution("confirm_order", confirm_adapter, ctx,
                                     payment_status=payment_status, userId=userId,
                                     deviceId=deviceId, session_id=session_id)

# ============================================================================
# SESSION MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def initialize_shopping(
    ctx: Context,
    user_preferences: Optional[Dict[str, Any]] = None,
    location: Optional[str] = None,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Initialize a new shopping session."""
    return await handle_tool_execution("initialize_shopping", init_session_adapter, ctx,
                                     user_preferences=user_preferences, location=location,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def get_session_info(
    ctx: Context,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Get current session information and status."""
    return await handle_tool_execution("get_session_info", session_info_adapter, ctx,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

# ============================================================================
# ORDER MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def initiate_payment(
    ctx: Context,
    order_id: str,
    payment_details: Dict[str, Any],
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Initiate payment for an existing order (real implementation)."""
    return await handle_tool_execution("initiate_payment", payment_init_adapter, ctx,
                                     order_id=order_id, payment_details=payment_details,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def confirm_order_simple(
    ctx: Context,
    order_id: str,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Confirm an order with simplified parameters (real implementation)."""
    return await handle_tool_execution("confirm_order_simple", confirm_simple_adapter, ctx,
                                     order_id=order_id, userId=userId, deviceId=deviceId,
                                     session_id=session_id)

@mcp.tool()
async def get_order_status(
    ctx: Context,
    order_id: str,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Get the status of an existing order."""
    return await handle_tool_execution("get_order_status", order_status_adapter, ctx,
                                     order_id=order_id, userId=userId, deviceId=deviceId,
                                     session_id=session_id)

@mcp.tool()
async def track_order(
    ctx: Context,
    order_id: str,
    userId: Optional[str] = None,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Track the delivery status of an order."""
    return await handle_tool_execution("track_order", track_adapter, ctx,
                                     order_id=order_id, userId=userId, deviceId=deviceId,
                                     session_id=session_id)

# ============================================================================
# ADDRESS MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def get_delivery_addresses(
    ctx: Context,
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Get user's delivery addresses."""
    return await handle_tool_execution("get_delivery_addresses", address_get_adapter, ctx,
                                     user_id=userId, device_id=deviceId, session_id=session_id)

@mcp.tool()
async def add_delivery_address(
    ctx: Context,
    address_data: Dict[str, Any],
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Add new delivery address."""
    return await handle_tool_execution("add_delivery_address", address_add_adapter, ctx,
                                     address_data=address_data, user_id=userId, 
                                     device_id=deviceId, session_id=session_id)

@mcp.tool()
async def update_delivery_address(
    ctx: Context,
    address_id: str,
    address_data: Dict[str, Any],
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Update existing delivery address."""
    return await handle_tool_execution("update_delivery_address", address_update_adapter, ctx,
                                     address_id=address_id, address_data=address_data,
                                     user_id=userId, device_id=deviceId, session_id=session_id)

@mcp.tool()
async def delete_delivery_address(
    ctx: Context,
    address_id: str,
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Delete delivery address."""
    return await handle_tool_execution("delete_delivery_address", address_delete_adapter, ctx,
                                     address_id=address_id, user_id=userId,
                                     device_id=deviceId, session_id=session_id)

# ============================================================================
# OFFER MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def get_active_offers(
    ctx: Context,
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Get active offers available to user."""
    return await handle_tool_execution("get_active_offers", offer_get_active_adapter, ctx,
                                     user_id=userId, device_id=deviceId, session_id=session_id)

@mcp.tool()
async def get_applied_offers(
    ctx: Context,
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Get offers already applied to user's cart/order."""
    return await handle_tool_execution("get_applied_offers", offer_get_applied_adapter, ctx,
                                     user_id=userId, device_id=deviceId, session_id=session_id)

@mcp.tool()
async def apply_offer(
    ctx: Context,
    offer_id: str,
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Apply an offer to user's cart."""
    return await handle_tool_execution("apply_offer", offer_apply_adapter, ctx,
                                     offer_id=offer_id, user_id=userId,
                                     device_id=deviceId, session_id=session_id)

@mcp.tool()
async def clear_offers(
    ctx: Context,
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Clear all applied offers from user's cart."""
    return await handle_tool_execution("clear_offers", offer_clear_adapter, ctx,
                                     user_id=userId, device_id=deviceId, session_id=session_id)

@mcp.tool()
async def delete_offer(
    ctx: Context,
    offer_id: str,
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Remove a specific applied offer from user's cart."""
    return await handle_tool_execution("delete_offer", offer_delete_adapter, ctx,
                                     offer_id=offer_id, user_id=userId,
                                     device_id=deviceId, session_id=session_id)

# ============================================================================
# USER PROFILE MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def get_user_profile(
    ctx: Context,
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Get user profile information."""
    return await handle_tool_execution("get_user_profile", profile_get_adapter, ctx,
                                     user_id=userId, device_id=deviceId, session_id=session_id)

@mcp.tool()
async def update_user_profile(
    ctx: Context,
    profile_data: Dict[str, Any],
    userId: str,
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Update user profile information."""
    return await handle_tool_execution("update_user_profile", profile_update_adapter, ctx,
                                     profile_data=profile_data, user_id=userId,
                                     device_id=deviceId, session_id=session_id)

# ============================================================================
# RESOURCES - MCP Compliance
# ============================================================================

@mcp.resource("ondc://categories")
async def get_categories_resource() -> str:
    """List all available ONDC product categories"""
    try:
        # Use the browse_categories adapter to get categories
        from .adapters.search import browse_categories
        result = await browse_categories()
        if isinstance(result, dict):
            categories = result.get("categories", [])
            return f"Available ONDC Categories ({len(categories)} total):\n" + \
                   "\n".join([f"- {cat.get('name', 'Unknown')}: {cat.get('description', '')}" 
                             for cat in categories[:20]])
        return "Categories resource: Unable to load categories"
    except Exception as e:
        logger.error(f"Error loading categories resource: {e}")
        return f"Categories resource error: {str(e)}"

@mcp.resource("ondc://session/{session_id}")
async def get_session_resource(session_id: str) -> str:
    """Get session information for a specific session ID"""
    try:
        session_obj, _ = get_persistent_session(session_id)
        return f"Session {session_id}:\n" + \
               f"- User ID: {getattr(session_obj, 'user_id', 'Unknown')}\n" + \
               f"- Authenticated: {getattr(session_obj, 'user_authenticated', False)}\n" + \
               f"- Cart Items: {len(getattr(session_obj, 'cart', []))}\n" + \
               f"- Location: {getattr(session_obj, 'location', 'Not set')}"
    except Exception as e:
        logger.error(f"Error loading session resource: {e}")
        return f"Session resource error: {str(e)}"

# ============================================================================
# MAIN SERVER RUNNER
# ============================================================================

def main():
    """Main entry point with comprehensive logging and error handling"""
    try:
        # Setup logging
        setup_mcp_logging(debug=config.logging.level == "DEBUG")
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid configuration. Please check your .env file.")
        
        logger.info("=" * 60)
        logger.info("ONDC Shopping MCP Server - Official FastMCP Implementation")
        logger.info("=" * 60)
        logger.info(f"Protocol Version: 2025-03-26")
        logger.info(f"Server Name: ondc-shopping")
        logger.info(f"Server Version: 4.0.0 (FastMCP)")
        logger.info(f"Vector Search: {'Enabled' if config.vector.enabled else 'Disabled'}")
        logger.info(f"Total Tools: 27 (Cart: 6, Search: 3, Orders: 8, Session: 2, Address: 4, Offers: 5, Profile: 2)")
        logger.info(f"Session Support: biap-client-node-js compatible")
        logger.info("=" * 60)
        logger.info("🚀 Starting FastMCP server...")
        logger.info("📡 STDIO transport ready for MCP client connection")
        logger.info("✅ Server startup completed successfully")
        
        # Run the FastMCP server (handles its own event loop)
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("⏹️  Server shutdown requested by user")
    except Exception as e:
        logger.error("❌ MCP Server startup FAILED!")
        logger.error(f"Error details: {e}")
        logger.error("This error prevents MCP client from connecting")
        logger.error("Check the error above and fix before retrying")
        raise


if __name__ == "__main__":
    main()