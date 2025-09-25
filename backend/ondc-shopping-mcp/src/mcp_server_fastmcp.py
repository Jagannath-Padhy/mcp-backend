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

Agent instructions are maintained in: mcp_agent_instructions.md
"""

import asyncio
import logging
import time
import uuid
import traceback
import os
import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json

# Fix Python path for tool execution context
# This ensures all tool imports work correctly even when called from FastMCP
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Official MCP SDK imports
from mcp.server.fastmcp import FastMCP, Context

# Import existing functionality
from src.adapters.utils import (
    get_persistent_session,
    save_persistent_session,
    extract_session_id,
    format_mcp_response,
    format_products_for_display,
    get_services
)

# Import all tool adapters
from src.adapters.cart import (
    add_to_cart as cart_add_adapter,
    view_cart as cart_view_adapter,
    remove_from_cart as cart_remove_adapter,
    update_cart_quantity as cart_update_adapter,
    clear_cart as cart_clear_adapter,
    get_cart_total as cart_total_adapter
)

from src.adapters.search import (
    search_products as search_adapter,
    advanced_search as advanced_search_adapter,
    browse_categories as categories_adapter
)

from src.adapters.checkout import (
    select_items_for_order as select_items_adapter,
    initialize_order as init_order_adapter,
    create_payment as payment_adapter,
    confirm_order as confirm_adapter
)

# GUEST MODE ONLY - Authentication disabled
# from src.adapters.auth import phone_login as auth_adapter
from src.adapters.session import (
    initialize_shopping as init_session_adapter,
    get_session_info as session_info_adapter
)

from src.adapters.orders import (
    initiate_payment as payment_init_adapter,
    confirm_order_simple as confirm_simple_adapter,
    get_order_status as order_status_adapter,
    track_order as track_adapter
)

from src.adapters.address import (
    get_delivery_addresses as address_get_adapter,
    add_delivery_address as address_add_adapter,
    update_delivery_address as address_update_adapter,
    delete_delivery_address as address_delete_adapter
)

from src.adapters.offer import (
    get_active_offers as offer_get_active_adapter,
    get_applied_offers as offer_get_applied_adapter,
    apply_offer as offer_apply_adapter,
    clear_offers as offer_clear_adapter,
    delete_offer as offer_delete_adapter
)

from src.adapters.profile import (
    get_user_profile as profile_get_adapter,
    update_user_profile as profile_update_adapter
)

# Import existing configuration and logging
from src.config import config
from src.utils import setup_mcp_logging, get_logger
from src.utils.logger import get_mcp_operations_logger

logger = get_logger(__name__)
mcp_ops_logger = get_mcp_operations_logger()

# Initialize FastMCP server with official SDK
mcp = FastMCP("ondc-shopping")

# ============================================================================
# SESSION HELPER FUNCTIONS
# ============================================================================

def extract_session_from_context(ctx: Context, **kwargs) -> str:
    """Extract or generate unique session ID for conversation isolation
    
    Maintains session persistence within conversations while ensuring
    isolation between different conversations.
    """
    # Use imports from module level
    
    # 1. Try to use MCP's built-in session_id if available
    if hasattr(ctx, 'session_id') and ctx.session_id:
        logger.info(f"[Session] Using MCP session: {ctx.session_id}")
        return ctx.session_id
    
    # 2. Check if session_id was provided in tool parameters
    if 'session_id' in kwargs and kwargs['session_id']:
        logger.info(f"[Session] Using provided session: {kwargs['session_id']}")
        return kwargs['session_id']
    
    # 3. Check if there's a chat API session context (file-based approach)
    # This allows chat API to set the session context that MCP tools can use
    try:
        import os
        context_file = os.path.expanduser("~/.ondc-mcp/chat_session_context.txt")
        if os.path.exists(context_file):
            with open(context_file, 'r') as f:
                chat_session = f.read().strip()
                if chat_session:
                    logger.info(f"[Session] Using chat API session from file: {chat_session}")
                    return chat_session
    except Exception as e:
        logger.debug(f"[Session] Failed to read chat API session context: {e}")
    
    # 4. Generate new session only if no existing session found
    session_id = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    logger.info(f"[Session] Generated new session: {session_id}")
    return session_id

async def handle_tool_execution(tool_name: str, adapter_func, ctx: Context, **kwargs):
    """Generic handler for tool execution with comprehensive request/response logging"""
    session_id = None
    start_time = time.time()
    backend_calls = []
    
    try:
        # Extract session ID using biap-client patterns
        session_id = extract_session_from_context(ctx, **kwargs)
        
        # Session-first authentication enforcement
        userId = kwargs.get('userId')
        deviceId = kwargs.get('deviceId')
        
        if tool_name == 'initialize_shopping':
            # Initialize: require credentials, allow session_id for updates
            if not (userId and deviceId):
                return json.dumps({
                    "success": False, 
                    "error": "CREDENTIALS_REQUIRED",
                    "message": "initialize_shopping requires both userId and deviceId parameters",
                    "required_params": ["userId", "deviceId"]
                }, indent=2)
            # Credentials are valid, proceed with initialization (with or without session_id)
            # The adapter function will handle both new session creation and existing session updates
        
        elif tool_name in ['get_session_info']:
            # Session info tools don't need credentials
            pass
            
        else:
            # All other tools: require session_id, forbid credentials
            if userId or deviceId:
                return json.dumps({
                    "success": False,
                    "error": "INVALID_PARAMS", 
                    "message": f"Don't provide userId/deviceId to {tool_name} - only session_id from initialize_shopping",
                    "required_action": "initialize_shopping(userId='...', deviceId='...') first, then use session_id"
                }, indent=2)
            
            if not session_id:
                return json.dumps({
                    "success": False,
                    "error": "SESSION_REQUIRED",
                    "message": f"{tool_name} requires session_id. Call initialize_shopping(userId='...', deviceId='...') first",
                    "required_action": "initialize_shopping"
                }, indent=2)
            
            # Validate session exists and load credentials
            from src.adapters.utils import get_persistent_session
            try:
                session_obj, _ = get_persistent_session(session_id, tool_name=tool_name)
                
                if not session_obj:
                    return json.dumps({
                        "success": False,
                        "error": "INVALID_SESSION",
                        "message": f"Session '{session_id}' not found. Call initialize_shopping(userId='...', deviceId='...') first",
                        "required_action": "initialize_shopping"
                    }, indent=2)
                
                if not (session_obj.user_id and session_obj.device_id):
                    return json.dumps({
                        "success": False,
                        "error": "INVALID_SESSION",
                        "message": "Session missing credentials. Call initialize_shopping(userId='...', deviceId='...') first",
                        "required_action": "initialize_shopping"
                    }, indent=2)
                
                # Auto-inject stored credentials from session
                kwargs['userId'] = session_obj.user_id
                kwargs['deviceId'] = session_obj.device_id
                logger.info(f"[{tool_name}] Using session {session_id[:16]}... - credentials: userId={session_obj.user_id[:8]}..., deviceId={session_obj.device_id[:8]}...")
                
            except Exception as e:
                logger.error(f"[{tool_name}] Session validation failed: {e}")
                return json.dumps({
                    "success": False,
                    "error": "SESSION_ERROR",
                    "message": f"Session validation failed: {str(e)}. Call initialize_shopping(userId='...', deviceId='...') first",
                    "required_action": "initialize_shopping"
                }, indent=2)
        
        # Log comprehensive request information
        request_data = {
            "tool": tool_name,
            "session_id": session_id,
            "parameters": kwargs.copy()
        }
        mcp_ops_logger.log_tool_request(tool_name, session_id, request_data)
        
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
        
        # Basic logging (keep existing for backward compatibility)
        logger.info(f"[{tool_name}] Execution successful in {execution_time_ms:.2f}ms")
        
        # Return formatted result
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
    - userId (default: guestUser)
    - deviceId (optional)
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
        'session_id': 'str'  # Required for all tools except initialize_shopping
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
    session_id: str,
    quantity: int = 1
) -> str:
    """Add an item to the shopping cart.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("add_to_cart", cart_add_adapter, ctx, 
                                     item=item, quantity=quantity, session_id=session_id)

@mcp.tool()
async def view_cart(
    ctx: Context,
    session_id: str
) -> str:
    """View all items in the shopping cart.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("view_cart", cart_view_adapter, ctx,
                                     session_id=session_id)

@mcp.tool() 
async def update_cart_quantity(
    ctx: Context,
    item_id: str,
    quantity: int,
    session_id: str
) -> str:
    """Update the quantity of an item in the cart.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("update_cart_quantity", cart_update_adapter, ctx,
                                     item_id=item_id, quantity=quantity, session_id=session_id)

@mcp.tool()
async def remove_from_cart(
    ctx: Context,
    item_id: str,
    session_id: str
) -> str:
    """Remove an item from the shopping cart.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("remove_from_cart", cart_remove_adapter, ctx,
                                     item_id=item_id, session_id=session_id)

@mcp.tool()
async def clear_cart(
    ctx: Context,
    session_id: str
) -> str:
    """Clear all items from the shopping cart.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("clear_cart", cart_clear_adapter, ctx,
                                     session_id=session_id)

@mcp.tool()
async def get_cart_total(
    ctx: Context,
    session_id: str
) -> str:
    """Get the total price of items in the cart.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("get_cart_total", cart_total_adapter, ctx,
                                     session_id=session_id)

# ============================================================================
# SEARCH OPERATIONS - FastMCP Tools  
# ============================================================================

@mcp.tool()
async def search_products(
    ctx: Context,
    query: str,
    session_id: str,
    category: Optional[str] = None,
    location: Optional[str] = None,
    max_results: int = 10
) -> str:
    """Search for products in the ONDC network.
    
    Requires session_id from initialize_shopping.
    
    Args:
        query: Search term for products
        session_id: Session ID from initialize_shopping (required)
        category: Optional category filter
        location: Optional location filter
        max_results: Maximum number of results
        
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("search_products", search_adapter, ctx,
                                     query=query, category=category, location=location,
                                     max_results=max_results, session_id=session_id)

@mcp.tool()
async def advanced_search(
    ctx: Context,
    filters: Dict[str, Any],
    session_id: str
) -> str:
    """Perform advanced product search with multiple filters.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("advanced_search", advanced_search_adapter, ctx,
                                     filters=filters, session_id=session_id)

@mcp.tool()
async def browse_categories(
    ctx: Context,
    session_id: str,
    parent_category: Optional[str] = None
) -> str:
    """Browse available product categories in the ONDC network.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("browse_categories", categories_adapter, ctx,
                                     parent_category=parent_category, session_id=session_id)

# ============================================================================
# ORDER & CHECKOUT OPERATIONS - FastMCP Tools
# ============================================================================

@mcp.tool()
async def select_items_for_order(
    ctx: Context,
    delivery_city: str,
    delivery_state: str,
    delivery_pincode: str,
    delivery_gps: str,
    session_id: str
) -> str:
    """Get delivery quotes and options for items in cart (ONDC SELECT stage).
    
    This initiates the checkout process by checking delivery availability
    and getting quotes for the items in your cart.
    
    Args:
        delivery_city: City name (e.g., "Bangalore")
        delivery_state: State name (e.g., "Karnataka")
        delivery_pincode: PIN code (e.g., "560001")
        delivery_gps: GPS coordinates in 'latitude,longitude' format (e.g., "12.9716,77.5946")
        session_id: Session identifier (required - from initialize_shopping)
        
    Returns:
        Delivery quotes and availability information
    """
    return await handle_tool_execution("select_items_for_order", select_items_adapter, ctx,
                                     delivery_city=delivery_city, delivery_state=delivery_state,
                                     delivery_pincode=delivery_pincode, delivery_gps=delivery_gps, session_id=session_id)

@mcp.tool()
async def initialize_order(
    ctx: Context,
    customer_name: str,
    building: str,
    street: str, 
    locality: str,
    city: str,
    state: str,
    pincode: str,
    delivery_gps: str,
    phone: str,
    email: str,
    session_id: str,
    payment_method: str = "razorpay"
) -> str:
    """Initialize order with customer and delivery details (ONDC INIT stage).
    
    Prepares the order with complete billing and shipping information
    after getting delivery quotes from SELECT stage.
    
    Args:
        customer_name: Full name of the customer
        building: Building/apartment number  
        street: Street name
        locality: Area/locality name
        city: City name
        state: State name
        pincode: PIN/ZIP code
        delivery_gps: GPS coordinates in "latitude,longitude" format (e.g. "12.9716,77.5946")
        phone: Contact phone number
        email: Email address
        session_id: Session ID from initialize_shopping
        payment_method: Payment method (default: "razorpay") - cod not supported by Himira
        
    Returns:
        Order initialization status with next steps
    """
    # Combine structured address into full address for backend processing
    full_delivery_address = f"{building} {street} {locality}"
    
    return await handle_tool_execution("initialize_order", init_order_adapter, ctx,
                                     customer_name=customer_name, delivery_address=full_delivery_address,
                                     phone=phone, email=email, payment_method=payment_method,
                                     city=city, state=state, pincode=pincode, 
                                     delivery_gps=delivery_gps, session_id=session_id)

@mcp.tool()
async def create_payment(
    ctx: Context,
    payment_method: str,
    amount: float,
    session_id: str
) -> str:
    """Create payment for the order.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("create_payment", payment_adapter, ctx,
                                     payment_method=payment_method, amount=amount,
                                     session_id=session_id)

@mcp.tool()
async def confirm_order(
    ctx: Context,
    session_id: str,
    payment_status: str = "PAID"
) -> str:
    """Confirm and finalize the order (ONDC CONFIRM stage).
    
    Finalizes the order after payment processing. In production mode,
    this should be called after successful payment. In mock mode,
    it can be called directly with payment_status="PAID".
    
    Args:
        session_id: Session identifier (required - from initialize_shopping)
        payment_status: Payment status - "PAID", "PENDING", or "FAILED" (default: "PAID" for mock mode)
        
    Returns:
        Order confirmation with order ID and tracking information
    """
    return await handle_tool_execution("confirm_order", confirm_adapter, ctx,
                                     payment_status=payment_status, session_id=session_id)

# ============================================================================
# SESSION MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def initialize_shopping(
    ctx: Context,
    userId: str,
    deviceId: str,
    user_preferences: Optional[Dict[str, Any]] = None,
    location: Optional[str] = None
) -> str:
    """Initialize a new shopping session with your Himira credentials.
    
    MANDATORY FIRST STEP: This must be called before any other shopping operations.
    
    Args:
        userId: Your Himira user ID from the frontend (required)
        deviceId: Your device ID from the frontend (required)
        user_preferences: Optional user preferences
        location: Optional location preference
        
    Returns:
        Session information with session_id that must be used for all subsequent operations
    """
    return await handle_tool_execution("initialize_shopping", init_session_adapter, ctx,
                                     userId=userId, deviceId=deviceId, 
                                     user_preferences=user_preferences, location=location)

@mcp.tool()
async def get_session_info(
    ctx: Context,
    session_id: str
) -> str:
    """Get current session information and status."""
    return await handle_tool_execution("get_session_info", session_info_adapter, ctx,
                                     session_id=session_id)

# ============================================================================
# ORDER MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def initiate_payment(
    ctx: Context,
    order_id: str,
    payment_details: Dict[str, Any],
    session_id: str
) -> str:
    """Initiate payment for an existing order.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("initiate_payment", payment_init_adapter, ctx,
                                     order_id=order_id, payment_details=payment_details,
                                     session_id=session_id)

@mcp.tool()
async def confirm_order_simple(
    ctx: Context,
    order_id: str,
    session_id: str
) -> str:
    """Confirm an order with simplified parameters.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("confirm_order_simple", confirm_simple_adapter, ctx,
                                     order_id=order_id, session_id=session_id)

@mcp.tool()
async def get_order_status(
    ctx: Context,
    order_id: str,
    session_id: str
) -> str:
    """Get the status of an existing order.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("get_order_status", order_status_adapter, ctx,
                                     order_id=order_id, session_id=session_id)

@mcp.tool()
async def track_order(
    ctx: Context,
    order_id: str,
    session_id: str
) -> str:
    """Track the delivery status of an order.
    
    Requires initialized session from initialize_shopping.
    """
    return await handle_tool_execution("track_order", track_adapter, ctx,
                                     order_id=order_id, session_id=session_id)

# ============================================================================
# ADDRESS MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def get_delivery_addresses(
    ctx: Context,
    session_id: str
) -> str:
    """Get user's delivery addresses.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("get_delivery_addresses", address_get_adapter, ctx,
                                     session_id=session_id)

@mcp.tool()
async def add_delivery_address(
    ctx: Context,
    address_data: Dict[str, Any],
    session_id: str
) -> str:
    """Add new delivery address.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("add_delivery_address", address_add_adapter, ctx,
                                     address_data=address_data, session_id=session_id)

@mcp.tool()
async def update_delivery_address(
    ctx: Context,
    address_id: str,
    address_data: Dict[str, Any],
    session_id: str
) -> str:
    """Update existing delivery address.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("update_delivery_address", address_update_adapter, ctx,
                                     address_id=address_id, address_data=address_data, session_id=session_id)

@mcp.tool()
async def delete_delivery_address(
    ctx: Context,
    address_id: str,
    session_id: str
) -> str:
    """Delete delivery address.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("delete_delivery_address", address_delete_adapter, ctx,
                                     address_id=address_id, session_id=session_id)

# ============================================================================
# OFFER MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def get_active_offers(
    ctx: Context,
    session_id: str
) -> str:
    """Get active offers available to user.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("get_active_offers", offer_get_active_adapter, ctx,
                                     session_id=session_id)

@mcp.tool()
async def get_applied_offers(
    ctx: Context,
    session_id: str
) -> str:
    """Get offers already applied to user's cart/order.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("get_applied_offers", offer_get_applied_adapter, ctx,
                                     session_id=session_id)

@mcp.tool()
async def apply_offer(
    ctx: Context,
    offer_id: str,
    session_id: str
) -> str:
    """Apply an offer to user's cart.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("apply_offer", offer_apply_adapter, ctx,
                                     offer_id=offer_id, session_id=session_id)

@mcp.tool()
async def clear_offers(
    ctx: Context,
    session_id: str
) -> str:
    """Clear all applied offers from user's cart.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("clear_offers", offer_clear_adapter, ctx,
                                     session_id=session_id)

@mcp.tool()
async def delete_offer(
    ctx: Context,
    offer_id: str,
    session_id: str
) -> str:
    """Remove a specific applied offer from user's cart.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("delete_offer", offer_delete_adapter, ctx,
                                     offer_id=offer_id, session_id=session_id)

# ============================================================================
# USER PROFILE MANAGEMENT - FastMCP Tools
# ============================================================================

@mcp.tool()
async def get_user_profile(
    ctx: Context,
    session_id: str
) -> str:
    """Get user profile information.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("get_user_profile", profile_get_adapter, ctx,
                                     session_id=session_id)

@mcp.tool()
async def update_user_profile(
    ctx: Context,
    profile_data: Dict[str, Any],
    session_id: str
) -> str:
    """Update user profile information.
    
    Requires session_id from initialize_shopping.
    
    If you get SESSION_REQUIRED error:
    Call initialize_shopping(userId='...', deviceId='...') first
    """
    return await handle_tool_execution("update_user_profile", profile_update_adapter, ctx,
                                     profile_data=profile_data, session_id=session_id)

# ============================================================================
# RESOURCES - MCP Compliance
# ============================================================================

@mcp.resource("ondc://agent-instructions")
async def get_agent_instructions() -> str:
    """Agent instructions for ONDC shopping - MANDATORY credential collection"""
    try:
        # Use os from module level
        instructions_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_agent_instructions.md")
        if os.path.exists(instructions_path):
            with open(instructions_path, 'r') as f:
                return f.read()
        return "⚠️ MANDATORY: Ask user for userId and deviceId before ANY shopping operations!"
    except Exception as e:
        return "🚨 CRITICAL: Always collect userId and deviceId from user first!"

@mcp.resource("ondc://categories")
async def get_categories_resource() -> str:
    """List all available ONDC product categories"""
    try:
        # Use the browse_categories adapter to get categories
        from src.adapters.search import browse_categories
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