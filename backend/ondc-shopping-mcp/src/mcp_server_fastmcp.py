#!/usr/bin/env python3
"""
ONDC Shopping MCP Server - Official FastMCP Implementation

A fully compliant MCP server implementation using the official Anthropic MCP SDK.
This implementation uses FastMCP with decorator patterns for maximum compatibility
with Claude Desktop and Langflow.

Key Features:
- Official MCP SDK compliance
- Decorator-based tool registration
- Session continuity with biap-client-node-js patterns  
- Full cart operations support
- Vector search integration
- Comprehensive error handling
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
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
    get_services
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

logger = get_logger(__name__)

# Initialize FastMCP server with official SDK
mcp = FastMCP("ondc-shopping")

# ============================================================================
# SESSION HELPER FUNCTIONS
# ============================================================================

def extract_session_from_context(ctx: Context, **kwargs) -> Optional[str]:
    """Extract session ID from MCP context and kwargs using biap-client patterns"""
    session_id = None
    
    # Method 1: Direct session_id parameter
    if kwargs.get('session_id'):
        session_id = kwargs['session_id']
        logger.info(f"[Session] Found session_id in kwargs: {session_id}")
    
    # Method 2: Extract from userId/deviceId (biap-client pattern)
    user_id = kwargs.get('userId') or kwargs.get('user_id')
    device_id = kwargs.get('deviceId') or kwargs.get('device_id')
    
    if user_id and device_id:
        # Create session ID using biap-client pattern
        session_id = f"{user_id}_{device_id}"
        logger.info(f"[Session] Created session from userId/deviceId: {session_id}")
    elif device_id and (not user_id or user_id == "guestUser"):
        # Guest user with device ID
        session_id = f"guest_{device_id}"
        logger.info(f"[Session] Created guest session: {session_id}")
    elif user_id and user_id != "guestUser":
        # Authenticated user without device ID
        session_id = f"user_{user_id}"
        logger.info(f"[Session] Created user session: {session_id}")
    
    # Method 3: Extract from MCP context (if available)
    if not session_id and hasattr(ctx, 'session'):
        session_id = getattr(ctx.session, 'id', None)
        if session_id:
            logger.info(f"[Session] Found session in MCP context: {session_id}")
    
    # Method 4: Generate default session for testing
    if not session_id:
        session_id = "default_mcp_session"
        logger.warning(f"[Session] No session found, using default: {session_id}")
    
    return session_id

async def handle_tool_execution(tool_name: str, adapter_func, ctx: Context, **kwargs):
    """Generic handler for tool execution with proper session management"""
    try:
        # Extract session ID using biap-client patterns
        session_id = extract_session_from_context(ctx, **kwargs)
        
        logger.info(f"[{tool_name}] Executing with session: {session_id}")
        logger.debug(f"[{tool_name}] Parameters: {json.dumps(kwargs, indent=2, default=str)}")
        
        # Add session_id to kwargs for adapter
        kwargs['session_id'] = session_id
        
        # Execute the adapter function
        result = await adapter_func(**kwargs)
        
        # Log successful execution
        logger.info(f"[{tool_name}] Execution successful")
        
        # Return formatted result
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        else:
            return str(result)
            
    except Exception as e:
        error_msg = f"Error in {tool_name}: {str(e)}"
        logger.error(f"[{tool_name}] {error_msg}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "session_id": session_id if 'session_id' in locals() else None
        }, indent=2)

# ============================================================================
# CART OPERATIONS - FastMCP Tools
# ============================================================================

@mcp.tool()
async def add_to_cart(
    ctx: Context,
    item: Dict[str, Any],
    quantity: int = 1,
    userId: Optional[str] = "guestUser",
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
    userId: Optional[str] = "guestUser",
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
    userId: Optional[str] = "guestUser",
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
    userId: Optional[str] = "guestUser", 
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
    userId: Optional[str] = "guestUser",
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Clear all items from the shopping cart."""
    return await handle_tool_execution("clear_cart", cart_clear_adapter, ctx,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def get_cart_total(
    ctx: Context,
    userId: Optional[str] = "guestUser",
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
    max_results: int = 10,
    userId: Optional[str] = "guestUser",
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Search for products in the ONDC network."""
    return await handle_tool_execution("search_products", search_adapter, ctx,
                                     query=query, category=category, location=location,
                                     max_results=max_results, userId=userId, deviceId=deviceId,
                                     session_id=session_id)

@mcp.tool()
async def advanced_search(
    ctx: Context,
    filters: Dict[str, Any],
    userId: Optional[str] = "guestUser",
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
    userId: Optional[str] = "guestUser",
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
    item_ids: List[str],
    userId: Optional[str] = "guestUser",
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Select items from cart for order processing."""
    return await handle_tool_execution("select_items_for_order", select_items_adapter, ctx,
                                     item_ids=item_ids, userId=userId, deviceId=deviceId,
                                     session_id=session_id)

@mcp.tool()
async def initialize_order(
    ctx: Context,
    delivery_address: Dict[str, Any],
    userId: Optional[str] = "guestUser",
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Initialize order with delivery details."""
    return await handle_tool_execution("initialize_order", init_order_adapter, ctx,
                                     delivery_address=delivery_address, userId=userId,
                                     deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def create_payment(
    ctx: Context,
    payment_method: str,
    amount: float,
    userId: Optional[str] = "guestUser",
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Create payment for the order."""
    return await handle_tool_execution("create_payment", payment_adapter, ctx,
                                     payment_method=payment_method, amount=amount,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def confirm_order(
    ctx: Context,
    order_details: Dict[str, Any],
    userId: Optional[str] = "guestUser",
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Confirm and finalize the order."""
    return await handle_tool_execution("confirm_order", confirm_adapter, ctx,
                                     order_details=order_details, userId=userId,
                                     deviceId=deviceId, session_id=session_id)

# ============================================================================
# AUTHENTICATION & SESSION MANAGEMENT - FastMCP Tools
# ============================================================================

# GUEST MODE ONLY - Phone login disabled for pure guest checkout
# @mcp.tool()
# async def phone_login(
#     ctx: Context,
#     phone_number: str,
#     otp: Optional[str] = None,
#     deviceId: Optional[str] = None,
#     session_id: Optional[str] = None
# ) -> str:
#     """Authenticate user with phone number and OTP."""
#     return await handle_tool_execution("phone_login", auth_adapter, ctx,
#                                      phone_number=phone_number, otp=otp,
#                                      deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def initialize_shopping(
    ctx: Context,
    user_preferences: Optional[Dict[str, Any]] = None,
    location: Optional[str] = None,
    userId: Optional[str] = "guestUser",
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
    userId: Optional[str] = "guestUser",
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
    userId: Optional[str] = "guestUser",
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Initiate payment for an existing order."""
    return await handle_tool_execution("initiate_payment", payment_init_adapter, ctx,
                                     order_id=order_id, payment_details=payment_details,
                                     userId=userId, deviceId=deviceId, session_id=session_id)

@mcp.tool()
async def confirm_order_simple(
    ctx: Context,
    order_id: str,
    userId: Optional[str] = "guestUser",
    deviceId: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Confirm an order with simplified parameters."""
    return await handle_tool_execution("confirm_order_simple", confirm_simple_adapter, ctx,
                                     order_id=order_id, userId=userId, deviceId=deviceId,
                                     session_id=session_id)

@mcp.tool()
async def get_order_status(
    ctx: Context,
    order_id: str,
    userId: Optional[str] = "guestUser",
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
    userId: Optional[str] = "guestUser",
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
        logger.info("📡 STDIO transport ready for Claude Desktop connection")
        logger.info("✅ Server startup completed successfully")
        
        # Run the FastMCP server (handles its own event loop)
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("⏹️  Server shutdown requested by user")
    except Exception as e:
        logger.error("❌ MCP Server startup FAILED!")
        logger.error(f"Error details: {e}")
        logger.error("This error prevents Claude Desktop from connecting")
        logger.error("Check the error above and fix before retrying")
        raise


if __name__ == "__main__":
    main()