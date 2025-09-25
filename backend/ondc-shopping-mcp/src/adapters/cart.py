"""Cart operations for MCP adapters"""

import sys
import os

# Ensure Python path is set for tool execution context
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from typing import Dict, Any, Optional
import json
from src.adapters.utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services
)
from src.utils.logger import get_logger
from src.utils.field_mapper import from_backend

logger = get_logger(__name__)

# Get services
services = get_services()
cart_service = services['cart_service']


async def add_to_cart(item: Optional[Dict] = None, quantity: int = 1, 
                         session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for add_to_cart"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="add_to_cart", **kwargs)
        
        logger.info(f"[Cart] Add to cart - Session ID: {session_obj.session_id}")
        
        # Enhanced item validation with auto-detection from search history
        if not item or not item.get('name'):
            logger.info(f"[Cart] No item or missing name field - attempting auto-detection from search history")
            
            # Try to auto-detect from recent search history
            if session_obj.search_history:
                last_search = session_obj.search_history[-1]  # Get most recent search
                if last_search.get('products') and len(last_search['products']) > 0:
                    # Use the first product from the last search
                    auto_detected_item = last_search['products'][0]
                    logger.info(f"[Cart] Auto-detected item from search: {auto_detected_item.get('name')}")
                    # Apply field mapping from backend format
                    item = from_backend(auto_detected_item)
                else:
                    logger.error(f"[Cart] No products in recent search history")
                    return format_mcp_response(
                        False, 
                        ' No recent products found. Please search for products first, then add to cart.',
                        session_obj.session_id
                    )
            else:
                logger.error(f"[Cart] No search history available")
                return format_mcp_response(
                    False, 
                    ' Please search for products first, then add to cart.',
                    session_obj.session_id
                )
        
        # Enhanced debugging for item validation
        logger.info(f"[Cart] Final item for cart: {json.dumps(item, indent=2)}")
        
        # Check for required fields after auto-detection
        required_fields = ['name']  # Minimum required field
        missing_fields = [field for field in required_fields if not item.get(field)]
        
        if missing_fields:
            logger.error(f"[Cart] Missing required fields even after auto-detection: {missing_fields}")
            return format_mcp_response(
                False,
                f' Missing required item fields: {", ".join(missing_fields)}',
                session_obj.session_id
            )
        
        # Add item using pure backend service
        try:
            # Pure backend add - no local storage
            success, message = await cart_service.add_item(session_obj, item, quantity)
            logger.info(f"[Cart] Backend add result - Success: {success}, Message: {message}")
            
        except Exception as e:
            logger.error(f"[Cart] Exception in cart_service.add_item: {e}")
            return format_mcp_response(
                False,
                f' Failed to add item to cart: {str(e)}',
                session_obj.session_id
            )
        
        # Get cart summary
        cart_summary = await cart_service.get_cart_summary(session_obj)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            success,
            message,
            session_obj.session_id,
            cart=cart_summary
        )
        
    except Exception as e:
        logger.error(f"Failed to add item to cart: {e}")
        return format_mcp_response(
            False,
            f' Failed to add item to cart: {str(e)}',
            session_id or 'unknown'
        )


async def view_cart(session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for view_cart"""
    logger.error(f"[EMERGENCY DEBUG] view_cart called with session_id={session_id}")
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="view_cart", **kwargs)
        logger.info(f"[Cart] View cart - Session ID: {session_obj.session_id}")
        
        # Get guest configuration for consistent device ID
        from ..config import Config
        config = Config()
        
        # If authenticated, fetch cart from backend
        if session_obj.user_authenticated and session_obj.user_id:
            logger.info(f"[Cart] Fetching cart from backend for user {session_obj.user_id}")
            # Use session device_id (mandatory initialization ensures this is always present)
            device_id = session_obj.device_id
            if not device_id:
                logger.error(f"[Cart] No device_id found in session {session_obj.session_id} - session not properly initialized")
                raise ValueError("Session not properly initialized - missing device_id")
            
            # Get backend cart using singleton with debug logging
            from ..buyer_backend_client import get_buyer_backend_client
            buyer_app = get_buyer_backend_client()
            backend_cart = await buyer_app.get_cart(session_obj.user_id, device_id)
            
            # Handle backend response properly - backend returns list, not dict with error field
            if backend_cart is not None and not (isinstance(backend_cart, dict) and backend_cart.get('error')):
                logger.info(f"[Cart] Backend cart fetched successfully")
                # TODO: Sync backend cart to local session if needed
        
        # Get cart display using pure backend service
        try:
            logger.debug(f"[Cart Debug] Starting view_cart for session {session_obj.session_id}")
            success, cart_display, cart_items = await cart_service.view_cart(session_obj)
            
            if not success:
                return format_mcp_response(
                    False,
                    f" {cart_display}",
                    session_obj.session_id
                )
            
            logger.debug(f"[Cart Debug] view_cart completed successfully with {len(cart_items)} items")
        except Exception as e:
            logger.error(f"[Cart Debug] Error in view_cart: {e}")
            import traceback
            logger.error(f"[Cart Debug] view_cart traceback: {traceback.format_exc()}")
            return format_mcp_response(
                False,
                f" Failed to view cart: {str(e)}",
                session_obj.session_id
            )
        
        # Get cart summary from backend data
        try:
            cart_summary = await cart_service.get_cart_summary(session_obj)
            logger.debug(f"[Cart Debug] get_cart_summary completed successfully")
        except Exception as e:
            logger.error(f"[Cart Debug] Error in get_cart_summary: {e}")
            # Continue anyway - display is more important than summary
            cart_summary = {'items': [], 'total_items': 0, 'total_value': 0.0, 'is_empty': True}
        
        # Save session with persistence
        save_persistent_session(session_obj, conversation_manager)
        
        # Return successful cart view
        return format_mcp_response(
            True,
            cart_display,
            session_obj.session_id,
            cart=cart_summary
        )
        
    except Exception as e:
        logger.error(f"Failed to view cart: {e}")
        return format_mcp_response(
            False,
            f' Failed to view cart: {str(e)}',
            session_id or 'unknown'
        )


async def remove_from_cart(item_id: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for remove_from_cart"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="remove_from_cart", **kwargs)
        
        # Remove item using service
        success, message = await cart_service.remove_item(session_obj, item_id)
        
        # Sync with backend
        await cart_service.sync_with_backend(session_obj)
        
        # Get cart summary
        cart_summary = await cart_service.get_cart_summary(session_obj)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            success,
            message,
            session_obj.session_id,
            cart=cart_summary
        )
        
    except Exception as e:
        logger.error(f"Failed to remove item from cart: {e}")
        return format_mcp_response(
            False,
            f' Failed to remove item: {str(e)}',
            session_id or 'unknown'
        )


async def update_cart_quantity(item_id: str, quantity: int, 
                                  session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for update_cart_quantity"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="update_cart_quantity", **kwargs)
        
        # Update quantity using service
        success, message = await cart_service.update_quantity(session_obj, item_id, quantity)
        
        # Sync with backend
        await cart_service.sync_with_backend(session_obj)
        
        # Get cart summary
        cart_summary = await cart_service.get_cart_summary(session_obj)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            success,
            message,
            session_obj.session_id,
            cart=cart_summary
        )
        
    except Exception as e:
        logger.error(f"Failed to update cart quantity: {e}")
        return format_mcp_response(
            False,
            f' Failed to update quantity: {str(e)}',
            session_id or 'unknown'
        )


async def clear_cart(session_id: Optional[str], **kwargs) -> Dict[str, Any]:
    """MCP adapter for clear_cart"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="clear_cart", **kwargs)
        
        # Clear cart using service
        success, message = await cart_service.clear_cart(session_obj)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            success,
            message,
            session_obj.session_id
        )
        
    except Exception as e:
        logger.error(f"Failed to clear cart: {e}")
        return format_mcp_response(
            False,
            f' Failed to clear cart: {str(e)}',
            session_id or 'unknown'
        )


async def get_cart_total(session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for get_cart_total"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="get_cart_total", **kwargs)
        
        # Get cart summary
        summary = await cart_service.get_cart_summary(session_obj)
        
        if summary['is_empty']:
            message = " Your cart is empty"
        else:
            message = f" Cart Total: {summary['total_items']} items - ₹{summary['total_value']:.2f}"
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            True,
            message,
            session_obj.session_id,
            total_items=summary['total_items'],
            total_value=summary['total_value']
        )
        
    except Exception as e:
        logger.error(f"Failed to get cart total: {e}")
        return format_mcp_response(
            False,
            f' Failed to get cart total: {str(e)}',
            session_id or 'unknown'
        )