"""Cart operations for MCP adapters"""

from typing import Dict, Any, Optional
import json
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services
)
from ..utils.logger import get_logger
from ..utils.field_mapper import from_backend

logger = get_logger(__name__)

# Get services
services = get_services()
cart_service = services['cart_service']


async def add_to_cart(session_id: Optional[str] = None, item: Optional[Dict] = None, 
                         quantity: int = 1, **kwargs) -> Dict[str, Any]:
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
        
        # Add item using service with error handling
        try:
            # First add to local cart
            success, message = await cart_service.add_item(session_obj, item, quantity)
            logger.info(f"[Cart] Add item result - Success: {success}, Message: {message}")
            
            # If local add succeeded and user is authenticated, sync with backend
            if success and session_obj.user_authenticated and session_obj.user_id:
                logger.info(f"[Cart] User authenticated, adding to backend cart")
                backend_success, backend_msg = await cart_service.add_item_to_backend(session_obj, item, quantity)
                logger.info(f"[Cart] Backend add result - Success: {backend_success}, Message: {backend_msg}")
                
                # Use backend result if available
                if not backend_success:
                    # Backend failed, but local succeeded - warn user
                    message = f" {message}\n(Note: Backend sync failed - {backend_msg})"
            
        except Exception as e:
            logger.error(f"[Cart] Exception in cart_service.add_item: {e}")
            return format_mcp_response(
                False,
                f' Failed to add item to cart: {str(e)}',
                session_obj.session_id
            )
        
        # Get cart summary
        cart_summary = cart_service.get_cart_summary(session_obj)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            success,
            message,
            session_obj.session_id,
            cart_summary=cart_summary
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
            # Use session device_id if available, otherwise fall back to configured guest device_id
            device_id = getattr(session_obj, 'device_id', config.guest.device_id)
            
            # Get backend cart
            from ..buyer_backend_client import BuyerBackendClient
            buyer_app = BuyerBackendClient()
            backend_cart = await buyer_app.get_cart(session_obj.user_id, device_id)
            
            if backend_cart and not backend_cart.get('error'):
                logger.info(f"[Cart] Backend cart fetched successfully")
                # TODO: Sync backend cart to local session if needed
        
        # Get cart display
        cart_display = cart_service.format_cart_display(session_obj)
        cart_summary = cart_service.get_cart_summary(session_obj)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
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


async def remove_from_cart(session_id: Optional[str], item_id: str, **kwargs) -> Dict[str, Any]:
    """MCP adapter for remove_from_cart"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="remove_from_cart", **kwargs)
        
        # Remove item using service
        success, message = await cart_service.remove_item(session_obj, item_id)
        
        # Sync with backend
        await cart_service.sync_with_backend(session_obj)
        
        # Get cart summary
        cart_summary = cart_service.get_cart_summary(session_obj)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            success,
            message,
            session_obj.session_id,
            cart_summary=cart_summary
        )
        
    except Exception as e:
        logger.error(f"Failed to remove item from cart: {e}")
        return format_mcp_response(
            False,
            f' Failed to remove item: {str(e)}',
            session_id or 'unknown'
        )


async def update_cart_quantity(session_id: Optional[str], item_id: str, 
                                  quantity: int, **kwargs) -> Dict[str, Any]:
    """MCP adapter for update_cart_quantity"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="update_cart_quantity", **kwargs)
        
        # Update quantity using service
        success, message = await cart_service.update_quantity(session_obj, item_id, quantity)
        
        # Sync with backend
        await cart_service.sync_with_backend(session_obj)
        
        # Get cart summary
        cart_summary = cart_service.get_cart_summary(session_obj)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            success,
            message,
            session_obj.session_id,
            cart_summary=cart_summary
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
        summary = cart_service.get_cart_summary(session_obj)
        
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