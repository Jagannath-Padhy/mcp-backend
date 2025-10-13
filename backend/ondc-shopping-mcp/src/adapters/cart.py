"""Cart operations for MCP adapters"""

from typing import Dict, Any, Optional
import json
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services,
    send_raw_data_to_frontend
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
        
        # Get raw backend cart data for SSE streaming (after successful add)
        raw_backend_data = None
        logger.debug(f"[Cart Debug] Starting backend data capture. Success: {success}")
        if success:
            try:
                user_id = session_obj.user_id or "guestUser"
                device_id = getattr(session_obj, 'device_id', 'device_9bca8c59')
                logger.debug(f"[Cart Debug] About to call get_cart with user_id: {user_id}, device_id: {device_id}")
                
                # First try backend cart for authenticated users
                if session_obj.user_authenticated and session_obj.user_id:
                    logger.debug(f"[Cart Debug] User authenticated, fetching from backend")
                    backend_result = await cart_service.buyer_app.get_cart(user_id, device_id)
                    logger.debug(f"[Cart Debug] Backend call completed. Result type: {type(backend_result)}, Result: {backend_result}")
                    if backend_result:
                        raw_backend_data = backend_result
                        logger.info(f"[Cart] Raw backend data captured from backend: {len(raw_backend_data) if isinstance(raw_backend_data, list) else 'dict'}")
                    else:
                        logger.warning(f"[Cart Debug] Backend result is empty for authenticated user: {backend_result}")
                
                # For guest users or if backend is empty, use local cart data
                if not raw_backend_data and session_obj.cart and session_obj.cart.items:
                    logger.debug(f"[Cart Debug] Using local cart data for SSE transmission")
                    # Convert local cart items to dictionary format for SSE
                    raw_backend_data = []
                    for item in session_obj.cart.items:
                        if hasattr(item, 'to_dict'):
                            cart_item_dict = item.to_dict()
                            # Add BIAP-compatible structure
                            cart_item_dict['biap_format'] = True
                            cart_item_dict['source'] = 'local_session'
                            raw_backend_data.append(cart_item_dict)
                    logger.info(f"[Cart] Raw cart data captured from local session: {len(raw_backend_data)} items")
                
            except Exception as e:
                logger.error(f"[Cart] Failed to capture raw backend data after add: {e}", exc_info=True)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        # Send raw cart data to frontend via SSE (Universal Pattern)
        if raw_backend_data:
            raw_data_for_sse = {
                'cart_items': raw_backend_data,
                'cart_summary': cart_summary,
                'biap_specifications': True
            }
            send_raw_data_to_frontend(session_obj.session_id, 'add_to_cart', raw_data_for_sse)
        
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
        
        
        # SYNC: Fetch from backend and sync to local session.cart (the missing piece!)
        logger.info(f"[Cart] Syncing backend cart to local session for display")
        sync_success = await cart_service.sync_backend_to_local_cart(session_obj)
        
        if not sync_success:
            logger.warning(f"[Cart] Backend sync failed, proceeding with local cart display")
        
        # Get cart display using the original working service method
        cart_display = cart_service.format_cart_display(session_obj)
        cart_summary = cart_service.get_cart_summary(session_obj)
        
        # Get raw backend cart data for SSE streaming
        raw_backend_data = None
        try:
            user_id = session_obj.user_id or "guestUser"
            device_id = getattr(session_obj, 'device_id', 'device_9bca8c59')
            backend_result = await cart_service.buyer_app.get_cart(user_id, device_id)
            if backend_result:
                raw_backend_data = backend_result
                logger.info(f"[Cart] Raw backend data captured for SSE: {len(raw_backend_data) if isinstance(raw_backend_data, list) else 'dict'}")
        except Exception as e:
            logger.warning(f"[Cart] Failed to capture raw backend data: {e}")
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        # Send raw cart data to frontend via SSE (Universal Pattern)
        if raw_backend_data:
            raw_data_for_sse = {
                'cart_items': raw_backend_data,
                'cart_summary': cart_summary,
                'biap_specifications': True
            }
            send_raw_data_to_frontend(session_obj.session_id, 'view_cart', raw_data_for_sse)
        
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
    """MCP adapter for clear_cart - Uses real-time backend data"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="clear_cart", **kwargs)
        logger.info(f"[Cart] Clear cart - Session ID: {session_obj.session_id}")
        
        
        # Step 1: Get real current cart data from backend first
        logger.info(f"[Cart] Step 1: Getting real cart data from backend...")
        try:
            user_id = session_obj.user_id or "guestUser"
            device_id = getattr(session_obj, 'device_id', 'device_9bca8c59')
            backend_cart_data = await cart_service.buyer_app.get_cart(user_id, device_id)
            logger.info(f"[Cart] Backend cart data retrieved: {len(backend_cart_data) if backend_cart_data else 0} items")
        except Exception as e:
            logger.warning(f"[Cart] Failed to get backend cart data: {e}")
            backend_cart_data = None
        
        # Step 2: Use direct backend clear cart API instead of individual item removal
        if backend_cart_data and isinstance(backend_cart_data, list) and len(backend_cart_data) > 0:
            logger.info(f"[Cart] Step 2: Found {len(backend_cart_data)} items to clear using direct backend API")
            user_id = session_obj.user_id or "guestUser"
            device_id = getattr(session_obj, 'device_id', 'device_9bca8c59')
            
            # Use direct backend clear cart API
            logger.info(f"[Cart] Step 3: Calling backend clear_cart API for user_id: {user_id}, device_id: {device_id}")
            backend_result = await cart_service.buyer_app.clear_cart(user_id, device_id)
            
            if backend_result is not None:
                success = True
                message = f" Cart cleared successfully - removed {len(backend_cart_data)} items"
                logger.info(f"[Cart] Backend clear cart successful: {backend_result}")
                
                # Clear local session cart to match backend
                from ..models.session import Cart
                session_obj.cart = Cart()
                
            else:
                success = False
                message = " Failed to clear cart from backend"
                logger.error(f"[Cart] Backend clear cart failed - returned None")
        else:
            # Cart is already empty
            success = True
            message = " Your cart is already empty"
            logger.info(f"[Cart] Cart already empty, no items to clear")
        
        # Step 4: Refresh cart view and get real empty cart data
        logger.info(f"[Cart] Step 4: Refreshing cart view to show real empty state...")
        cart_view_result = await view_cart(session_id, **kwargs)
        
        # Get the final cart summary from the view result
        final_cart_summary = cart_view_result.get('cart', {})
        
        # Send empty cart raw data to frontend via SSE
        try:
            raw_data_for_sse = {
                'cart_items': [],  # Empty cart
                'cart_summary': final_cart_summary,
                'biap_specifications': True,
                'operation': 'clear_cart_complete'
            }
            send_raw_data_to_frontend(session_obj.session_id, 'clear_cart', raw_data_for_sse)
            logger.info(f"[Cart] Empty cart data sent to frontend via SSE")
        except Exception as e:
            logger.warning(f"[Cart] Failed to send empty cart data to frontend: {e}")
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            success,
            message + "\n✅ Cart cleared and refreshed with real backend data",
            session_obj.session_id,
            cart=final_cart_summary
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