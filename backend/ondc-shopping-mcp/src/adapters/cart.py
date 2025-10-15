"""Cart Operations for MCP Adapters with AI-Enhanced Functionality

Provides comprehensive cart management with:
- Intelligent item auto-detection from search history
- Real-time backend synchronization
- Context-aware cart operations
- Enhanced error handling and logging
- Universal SSE data transmission for frontend updates
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime
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
            
            # CRITICAL: Check and log authentication status for debugging
            logger.info(f"[Cart] Authentication check - success: {success}, user_authenticated: {session_obj.user_authenticated}, user_id: {session_obj.user_id}")
            
            # If local add succeeded and user is authenticated, sync with backend
            if success and session_obj.user_authenticated and session_obj.user_id:
                logger.info(f"[Cart] ✅ User authenticated, adding to backend cart")
                backend_success, backend_msg = await cart_service.add_item_to_backend(session_obj, item, quantity)
                logger.info(f"[Cart] Backend add result - Success: {backend_success}, Message: {backend_msg}")
                
                # Use backend result if available
                if not backend_success:
                    # Backend failed, but local succeeded - warn user
                    message = f" {message}\n(Note: Backend sync failed - {backend_msg})"
            else:
                logger.warning(f"[Cart] ❌ Skipping backend sync - Auth check failed")
            
        except Exception as e:
            logger.error(f"[Cart] Exception in cart_service.add_item: {e}")
            return format_mcp_response(
                False,
                f' Failed to add item to cart: {str(e)}',
                session_obj.session_id
            )
        
        # DRY: Use the same cart view service for consistent parsing after add operation
        if success and session_obj.user_authenticated and session_obj.user_id:
            try:
                logger.info(f"[Cart] 🔄 Using DRY service to get fresh cart state after add operation")
                
                # DRY: Reuse the same cart parsing logic
                cart_result = await cart_service.get_formatted_cart_view(session_obj)
                cart_summary = cart_result['cart_summary']
                
                # Store backend response in session for continuity
                if cart_result['raw_backend_data']:
                    if not hasattr(session_obj, 'backend_responses'):
                        session_obj.backend_responses = {}
                    
                    session_obj.backend_responses['latest_cart_state'] = cart_result['raw_backend_data']
                    session_obj.backend_responses['last_cart_operation'] = {
                        'operation': 'add_to_cart',
                        'timestamp': datetime.utcnow().isoformat(),
                        'raw_response': cart_result['raw_backend_data']
                    }
                
                # Update message with REAL backend data instead of assumption
                item_name = item.get('name', 'item')
                total_items = cart_summary.get('total_items', 0)
                total_value = cart_summary.get('total_value', 0)
                message = f"✅ Added {quantity}x {item_name} to cart\nCart total: {total_items} items - ₹{total_value:.2f}"
                
            except Exception as e:
                logger.error(f"[Cart] ❌ Failed to get fresh cart via DRY service: {e}")
                # Fallback to local cart summary only if backend fails
                cart_summary = cart_service.get_cart_summary(session_obj)
                cart_result = {'raw_backend_data': None, 'source': 'local_fallback'}
        else:
            # Fallback for unauthenticated users
            logger.info(f"[Cart] Using local cart summary (unauthenticated user)")
            cart_summary = cart_service.get_cart_summary(session_obj)
            cart_result = {'raw_backend_data': None, 'source': 'local_session'}
        
        # Use fresh backend data for SSE streaming (from DRY service)
        raw_backend_data = cart_result.get('raw_backend_data') if 'cart_result' in locals() else None
        
        if raw_backend_data:
            logger.info(f"[Cart] 📡 Using fresh backend data for SSE: {len(raw_backend_data)} items")
        else:
            logger.info(f"[Cart] 📡 No backend data available for SSE, using local fallback")
            # Fallback to local cart data only if no backend data available
            if session_obj.cart and session_obj.cart.items:
                raw_backend_data = []
                for item in session_obj.cart.items:
                    if hasattr(item, 'to_dict'):
                        cart_item_dict = item.to_dict()
                        cart_item_dict['biap_format'] = True
                        cart_item_dict['source'] = 'local_session'
                        raw_backend_data.append(cart_item_dict)
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        # Send fresh backend data to frontend via SSE (Universal Pattern)
        if raw_backend_data:
            raw_data_for_sse = {
                'cart_items': raw_backend_data,
                'cart_summary': cart_summary,
                'biap_specifications': True,
                'data_source': 'fresh_backend',  # Indicate this is fresh backend data
                'operation': 'add_to_cart',
                'timestamp': datetime.utcnow().isoformat()
            }
            send_raw_data_to_frontend(session_obj.session_id, 'add_to_cart', raw_data_for_sse)
            logger.info(f"[Cart] 📡 Fresh backend data sent to frontend via SSE")
        
        # Enhance message to encourage agent to refresh cart view
        if success:
            enhanced_message = f"{message}\n\n💡 Call view_cart to see the updated cart contents."
        else:
            enhanced_message = message
        
        return format_mcp_response(
            success,
            enhanced_message,
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
    """MCP adapter for view_cart - DRY architecture using reusable service"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="view_cart", **kwargs)
        logger.info(f"[Cart] View cart - Session ID: {session_obj.session_id}")
        
        # DRY: Use reusable cart view service with fixed parsing logic
        # Note: cart_service is already imported at module level
        cart_result = await cart_service.get_formatted_cart_view(session_obj)
        
        # Store backend response in session for continuity
        if cart_result['raw_backend_data']:
            if not hasattr(session_obj, 'backend_responses'):
                session_obj.backend_responses = {}
            
            session_obj.backend_responses['latest_cart_state'] = cart_result['raw_backend_data']
            session_obj.backend_responses['last_cart_operation'] = {
                'operation': 'view_cart',
                'timestamp': datetime.utcnow().isoformat(),
                'raw_response': cart_result['raw_backend_data']
            }
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        # Send fresh backend data to frontend via SSE (Universal Pattern)
        if cart_result['raw_backend_data']:
            raw_data_for_sse = {
                'cart_items': cart_result['raw_backend_data'],
                'cart_summary': cart_result['cart_summary'],
                'biap_specifications': True,
                'data_source': cart_result['source'],
                'operation': 'view_cart',
                'timestamp': datetime.utcnow().isoformat()
            }
            send_raw_data_to_frontend(session_obj.session_id, 'view_cart', raw_data_for_sse)
            logger.info(f"[Cart] 📡 Cart data sent to frontend via SSE (source: {cart_result['source']})")
        else:
            logger.info(f"[Cart] 📡 No backend data for SSE (source: {cart_result['source']})")
        
        return format_mcp_response(
            True,
            cart_result['cart_display'],
            session_obj.session_id,
            cart=cart_result['cart_summary']
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
        
        # Enhance message to encourage agent to refresh cart view
        if success:
            enhanced_message = f"{message}\n\n💡 Call view_cart to see the updated cart."
        else:
            enhanced_message = message
        
        return format_mcp_response(
            success,
            enhanced_message,
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
        
        # Enhance message to encourage agent to refresh cart view  
        if success:
            enhanced_message = f"{message}\n\n💡 Call view_cart to see the updated cart."
        else:
            enhanced_message = message
        
        return format_mcp_response(
            success,
            enhanced_message,
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
                message = f"✅ Cart cleared successfully - removed {len(backend_cart_data)} items"
                logger.info(f"[Cart] Backend clear cart successful: {backend_result}")
                
                # Clear local session cart to match backend
                from ..models.session import Cart
                session_obj.cart = Cart()
                
            else:
                success = False
                message = "❌ Failed to clear cart from backend"
                logger.error(f"[Cart] Backend clear cart failed - returned None")
        else:
            # Cart is already empty
            success = True
            message = "ℹ️ Your cart is already empty"
            logger.info(f"[Cart] Cart already empty, no items to clear")
        
        # Refresh cart view and get real empty cart data
        logger.info(f"[Cart] Refreshing cart view to show real empty state...")
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