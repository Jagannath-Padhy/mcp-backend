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
    format_cart_for_ai,
    create_enhanced_response,
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

        # 🚀 INTELLIGENT WORKFLOW: Enable smart search-then-add behavior
        # If item is missing or incomplete, allow agent to auto-search from recent results
        intelligent_selection = False
        intelligent_message_prefix = ""
        
        # Check if item is incomplete (missing critical fields beyond just name)
        item_incomplete = (
            not item or 
            not isinstance(item, dict) or 
            not item.get('name') or
            not item.get('price') or 
            not item.get('id') and not item.get('local_id')
        )
        
        if item_incomplete:
            # Try intelligent matching based on partial name provided
            partial_name = item.get('name', '').lower() if item else ''
            logger.info(f"[Cart] Item incomplete - searching for match to '{partial_name}' in recent results")
            
            # Check if we have recent search results that can be used
            if session_obj.search_history:
                last_search = session_obj.search_history[-1]  # Get most recent search
                if last_search.get('products') and len(last_search['products']) > 0:
                    # Smart matching: find best match by name similarity
                    best_match = None
                    if partial_name:
                        # Find product that contains the partial name
                        for product in last_search['products']:
                            product_name = product.get('name', '').lower()
                            if partial_name in product_name or product_name in partial_name:
                                best_match = product
                                logger.info(f"[Cart] 🎯 INTELLIGENT MATCH: Found '{product.get('name')}' matching '{partial_name}'")
                                break
                    
                    # Fallback to first product if no name match found
                    if not best_match:
                        best_match = last_search['products'][0]
                        logger.info(f"[Cart] 🤖 INTELLIGENT FALLBACK: Using first product '{best_match.get('name')}' from recent search")
                    
                    # Apply field mapping from backend format
                    item = from_backend(best_match)
                    
                    # Mark as intelligent selection and prepare enhanced message
                    intelligent_selection = True
                    product_name = item.get('name', 'Unknown Product')
                    product_price = item.get('price', 0)
                    intelligent_message_prefix = f"🤖 Smart Selection: {product_name} (₹{product_price}) from recent search. "
                    
                else:
                    logger.info("[Cart] No suitable products in recent search history for auto-selection")
                    return format_mcp_response(
                        False,
                        ' No recent products found for auto-selection. Please search for products first, then add to cart.',
                        session_obj.session_id
                    )
            else:
                logger.info("[Cart] No search history available for intelligent auto-selection")
                return format_mcp_response(
                    False,
                    ' Please search for products first to enable smart cart additions.',
                    session_obj.session_id
                )
        
        # Validate essential product fields (after auto-detection)
        if not item.get('name'):
            logger.error("[Cart] Product still missing required 'name' field after auto-detection")
            return format_mcp_response(
                False,
                ' Invalid product data: missing product name even after auto-detection.',
                session_obj.session_id
            )
        
        # Validate product has essential identifiers (relaxed for intelligent behavior)
        if not item.get('id') and not item.get('local_id'):
            logger.warning(f"[Cart] Product {item.get('name')} missing ID fields - attempting to proceed with available data")
            # Don't fail immediately - let backend validation handle this

        # Enhanced debugging for validated item
        logger.info(f"[Cart] Validated item for cart: {json.dumps(item, indent=2)}")

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

        # ✅ FIXED: Clean separation - add_to_cart only adds, doesn't auto-display cart
        if success:
            # Enhanced message with intelligent selection info (no cart display)
            if intelligent_selection:
                enhanced_message = f"{intelligent_message_prefix}{message}"
            else:
                enhanced_message = message
                
            # Create minimal cart context for AI (just success indicator)
            cart_for_ai = {
                'operation_success': True,
                'item_added': True,
                'ready_for_checkout': False  # User should call view_cart to get real status
            }
        else:
            # Add failed - return error immediately with no fake data
            return format_mcp_response(
                False,
                message,  # Real error message from backend
                session_obj.session_id
            )

        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)

        # ✅ FIXED: Clean add operation without automatic cart display
        return create_enhanced_response(
            success, enhanced_message, session_obj.session_id,
            ai_data={'cart_context': cart_for_ai},      # Minimal success indicator for AI
            ui_data={'operation_success': success, 'item_added': True},  # Simple UI confirmation
            operation_type='add_to_cart',
            journey_context={
                'stage': 'item_added',
                'next_operations': ['view_cart', 'continue_shopping'],  # User chooses next action
                'suggestion': 'Call view_cart to see your cart contents'
            }
        )

    except Exception as e:
        logger.error(f"Failed to add item to cart: {e}")
        return format_mcp_response(
            False,
            f' Failed to add item to cart: {str(e)}',
            session_id or 'unknown'
        )


async def view_cart(session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for view_cart - simplified and clean"""
    logger.info(f"[Cart] 👁️  VIEW CART called with session_id={session_id}")
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="view_cart", **kwargs)
        logger.info(f"[Cart] View cart - Session ID: {session_obj.session_id}")

        # ✅ SIMPLIFIED: Direct cart service call (no redundant backend calls)
        logger.info(f"[Cart] Calling cart_service.view_cart for session {session_obj.session_id}")
        success, cart_display, cart_items = await cart_service.view_cart(session_obj)

        if not success:
            logger.error(f"[Cart] view_cart failed: {cart_display}")
            return format_mcp_response(
                False,
                cart_display,
                session_obj.session_id
            )

        logger.info(f"[Cart] ✅ view_cart successful with {len(cart_items)} items")

        # Get cart summary for AI processing
        cart_summary = await cart_service.get_cart_summary(session_obj)

        # Create simplified cart for AI processing
        cart_for_ai = format_cart_for_ai(cart_summary)

        # Enhanced display message with cart status
        enhanced_display = cart_display
        if cart_for_ai['total_items'] > 0:
            enhanced_display += (
                f"\n\n📊 Summary: {cart_for_ai['total_items']} items, "
                f"₹{cart_for_ai['total_value']:.2f} total"
            )
            if cart_for_ai['ready_for_checkout']:
                enhanced_display += "\n🚚 Ready for checkout with delivery quotes"

        # Save session with persistence
        save_persistent_session(session_obj, conversation_manager)

        # ✅ FIXED: Use separated AI/UI data to prevent token bloat
        return create_enhanced_response(
            True, enhanced_display, session_obj.session_id,
            ai_data={'cart_context': cart_for_ai},      # Minimal cart data for AI
            ui_data={'cart': cart_summary},             # Full ONDC data for frontend
            operation_type='view_cart',
            journey_context={
                'stage': 'cart_viewed',
                'next_operations': (
                    ['select_items_for_order', 'add_to_cart', 'update_cart_quantity']
                    if cart_for_ai['total_items'] > 0 else ['search_products']
                ),
                'ready_for_checkout': cart_for_ai['ready_for_checkout']
            }
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

        # Create simplified cart for AI processing
        cart_for_ai = format_cart_for_ai(cart_summary)

        # Enhanced message with updated cart context
        enhanced_message = f"{message}\n📦 Cart: {cart_for_ai['total_items']} items, ₹{cart_for_ai['total_value']:.2f}"

        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)

        return create_enhanced_response(
            success, enhanced_message, session_obj.session_id,
            ai_data={'cart_context': cart_for_ai},      # Minimal cart data for AI
            ui_data={'cart': cart_summary},             # Full ONDC data for frontend
            operation_type='remove_from_cart'
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
        session_obj, conversation_manager = get_persistent_session(
            session_id, tool_name="update_cart_quantity", **kwargs
        )

        # Update quantity using service
        success, message = await cart_service.update_quantity(session_obj, item_id, quantity)

        # Sync with backend
        await cart_service.sync_with_backend(session_obj)

        # Get cart summary
        cart_summary = await cart_service.get_cart_summary(session_obj)

        # Create simplified cart for AI processing
        cart_for_ai = format_cart_for_ai(cart_summary)

        # Enhanced message with updated cart context
        enhanced_message = f"{message}\n📦 Cart: {cart_for_ai['total_items']} items, ₹{cart_for_ai['total_value']:.2f}"

        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)

        return create_enhanced_response(
            success, enhanced_message, session_obj.session_id,
            ai_data={'cart_context': cart_for_ai},      # Minimal cart data for AI
            ui_data={'cart': cart_summary},             # Full ONDC data for frontend
            operation_type='update_cart_quantity'
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

        # Create simplified cart for AI processing
        cart_for_ai = format_cart_for_ai(summary)

        if summary['is_empty']:
            message = " Your cart is empty"
        else:
            message = f" Cart Total: {summary['total_items']} items - ₹{summary['total_value']:.2f}"
            if cart_for_ai['ready_for_checkout']:
                message += "\n🚚 Ready for checkout"

        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)

        return create_enhanced_response(
            True, message, session_obj.session_id,
            ai_data={'cart_context': cart_for_ai},      # Minimal cart data for AI
            ui_data={'cart': summary},                  # Full ONDC data for frontend
            operation_type='get_cart_total'
        )

    except Exception as e:
        logger.error(f"Failed to get cart total: {e}")
        return format_mcp_response(
            False,
            f' Failed to get cart total: {str(e)}',
            session_id or 'unknown'
        )
