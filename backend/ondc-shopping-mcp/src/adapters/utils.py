"""Shared utilities for MCP adapters"""

import sys
import os

# Ensure Python path is set for tool execution context
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from typing import Dict, Any, Optional, List
import logging
import json
import traceback
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_persistent_session(session_id: Optional[str] = None, tool_name: str = "unknown", **kwargs):
    """Simplified helper to get session directly from session service
    
    This function directly uses the session service without the over-engineered
    ConversationSessionManager layer. This ensures Langflow agent session IDs
    and MCP session IDs remain consistent and work universally as MCP.
    
    Args:
        session_id: Optional session ID from Langflow Agent (used directly)
        tool_name: Name of the MCP tool calling this function (for logging)
        **kwargs: Additional context (currently unused but kept for compatibility)
        
    Returns:
        tuple: (session_obj, None) - None for conversation_manager (removed)
    """
    # Log MCP tool call for debugging
    logger.info(f"[MCP Tool Call] {tool_name} - Session: {session_id[:16] if session_id else 'NEW'}")
    logger.debug(f"[MCP Tool Call] Full kwargs: {json.dumps(kwargs, default=str)}")
    from src.services.session_service import get_session_service
    session_service = get_session_service()
    
    if session_id:
        # Use explicit session ID from Langflow agent
        session_obj = session_service.get(session_id)
        if session_obj is None:
            # Create new session with the provided ID
            session_obj = session_service.create_with_id(session_id)
        logger.debug(f"[Session] Tool: {tool_name}, Using explicit session: {session_obj.session_id}")
    else:
        # Create new session if no session ID provided
        session_obj = session_service.get_or_create()
        logger.debug(f"[Session] Tool: {tool_name}, Created new session: {session_obj.session_id}")
    
    return session_obj, None


def save_persistent_session(session_obj, conversation_manager):
    """Simplified helper to save session directly to session service"""
    # conversation_manager is now None (removed), so save directly to session service
    from src.services.session_service import get_session_service
    session_service = get_session_service()
    session_service.update(session_obj)
    logger.debug(f"[Session] Saved session: {session_obj.session_id}")


def extract_session_id(session_param: Any) -> Optional[str]:
    """Extract session_id from MCP session parameter
    
    MCP sends session as either:
    - A dictionary with session_id key
    - A session_id string
    - None/empty
    """
    if session_param is None:
        return None
    
    if isinstance(session_param, str):
        return session_param
    
    if isinstance(session_param, dict):
        return session_param.get('session_id')
    
    return None


def format_mcp_response(success: bool, message: str, session_id: str, 
                       **extra_data) -> Dict[str, Any]:
    """Format response for MCP protocol
    
    MCP expects responses with:
    - success: bool
    - message: str
    - session: dict (for session persistence)
    - Additional data fields
    - Anti-caching flags to force fresh AI agent execution
    - _structured_data: Enhanced structured data for chat API extraction
    """
    import uuid
    from datetime import datetime
    
    response = {
        'success': success,
        'message': message,
        'session': {'session_id': session_id}  # MCP expects session dict
    }
    
    # Add anti-caching flags to bypass AI agent learned failures
    response.update({
        '_debug_force_execution': True,          # Force AI agent to call tools
        '_operation_id': str(uuid.uuid4()),     # Unique ID prevents response caching
        '_fresh_execution_requested': True,     # Explicit fresh execution flag
        '_timestamp': datetime.utcnow().isoformat(),  # Timestamp for uniqueness
        '_bypass_cache': True                   # Clear bypass instruction
    })
    
    # Format products for better display if present
    if 'products' in extra_data:
        extra_data['products'] = format_products_for_display(extra_data['products'])
    
    # Add any extra data
    response.update(extra_data)
    
    # 🚀 NEW: Add structured data for chat API extraction
    structured_data = _extract_structured_data(extra_data)
    if structured_data:
        response['_structured_data'] = structured_data
        response['_context_type'] = _determine_data_context(structured_data)
        response['_ui_hints'] = _generate_ui_hints(structured_data)
    
    # Log MCP response for debugging
    logger.debug(f"[MCP Response] Session: {session_id[:16]}... Success: {success}")
    if structured_data:
        logger.debug(f"[MCP Structured Data] Context: {response.get('_context_type')}, Keys: {list(structured_data.keys())}")
    if not success:
        logger.error(f"[MCP Error] {message}")
    
    return response


def format_products_for_display(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format products for better display in Desktop Client
    
    Simplifies complex nested structures and ensures products display properly.
    """
    formatted = []
    
    logger.debug(f"Formatting {len(products)} products for display")
    
    for product in products:
        try:
            # Handle both flat and nested structures
            logger.debug(f"Product keys: {list(product.keys())[:5]}")
            if 'item_details' in product:
                # Nested structure from BIAP backend
                item_details = product.get('item_details', {})
                descriptor = item_details.get('descriptor', {})
                price = item_details.get('price', {})
                provider_details = product.get('provider_details', {})
                location_details = product.get('location_details', [])
                
                # Handle price that might already be extracted as float
                price_value = 0
                if isinstance(price, (int, float)):
                    price_value = float(price)
                elif isinstance(price, dict):
                    price_value = float(price.get('value', 0))
                
                formatted_product = {
                    'id': item_details.get('id', ''),
                    'name': descriptor.get('name', 'Unknown Product') if isinstance(descriptor, dict) else product.get('name', 'Unknown Product'),
                    'description': (descriptor.get('short_desc', '') or descriptor.get('long_desc', '')) if isinstance(descriptor, dict) else product.get('description', ''),
                    'price': price_value,
                    'currency': price.get('currency', 'INR') if isinstance(price, dict) else product.get('currency', 'INR'),
                    'category': item_details.get('category_id', '') or product.get('category', ''),
                    'brand': descriptor.get('brand', '') if isinstance(descriptor, dict) else product.get('brand', ''),
                    'images': descriptor.get('images', []) if isinstance(descriptor, dict) else product.get('images', []),
                    'provider': {
                        'id': provider_details.get('id', '') if isinstance(provider_details, dict) else '',
                        'name': provider_details.get('descriptor', {}).get('name', '') if isinstance(provider_details, dict) and isinstance(provider_details.get('descriptor'), dict) else product.get('provider_name', ''),
                        'rating': provider_details.get('rating', 0) if isinstance(provider_details, dict) else 0
                    },
                    'availability': {
                        'in_stock': True,  # Default to available
                        'quantity': item_details.get('quantity', {}).get('available', {}).get('count', '0')
                    },
                    'fulfillment': product.get('fulfillment_details', []),
                    'location': location_details[0] if isinstance(location_details, list) and len(location_details) > 0 else (location_details if isinstance(location_details, dict) else {}),
                    '_raw': product  # Keep raw data for cart operations
                }
            else:
                # Already flat structure or from vector DB
                # Extract price value properly from dict structure
                price_value = 0
                if isinstance(product.get('price'), (int, float)):
                    price_value = float(product.get('price'))
                elif isinstance(product.get('price'), str):
                    try:
                        price_value = float(product.get('price'))
                    except:
                        price_value = 0
                elif isinstance(product.get('price'), dict):
                    price_value = float(product.get('price', {}).get('value', 0))
                
                formatted_product = {
                    'id': product.get('id', ''),
                    'name': product.get('name', 'Unknown Product'),
                    'description': product.get('description', ''),
                    'price': price_value,  # Use extracted price value
                    'currency': product.get('currency', 'INR'),
                    'category': product.get('category', '') if isinstance(product.get('category'), str) else product.get('category', {}).get('name', ''),
                    'brand': product.get('brand', ''),
                    'images': product.get('images', []),
                    'provider': product.get('provider', {}) if isinstance(product.get('provider'), dict) else {'name': product.get('provider', ''), 'id': ''},
                    'availability': product.get('availability', {'in_stock': True}),
                    'fulfillment': product.get('fulfillment', []),
                    'location': product.get('location', {}),
                    '_raw': product  # Keep raw data for cart operations
                }
            
            # Clean up images format for display (unindented to run for both branches)
            if formatted_product['images']:
                if isinstance(formatted_product['images'][0], str):
                    # Simple string URLs
                    formatted_product['image_url'] = formatted_product['images'][0]
                elif isinstance(formatted_product['images'][0], dict):
                    # Complex image objects
                    formatted_product['image_url'] = formatted_product['images'][0].get('url', '')
                else:
                    formatted_product['image_url'] = ''
            else:
                formatted_product['image_url'] = ''
            
            # Create a display-friendly string representation (unindented to run for both branches)
            price_str = f"₹{formatted_product['price']:.2f}" if formatted_product['price'] else "Price not available"
            provider_str = formatted_product['provider'].get('name', '') if isinstance(formatted_product['provider'], dict) else ''
            
            logger.debug(f"Creating display_text for {formatted_product.get('name', 'Unknown')}")
            formatted_product['display_text'] = (
                f"{formatted_product['name']}\n"
                + (f"{formatted_product['description'][:100]}...\n" if formatted_product['description'] else "")
                + f"Price: {price_str}\n"
                + f"Category: {formatted_product['category']}\n"
                + (f"Provider: {provider_str}" if provider_str else "")
            ).strip()
            
            formatted.append(formatted_product)
            
        except Exception as e:
            logger.warning(f"Failed to format product: {e}")
            # Use traceback from module level
            logger.warning(f"Traceback: {traceback.format_exc()}")
            # Return original product if formatting fails
            formatted.append(product)
    
    return formatted


# Get singleton service instances (shared across all adapters)
def get_services():
    """Get all service instances in one place to avoid duplicate imports"""
    from src.services.session_service import get_session_service
    from src.services.cart_service import get_cart_service
    from src.services.checkout_service import get_checkout_service
    from src.services.search_service import get_search_service
    from src.services.user_service import get_user_service
    from src.services.order_service import get_order_service
    from src.services.payment_service import get_payment_service
    
    return {
        'session_service': get_session_service(),
        'cart_service': get_cart_service(),
        'checkout_service': get_checkout_service(),
        'search_service': get_search_service(),
        'user_service': get_user_service(),
        'order_service': get_order_service(),
        'payment_service': get_payment_service()
    }


# 🚀 NEW: Enhanced structured data extraction for chat API

def _extract_structured_data(extra_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract and organize structured data for frontend consumption
    
    Args:
        extra_data: Raw MCP tool response data
        
    Returns:
        Structured data dictionary ready for frontend or None
    """
    if not extra_data:
        return None
    
    structured = {}
    
    # Extract products (search results)
    if 'products' in extra_data:
        products = extra_data['products']
        if products and len(products) > 0:
            structured['products'] = products  # Already formatted by format_products_for_display
            structured['total_results'] = len(products)
            
            # Extract search metadata if present
            if 'search_query' in extra_data:
                structured['search_query'] = extra_data['search_query']
            if 'search_metadata' in extra_data:
                structured.update(extra_data['search_metadata'])
    
    # Extract cart data
    if 'cart' in extra_data or 'cart_summary' in extra_data:
        cart_data = extra_data.get('cart') or extra_data.get('cart_summary')
        if cart_data:
            structured['cart'] = cart_data
            # Extract cart metadata
            for key in ['total', 'total_amount', 'item_count', 'items']:
                if key in extra_data:
                    structured[key] = extra_data[key]
    
    # Extract single item (add to cart result)
    if 'item_added' in extra_data or 'product' in extra_data:
        item_data = extra_data.get('item_added') or extra_data.get('product')
        if item_data:
            structured['item'] = item_data
            # Include quantity if present
            if 'quantity' in extra_data:
                structured['quantity'] = extra_data['quantity']
    
    # Extract order data
    if 'order_id' in extra_data or 'order_details' in extra_data:
        structured['order'] = {}
        for key in ['order_id', 'order_details', 'tracking_info', 'status']:
            if key in extra_data:
                structured['order'][key] = extra_data[key]
    
    # Extract checkout/delivery data
    if any(key in extra_data for key in ['quote_data', 'delivery', 'quotes', 'delivery_options']):
        structured['checkout'] = {}
        for key in ['quote_data', 'delivery', 'quotes', 'delivery_options', 'selected_quote']:
            if key in extra_data:
                structured['checkout'][key] = extra_data[key]
    
    # Extract payment data
    if any(key in extra_data for key in ['payment_id', 'payment_status', 'transaction_id']):
        structured['payment'] = {}
        for key in ['payment_id', 'payment_status', 'transaction_id', 'amount']:
            if key in extra_data:
                structured['payment'][key] = extra_data[key]
    
    # Extract error information
    if 'error' in extra_data or 'errors' in extra_data:
        structured['error'] = extra_data.get('error') or extra_data.get('errors')
    
    logger.debug(f"[Structured Data] Extracted keys: {list(structured.keys())}")
    return structured if structured else None


def _determine_data_context(structured_data: Dict[str, Any]) -> str:
    """Determine the context type based on structured data
    
    Args:
        structured_data: Structured data dictionary
        
    Returns:
        Context type string for frontend routing
    """
    if not structured_data:
        return 'message'
    
    # Product search results
    if 'products' in structured_data:
        product_count = structured_data.get('total_results', 0)
        if product_count > 0:
            return 'search_results'
        else:
            return 'no_results'
    
    # Cart operations
    if 'cart' in structured_data:
        return 'cart_view'
    
    # Single item added
    if 'item' in structured_data:
        return 'cart_updated'
    
    # Order operations
    if 'order' in structured_data:
        if 'order_id' in structured_data['order']:
            return 'order_confirmed'
        else:
            return 'order_details'
    
    # Checkout flow
    if 'checkout' in structured_data:
        if 'quotes' in structured_data['checkout'] or 'delivery_options' in structured_data['checkout']:
            return 'checkout_quotes'
        else:
            return 'checkout_flow'
    
    # Payment operations
    if 'payment' in structured_data:
        return 'payment_status'
    
    # Error states
    if 'error' in structured_data:
        return 'error_state'
    
    # Default fallback
    return 'data_response'


def _generate_ui_hints(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate UI hints for frontend rendering
    
    Args:
        structured_data: Structured data dictionary
        
    Returns:
        UI hints dictionary with rendering guidance
    """
    hints = {
        'render_as': 'default',
        'enable_actions': [],
        'suggested_actions': [],
        'display_priority': 'normal'
    }
    
    # Product search results
    if 'products' in structured_data:
        product_count = structured_data.get('total_results', 0)
        if product_count > 0:
            hints.update({
                'render_as': 'product_grid',
                'enable_actions': ['add_to_cart', 'view_details', 'compare'],
                'suggested_actions': ['Would you like to add any of these to your cart?'],
                'display_priority': 'high'
            })
        else:
            hints.update({
                'render_as': 'empty_state',
                'suggested_actions': ['Try a different search term', 'Browse categories'],
                'display_priority': 'normal'
            })
    
    # Cart view
    elif 'cart' in structured_data:
        hints.update({
            'render_as': 'cart_summary',
            'enable_actions': ['update_quantity', 'remove_item', 'checkout'],
            'suggested_actions': ['Ready to checkout?', 'Continue shopping'],
            'display_priority': 'high'
        })
    
    # Item added to cart
    elif 'item' in structured_data:
        hints.update({
            'render_as': 'success_notification',
            'enable_actions': ['view_cart', 'checkout', 'continue_shopping'],
            'suggested_actions': ['View your cart', 'Continue shopping'],
            'display_priority': 'medium'
        })
    
    # Checkout flow
    elif 'checkout' in structured_data:
        if 'quotes' in structured_data['checkout']:
            hints.update({
                'render_as': 'delivery_options',
                'enable_actions': ['select_delivery', 'change_address'],
                'suggested_actions': ['Please select a delivery option'],
                'display_priority': 'high'
            })
        else:
            hints.update({
                'render_as': 'checkout_form',
                'enable_actions': ['confirm_order', 'edit_details'],
                'suggested_actions': ['Review and confirm your order'],
                'display_priority': 'high'
            })
    
    # Order confirmed
    elif 'order' in structured_data and 'order_id' in structured_data['order']:
        hints.update({
            'render_as': 'order_confirmation',
            'enable_actions': ['track_order', 'download_receipt'],
            'suggested_actions': ['Track your order', 'Continue shopping'],
            'display_priority': 'high'
        })
    
    # Error states
    elif 'error' in structured_data:
        hints.update({
            'render_as': 'error_message',
            'enable_actions': ['retry', 'help'],
            'suggested_actions': ['Please try again or contact support'],
            'display_priority': 'high'
        })
    
    logger.debug(f"[UI Hints] Generated hints: {hints}")
    return hints