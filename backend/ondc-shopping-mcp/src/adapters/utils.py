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
    """
    response = {
        'success': success,
        'message': message,
        'session': {'session_id': session_id}  # MCP expects session dict
    }
    
    # Format products for better display if present
    if 'products' in extra_data:
        extra_data['products'] = format_products_for_display(extra_data['products'])
    
    # Add any extra data
    response.update(extra_data)
    
    # Log MCP response for debugging
    logger.debug(f"[MCP Response] Session: {session_id[:16]}... Success: {success}")
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