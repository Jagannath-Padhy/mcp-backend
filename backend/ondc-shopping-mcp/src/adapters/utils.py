"""Shared utilities for MCP adapters"""

from typing import Dict, Any, Optional, List
import logging
import json
from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_persistent_session(session_id: Optional[str] = None, tool_name: str = "unknown", **kwargs):
    """Get or create session for MCP tool execution
    
    This function manages session persistence across MCP tool calls, ensuring
    cart data and user state is maintained throughout the shopping journey.
    
    Args:
        session_id: Session ID from MCP agent (if provided, used as-is)
        tool_name: Name of the MCP tool calling this function (for logging)
        **kwargs: Additional context including userId/deviceId for session properties
        
    Returns:
        tuple: (session_obj, None) - session object and None for compatibility
        
    Note:
        When the agent provides a session_id, it's used directly as the single
        source of truth. If no session_id is provided, a new session is created.
    """
    from ..services.session_service import get_session_service
    session_service = get_session_service()
    
    if session_id:
        logger.debug(f"[Session] {tool_name} using session_id: {session_id}")
        
        # Use agent session_id as-is (single source of truth)
        session_obj = session_service.get(session_id)
        
        if session_obj is None:
            # Create new session with agent's session_id
            session_obj = session_service.create_with_id(session_id)
            logger.info(f"[Session] Created new session: {session_obj.session_id}")
        else:
            # Log cart state for monitoring
            cart_items = len(session_obj.cart.items) if session_obj.cart else 0
            logger.debug(f"[Session] Retrieved session with {cart_items} cart items")
    else:
        # Create new session if agent doesn't provide one
        session_obj = session_service.create()
        logger.warning(f"[Session] No session_id provided to {tool_name}, created new: {session_obj.session_id}")
    
    # CRITICAL FIX: Apply userId/deviceId from kwargs to session if not set
    # This ensures all MCP tools can access backend APIs, not just initialize_shopping
    provided_user_id = kwargs.get('userId') or kwargs.get('user_id')
    provided_device_id = kwargs.get('deviceId') or kwargs.get('device_id')
    
    # CRITICAL FIX: Don't overwrite real authenticated credentials with guestUser defaults
    # Only apply kwargs if session doesn't have authenticated credentials or if kwargs contain real auth data
    
    # Handle user_id - protect against guestUser contamination
    if provided_user_id and provided_user_id != "guestUser":
        # Only apply real user IDs (not guestUser fallback)
        if not session_obj.user_id or session_obj.user_id == "guestUser":
            session_obj.user_id = provided_user_id
            logger.debug(f"[Session] Applied REAL userId from kwargs: {provided_user_id}")
    elif provided_user_id == "guestUser" and session_obj.user_id and session_obj.user_id != "guestUser":
        # Don't overwrite real credentials with guestUser
        logger.debug(f"[Session] PROTECTED stored userId {session_obj.user_id} from guestUser contamination")
    elif provided_user_id and not session_obj.user_id:
        # Only as last resort when session has no user_id
        session_obj.user_id = provided_user_id
        logger.debug(f"[Session] Applied fallback userId from kwargs: {provided_user_id}")
    
    # Handle device_id - protect against device_* contamination  
    if provided_device_id and not provided_device_id.startswith("device_"):
        # Only apply real device IDs (not generated device_* patterns)
        if not session_obj.device_id or session_obj.device_id.startswith("device_"):
            session_obj.device_id = provided_device_id
            logger.debug(f"[Session] Applied REAL deviceId from kwargs: {provided_device_id}")
    elif provided_device_id and provided_device_id.startswith("device_") and session_obj.device_id and not session_obj.device_id.startswith("device_"):
        # Don't overwrite real credentials with generated device_*
        logger.debug(f"[Session] PROTECTED stored deviceId {session_obj.device_id} from device_* contamination")
    elif provided_device_id and not session_obj.device_id:
        # Only as last resort when session has no device_id
        session_obj.device_id = provided_device_id
        logger.debug(f"[Session] Applied fallback deviceId from kwargs: {provided_device_id}")
    elif not provided_device_id and session_obj.device_id:
        # Preserve existing session device_id when kwargs don't provide one
        logger.debug(f"[Session] Preserving existing deviceId: {session_obj.device_id}")
    
    # CRITICAL: Mark session as authenticated for backend operations if we have both user_id and device_id
    if session_obj.user_id and session_obj.device_id and not session_obj.user_authenticated:
        session_obj.user_authenticated = True
        logger.info(f"[Session] {tool_name} - Enabled backend authentication for userId: {session_obj.user_id}, deviceId: {session_obj.device_id}")
    
    return session_obj, None


def save_persistent_session(session_obj, conversation_manager):
    """Save session data to persistent storage
    
    Args:
        session_obj: Session object to save
        conversation_manager: Legacy parameter (ignored for compatibility)
        
    Note:
        The conversation_manager parameter is retained for backward compatibility
        but is no longer used. Session persistence is handled directly by the
        session service.
    """
    from ..services.session_service import get_session_service
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
    
    # Send raw product data to agent instead of formatting
    # This preserves full BIAP specifications for agent analysis
    # if 'products' in extra_data:
    #     extra_data['products'] = format_products_for_display(extra_data['products'])
    
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
                    'location': location_details[0] if isinstance(location_details, list) and location_details else location_details if isinstance(location_details, dict) else {},
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
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            # Return original product if formatting fails
            formatted.append(product)
    
    return formatted


# Get singleton service instances (shared across all adapters)
def get_services():
    """Get all service instances in one place to avoid duplicate imports"""
    from ..services.session_service import get_session_service
    from ..services.cart_service import get_cart_service
    from ..services.checkout_service import get_checkout_service
    from ..services.search_service import get_search_service
    from ..services.user_service import get_user_service
    from ..services.order_service import get_order_service
    from ..services.payment_service import get_payment_service
    
    return {
        'session_service': get_session_service(),
        'cart_service': get_cart_service(),
        'checkout_service': get_checkout_service(),
        'search_service': get_search_service(),
        'user_service': get_user_service(),
        'order_service': get_order_service(),
        'payment_service': get_payment_service()
    }


def send_raw_data_to_frontend(session_id: str, tool_name: str, raw_data: Dict[str, Any]):
    """Universal helper to send raw data to frontend via SSE stream
    
    This function sends tool response data to the frontend via the internal API endpoint
    that manages SSE streams. Used for cart, orders, and other tools that need structured
    data transmission to frontend for rendering.
    
    Args:
        session_id: Session identifier for routing data to correct SSE stream
        tool_name: Name of the MCP tool (for event type mapping)
        raw_data: Structured data to send to frontend
    """
    try:
        import requests
        import os
        import time
        
        # Get configurable API settings from environment
        api_url = os.getenv('API_URL', 'http://localhost:8001')
        api_timeout = float(os.getenv('API_TIMEOUT', 2.0))
        retry_attempts = int(os.getenv('API_RETRY_ATTEMPTS', 2))
        
        # Prepare callback data for internal API endpoint
        callback_data = {
            'session_id': session_id,
            'tool_name': tool_name, 
            'raw_data': raw_data
        }
        
        # Send to internal endpoint with retry logic
        last_exception = None
        for attempt in range(retry_attempts):
            try:
                response = requests.post(f"{api_url}/internal/tool-result", 
                    json=callback_data, timeout=api_timeout)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                logger.info(f"[Universal SSE] Sent {tool_name} data for session {session_id} (attempt {attempt + 1})")
                return  # Success, exit the function
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < retry_attempts - 1:  # Not the last attempt
                    wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"[Universal SSE] Attempt {attempt + 1} failed for {tool_name}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[Universal SSE] All {retry_attempts} attempts failed for {tool_name}: {e}")
        
        # If we get here, all attempts failed
        raise last_exception
        
    except Exception as e:
        logger.warning(f"[Universal SSE] Failed to send {tool_name} data after {retry_attempts} attempts: {e}")