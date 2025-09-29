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


def format_products_for_ai(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create minimal product data for AI context (no bloat)
    
    Extracts only essential fields needed for AI decision-making:
    - Product identification (name, id)
    - Pricing information (price, currency)
    - Basic categorization (provider name, category)
    
    Removes all bloat: images, descriptions, fulfillment details, ONDC schemas, etc.
    
    Args:
        products: Full product data from search results
        
    Returns:
        Minimal product list optimized for AI processing
    """
    logger.debug(f"[Products AI] Creating minimal product data for {len(products)} products")
    
    minimal_products = []
    
    for product in products:
        try:
            # Extract core identification
            product_id = product.get('id', '')
            name = product.get('name', 'Unknown Product')
            
            # Extract price (handle various formats)
            price_value = 0.0
            if isinstance(product.get('price'), (int, float)):
                price_value = float(product.get('price'))
            elif isinstance(product.get('price'), dict):
                price_value = float(product.get('price', {}).get('value', 0))
            
            # Extract provider name only (not full object)
            provider_name = ''
            if isinstance(product.get('provider'), dict):
                provider_name = product.get('provider', {}).get('name', '')
            elif isinstance(product.get('provider'), str):
                provider_name = product.get('provider', '')
            
            # Extract category name only
            category = ''
            if isinstance(product.get('category'), str):
                category = product.get('category', '')
            elif isinstance(product.get('category'), dict):
                category = product.get('category', {}).get('name', '')
            
            # ✅ NEW: Extract stock availability from ONDC schema
            stock_quantity = 0
            in_stock = False
            try:
                # Parse ONDC stock data: item_details.quantity.available.count
                item_details = product.get('item_details', {})
                quantity = item_details.get('quantity', {})
                available = quantity.get('available', {})
                stock_quantity = int(available.get('count', 0))
                in_stock = stock_quantity > 0
            except (ValueError, TypeError, AttributeError):
                # Fallback: assume in stock if no clear availability data
                stock_quantity = 1
                in_stock = True
            
            minimal_product = {
                'id': product_id,
                'name': name,
                'price': price_value,
                'currency': product.get('currency', 'INR'),
                'provider': provider_name,
                'category': category,
                'stock_quantity': stock_quantity,  # NEW: For AI decision making
                'in_stock': in_stock              # NEW: Simple boolean for AI
            }
            
            minimal_products.append(minimal_product)
            logger.debug(f"[Products AI] Minimal: {name} - ₹{price_value} from {provider_name}")
            
        except Exception as e:
            logger.warning(f"[Products AI] Failed to create minimal product: {e}")
            # Skip problematic products rather than including bloat
            continue
    
    # ✅ CRITICAL DEBUG: Log token count for debugging massive overflow
    import json
    ai_data_str = json.dumps(minimal_products, ensure_ascii=False)
    token_count = len(ai_data_str.split())  # Rough token approximation
    
    logger.info(f"[Products AI] Created minimal data for {len(minimal_products)} products")
    logger.error(f"[TOKEN DEBUG] AI products data size: {len(ai_data_str)} characters, ~{token_count} tokens")
    
    if token_count > 10000:  # If still too large
        logger.error(f"[TOKEN DEBUG] ⚠️  AI data still too large! Truncating to first 5 products")
        minimal_products = minimal_products[:5]  # Emergency truncation
        
    return minimal_products


# ✅ NEW: Frontend data extraction functions for complete product info

def extract_price_value(product: Dict[str, Any]) -> float:
    """Extract price value from various ONDC price formats"""
    try:
        if isinstance(product.get('price'), (int, float)):
            return float(product.get('price'))
        elif isinstance(product.get('price'), dict):
            return float(product.get('price', {}).get('value', 0))
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


def extract_product_images(product: Dict[str, Any]) -> List[str]:
    """Extract product images from ONDC schema for frontend"""
    images = []
    try:
        # Primary image from descriptor.symbol
        if product.get('descriptor', {}).get('symbol'):
            images.append(product['descriptor']['symbol'])
            
        # Additional images from descriptor.images array
        if product.get('descriptor', {}).get('images'):
            for img in product['descriptor']['images']:
                if isinstance(img, str):
                    images.append(img)
                elif isinstance(img, dict) and img.get('url'):
                    images.append(img['url'])
                    
        # Fallback from item_details
        item_details = product.get('item_details', {})
        if item_details.get('descriptor', {}).get('symbol'):
            images.append(item_details['descriptor']['symbol'])
            
    except (TypeError, AttributeError) as e:
        logger.warning(f"[Frontend] Failed to extract images: {e}")
        
    return list(set(images))  # Remove duplicates


def extract_product_description(product: Dict[str, Any]) -> Dict[str, str]:
    """Extract product descriptions from ONDC schema for frontend"""
    try:
        descriptor = product.get('descriptor', {}) or product.get('item_details', {}).get('descriptor', {})
        
        return {
            'short_desc': descriptor.get('short_desc', ''),
            'long_desc': descriptor.get('long_desc', ''),
            'name': descriptor.get('name', product.get('name', '')),
            'code': descriptor.get('code', ''),
            'symbol': descriptor.get('symbol', '')
        }
    except (TypeError, AttributeError):
        return {'short_desc': '', 'long_desc': '', 'name': product.get('name', ''), 'code': '', 'symbol': ''}


def extract_product_rating(product: Dict[str, Any]) -> Dict[str, Any]:
    """Extract product ratings from ONDC schema for frontend"""
    try:
        # Look for rating in tags or rateable fields
        rating_data = {'rating': 0.0, 'review_count': 0, 'rating_breakdown': {}}
        
        # Check tags for rating info
        tags = product.get('tags', []) or product.get('item_details', {}).get('tags', [])
        for tag in tags:
            if isinstance(tag, dict):
                if tag.get('code') == 'rating':
                    rating_list = tag.get('list', [])
                    for item in rating_list:
                        if item.get('code') == 'rating':
                            rating_data['rating'] = float(item.get('value', 0))
                        elif item.get('code') == 'count':
                            rating_data['review_count'] = int(item.get('value', 0))
                            
        return rating_data
    except (TypeError, AttributeError, ValueError):
        return {'rating': 0.0, 'review_count': 0, 'rating_breakdown': {}}


def extract_product_specifications(product: Dict[str, Any]) -> Dict[str, Any]:
    """Extract product specifications from ONDC schema for frontend"""
    try:
        specifications = {}
        
        # Extract from tags
        tags = product.get('tags', []) or product.get('item_details', {}).get('tags', [])
        for tag in tags:
            if isinstance(tag, dict) and tag.get('list'):
                tag_code = tag.get('code', 'general')
                tag_specs = {}
                for item in tag.get('list', []):
                    if isinstance(item, dict):
                        key = item.get('code', 'property')
                        value = item.get('value', '')
                        tag_specs[key] = value
                if tag_specs:
                    specifications[tag_code] = tag_specs
                    
        return specifications
    except (TypeError, AttributeError):
        return {}


def extract_complete_stock_info(product: Dict[str, Any]) -> Dict[str, Any]:
    """Extract complete stock information for frontend"""
    try:
        item_details = product.get('item_details', {})
        quantity = item_details.get('quantity', {})
        available = quantity.get('available', {})
        
        return {
            'stock_quantity': int(available.get('count', 0)),
            'unit_of_measure': available.get('measure', {}).get('unit', 'piece'),
            'availability_status': 'available' if int(available.get('count', 0)) > 0 else 'out_of_stock',
            'maximum': quantity.get('maximum', {}).get('count', 999),
            'minimum': quantity.get('minimum', {}).get('count', 1)
        }
    except (TypeError, AttributeError, ValueError):
        return {
            'stock_quantity': 1,
            'unit_of_measure': 'piece', 
            'availability_status': 'available',
            'maximum': 999,
            'minimum': 1
        }


def extract_provider_details(product: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive provider details for frontend"""
    try:
        provider = product.get('provider', {})
        
        return {
            'id': provider.get('id', ''),
            'local_id': provider.get('local_id', ''),
            'name': provider.get('name', 'Unknown Provider'),
            'description': provider.get('descriptor', {}).get('long_desc', ''),
            'locations': provider.get('locations', []),
            'rating': provider.get('rating', 0.0),
            'categories': provider.get('categories', [])
        }
    except (TypeError, AttributeError):
        return {
            'id': '',
            'local_id': '', 
            'name': 'Unknown Provider',
            'description': '',
            'locations': [],
            'rating': 0.0,
            'categories': []
        }


def create_enhanced_product_for_frontend(product: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive product data for frontend rendering"""
    try:
        return {
            # Basic product info (same as AI)
            'id': product.get('id', ''),
            'name': product.get('name', ''),
            'price': extract_price_value(product),
            'currency': product.get('currency', 'INR'),
            'category': product.get('category', ''),
            
            # Rich frontend data
            'images': extract_product_images(product),
            'description': extract_product_description(product),
            'rating': extract_product_rating(product),
            'specifications': extract_product_specifications(product),
            'stock': extract_complete_stock_info(product),
            'provider': extract_provider_details(product),
            
            # ONDC metadata for cart operations
            'ondc_data': {
                'bpp_id': product.get('bpp_id', ''),
                'bpp_uri': product.get('bpp_uri', ''),
                'provider_id': product.get('provider', {}).get('id', ''),
                'fulfillment_id': product.get('fulfillment_id', 'Fulfillment1'),
                'location_id': product.get('location_id', ''),
                'context_city': product.get('context_city', ''),
                'tags': product.get('tags', [])
            }
        }
    except Exception as e:
        logger.error(f"[Frontend] Failed to create enhanced product: {e}")
        # Return basic fallback
        return {
            'id': product.get('id', ''),
            'name': product.get('name', 'Unknown Product'),
            'price': 0.0,
            'currency': 'INR',
            'category': '',
            'images': [],
            'description': {'short_desc': '', 'long_desc': '', 'name': product.get('name', '')},
            'rating': {'rating': 0.0, 'review_count': 0},
            'specifications': {},
            'stock': {'stock_quantity': 1, 'availability_status': 'available'},
            'provider': {'name': 'Unknown Provider'},
            'ondc_data': {}
        }


def format_cart_for_ai(cart_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify cart data for AI processing while preserving order journey context
    
    Reduces complex ONDC metadata while keeping essential fields needed for:
    - SELECT operations (provider_id, location_id, fulfillment_id)
    - Order journey continuity (totals, quantities, basic item info)
    - AI context understanding (readable names, prices)
    
    Args:
        cart_summary: Full cart summary from cart_service
        
    Returns:
        Simplified cart data optimized for AI processing
    """
    logger.debug(f"[Cart AI] Simplifying cart data for AI processing")
    
    if not cart_summary or cart_summary.get('is_empty', True):
        return {
            'items': [],
            'total_items': 0,
            'total_value': 0.0,
            'is_empty': True,
            'ready_for_checkout': False
        }
    
    simplified_items = []
    cart_items = cart_summary.get('items', [])
    
    for item in cart_items:
        try:
            # Extract essential fields with fallbacks
            item_id = item.get('id', '')
            local_id = item.get('local_id', '')
            
            # Extract name from actual cart structure (item.item.product.descriptor.name)
            name = 'Unknown Product'
            if item.get('item', {}).get('product', {}).get('descriptor', {}).get('name'):
                name = item['item']['product']['descriptor']['name']
            elif item.get('item', {}).get('descriptor', {}).get('name'):
                name = item['item']['descriptor']['name']
            elif item.get('descriptor', {}).get('name'):
                name = item['descriptor']['name']
            elif item.get('name'):
                name = item['name']
            
            # Extract price from actual cart structure (item.item.product.price.value)
            price = 0.0
            if item.get('item', {}).get('product', {}).get('price', {}).get('value'):
                price = float(item['item']['product']['price']['value'])
            elif item.get('item', {}).get('price', {}).get('value'):
                price = float(item['item']['price']['value'])
            elif item.get('price'):
                if isinstance(item['price'], dict):
                    price = float(item['price'].get('value', 0))
                else:
                    price = float(item['price'])
            
            # Extract quantity
            quantity = item.get('count', 1)
            if isinstance(quantity, dict):
                quantity = quantity.get('count', 1)
            quantity = int(quantity)
            
            # Extract provider information from actual cart structure (CRITICAL for SELECT)
            provider_id = item.get('provider_id', '')
            provider_name = ''
            if item.get('item', {}).get('provider', {}).get('descriptor', {}).get('name'):
                provider_name = item['item']['provider']['descriptor']['name']
            elif item.get('provider', {}).get('descriptor', {}).get('name'):
                provider_name = item['provider']['descriptor']['name']
            elif item.get('provider', {}).get('name'):
                provider_name = item['provider']['name']
            
            # Extract location ID from actual cart structure (CRITICAL for SELECT)
            location_id = ''
            if item.get('item', {}).get('product', {}).get('location_id'):
                location_id = item['item']['product']['location_id']
            elif item.get('location', {}).get('id'):
                location_id = item['location']['id']
            
            # Extract fulfillment ID from actual cart structure (CRITICAL for SELECT)
            fulfillment_id = ''
            if item.get('item', {}).get('product', {}).get('fulfillment_id'):
                fulfillment_id = item['item']['product']['fulfillment_id']
            elif item.get('fulfillment_id'):
                fulfillment_id = item['fulfillment_id']
            
            # Extract category from actual cart structure
            category = ''
            if item.get('item', {}).get('product', {}).get('category_id'):
                category = item['item']['product']['category_id']
            elif item.get('category'):
                category = item['category']
            
            simplified_item = {
                'id': item_id,
                'local_id': local_id,
                'name': name,
                'quantity': quantity,
                'price': price,
                'total_price': price * quantity,
                'provider_id': provider_id,
                'provider_name': provider_name,
                'location_id': location_id,
                'fulfillment_id': fulfillment_id,
                'category': category,
                'currency': 'INR'
            }
            
            simplified_items.append(simplified_item)
            logger.debug(f"[Cart AI] Simplified item: {name} (qty: {quantity}, provider: {provider_name})")
            
        except Exception as e:
            logger.warning(f"[Cart AI] Failed to simplify cart item: {e}")
            # Continue with other items rather than failing completely
            continue
    
    # Calculate totals
    total_items = sum(item['quantity'] for item in simplified_items)
    total_value = sum(item['total_price'] for item in simplified_items)
    
    # Extract unique providers for SELECT grouping
    providers = list(set(item['provider_id'] for item in simplified_items if item['provider_id']))
    
    simplified_cart = {
        'items': simplified_items,
        'total_items': total_items,
        'total_value': round(total_value, 2),
        'is_empty': len(simplified_items) == 0,
        'ready_for_checkout': len(simplified_items) > 0 and all(
            item['provider_id'] and item['location_id'] for item in simplified_items
        ),
        'providers': providers,
        'provider_count': len(providers),
        'contains_required_fields': True  # Validation flag for order journey
    }
    
    logger.info(f"[Cart AI] Simplified cart: {total_items} items, ₹{total_value:.2f}, {len(providers)} providers")
    return simplified_cart


def format_quotes_for_ai(quotes_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify delivery quotes for AI processing while preserving order journey context
    
    Reduces complex ONDC quote metadata while keeping essential fields needed for:
    - Order journey continuity (delivery options, pricing)
    - AI context understanding (readable delivery info, costs)
    
    Args:
        quotes_data: Full quotes data from checkout service
        
    Returns:
        Simplified quotes data optimized for AI processing
    """
    logger.debug("[Quotes AI] Simplifying quotes data for AI processing")
    
    if not quotes_data or not quotes_data.get('quotes'):
        return {
            'delivery_options': [],
            'total_options': 0,
            'available': False,
            'message': 'No delivery options available'
        }
    
    simplified_options = []
    quotes = quotes_data.get('quotes', [])
    
    for quote in quotes:
        try:
            # Extract essential delivery information
            provider_name = 'Unknown Provider'
            if quote.get('provider', {}).get('descriptor', {}).get('name'):
                provider_name = quote['provider']['descriptor']['name']
            
            # Extract delivery cost and time
            delivery_cost = 0.0
            delivery_time = 'Standard delivery'
            
            if quote.get('price', {}).get('value'):
                delivery_cost = float(quote['price']['value'])
            
            if quote.get('fulfillments'):
                fulfillment = quote['fulfillments'][0] if quote['fulfillments'] else {}
                if fulfillment.get('end', {}).get('time', {}).get('range'):
                    delivery_time = 'Express delivery'
            
            simplified_option = {
                'provider': provider_name,
                'delivery_cost': delivery_cost,
                'delivery_time': delivery_time,
                'currency': quote.get('price', {}).get('currency', 'INR'),
                'available': True
            }
            
            simplified_options.append(simplified_option)
            logger.debug(f"[Quotes AI] Simplified option: {provider_name} - ₹{delivery_cost}")
            
        except Exception as e:
            logger.warning(f"[Quotes AI] Failed to simplify quote: {e}")
            continue
    
    simplified_quotes = {
        'delivery_options': simplified_options,
        'total_options': len(simplified_options),
        'available': len(simplified_options) > 0,
        'message': f'{len(simplified_options)} delivery options available' if simplified_options else 'No delivery options'
    }
    
    logger.info(f"[Quotes AI] Simplified quotes: {len(simplified_options)} options available")
    return simplified_quotes


def format_order_for_ai(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify order data for AI processing while preserving order journey context
    
    Reduces complex ONDC order metadata while keeping essential fields needed for:
    - Order confirmation and tracking
    - AI context understanding (order status, amounts)
    
    Args:
        order_data: Full order data from order service
        
    Returns:
        Simplified order data optimized for AI processing
    """
    logger.debug("[Order AI] Simplifying order data for AI processing")
    
    if not order_data:
        return {
            'order_id': None,
            'status': 'unknown',
            'total_amount': 0.0,
            'currency': 'INR',
            'ready_for_payment': False
        }
    
    try:
        order_id = order_data.get('order_id', order_data.get('id', 'Unknown'))
        status = order_data.get('status', order_data.get('state', 'pending'))
        
        # Extract total amount
        total_amount = 0.0
        if order_data.get('quote', {}).get('price', {}).get('value'):
            total_amount = float(order_data['quote']['price']['value'])
        elif order_data.get('billing', {}).get('total_price'):
            total_amount = float(order_data['billing']['total_price'])
        
        simplified_order = {
            'order_id': order_id,
            'status': status,
            'total_amount': total_amount,
            'currency': order_data.get('quote', {}).get('price', {}).get('currency', 'INR'),
            'ready_for_payment': status in ['created', 'confirmed', 'pending'],
            'customer_name': order_data.get('billing', {}).get('name', 'Customer'),
            'delivery_address': order_data.get('fulfillments', [{}])[0].get('end', {}).get('location', {}).get('address', {}).get('locality', 'Address on file')
        }
        
        logger.info(f"[Order AI] Simplified order: {order_id} - ₹{total_amount:.2f} - {status}")
        return simplified_order
        
    except Exception as e:
        logger.warning(f"[Order AI] Failed to simplify order: {e}")
        return {
            'order_id': 'Error',
            'status': 'error',
            'total_amount': 0.0,
            'currency': 'INR',
            'ready_for_payment': False
        }


def create_enhanced_response(success: bool, message: str, session_id: str, 
                           ai_data: Optional[Dict] = None,      # NEW: AI-specific minimal data
                           ui_data: Optional[Dict] = None,      # Frontend-specific full data  
                           operation_type: str = "",
                           journey_context: Optional[Dict] = None,
                           **kwargs) -> Dict[str, Any]:
    """DRY helper to create consistent enhanced responses for all adapters
    
    ✅ FIXED: Separates AI context data from frontend UI data to prevent token bloat
    
    Creates standardized response format with:
    - AI data: Minimal data for AI decision-making (names, prices, IDs only)
    - UI data: Complete data for frontend rendering (images, schemas, etc.)
    - Clear separation prevents massive tokens being sent to AI
    
    Args:
        success: Operation success status
        message: Enhanced message for AI
        session_id: Session identifier
        ai_data: Minimal data for AI context (optional)
        ui_data: Full data for frontend UI (optional)
        operation_type: Type of operation for frontend
        journey_context: Context for next journey steps (optional)
        **kwargs: Additional data for response
        
    Returns:
        Optimized response with separated AI/UI data streams
    """
    response_data = {}
    
    # Add minimal AI data directly to response root (AI context sees this)
    if ai_data:
        response_data.update(ai_data)
        logger.debug(f"[Response] AI data: {list(ai_data.keys())}")
    
    # Add full UI data under _ui_data key (frontend uses, AI ignores)
    if ui_data:
        frontend_data = {
            'operation_type': operation_type,
            'backend_success': success,
            **ui_data
        }
        response_data['_ui_data'] = frontend_data  # Underscore prefix = "not for AI"
        logger.debug(f"[Response] UI data: {list(ui_data.keys())}")
    
    # Add journey context for AI workflow decisions
    if journey_context:
        response_data['journey_context'] = journey_context
    
    # Add any additional kwargs for AI context
    response_data.update(kwargs)
    
    return format_mcp_response(success, message, session_id, **response_data)


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