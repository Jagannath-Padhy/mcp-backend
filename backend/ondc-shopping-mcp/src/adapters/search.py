"""Search operations for MCP adapters"""

from typing import Dict, Any, Optional
from datetime import datetime
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services
)
from ..utils.logger import get_logger
from ..utils.field_mapper import enhance_for_mcp

logger = get_logger(__name__)

# Get services
services = get_services()
search_service = services['search_service']
session_service = services['session_service']


async def search_products(session_id: Optional[str] = None, query: str = '', 
                             latitude: Optional[float] = None,
                             longitude: Optional[float] = None,
                             page: int = 1, limit: int = 10, 
                             **kwargs) -> Dict[str, Any]:
    """MCP adapter for search_products"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="search_products", **kwargs)
        
        # Get session pincode if available
        session_pincode = None
        delivery_location = getattr(session_obj, 'delivery_location', None)
        if delivery_location and delivery_location.get('pincode'):
            session_pincode = delivery_location['pincode']
        
        # Perform search using service
        results = await search_service.search_products(
            query, latitude, longitude, page, limit, session_pincode
        )
        
        # Extract the actual items from the reranked search results
        products = []
        search_results = results.get('search_results', [])
        
        for result in search_results:
            item_data = None
            if isinstance(result, dict) and 'item' in result:
                # Handle nested item structure
                item_data = result['item']
            elif isinstance(result, dict):
                # Handle flat structure
                item_data = result
            
            if item_data:
                # Apply field mapping for MCP compatibility
                mcp_item = enhance_for_mcp(item_data)
                products.append(mcp_item)
        
        # Update session history with actual product results for conversation context
        session_obj.search_history.append({
            'query': query,
            'timestamp': datetime.utcnow().isoformat(),
            'results_count': len(results.get('search_results', [])),
            'products': products[:5] if products else []  # Store top 5 products for context
        })
        
        # Enhanced message with result summary
        if products:
            message = f" Found {len(products)} products for '{query}' ({results.get('search_type', 'hybrid')} search)"
        else:
            message = f"No products found for '{query}'. Try a different search term."
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            results.get('success', True),
            message,
            session_obj.session_id,
            products=products,  # Will be formatted by format_mcp_response
            total_results=results.get('total_results', 0),
            search_type=results.get('search_type', 'unknown'),
            page=results.get('page', 1),
            page_size=results.get('page_size', 10)
        )
        
    except Exception as e:
        logger.error(f"Failed to search products: {e}")
        return format_mcp_response(
            False,
            f' Failed to search products: {str(e)}',
            session_id or 'unknown'
        )


async def advanced_search(session_id: Optional[str] = None, query: Optional[str] = None,
                             category: Optional[str] = None,
                             brand: Optional[str] = None,
                             price_min: Optional[float] = None,
                             price_max: Optional[float] = None,
                             location: Optional[str] = None,
                             page: int = 1, limit: int = 10,
                             **kwargs) -> Dict[str, Any]:
    """MCP adapter for advanced_search"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="advanced_search", **kwargs)
        
        # Perform search using service
        results = await search_service.advanced_search(
            query, category, brand, price_min, price_max, 
            location, page, limit
        )
        
        # Extract the actual items from the reranked search results (same as search_products)
        products = []
        search_results = results.get('search_results', [])
        
        for result in search_results:
            if isinstance(result, dict) and 'item' in result:
                # Handle nested item structure
                products.append(result['item'])
            elif isinstance(result, dict):
                # Handle flat structure
                products.append(result)
        
        # Update session history with actual product results for conversation context
        session_obj.search_history.append({
            'query': query or f"Advanced search: {category or 'all categories'}",
            'timestamp': datetime.utcnow().isoformat(),
            'results_count': len(results.get('search_results', [])),
            'products': products[:5] if products else []  # Store top 5 products for context
        })

        # Enhanced message with result summary
        if products:
            filter_desc = f" with category: {category}" if category else ""
            message = f" Found {len(products)} products{filter_desc} ({results.get('search_type', 'filtered')} search)"
        else:
            message = f"No products found matching your filters. Try adjusting your search criteria."

        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)

        return format_mcp_response(
            results.get('success', True),
            message,
            session_obj.session_id,
            products=products,  # Will be formatted by simple formatter
            total_results=results.get('total_results', 0),
            search_type=results.get('search_type', 'filtered'),
            page=results.get('page', 1),
            page_size=results.get('page_size', 10)
        )
        
    except Exception as e:
        logger.error(f"Failed to advanced search: {e}")
        return format_mcp_response(
            False,
            f' Failed to search: {str(e)}',
            session_id or 'unknown'
        )


async def browse_categories(session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for browse_categories"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="browse_categories", **kwargs)
        
        # Get categories using service
        results = await search_service.browse_categories()
        
        # Debug logging
        logger.info(f"[MCP Adapter] browse_categories results keys: {list(results.keys())}")
        logger.info(f"[MCP Adapter] Number of categories: {len(results.get('categories', []))}")
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        # Format response
        response = format_mcp_response(
            True,
            results.get('message', 'Categories retrieved'),
            session_obj.session_id,
            categories=results.get('categories', [])
        )
        
        # Log what we're returning
        logger.info(f"[MCP Adapter] Response keys: {list(response.keys())}")
        if 'categories' in response:
            logger.info(f"[MCP Adapter] Returning {len(response['categories'])} categories")
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to browse categories: {e}")
        return format_mcp_response(
            False,
            f' Failed to browse categories: {str(e)}',
            session_id or 'unknown'
        )