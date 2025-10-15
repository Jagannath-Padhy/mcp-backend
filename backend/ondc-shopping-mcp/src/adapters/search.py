"""Search operations for MCP adapters - Agent-Driven Design"""

from typing import Dict, Any, Optional
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
from ..utils.field_mapper import enhance_for_mcp

logger = get_logger(__name__)

# Get services
services = get_services()
search_service = services['search_service']
session_service = services['session_service']


async def search_products(session_id: Optional[str] = None, query: str = '', 
                             latitude: Optional[float] = None,
                             longitude: Optional[float] = None,
                             page: int = 1, 
                             limit: int = 10,  # Agent decides appropriate limit
                             relevance_threshold: float = 0.7,  # Good default, agent can override
                             **kwargs) -> Dict[str, Any]:
    """MCP adapter for search_products - Agent-driven intelligent search
    
    Simple, flexible search that lets the MCP agent be intelligent about:
    - Search query formulation
    - Result limits based on user context
    - Relevance thresholds based on search precision needs
    - Follow-up search strategy
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="search_products", **kwargs)
        
        logger.info(f"[Search] Agent-driven search - Query: '{query}', Limit: {limit}, Threshold: {relevance_threshold}")
        
        # Get session pincode if available
        session_pincode = None
        delivery_location = getattr(session_obj, 'delivery_location', None)
        if delivery_location and delivery_location.get('pincode'):
            session_pincode = delivery_location['pincode']
        
        # Enhanced search with intelligent query processing
        results = await search_service.search_products(
            query, latitude, longitude, page, limit, session_pincode,
            relevance_threshold=relevance_threshold
        )
        
        # If no good results found, let agent try variations naturally
        search_results = results.get('search_results', [])
        if not search_results or len(search_results) < 2:
            # Agent will naturally suggest variations like "basmati rice" for "rice"
            logger.info(f"[Search] Limited results for '{query}' - agent will suggest variations")
        
        # Extract and format products
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
        
        # Update session history with product results for context
        session_obj.search_history.append({
            'query': query,
            'timestamp': datetime.utcnow().isoformat(),
            'results_count': len(results.get('search_results', [])),
            'products': products[:5] if products else [],  # Store top 5 for context
            'agent_params': {
                'limit': limit,
                'relevance_threshold': relevance_threshold,
                'search_type': results.get('search_type', 'hybrid')
            }
        })
        
        # Simple success message - let agent interpret results
        if products:
            message = f" Found {len(products)} products for '{query}'"
        else:
            message = f"No products found for '{query}'. Try different search terms or broader criteria."
        
        # Save session with enhanced persistence
        save_persistent_session(session_obj, conversation_manager)
        
        # Send raw product data to frontend via SSE stream
        if products:
            raw_data_for_sse = {
                'products': products,
                'total_results': results.get('total_results', 0),
                'search_type': results.get('search_type', 'hybrid'),
                'page': results.get('page', 1),
                'page_size': results.get('page_size', limit),
                'search_metadata': {
                    'original_query': query,
                    'limit_requested': limit,
                    'relevance_threshold': relevance_threshold,
                    'results_returned': len(products),
                    'search_timestamp': datetime.utcnow().isoformat(),
                    'session_id': session_obj.session_id
                }
            }
            
            send_raw_data_to_frontend(session_obj.session_id, 'search_products', raw_data_for_sse)
            logger.info(f"[Search] Sent {len(products)} products with query '{query}' to SSE stream")
        
        return format_mcp_response(
            results.get('success', True),
            message,
            session_obj.session_id,
            products=products,
            total_results=results.get('total_results', 0),
            search_type=results.get('search_type', 'hybrid'),
            page=results.get('page', 1),
            page_size=results.get('page_size', limit),
            # Agent-friendly metadata
            search_params={
                'query': query,
                'limit_requested': limit,
                'relevance_threshold': relevance_threshold,
                'results_returned': len(products)
            }
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
                             page: int = 1, 
                             limit: int = 10,  # Agent decides
                             **kwargs) -> Dict[str, Any]:
    """MCP adapter for advanced_search - Agent-driven filtering"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="advanced_search", **kwargs)
        
        logger.info(f"[Search] Advanced search - Category: {category}, Brand: {brand}, Limit: {limit}")
        
        # Perform search using service with agent-specified limit
        results = await search_service.advanced_search(
            query, category, brand, price_min, price_max, 
            location, page, limit
        )
        
        # Extract products
        products = []
        search_results = results.get('search_results', [])
        
        for result in search_results:
            if isinstance(result, dict) and 'item' in result:
                products.append(result['item'])
            elif isinstance(result, dict):
                products.append(result)
        
        # Update session history
        session_obj.search_history.append({
            'query': query or f"Advanced search: {category or 'filtered'}",
            'timestamp': datetime.utcnow().isoformat(),
            'results_count': len(results.get('search_results', [])),
            'products': products[:5] if products else [],
            'filters_applied': {
                'category': category,
                'brand': brand,
                'price_range': f"{price_min}-{price_max}" if price_min or price_max else None
            }
        })

        # Simple message
        if products:
            filter_desc = f" in {category}" if category else ""
            message = f" Found {len(products)} products{filter_desc}"
        else:
            message = f"No products found matching your filters. Try adjusting search criteria."

        # Save session
        save_persistent_session(session_obj, conversation_manager)

        return format_mcp_response(
            results.get('success', True),
            message,
            session_obj.session_id,
            products=products,
            total_results=results.get('total_results', 0),
            search_type=results.get('search_type', 'filtered'),
            page=results.get('page', 1),
            page_size=results.get('page_size', limit)
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
        
        logger.info(f"[Categories] Retrieved {len(results.get('categories', []))} categories")
        
        # Save session
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            True,
            results.get('message', 'Categories retrieved'),
            session_obj.session_id,
            categories=results.get('categories', [])
        )
        
    except Exception as e:
        logger.error(f"Failed to browse categories: {e}")
        return format_mcp_response(
            False,
            f' Failed to browse categories: {str(e)}',
            session_id or 'unknown'
        )