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
                             page: int = 1, limit: Optional[int] = None,
                             max_results: Optional[int] = None,
                             relevance_threshold: Optional[float] = None,
                             adaptive_results: bool = True,
                             context_aware: bool = True,
                             **kwargs) -> Dict[str, Any]:
    """MCP adapter for search_products with intelligent result sizing"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="search_products", **kwargs)
        
        # Import QueryAnalyzer and Config for intelligent sizing
        from ..utils.query_analyzer import get_query_analyzer
        from ..config import config
        
        # Determine optimal search configuration using AI query analysis
        search_context = {}
        
        if context_aware and session_obj:
            # Build search context from session data
            search_context = {
                'user_preferences': getattr(session_obj, 'user_preferences', {}),
                'session_history': {
                    'recent_search_count': len(getattr(session_obj, 'search_history', [])),
                },
                'cart_items': len(getattr(session_obj, 'cart', {}).get('items', []))
            }
        
        # Use QueryAnalyzer for intelligent configuration
        query_config = None
        if adaptive_results and config.search.query_analysis_enabled:
            analyzer = get_query_analyzer()
            query_config = analyzer.analyze_query(query, search_context)
        
        # Determine final search parameters
        if query_config and adaptive_results:
            # Use AI-determined configuration
            final_limit = max_results or query_config.max_results
            final_threshold = relevance_threshold or query_config.relevance_threshold
            search_intent = query_config.intent.value
        else:
            # Use provided parameters or defaults
            final_limit = max_results or limit or config.search.default_limit
            final_threshold = relevance_threshold or config.search.default_relevance_threshold
            search_intent = "manual"
        
        # Ensure limits stay within configured bounds
        final_limit = max(config.search.min_results, min(final_limit, config.search.max_results))
        final_threshold = max(config.search.min_relevance_threshold, 
                             min(final_threshold, config.search.max_relevance_threshold))
        
        logger.info(f"[SearchAdapter] Query: '{query}' -> Intent: {search_intent}, "
                   f"Limit: {final_limit}, Threshold: {final_threshold}")
        
        # Get session pincode if available
        session_pincode = None
        delivery_location = getattr(session_obj, 'delivery_location', None)
        if delivery_location and delivery_location.get('pincode'):
            session_pincode = delivery_location['pincode']
        
        # Perform search using service with intelligent parameters
        results = await search_service.search_products(
            query, latitude, longitude, page, final_limit, session_pincode,
            relevance_threshold=final_threshold,
            query_intent=search_intent
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
        
        # Update session history with intelligent search metadata
        session_obj.search_history.append({
            'query': query,
            'timestamp': datetime.utcnow().isoformat(),
            'results_count': len(results.get('search_results', [])),
            'products': products[:5] if products else [],  # Store top 5 products for context
            'search_intent': search_intent,
            'relevance_threshold': final_threshold,
            'adaptive_sizing': adaptive_results,
            'final_limit': final_limit
        })
        
        # Enhanced message with intelligent search summary
        if products:
            intent_desc = f" ({search_intent} intent)" if search_intent != "manual" else ""
            message = f" Found {len(products)} relevant products for '{query}'{intent_desc} ({results.get('search_type', 'hybrid')} search)"
        else:
            # Get search suggestions if query analysis was performed
            suggestions = []
            if query_config and hasattr(query_config, 'intent'):
                analyzer = get_query_analyzer()
                suggestions = analyzer.get_search_suggestions(query, query_config.intent)
            
            suggestion_text = f" Try: {suggestions[0]}" if suggestions else " Try a different search term."
            message = f"No products found for '{query}'.{suggestion_text}"
        
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
            page_size=results.get('page_size', final_limit),
            # Enhanced search metadata for UI
            search_metadata={
                'query_intent': search_intent,
                'relevance_threshold': final_threshold,
                'adaptive_results': adaptive_results,
                'context_aware': context_aware,
                'original_limit_requested': max_results or limit or "auto",
                'final_limit_applied': final_limit,
                'search_suggestions': analyzer.get_search_suggestions(query, query_config.intent) if query_config else []
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