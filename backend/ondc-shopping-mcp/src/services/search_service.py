"""Search Service for ONDC Products

Handles all product search operations including vector and API search.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from ..buyer_backend_client import BuyerBackendClient
from ..vector_search import VectorSearchClient, SearchFilters
from ..utils.logger import get_logger
from ..utils.ondc_constants import DEFAULT_CITY_CODE

logger = get_logger(__name__)


class SearchService:
    """Service for handling product searches"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.buyer_app = BuyerBackendClient()
            self.vector_client: Optional[VectorSearchClient] = None
            self._initialize_vector_client()
            self.initialized = True
    
    def _initialize_vector_client(self):
        """Initialize vector search client if enabled"""
        try:
            from ..config import Config
            config = Config()
            if config.vector.enabled:
                self.vector_client = VectorSearchClient(config.vector)
                logger.info(f"Vector search initialized: {self.vector_client.is_available()}")
        except Exception as e:
            logger.warning(f"Failed to initialize vector search: {e}")
    
    def _extract_price(self, item: Dict[str, Any]) -> float:
        """Extract price from various item formats."""
        # Try direct price field
        if isinstance(item.get("price"), (int, float)):
            return float(item["price"])
        elif isinstance(item.get("price"), dict):
            return float(item["price"].get("value", 0))
        # Try nested item_details
        elif item.get("item_details", {}).get("price", {}).get("value"):
            return float(item["item_details"]["price"]["value"])
        return 0.0
    
    def _extract_category(self, item: Dict[str, Any]) -> str:
        """Extract category from various item formats."""
        if isinstance(item.get("category"), dict):
            return item["category"].get("name", "")
        elif isinstance(item.get("category"), str):
            return item["category"]
        elif item.get("item_details", {}).get("category_id"):
            return item["item_details"]["category_id"]
        return ""
    
    def _extract_brand(self, item: Dict[str, Any]) -> str:
        """Extract brand from various item formats."""
        if item.get("brand"):
            return str(item["brand"])
        elif item.get("item_details", {}).get("descriptor", {}).get("brand"):
            return item["item_details"]["descriptor"]["brand"]
        elif item.get("provider_details", {}).get("descriptor", {}).get("name"):
            return item["provider_details"]["descriptor"]["name"]
        return ""
    
    def _matches_category(self, item: Dict[str, Any], category: str) -> bool:
        """Check if item matches category filter."""
        item_category = self._extract_category(item)
        return category.lower() in item_category.lower()
    
    def _matches_brand(self, item: Dict[str, Any], brand: str) -> bool:
        """Check if item matches brand filter."""
        item_brand = self._extract_brand(item)
        return brand.lower() in item_brand.lower()
    
    def _format_api_results(self, api_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format API results to standard product structure
        
        Args:
            api_data: Raw API response data
            
        Returns:
            Formatted product list
        """
        formatted_results = []
        for item in api_data:
            try:
                # Extract item_details which contains the actual product info
                item_details = item.get("item_details", item)  # Fallback to item if no item_details
                descriptor = item_details.get("descriptor", {})
                price = item_details.get("price", {})
                
                # Extract provider details from the response
                provider_details = item.get("provider_details", {})
                location_details = item.get("location_details", {})
                
                # Provider ID handling - use local_id or full id
                provider_id = provider_details.get("local_id", "")
                if not provider_id:
                    provider_id = provider_details.get("id", "")
                    # If it's the long ONDC format, try to extract the local part
                    if provider_id and "_" in provider_id:
                        # Format: hp-seller-preprod.himira.co.in_ONDC:RET10_d871c2ae-bf3f-4d3c-963f-f85f94848e8c
                        parts = provider_id.split("_")
                        if len(parts) >= 3:
                            provider_id = parts[-1]  # Use the last UUID part
                
                # Keep original structure but add simplified fields for easy access
                # This preserves compatibility with ProductFormatter
                product = {
                    # Keep original nested structure
                    "item_details": item_details,
                    "provider_details": provider_details,
                    "location_details": location_details,
                    # Add simplified fields for cart/checkout operations
                    "id": item.get("id") or item_details.get("id", ""),  # Use top-level full ONDC ID first
                    "name": descriptor.get("name", "Unknown Product"),
                    "description": descriptor.get("short_desc", ""),
                    "long_description": descriptor.get("long_desc", ""),
                    "price": price,  # Keep full price dict structure
                    "currency": price.get("currency", "INR"),
                    "images": descriptor.get("images", []),
                    "category": item_details.get("category_id", ""),
                    "provider_id": provider_id,
                    "provider_name": provider_details.get("descriptor", {}).get("name", ""),
                    "provider_location": location_details.get("local_id", location_details.get("id", "")),
                    "returnable": item_details.get("@ondc/org/returnable", False),
                    "cod_available": item_details.get("@ondc/org/available_on_cod", False),
                    "available": True  # Assume available if returned by search
                }
                formatted_results.append(product)
            except Exception as e:
                logger.warning(f"Failed to format API item: {e}")
                continue
        
        return formatted_results
    
    async def search_products(self, query: str = '', 
                            latitude: Optional[float] = None,
                            longitude: Optional[float] = None,
                            page: int = 1, limit: int = 20,
                            session_pincode: Optional[str] = None,
                            relevance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Search for products with hybrid search capabilities
        
        Combines results from both Vector DB and MongoDB (via API) for comprehensive search.
        
        Args:
            query: Search query
            latitude: Optional latitude
            longitude: Optional longitude
            page: Page number
            limit: Results per page
            
        Returns:
            Search results dictionary with combined and reranked results
        """
        if not query:
            return {
                "success": False,
                "message": " Please provide a search query",
                "search_results": []
            }
        
        # Prepare for parallel execution
        search_types = []
        
        # Build search parameters
        params = {
            "name": query,
            "page": page,
            "limit": limit
        }
        
        # Add location parameters
        if latitude and longitude:
            params.update({"latitude": latitude, "longitude": longitude})
        elif session_pincode:
            # Use coordinates from session's delivery pincode
            from ..utils.city_code_mapping import get_coordinates_by_pincode, get_city_code_by_pincode
            coords = get_coordinates_by_pincode(session_pincode)
            city_code = get_city_code_by_pincode(session_pincode)
            params.update({
                "latitude": str(coords["latitude"]),
                "longitude": str(coords["longitude"]),
                "city": city_code
            })
        else:
            # Use consistent Bangalore defaults from constants
            params.update({
                "latitude": "12.9719",
                "longitude": "77.5937",
                "city": DEFAULT_CITY_CODE
            })
        
        # Step 1: Create parallel search tasks
        tasks = []
        
        # API search task
        async def api_search():
            try:
                logger.info(f"[Parallel] Starting API search for: '{query}'")
                user_id = "searchUser"
                response = await self.buyer_app.search_products(user_id, params)
                if response and "response" in response:
                    data = response.get("response", {}).get("data", [])
                    if data:
                        results = self._format_api_results(data)
                        logger.info(f"[Parallel] API returned {len(results)} results")
                        return results
                return []
            except Exception as e:
                logger.error(f"[Parallel] API search failed: {e}")
                return []
        
        tasks.append(api_search())
        
        # Vector search task (if available)
        async def vector_search():
            if self.vector_client and self.vector_client.is_available():
                try:
                    logger.info(f"[Parallel] Starting vector search for: '{query}'")
                    filters = SearchFilters(available_only=False)
                    results = await self.vector_client.search(query=query, filters=filters, limit=limit)
                    if results:
                        logger.info(f"[Parallel] Vector returned {len(results)} results")
                        return results
                except Exception as e:
                    logger.error(f"[Parallel] Vector search failed: {e}")
            return []
        
        tasks.append(vector_search())
        
        # Execute searches in parallel
        logger.info(f"Executing parallel searches for: '{query}'")
        search_results = await asyncio.gather(*tasks)
        api_results = search_results[0]
        vector_results = search_results[1]
        
        if api_results:
            search_types.append("mongodb")
        if vector_results:
            search_types.append("vector")
        
        # Step 2: Perform AI-based reranking on combined results
        if api_results or vector_results:
            logger.info(f"Reranking {len(api_results)} API + {len(vector_results)} vector results")
            
            # Import reranker
            from ..vector_search.reranker import ResultReranker
            reranker = ResultReranker()
            
            # Convert vector results to standard format if needed
            formatted_vector_results = []
            for vr in vector_results:
                if isinstance(vr, dict) and "item" in vr:
                    item = vr["item"]
                    item["_vector_score"] = vr.get("score", 0.0)
                    formatted_vector_results.append(item)
                else:
                    formatted_vector_results.append(vr)
            
            # Rerank combined results with AI scoring and intelligent threshold
            final_results = reranker.rerank(
                api_results=api_results,
                vector_results=formatted_vector_results,
                query=query,
                custom_threshold=relevance_threshold
            )
            
            logger.info(f"AI reranking completed: {len(final_results)} final results")
            
            # Let the agent naturally understand what's relevant through improved scoring
            # Agent will see all results and can intelligently filter what to show users
            
            # Smart fallback: If reranking filtered out everything but we had good vector results
            if not final_results and vector_results:
                # Check if we had high-quality vector matches (score > 0.5)
                high_quality_vector = [vr for vr in vector_results if vr.get("score", 0) > 0.5]
                if high_quality_vector:
                    logger.info(f"Applying vector fallback: returning {len(high_quality_vector)} high-quality vector results")
                    # Convert vector results to final format
                    final_results = []
                    for vr in high_quality_vector[:5]:  # Limit to top 5
                        item = vr.get("item", {})
                        item["rerank_score"] = vr.get("score", 0.0)
                        item["_fallback_source"] = "vector_bypass"
                        final_results.append({"item": item})
                        
                # Last resort: return any vector results if we have them
                elif vector_results:
                    logger.info(f"Applying emergency fallback: returning top {min(3, len(vector_results))} vector results")
                    final_results = []
                    for vr in vector_results[:3]:  # Limit to top 3
                        item = vr.get("item", {})
                        item["rerank_score"] = vr.get("score", 0.0) * 0.8  # Slight penalty for emergency fallback
                        item["_fallback_source"] = "vector_emergency"
                        final_results.append({"item": item})
        else:
            logger.info(f"No products found for query: '{query}'")
            final_results = []
        
        # Format final results
        total_results = len(final_results)
        
        # Create search type description
        if len(search_types) > 1:
            search_type_str = "hybrid (" + "+".join(search_types) + ")"
        elif search_types:
            search_type_str = search_types[0]
        else:
            search_type_str = "none"
        
        if total_results == 0:
            message = f" No products found for '{query}'"
        else:
            intent_info = ""
            threshold_info = f", threshold: {relevance_threshold:.2f}" if relevance_threshold else ""
            message = f" Found {total_results} products{intent_info} ({search_type_str} search{threshold_info})"
        
        return {
            "success": True,
            "message": message,
            "search_results": final_results[:limit],  # Limit final results
            "total_results": total_results,
            "page": page,
            "page_size": limit,
            "search_type": search_type_str,
            "relevance_threshold": relevance_threshold,
            "filtered_by_relevance": relevance_threshold is not None
        }
    
    async def advanced_search(self, query: Optional[str] = None,
                            category: Optional[str] = None,
                            brand: Optional[str] = None,
                            price_min: Optional[float] = None,
                            price_max: Optional[float] = None,
                            location: Optional[str] = None,
                            page: int = 1, limit: int = 20) -> Dict[str, Any]:
        """Advanced search with multiple filters
        
        Performs broad search then applies filters on results since backend
        doesn't support structured queries like "category:Fashion".
        
        Args:
            query: Optional search query
            category: Category filter
            brand: Brand filter
            price_min: Minimum price
            price_max: Maximum price
            location: Location filter
            page: Page number
            limit: Results per page
            
        Returns:
            Search results dictionary
        """
        # Use query directly or infer from category for better results
        base_query = query
        if not base_query and category:
            # Generate search terms from category for better results than "all"
            category_keywords = {
                "Oil & Ghee": "oil ghee",
                "Dairy and Cheese": "milk cheese dairy",
                "Fruits and Vegetables": "fruits vegetables",
                "Rice and Rice Products": "rice",
                "Masala & Seasoning": "masala spice seasoning",
                "Tea and Coffee": "tea coffee",
                "Snacks, Dry Fruits, Nuts": "snacks nuts",
                "Bakery, Cakes & Dairy": "bakery cake bread",
                "Atta, Flours and Sooji": "flour atta",
                "Salt, Sugar and Jaggery": "salt sugar jaggery"
            }
            base_query = category_keywords.get(category, category.split()[0].lower())
        
        if not base_query:
            base_query = "food"  # Better fallback than "all"
        
        # Get more results initially to filter
        initial_limit = limit * 3  # Get 3x to account for filtering
        
        # Perform broad search
        results = await self.search_products(
            query=base_query,
            page=1,  # Always get first page for filtering
            limit=initial_limit
        )
        
        # Apply filters on search results
        filtered = []
        search_results = results.get("search_results", [])
        
        for result in search_results:
            # Extract item from result structure
            item = result.get("item", result) if isinstance(result, dict) else result
            
            # For debugging - track what we're filtering
            item_name = item.get("name", "Unknown")
            
            # Apply filters using helper methods
            if category and not self._matches_category(item, category):
                continue
            
            if brand and not self._matches_brand(item, brand):
                continue
            
            if (price_min is not None or price_max is not None):
                item_price = self._extract_price(item)
                if price_min and item_price < price_min:
                    continue
                if price_max and item_price > price_max:
                    continue
            
            filtered.append(result)
        
        # Apply pagination on filtered results
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated = filtered[start_idx:end_idx]
        
        results["search_results"] = paginated
        results["total_results"] = len(filtered)
        results["page"] = page
        results["page_size"] = limit
        
        # Update message
        filters_applied = []
        if category:
            filters_applied.append(f"category: {category}")
        if brand:
            filters_applied.append(f"brand: {brand}")
        if price_min or price_max:
            price_range = f"₹{price_min or 0}-{price_max or '∞'}"
            filters_applied.append(f"price: {price_range}")
        
        if filters_applied:
            results["message"] = f" Advanced search with {', '.join(filters_applied)}"
        
        return results
    
    async def browse_categories(self) -> Dict[str, Any]:
        """Browse available product categories
        
        Fetches categories from real API products or provides fallback categories.
        
        Returns:
            Categories dictionary with available product categories
        """
        try:
            logger.info("Fetching categories from ONDC network")
            
            # Try to get categories from real API
            # Use a broad search to get various products and extract categories
            params = {"name": "", "page": 1, "limit": 100}
            user_id = "guestUser"
            
            categories_map = {}  # Use map to track category details
            
            try:
                # Use dedicated categories endpoint instead of extracting from search
                api_response = await self.buyer_app.get_categories(limit=100, page=1)
                logger.info(f"[Categories] Categories API response type: {type(api_response)}")
                logger.info(f"[Categories] Categories API response: {api_response}")
                
                # Handle categories response format
                if api_response:
                    categories_list = None
                    
                    # Add detailed logging to debug the response format
                    logger.info(f"[Categories] API response type: {type(api_response)}")
                    if isinstance(api_response, dict):
                        logger.info(f"[Categories] API response keys: {list(api_response.keys())}")
                    elif isinstance(api_response, list) and len(api_response) > 0:
                        logger.info(f"[Categories] API response is list with {len(api_response)} items")
                        if isinstance(api_response[0], dict):
                            logger.info(f"[Categories] First item keys: {list(api_response[0].keys())}")
                    
                    # Handle Himira backend wrapped response format
                    if isinstance(api_response, list) and len(api_response) > 0:
                        # Response is wrapped: [{result: {data: {json: {categories: []}}}}]
                        first_item = api_response[0]
                        if isinstance(first_item, dict) and 'result' in first_item:
                            # Extract categories from nested structure
                            categories_data = first_item.get('result', {}).get('data', {}).get('json', {})
                            categories_list = categories_data.get('categories', [])
                    elif "data" in api_response:
                        # Nested data structure
                        categories_list = api_response["data"]
                    elif "categories" in api_response:
                        # Categories key
                        categories_list = api_response["categories"]
                    else:
                        # Check if response has array at root level
                        if isinstance(api_response.get("result", {}).get("data", {}), list):
                            categories_list = api_response["result"]["data"]
                        elif isinstance(api_response, dict) and len(api_response) == 1:
                            # Single key response - might be TRPC format
                            first_key = list(api_response.keys())[0]
                            if isinstance(api_response[first_key], dict) and "result" in api_response[first_key]:
                                result_data = api_response[first_key]["result"]
                                if "data" in result_data and isinstance(result_data["data"], list):
                                    categories_list = result_data["data"]
                    
                    logger.info(f"[Categories] Found {len(categories_list) if isinstance(categories_list, list) else 0} categories from API")
                    
                    # Process categories
                    if isinstance(categories_list, list) and categories_list:
                        formatted_categories = []
                        for category in categories_list:
                            if isinstance(category, dict):
                                formatted_categories.append({
                                    "id": category.get("id", category.get("_id", category.get("categoryId", ""))),
                                    "name": category.get("name", category.get("title", category.get("categoryName", ""))),
                                    "description": category.get("description", f"Browse {category.get('name', category.get('title', 'items'))} products"),
                                    "image": category.get("image", category.get("imageUrl", category.get("icon", ""))),
                                    "item_count": category.get("productCount", category.get("item_count", category.get("count", 0)))
                                })
                        
                        if formatted_categories:
                            logger.info(f"[Categories] Returning {len(formatted_categories)} formatted categories")
                            return {
                                "success": True,
                                "message": f" Found {len(formatted_categories)} categories from ONDC buyer backend",
                                "categories": formatted_categories
                            }
                else:
                    logger.warning(f"[Categories] Categories API failed or returned error: {api_response}")
                    
                # Categories processed above in new dedicated endpoint call
                        
            except Exception as e:
                logger.error(f"[Categories] Error fetching categories from API: {e}")
                logger.error(f"[Categories] API response was: {api_response if 'api_response' in locals() else 'No response'}")
            
            # No fallback categories - only return real data from API
            logger.info("[Categories] No categories found from buyer backend API")
            
            return {
                "success": True,
                "message": " No categories available from ONDC network at the moment",
                "categories": []
            }
            
        except Exception as e:
            logger.error(f"Failed to browse categories: {e}")
            return {
                "success": False,
                "message": f" Failed to browse categories: {str(e)}",
                "categories": []
            }


# Singleton getter
def get_search_service() -> SearchService:
    """Get the singleton SearchService instance
    
    Returns:
        SearchService instance
    """
    return SearchService()