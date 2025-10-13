"""RelevanceFilter for LLM-based semantic validation of search results

Uses Gemini AI to validate if search results are truly relevant to the user's query intent.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RelevanceFilter:
    """LLM-based semantic relevance validation for search results"""
    
    def __init__(self):
        """Initialize RelevanceFilter with AI model configuration"""
        self.model = None
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini AI client for relevance validation"""
        try:
            from ..config import config
            if config.vector.gemini_api_key:
                import google.generativeai as genai
                genai.configure(api_key=config.vector.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("[RelevanceFilter] Gemini AI initialized for semantic validation")
            else:
                logger.warning("[RelevanceFilter] No Gemini API key - semantic validation disabled")
        except Exception as e:
            logger.error(f"[RelevanceFilter] Failed to initialize Gemini AI: {e}")
    
    def is_available(self) -> bool:
        """Check if RelevanceFilter is available for use"""
        return self.model is not None
    
    async def validate_results(self, query: str, products: List[Dict[str, Any]], 
                             query_intent: str = "unknown", 
                             max_validate: int = 10) -> List[Dict[str, Any]]:
        """
        Validate search results for semantic relevance using LLM
        
        Args:
            query: Original search query
            products: List of product results to validate
            query_intent: Query intent (specific, broad, discovery, etc.)
            max_validate: Maximum number of products to validate (for performance)
            
        Returns:
            Filtered list of products that passed semantic validation
        """
        if not self.is_available() or not products:
            return products
        
        # Limit validation for performance (validate top results only)
        products_to_validate = products[:max_validate]
        
        try:
            # Batch validation for efficiency
            validated_products = await self._batch_validate(query, products_to_validate, query_intent)
            
            # Add remaining unvalidated products if any
            if len(products) > max_validate:
                validated_products.extend(products[max_validate:])
            
            logger.info(f"[RelevanceFilter] Validated {len(products_to_validate)} products for '{query}', "
                       f"kept {len(validated_products)} relevant results")
            
            return validated_products
            
        except Exception as e:
            logger.error(f"[RelevanceFilter] Validation failed: {e}")
            # Return original products if validation fails
            return products
    
    async def _batch_validate(self, query: str, products: List[Dict[str, Any]], 
                             query_intent: str) -> List[Dict[str, Any]]:
        """Batch validate multiple products for efficiency"""
        
        # Create validation prompt
        validation_prompt = self._create_validation_prompt(query, products, query_intent)
        
        try:
            # Call Gemini for validation
            response = await asyncio.wait_for(
                self._call_gemini_async(validation_prompt), 
                timeout=10.0
            )
            
            # Parse validation results
            validated_products = self._parse_validation_response(response, products)
            return validated_products
            
        except asyncio.TimeoutError:
            logger.warning(f"[RelevanceFilter] Validation timeout for query: '{query}'")
            return products
        except Exception as e:
            logger.error(f"[RelevanceFilter] Batch validation error: {e}")
            return products
    
    def _create_validation_prompt(self, query: str, products: List[Dict[str, Any]], 
                                 query_intent: str) -> str:
        """Create validation prompt for LLM"""
        
        # Extract key product information for validation
        products_info = []
        for i, product in enumerate(products):
            product_name = product.get('name', 'Unknown Product')
            product_desc = product.get('description', '')
            category = product.get('category', '')
            brand = product.get('brand', '')
            
            products_info.append({
                'id': i,
                'name': product_name,
                'description': product_desc[:100],  # Limit description length
                'category': category,
                'brand': brand
            })
        
        intent_guidance = self._get_intent_guidance(query_intent)
        
        prompt = f"""You are a shopping assistant evaluating product search results for relevance.

User Query: "{query}"
Query Intent: {query_intent}
Intent Guidance: {intent_guidance}

Products to evaluate:
{json.dumps(products_info, indent=2)}

Task: Evaluate each product's relevance to the user's query and intent. Return a JSON array of product IDs that are relevant.

Relevance Criteria:
1. Product name/description matches query terms
2. Product category aligns with query intent
3. Product brand is relevant if mentioned in query
4. Product serves the user's intended purpose

For "{query_intent}" queries: {intent_guidance}

Return ONLY a JSON array of relevant product IDs (numbers): [0, 2, 4, ...]

Example response: [0, 1, 3]
"""
        
        return prompt
    
    def _get_intent_guidance(self, query_intent: str) -> str:
        """Get intent-specific guidance for validation"""
        guidance = {
            "specific": "Be strict - only include products that closely match the specific request",
            "broad": "Include products within the category/brand, allow some variety",
            "discovery": "Be inclusive - include products that could help user discover options",
            "generic": "Include diverse products that match basic query terms"
        }
        return guidance.get(query_intent, "Use balanced relevance criteria")
    
    async def _call_gemini_async(self, prompt: str) -> str:
        """Call Gemini API asynchronously"""
        def _sync_call():
            response = self.model.generate_content(prompt)
            return response.text
        
        # Run synchronous call in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)
    
    def _parse_validation_response(self, response: str, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse LLM validation response and filter products"""
        try:
            # Extract JSON array from response
            response_clean = response.strip()
            
            # Handle potential markdown formatting
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0]
            
            # Parse JSON array
            relevant_ids = json.loads(response_clean)
            
            if not isinstance(relevant_ids, list):
                logger.warning(f"[RelevanceFilter] Invalid response format: {response}")
                return products
            
            # Filter products based on validation results
            validated_products = []
            for product_id in relevant_ids:
                if isinstance(product_id, int) and 0 <= product_id < len(products):
                    validated_products.append(products[product_id])
            
            logger.debug(f"[RelevanceFilter] Validation kept {len(validated_products)}/{len(products)} products")
            return validated_products
            
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            logger.warning(f"[RelevanceFilter] Failed to parse validation response: {e}")
            logger.debug(f"[RelevanceFilter] Raw response: {response}")
            # Return all products if parsing fails
            return products
    
    async def validate_single_product(self, query: str, product: Dict[str, Any], 
                                    query_intent: str = "unknown") -> bool:
        """
        Validate a single product for relevance (useful for individual checks)
        
        Args:
            query: Search query
            product: Product to validate
            query_intent: Query intent
            
        Returns:
            True if product is relevant, False otherwise
        """
        if not self.is_available():
            return True  # Default to relevant if validation unavailable
        
        try:
            validated = await self.validate_results(query, [product], query_intent, max_validate=1)
            return len(validated) > 0
        except Exception as e:
            logger.error(f"[RelevanceFilter] Single product validation failed: {e}")
            return True  # Default to relevant if validation fails


# Global instance
_relevance_filter = None


def get_relevance_filter() -> RelevanceFilter:
    """Get singleton RelevanceFilter instance"""
    global _relevance_filter
    if _relevance_filter is None:
        _relevance_filter = RelevanceFilter()
    return _relevance_filter