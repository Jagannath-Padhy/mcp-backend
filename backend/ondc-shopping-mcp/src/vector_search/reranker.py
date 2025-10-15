"""Result reranking for hybrid search"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import re

from ..utils import get_logger

logger = get_logger(__name__)


class ResultReranker:
    """
    Intelligent reranking of search results
    
    Combines results from multiple sources (API, vector search) and
    reranks them based on multiple factors:
    - Semantic relevance (vector similarity)
    - Exact match score (API search)
    - Product attributes (price, availability, ratings)
    - Source diversity
    """
    
    def __init__(self):
        """Initialize reranker with balanced weights favoring vector similarity"""
        self.weights = {
            "relevance": 0.35,        # AI relevance scoring (reduced from 0.45)
            "vector_score": 0.40,     # Semantic similarity (increased from 0.25) 
            "exact_match": 0.15,      # Keyword match
            "availability": 0.04,     # In stock
            "price_score": 0.03,      # Price competitiveness
            "popularity": 0.03        # Reviews/ratings
        }
        
        # Minimum score threshold - products below this are filtered out
        self.min_score_threshold = 0.4  # Increased for better relevance filtering
    
    def rerank(self, api_results: List[Dict], vector_results: List[Dict], 
              query: str, custom_threshold: Optional[float] = None) -> List[Dict]:
        """
        Rerank combined results from multiple sources with intelligent filtering
        
        Args:
            api_results: Results from API search
            vector_results: Results from vector search
            query: Original search query
            custom_threshold: Custom relevance threshold
            
        Returns:
            Reranked list of results filtered by relevance threshold
        """
        # Use custom threshold if provided, otherwise use default
        effective_threshold = custom_threshold if custom_threshold is not None else self.min_score_threshold
        
        # Combine all results
        all_results = self._merge_results(api_results, vector_results)
        
        if not all_results:
            return []
        
        # Calculate scores for each result
        scored_results = []
        for result in all_results:
            score = self._calculate_score(result, query)
            result["rerank_score"] = score
            
            # Apply intelligent threshold filtering
            if score >= effective_threshold:
                scored_results.append(result)
            else:
                # Log filtered results for debugging
                item_name = self._get_item_name(result.get("item", {}))
                logger.debug(f"Filtered out '{item_name}' (score: {score:.3f} < {effective_threshold:.3f})")
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Enhanced logging with score distribution and intelligent filtering info
        if scored_results:
            avg_score = sum(r["rerank_score"] for r in scored_results) / len(scored_results)
            max_score = max(r["rerank_score"] for r in scored_results)
            min_score = min(r["rerank_score"] for r in scored_results)
            logger.info(f"Reranked {len(scored_results)}/{len(all_results)} results for '{query}' "
                       f"(threshold: {effective_threshold:.3f}, "
                       f"scores: {min_score:.3f}-{max_score:.3f}, avg: {avg_score:.3f})")
        else:
            logger.warning(f"All {len(all_results)} results filtered out for '{query}' "
                          f"(threshold: {effective_threshold:.3f}) - "
                          f"consider lowering relevance requirements!")
        
        return scored_results
    
    def _merge_results(self, api_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
        """
        Merge results from different sources, deduplicating by item ID
        
        Args:
            api_results: API search results
            vector_results: Vector search results
            
        Returns:
            Merged results with source tracking
        """
        merged = {}
        
        # Process API results
        for idx, result in enumerate(api_results):
            item_id = self._get_item_id(result)
            if item_id:
                merged[item_id] = {
                    "item": result,
                    "sources": ["api"],
                    "api_rank": idx + 1,
                    "vector_score": 0.0
                }
        
        # Process vector results
        for idx, result in enumerate(vector_results):
            # Vector results are already unwrapped items (just the payload)
            item_id = self._get_item_id(result)  # result IS already the item
            if item_id:
                if item_id in merged:
                    # Update existing entry
                    merged[item_id]["sources"].append("vector")
                    merged[item_id]["vector_rank"] = idx + 1
                    merged[item_id]["vector_score"] = result.get("_vector_score", 0.0)  # Score was stored in _vector_score
                else:
                    # Add new entry
                    merged[item_id] = {
                        "item": result,  # result IS already the item
                        "sources": ["vector"],
                        "vector_rank": idx + 1,
                        "vector_score": result.get("_vector_score", 0.0),  # Score was stored in _vector_score
                        "api_rank": float('inf')
                    }
        
        return list(merged.values())
    
    def _calculate_score(self, result: Dict, query: str) -> float:
        """
        Calculate AI-enhanced reranking score for a result
        
        Args:
            result: Result with item and metadata
            query: Search query
            
        Returns:
            Combined score between 0 and 1
        """
        scores = {}
        item = result.get("item", {})
        
        # 1. AI Relevance Score (NEW) - most important
        scores["relevance"] = self._calculate_ai_relevance(item, query)
        
        # 2. Vector similarity score
        scores["vector_score"] = result.get("vector_score", 0.0)
        
        # 3. Exact match score
        scores["exact_match"] = self._calculate_exact_match_score(item, query)
        
        # 4. Availability score
        scores["availability"] = self._calculate_availability_score(item)
        
        # 5. Price competitiveness score
        scores["price_score"] = self._calculate_price_score(item)
        
        # 6. Popularity score
        scores["popularity"] = self._calculate_popularity_score(item)
        
        # Apply source diversity bonus
        if len(result.get("sources", [])) > 1:
            diversity_bonus = 0.1
        else:
            diversity_bonus = 0.0
        
        # Calculate weighted sum
        total_score = sum(
            self.weights.get(key, 0) * score 
            for key, score in scores.items()
        )
        
        # Add diversity bonus
        total_score = min(1.0, total_score + diversity_bonus)
        
        # Store component scores for debugging
        result["score_components"] = scores
        
        return total_score
    
    def _calculate_ai_relevance(self, item: Dict, query: str) -> float:
        """
        Calculate AI-based relevance score using advanced text matching
        
        Args:
            item: Product item (can be API format with item_details or Vector format flat structure)
            query: Search query
            
        Returns:
            Relevance score between 0 and 1
        """
        query_tokens = query.lower().split()
        total_score = 0.0
        
        # Extract text fields from different item formats with better handling
        name, desc, long_desc, category, brand = "", "", "", "", ""
        
        # Handle API format (nested item_details) vs Vector format (flat)
        if "item_details" in item:
            # API format - nested structure
            item_details = item.get("item_details", {})
            descriptor = item_details.get("descriptor", {})
            name = str(descriptor.get("name", "")).lower()
            desc = str(descriptor.get("short_desc", "")).lower()
            long_desc = str(descriptor.get("long_desc", "")).lower()
            category = str(item_details.get("category_id", "")).lower()
            brand = str(item.get("provider_details", {}).get("descriptor", {}).get("name", "")).lower()
        else:
            # Vector format - flat structure (this is what we're getting!)
            name = str(item.get("name", "")).lower()
            
            # Handle various description field names from vector results
            desc_candidates = [
                item.get("description", ""),
                item.get("short_desc", ""), 
                item.get("desc", ""),
                item.get("short_description", "")
            ]
            desc = str(next((d for d in desc_candidates if d), "")).lower()
            
            long_desc_candidates = [
                item.get("long_desc", ""),
                item.get("long_description", ""),
                item.get("detailed_description", "")
            ]
            long_desc = str(next((ld for ld in long_desc_candidates if ld), desc)).lower()
            
            # Handle category - could be string, dict, or nested object
            category_obj = item.get("category", item.get("category_id", ""))
            if isinstance(category_obj, dict):
                category = str(category_obj.get("name", category_obj.get("id", ""))).lower()
            else:
                category = str(category_obj).lower()
            
            # Handle provider/brand - similar flexible approach
            provider_obj = item.get("provider", item.get("brand", {}))
            if isinstance(provider_obj, dict):
                brand = str(provider_obj.get("name", provider_obj.get("brand", ""))).lower()
            else:
                brand = str(provider_obj).lower()
        
        # Score each field
        field_scores = []
        
        # Title/Name matching (highest weight)
        name_score = 0.0
        for token in query_tokens:
            if token in name:
                # Exact word match
                if f" {token} " in f" {name} " or name.startswith(token) or name.endswith(token):
                    name_score += 1.0
                # Partial match
                else:
                    name_score += 0.5
        if query_tokens:
            name_score = min(name_score / len(query_tokens), 1.0)
        field_scores.append(("name", name_score, 0.4))  # 40% weight
        
        # Description matching
        desc_score = 0.0
        combined_desc = f"{desc} {long_desc}"
        for token in query_tokens:
            if token in combined_desc:
                desc_score += 1.0
        if query_tokens:
            desc_score = min(desc_score / len(query_tokens), 1.0)
        field_scores.append(("description", desc_score, 0.25))  # 25% weight
        
        # Category matching
        cat_score = 0.0
        for token in query_tokens:
            if token in category:
                cat_score += 1.0
        if query_tokens:
            cat_score = min(cat_score / len(query_tokens), 1.0)
        field_scores.append(("category", cat_score, 0.2))  # 20% weight
        
        # Brand matching
        brand_score = 0.0
        for token in query_tokens:
            if token in brand:
                brand_score += 1.0
        if query_tokens:
            brand_score = min(brand_score / len(query_tokens), 1.0)
        field_scores.append(("brand", brand_score, 0.15))  # 15% weight
        
        # Calculate weighted total
        for field_name, score, weight in field_scores:
            total_score += score * weight
        
        # Define stop words to ignore in token matching
        stop_words = {
            'for', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'from',
            'with', 'by', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # LLM-based relevance validation will be handled at service level
        
        # Filter out stop words and get important tokens
        important_tokens = [token for token in query_tokens if token not in stop_words and len(token) > 2]
        
        # Boost for exact phrase match in name
        if query.lower() in name:
            total_score = min(total_score * 1.5, 1.0)
        
        # Flexible token matching - require majority of important tokens to be found
        if important_tokens:
            searchable_text = f"{name} {combined_desc} {category} {brand}"
            important_tokens_found = sum(1 for token in important_tokens if token in searchable_text)
            token_coverage = important_tokens_found / len(important_tokens)
            
            # Boost based on token coverage (not requiring 100%)
            if token_coverage >= 0.6:  # 60% of important tokens found
                boost_factor = 1.0 + (token_coverage * 0.3)  # Up to 30% boost
                total_score = min(total_score * boost_factor, 1.0)
            elif token_coverage >= 0.3:  # 30% coverage - minimal boost
                total_score = min(total_score * 1.1, 1.0)
        
        # Removed artificial vector score boosting to prevent irrelevant results
        
        return total_score
    
    def _calculate_exact_match_score(self, item: Dict, query: str) -> float:
        """Calculate keyword match score"""
        # Get searchable text
        item_details = item.get("item_details", {})
        descriptor = item_details.get("descriptor", {})
        
        searchable_text = " ".join([
            descriptor.get("name", ""),
            descriptor.get("short_desc", ""),
            item_details.get("category_id", ""),
            item.get("provider_details", {}).get("descriptor", {}).get("name", "")
        ]).lower()
        
        # Normalize query
        query_terms = query.lower().split()
        
        # Count matches
        matches = sum(1 for term in query_terms if term in searchable_text)
        
        # Calculate score
        if len(query_terms) > 0:
            return matches / len(query_terms)
        return 0.0
    
    def _calculate_availability_score(self, item: Dict) -> float:
        """Calculate availability score"""
        item_details = item.get("item_details", item)  # Fallback to item itself
        
        # Check stock status
        quantity = item_details.get("quantity", {})
        available_count = quantity.get("available", {}).get("count", 0)
        # Convert to int if it's a string
        if isinstance(available_count, str):
            try:
                available_count = int(available_count)
            except:
                available_count = 0
        
        if available_count > 0:
            return 1.0
        
        # Check COD availability
        if item_details.get("@ondc/org/available_on_cod", False):
            return 0.8
        
        # Default availability
        return 0.5
    
    def _calculate_price_score(self, item: Dict) -> float:
        """Calculate price competitiveness score"""
        item_details = item.get("item_details", item)  # Fallback to item itself
        
        # Try multiple price locations
        price = None
        if "price" in item_details:
            price_info = item_details.get("price", {})
            price = price_info.get("value", 0)
        elif "price" in item:
            price = item.get("price", 0)
        else:
            price = 0
        
        try:
            price = float(price) if price else 0
            
            # Simple price scoring (lower is better)
            # This should ideally compare with category average
            if price <= 100:
                return 1.0
            elif price <= 500:
                return 0.8
            elif price <= 1000:
                return 0.6
            elif price <= 5000:
                return 0.4
            else:
                return 0.2
                
        except (ValueError, TypeError):
            return 0.5
    
    def _calculate_popularity_score(self, item: Dict) -> float:
        """Calculate popularity score based on ratings/reviews"""
        # This is a placeholder - actual implementation would use
        # ratings, review count, sales data, etc.
        
        # Check for returnable items (proxy for quality)
        item_details = item.get("item_details", {})
        if item_details.get("@ondc/org/returnable", False):
            return 0.8
        
        # Default middle score
        return 0.5
    
    def _get_item_id(self, item: Dict) -> Optional[str]:
        """Extract unique item ID from different item formats"""
        if isinstance(item, dict):
            # Try direct id first (vector search format)
            direct_id = item.get("id")
            if direct_id:
                return direct_id
            
            # Try nested item_details.id (API format)
            item_details = item.get("item_details", {})
            if isinstance(item_details, dict):
                nested_id = item_details.get("id")
                if nested_id:
                    return nested_id
            
            # Try original_id as fallback
            original_id = item.get("original_id")
            if original_id:
                return original_id
        
        return None
    
    def _get_item_name(self, item: Dict) -> str:
        """Extract item name for debugging purposes"""
        if isinstance(item, dict):
            # Try API format first
            if "item_details" in item:
                descriptor = item.get("item_details", {}).get("descriptor", {})
                name = descriptor.get("name", "")
                if name:
                    return str(name)
            
            # Try direct name field (vector format)
            name = item.get("name", "")
            if name:
                return str(name)
                
        return "Unknown Item"
    
