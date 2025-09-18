"""Qdrant vector search client for hybrid search"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..config import VectorConfig
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class SearchFilters:
    """Search filters for hybrid search"""
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    categories: Optional[List[str]] = None
    provider_ids: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    available_only: bool = False  # Don't filter by availability since field doesn't exist in DB


class VectorSearchClient:
    """
    Optional Qdrant integration for enhanced search
    
    This client provides hybrid search capabilities by combining:
    - Semantic search using vector embeddings
    - Traditional API search
    - Intelligent result reranking
    """
    
    def __init__(self, config: VectorConfig):
        """
        Initialize vector search client
        
        Args:
            config: Vector search configuration
        """
        self.config = config
        self.enabled = config.enabled
        self.client = None
        self.embeddings = None
        
        if self.enabled:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client and embeddings"""
        try:
            # Import only if vector search is enabled
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
            
            # Initialize Qdrant client
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port
            )
            
            # Verify collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.config.collection for c in collections)
            
            if not collection_exists:
                logger.warning(f"Qdrant collection '{self.config.collection}' not found")
                self.enabled = False
                return
            
            # Initialize embeddings
            from .embeddings import GeminiEmbeddings
            self.embeddings = GeminiEmbeddings(self.config.gemini_api_key)
            
            logger.info(f"Vector search initialized: {self.config.host}:{self.config.port}")
            
        except ImportError as e:
            logger.warning(f"Qdrant dependencies not installed: {e}")
            logger.warning("Install with: pip install qdrant-client")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            self.enabled = False
    
    async def search(self, query: str, filters: Optional[SearchFilters] = None, 
                    limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform vector search
        
        Args:
            query: Search query
            filters: Optional search filters
            limit: Maximum results
            
        Returns:
            List of search results with scores
        """
        if not self.enabled or not self.client:
            return []
        
        try:
            # Generate query embedding
            logger.info(f" Generating embedding for query: '{query}'")
            query_vector = await self.embeddings.generate_embedding(query)
            if not query_vector:
                logger.warning(" Failed to generate query embedding")
                return []
            
            logger.info(f" Generated embedding vector of length {len(query_vector)}")
            
            # Build Qdrant filters
            qdrant_filters = self._build_qdrant_filters(filters)
            logger.info(f" Built filters: {qdrant_filters}")
            
            # Search similar vectors
            logger.info(f" Searching collection '{self.config.collection}' with threshold {self.config.similarity_threshold}")
            search_result = self.client.search(
                collection_name=self.config.collection,
                query_vector=query_vector,
                query_filter=qdrant_filters,
                limit=limit,
                score_threshold=self.config.similarity_threshold
            )
            
            logger.info(f" Raw search returned {len(search_result)} hits")
            
            # Convert results to standard format
            results = []
            for i, hit in enumerate(search_result):
                logger.info(f"  Hit {i+1}: score={hit.score:.4f}, item={hit.payload.get('name', 'N/A')}")
                result = {
                    "score": hit.score,
                    "item": hit.payload,
                    "source": "vector"
                }
                results.append(result)
            
            logger.info(f" Vector search found {len(results)} results for '{query}' (threshold: {self.config.similarity_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _build_qdrant_filters(self, filters: Optional[SearchFilters]) -> Optional[Any]:
        """Build Qdrant filter conditions from search filters"""
        if not filters:
            return None
        
        try:
            from qdrant_client.models import Filter, FieldCondition, Range, MatchAny
            
            conditions = []
            
            # Price range filter
            if filters.price_min is not None or filters.price_max is not None:
                price_range = Range(
                    gte=filters.price_min if filters.price_min else 0,
                    lte=filters.price_max if filters.price_max else float('inf')
                )
                conditions.append(FieldCondition(
                    key="price",
                    range=price_range
                ))
            
            # Category filter
            if filters.categories:
                conditions.append(FieldCondition(
                    key="category",
                    match=MatchAny(any=filters.categories)
                ))
            
            # Brand filter
            if filters.brands:
                conditions.append(FieldCondition(
                    key="brand",
                    match=MatchAny(any=filters.brands)
                ))
            
            # Provider filter
            if filters.provider_ids:
                conditions.append(FieldCondition(
                    key="provider_id",
                    match=MatchAny(any=filters.provider_ids)
                ))
            
            # Availability filter
            if filters.available_only:
                conditions.append(FieldCondition(
                    key="available",
                    match={"value": True}
                ))
            
            if conditions:
                return Filter(must=conditions)
            
            return None
            
        except ImportError:
            logger.warning("Qdrant models not available")
            return None
    
    async def hybrid_search(self, query: str, api_results: List[Dict], 
                          filters: Optional[SearchFilters] = None) -> List[Dict]:
        """
        Combine vector and API search results
        
        Args:
            query: Search query
            api_results: Results from API search
            filters: Search filters
            
        Returns:
            Reranked combined results
        """
        if not self.enabled:
            return api_results
        
        # Get vector search results
        vector_results = await self.search(query, filters)
        
        if not vector_results:
            return api_results
        
        # Import reranker
        from .reranker import ResultReranker
        reranker = ResultReranker()
        
        # Combine and rerank results
        combined_results = reranker.rerank(api_results, vector_results, query)
        
        logger.info(f"Hybrid search: {len(api_results)} API + {len(vector_results)} vector = {len(combined_results)} combined")
        
        return combined_results
    
    def is_available(self) -> bool:
        """Check if vector search is available and enabled"""
        return self.enabled and self.client is not None