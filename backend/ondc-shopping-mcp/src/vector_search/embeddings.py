"""Embedding generation for vector search"""

import logging
from typing import List, Optional
import asyncio

from ..utils import get_logger

logger = get_logger(__name__)


class GeminiEmbeddings:
    """
    Google Gemini embeddings for vector search
    
    Uses the text-embedding-004 model which produces 768-dimensional vectors
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini embeddings
        
        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        self.model_name = "models/text-embedding-004"  # Match ETL model
        self.dimension = 768
        self.client = None
        
        if api_key:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            self.client = genai
            
            logger.info(f"Gemini embeddings initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning("Google Generative AI not installed")
            logger.warning("Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.client:
            return None
        
        try:
            # Use async wrapper for sync client
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_query"
                )
            )
            
            embedding = result['embedding']
            
            # Validate dimension
            if len(embedding) != self.dimension:
                logger.warning(f"Unexpected embedding dimension: {len(embedding)} vs {self.dimension}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embeddings (None for failed items)
        """
        if not self.client:
            return [None] * len(texts)
        
        try:
            # Process in batches to avoid rate limits
            batch_size = 20
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await asyncio.gather(*[
                    self.generate_embedding(text) for text in batch
                ])
                embeddings.extend(batch_embeddings)
                
                # Small delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)
    
    def is_available(self) -> bool:
        """Check if embeddings are available"""
        return self.client is not None


class MockEmbeddings:
    """Mock embeddings for testing without API key"""
    
    def __init__(self):
        self.dimension = 768
        logger.info("Using mock embeddings for testing")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate mock embedding"""
        # Simple hash-based mock embedding
        import hashlib
        
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to floats between -1 and 1
        embedding = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] - 128) / 128.0
            embedding.append(value)
        
        return embedding
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch"""
        return [await self.generate_embedding(text) for text in texts]
    
    def is_available(self) -> bool:
        """Mock embeddings are always available"""
        return True