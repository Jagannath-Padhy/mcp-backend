"""
Embedding Generator

Generates vector embeddings using Google Gemini AI for semantic search.
"""

import asyncio
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai
import json

from .base_transformer import BaseTransformer, TransformationConfig

logger = logging.getLogger(__name__)


class EmbeddingGenerator(BaseTransformer):
    """
    Transformer that generates vector embeddings using Google Gemini
    
    Features:
    - Text embedding generation
    - Caching for improved performance
    - Rate limiting for API calls
    - Batch processing for efficiency
    """
    
    def __init__(self, 
                 config: TransformationConfig,
                 ai_config: Dict[str, Any]):
        super().__init__(config, "embedding_generator")
        
        # AI Configuration
        self.api_key = ai_config.get("api_key", "")
        self.model_name = ai_config.get("model", "models/text-embedding-004")  # Use text-embedding-004 as default
        self.dimensions = ai_config.get("dimensions", 768)
        
        # Configure Gemini
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.warning("No Gemini API key provided - embeddings will fail")
            
        # Embedding configuration
        self.fields_to_embed = ai_config.get("fields_to_embed", ["name", "description"])
        self.embedding_batch_size = ai_config.get("batch_size", 10)
        self.max_text_length = ai_config.get("max_text_length", 2000)
        
        # Rate limiting
        self.rate_limit_delay = 1.0 / ai_config.get("rate_limit_rps", 10)
        self.last_request_time = 0.0
        
        # Cache stats (no actual caching)
        self.cache_stats = {"api_calls": 0}
                
    async def setup(self):
        """Initialize embedding generator"""
        await super().setup()
        
        # Verify Gemini connection
        if not await self._test_gemini_connection():
            logger.error("Failed to connect to Gemini API")
            
    async def _test_gemini_connection(self) -> bool:
        """Test connection to Gemini API"""
        try:
            # Try to generate a test embedding
            result = genai.embed_content(
                model=self.model_name,
                content="test connection",
                task_type="retrieval_document"
            )
            return len(result['embedding']) == self.dimensions
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            return False
            
    async def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single record by adding embeddings
        """
        try:
            if not self.validate_input(record):
                raise ValueError("Invalid input record")
                
            # Create a copy to avoid modifying original
            transformed_record = record.copy()
            
            # Extract text for embedding
            text_content = self._extract_embedding_text(record)
            
            if not text_content:
                logger.warning(f"No text content found for embedding in record {record.get('id', 'unknown')}")
                transformed_record["embedding"] = None
                transformed_record["embedding_error"] = "No text content found"
                return transformed_record
                
            # Generate embedding
            embedding = await self._generate_embedding(text_content)
            
            if embedding:
                transformed_record["embedding"] = embedding
                transformed_record["embedding_model"] = self.model_name
                transformed_record["embedding_dimensions"] = len(embedding)
                transformed_record["embedding_generated_at"] = datetime.utcnow().isoformat()
                transformed_record["embedding_text_length"] = len(text_content)
            else:
                transformed_record["embedding"] = None
                transformed_record["embedding_error"] = "Failed to generate embedding"
                
            return transformed_record
            
        except Exception as e:
            logger.error(f"Error transforming record: {e}")
            # Return original record with error info
            record["embedding"] = None
            record["embedding_error"] = str(e)
            return record
            
    def _extract_embedding_text(self, record: Dict[str, Any]) -> str:
        """
        Extract text content from record fields for embedding generation
        Handles both nested field paths (item_details.descriptor.name) and direct fields
        """
        text_parts = []
        
        for field in self.fields_to_embed:
            value = self._get_nested_value(record, field)
            
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
            elif isinstance(value, dict):
                # Handle nested objects (like category, provider)
                if "name" in value:
                    text_parts.append(str(value["name"]).strip())
                if "description" in value:
                    text_parts.append(str(value["description"]).strip())
            elif isinstance(value, list):
                # Handle arrays (like tags)
                for item in value:
                    if isinstance(item, str):
                        text_parts.append(item.strip())
                        
        # Combine all text parts
        combined_text = " ".join(text_parts)
        
        # Truncate if too long
        if len(combined_text) > self.max_text_length:
            combined_text = combined_text[:self.max_text_length].rsplit(' ', 1)[0]
            
        return combined_text
    
    def _get_nested_value(self, record: Dict[str, Any], field_path: str) -> Any:
        """
        Get value from nested field path like 'item_details.descriptor.name'
        Falls back to direct field access for compatibility
        """
        # Handle nested paths (e.g., "item_details.descriptor.name")
        if '.' in field_path:
            parts = field_path.split('.')
            current = record
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        else:
            # Direct field access (backward compatibility)
            return record.get(field_path)
        
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for given text using Gemini API
        """
        if not text or not text.strip():
            return None
            
        # Update stats
        self.cache_stats["api_calls"] += 1
        
        try:
            # Rate limiting
            await self._apply_rate_limit()
            
            # Generate embedding
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            
            # Validate embedding
            if not embedding or len(embedding) != self.dimensions:
                logger.error(f"Invalid embedding dimensions: got {len(embedding) if embedding else 0}, expected {self.dimensions}")
                return None
                
            # No caching
                
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
            
    async def _apply_rate_limit(self):
        """Apply rate limiting to API calls"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
            
        self.last_request_time = asyncio.get_event_loop().time()
        
            
    async def generate_batch_embeddings(self, 
                                      texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts efficiently
        """
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = await self._generate_embedding(text)
                batch_embeddings.append(embedding)
                
            embeddings.extend(batch_embeddings)
            
            # Add delay between batches
            if i + self.embedding_batch_size < len(texts):
                await asyncio.sleep(0.1)
                
        return embeddings
        
    def validate_input(self, record: Dict[str, Any]) -> bool:
        """Validate that record has embeddable content"""
        if not super().validate_input(record):
            return False
            
        # Check if any embeddable fields exist (including nested paths)
        for field in self.fields_to_embed:
            value = self._get_nested_value(record, field)
            if value and str(value).strip():
                return True
                
        return False
        
    def validate_output(self, 
                       records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
        """Validate that records have valid embeddings"""
        valid_records = []
        errors = []
        
        for i, record in enumerate(records):
            # Check basic record validity
            if not isinstance(record, dict):
                errors.append(f"Record {i}: Not a dictionary")
                continue
                
            # Check embedding validity
            embedding = record.get("embedding")
            
            if embedding is None:
                # Allow records without embeddings but log warning
                valid_records.append(record)
                continue
                
            if not isinstance(embedding, list):
                errors.append(f"Record {i}: Embedding is not a list")
                continue
                
            if len(embedding) != self.dimensions:
                errors.append(f"Record {i}: Invalid embedding dimensions {len(embedding)}, expected {self.dimensions}")
                continue
                
            # Check if embedding values are valid floats
            try:
                for val in embedding:
                    if not isinstance(val, (int, float)) or abs(val) > 1.0:
                        raise ValueError("Invalid embedding value")
            except (TypeError, ValueError):
                errors.append(f"Record {i}: Invalid embedding values")
                continue
                
            valid_records.append(record)
            
        return valid_records, errors
        
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get detailed embedding statistics"""
        base_stats = await self.get_stats()
        
        embedding_specific = {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "cache_enabled": self.cache_enabled,
            "cache_hit_rate": 0.0,
            "fields_embedded": self.fields_to_embed,
            "max_text_length": self.max_text_length
        }
        
        total_requests = (self.transformation_stats["cache_hits"] + 
                         self.transformation_stats["cache_misses"])
        
        if total_requests > 0:
            embedding_specific["cache_hit_rate"] = (
                self.transformation_stats["cache_hits"] / total_requests
            )
            
        base_stats["embedding_stats"] = embedding_specific
        return base_stats