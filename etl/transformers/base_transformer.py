"""
Base Transformer Class

Defines the interface and common functionality for all data transformers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """Result of a transformation operation"""
    success: bool
    data: List[Dict[str, Any]]
    errors: List[str]
    metadata: Dict[str, Any]
    transformed_at: datetime
    transformer: str
    input_records: int
    output_records: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "metadata": self.metadata,
            "transformed_at": self.transformed_at.isoformat(),
            "transformer": self.transformer,
            "input_records": self.input_records,
            "output_records": self.output_records
        }


@dataclass
class TransformationConfig:
    """Configuration for transformation operations"""
    batch_size: int = 50
    max_workers: int = 4
    timeout_seconds: int = 60
    retry_attempts: int = 3
    enable_caching: bool = True
    cache_ttl: int = 3600
    skip_on_error: bool = True
    validate_output: bool = True


class BaseTransformer(ABC):
    """
    Abstract base class for all data transformers.
    
    Provides common functionality like batching, error handling,
    and validation patterns.
    """
    
    def __init__(self, config: TransformationConfig, transformer_name: str):
        self.config = config
        self.transformer_name = transformer_name
        self.transformation_stats = {
            "total_processed": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "start_time": None,
            "end_time": None,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.setup()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        
    async def setup(self):
        """Initialize transformer resources"""
        self.transformation_stats["start_time"] = datetime.utcnow()
        logger.info(f"Initialized {self.transformer_name} transformer")
        
    async def cleanup(self):
        """Clean up transformer resources"""
        self.transformation_stats["end_time"] = datetime.utcnow()
        logger.info(f"Cleaned up {self.transformer_name} transformer")
        
    @abstractmethod
    async def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single record"""
        pass
        
    async def transform_batch(self, 
                             records: List[Dict[str, Any]], 
                             **kwargs) -> TransformationResult:
        """
        Transform a batch of records
        """
        try:
            logger.info(f"Transforming batch of {len(records)} records with {self.transformer_name}")
            
            transformed_records = []
            errors = []
            
            # Process records in parallel with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.config.max_workers)
            
            async def process_record(record, index):
                async with semaphore:
                    try:
                        self.transformation_stats["total_processed"] += 1
                        transformed = await self.transform_record(record)
                        
                        if transformed:
                            self.transformation_stats["successful_transformations"] += 1
                            return transformed, None
                        else:
                            error = f"Record {index}: Transformation returned None"
                            return None, error
                            
                    except Exception as e:
                        self.transformation_stats["failed_transformations"] += 1
                        error = f"Record {index}: {str(e)}"
                        logger.warning(error)
                        return None, error
                        
            # Execute transformations
            tasks = [
                process_record(record, i) 
                for i, record in enumerate(records)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                elif isinstance(result, tuple):
                    transformed_record, error = result
                    if transformed_record:
                        transformed_records.append(transformed_record)
                    if error:
                        errors.append(error)
                        
            # Validate output if enabled
            if self.config.validate_output:
                validated_records, validation_errors = self.validate_output(transformed_records)
                transformed_records = validated_records
                errors.extend(validation_errors)
                
            return TransformationResult(
                success=len(transformed_records) > 0,
                data=transformed_records,
                errors=errors,
                metadata={
                    "batch_size": len(records),
                    "transformer_config": {
                        "batch_size": self.config.batch_size,
                        "max_workers": self.config.max_workers
                    }
                },
                transformed_at=datetime.utcnow(),
                transformer=self.transformer_name,
                input_records=len(records),
                output_records=len(transformed_records)
            )
            
        except Exception as e:
            logger.error(f"Batch transformation failed for {self.transformer_name}: {e}")
            return TransformationResult(
                success=False,
                data=[],
                errors=[str(e)],
                metadata={},
                transformed_at=datetime.utcnow(),
                transformer=self.transformer_name,
                input_records=len(records),
                output_records=0
            )
            
    async def transform_stream(self, 
                              records: List[Dict[str, Any]], 
                              **kwargs):
        """
        Transform records in streaming fashion for large datasets
        """
        batch_size = kwargs.get("batch_size", self.config.batch_size)
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            result = await self.transform_batch(batch, **kwargs)
            yield result
            
    def validate_input(self, record: Dict[str, Any]) -> bool:
        """
        Validate input record before transformation
        Override in subclasses for specific validation
        """
        return isinstance(record, dict) and len(record) > 0
        
    def validate_output(self, 
                       records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate transformed records
        Override in subclasses for specific validation
        """
        valid_records = []
        errors = []
        
        for i, record in enumerate(records):
            if isinstance(record, dict) and len(record) > 0:
                valid_records.append(record)
            else:
                errors.append(f"Invalid output record at index {i}")
                
        return valid_records, errors
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        stats = self.transformation_stats.copy()
        
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            stats["duration_seconds"] = duration.total_seconds()
            
            if stats["total_processed"] > 0:
                stats["records_per_second"] = stats["total_processed"] / duration.total_seconds()
                stats["success_rate"] = stats["successful_transformations"] / stats["total_processed"]
                
        return {
            "transformer": self.transformer_name,
            "statistics": stats,
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
        
    def normalize_text(self, text: Optional[str]) -> str:
        """
        Normalize text fields for consistent processing
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Basic text normalization
        text = text.strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def normalize_price(self, price_data: Any) -> Dict[str, Any]:
        """
        Normalize price information to consistent format
        """
        if isinstance(price_data, (int, float)):
            return {
                "value": float(price_data),
                "currency": "INR",
                "formatted": f"₹{price_data}"
            }
        elif isinstance(price_data, dict):
            return {
                "value": float(price_data.get("value", 0)),
                "currency": price_data.get("currency", "INR"),
                "formatted": price_data.get("formatted", f"₹{price_data.get('value', 0)}")
            }
        else:
            return {
                "value": 0.0,
                "currency": "INR",
                "formatted": "₹0"
            }
            
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text for search enhancement
        Simple implementation - can be enhanced with NLP
        """
        if not text:
            return []
            
        # Simple keyword extraction
        import re
        
        # Remove special characters and split
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency and take most common
        from collections import Counter
        word_freq = Counter(keywords)
        
        return [word for word, _ in word_freq.most_common(max_keywords)]