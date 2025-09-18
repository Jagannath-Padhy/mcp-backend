"""
Base Extractor Class

Defines the interface and common functionality for all data extractors.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of an extraction operation"""
    success: bool
    data: List[Dict[str, Any]]
    errors: List[str]
    metadata: Dict[str, Any]
    extracted_at: datetime
    source: str
    total_records: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "metadata": self.metadata,
            "extracted_at": self.extracted_at.isoformat(),
            "source": self.source,
            "total_records": self.total_records
        }


@dataclass
class ExtractionConfig:
    """Configuration for extraction operations"""
    batch_size: int = 100
    max_workers: int = 4
    timeout_seconds: int = 30
    retry_attempts: int = 3
    rate_limit_rps: int = 10
    enable_caching: bool = True
    cache_ttl: int = 3600


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, rate_per_second: int = 10):
        self.rate_per_second = rate_per_second
        self.last_called = 0.0
        
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = asyncio.get_event_loop().time()
        time_since_last = now - self.last_called
        min_interval = 1.0 / self.rate_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
            
        self.last_called = asyncio.get_event_loop().time()


class BaseExtractor(ABC):
    """
    Abstract base class for all data extractors.
    
    Provides common functionality like rate limiting, error handling,
    and async processing patterns.
    """
    
    def __init__(self, config: ExtractionConfig, source_name: str):
        self.config = config
        self.source_name = source_name
        self.rate_limiter = RateLimiter(config.rate_limit_rps)
        self.session: Optional[aiohttp.ClientSession] = None
        self.extraction_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_records": 0,
            "start_time": None,
            "end_time": None
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.setup()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        
    async def setup(self):
        """Initialize resources (HTTP session, connections, etc.)"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            connector = aiohttp.TCPConnector(limit=self.config.max_workers)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        self.extraction_stats["start_time"] = datetime.utcnow()
        logger.info(f"Initialized {self.source_name} extractor")
        
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self.extraction_stats["end_time"] = datetime.utcnow()
        logger.info(f"Cleaned up {self.source_name} extractor")
        
    @abstractmethod
    async def extract_products(self, **kwargs) -> ExtractionResult:
        """Extract product data from the source"""
        pass
        
    @abstractmethod
    async def extract_categories(self, **kwargs) -> ExtractionResult:
        """Extract category data from the source"""
        pass
        
    @abstractmethod
    async def extract_providers(self, **kwargs) -> ExtractionResult:
        """Extract provider data from the source"""
        pass
        
    async def health_check(self) -> bool:
        """Check if the data source is healthy and accessible"""
        try:
            # Override in subclasses with source-specific health checks
            return True
        except Exception as e:
            logger.error(f"Health check failed for {self.source_name}: {e}")
            return False
            
    async def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about available data"""
        return {
            "source": self.source_name,
            "last_updated": datetime.utcnow().isoformat(),
            "extraction_stats": self.extraction_stats
        }
        
    async def extract_with_retry(self, 
                                extract_func,
                                *args, 
                                **kwargs) -> ExtractionResult:
        """
        Execute extraction function with retry logic
        """
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                await self.rate_limiter.acquire()
                self.extraction_stats["total_requests"] += 1
                
                result = await extract_func(*args, **kwargs)
                
                if result.success:
                    self.extraction_stats["successful_requests"] += 1
                    self.extraction_stats["total_records"] += result.total_records
                    return result
                else:
                    self.extraction_stats["failed_requests"] += 1
                    
            except Exception as e:
                last_exception = e
                self.extraction_stats["failed_requests"] += 1
                logger.warning(
                    f"Extraction attempt {attempt + 1} failed for {self.source_name}: {e}"
                )
                
                if attempt < self.config.retry_attempts - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    
        # All attempts failed
        error_msg = f"All {self.config.retry_attempts} attempts failed"
        if last_exception:
            error_msg += f": {last_exception}"
            
        return ExtractionResult(
            success=False,
            data=[],
            errors=[error_msg],
            metadata={"attempts": self.config.retry_attempts},
            extracted_at=datetime.utcnow(),
            source=self.source_name,
            total_records=0
        )
        
    async def extract_batch(self, 
                          items: List[Any], 
                          extract_func) -> List[ExtractionResult]:
        """
        Process a batch of items using semaphore for concurrency control
        """
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_item(item):
            async with semaphore:
                return await self.extract_with_retry(extract_func, item)
                
        tasks = [process_item(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    def validate_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate extracted data and return valid records + errors
        """
        valid_records = []
        errors = []
        
        for i, record in enumerate(data):
            try:
                # Basic validation - override in subclasses for specific validation
                if not isinstance(record, dict):
                    errors.append(f"Record {i}: Not a dictionary")
                    continue
                    
                if not record.get("id"):
                    errors.append(f"Record {i}: Missing required 'id' field")
                    continue
                    
                valid_records.append(record)
                
            except Exception as e:
                errors.append(f"Record {i}: Validation error - {e}")
                
        return valid_records, errors
        
    async def stream_extract(self, 
                           data_type: str, 
                           **kwargs) -> AsyncGenerator[ExtractionResult, None]:
        """
        Stream extraction results for large datasets
        """
        if data_type == "products":
            extract_func = self.extract_products
        elif data_type == "categories":
            extract_func = self.extract_categories
        elif data_type == "providers":
            extract_func = self.extract_providers
        else:
            raise ValueError(f"Unknown data type: {data_type}")
            
        # Yield results in batches
        result = await extract_func(**kwargs)
        
        # Split large results into smaller chunks
        chunk_size = self.config.batch_size
        data = result.data
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_result = ExtractionResult(
                success=result.success,
                data=chunk,
                errors=result.errors,
                metadata={**result.metadata, "chunk_index": i // chunk_size},
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=len(chunk)
            )
            yield chunk_result