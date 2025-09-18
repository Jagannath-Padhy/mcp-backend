"""
Base Loader Class

Defines the interface and common functionality for all data loaders.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of a load operation"""
    success: bool
    loaded_count: int
    failed_count: int
    errors: List[str]
    metadata: Dict[str, Any]
    loaded_at: datetime
    loader: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "loaded_count": self.loaded_count,
            "failed_count": self.failed_count,
            "errors": self.errors,
            "metadata": self.metadata,
            "loaded_at": self.loaded_at.isoformat(),
            "loader": self.loader
        }


@dataclass
class LoadConfig:
    """Configuration for load operations"""
    batch_size: int = 100
    max_workers: int = 4
    timeout_seconds: int = 60
    retry_attempts: int = 3
    create_collections: bool = True
    overwrite_existing: bool = False
    validate_before_load: bool = True


class BaseLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    Provides common functionality like batching, error handling,
    and validation patterns.
    """
    
    def __init__(self, config: LoadConfig, loader_name: str):
        self.config = config
        self.loader_name = loader_name
        self.load_stats = {
            "total_attempted": 0,
            "total_loaded": 0,
            "total_failed": 0,
            "batches_processed": 0,
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
        """Initialize loader resources"""
        self.load_stats["start_time"] = datetime.utcnow()
        logger.info(f"Initialized {self.loader_name} loader")
        
    async def cleanup(self):
        """Clean up loader resources"""
        self.load_stats["end_time"] = datetime.utcnow()
        logger.info(f"Cleaned up {self.loader_name} loader")
        
    @abstractmethod
    async def load_batch(self, 
                        records: List[Dict[str, Any]], 
                        collection_name: str,
                        **kwargs) -> LoadResult:
        """Load a batch of records to the target system"""
        pass
        
    @abstractmethod
    async def create_collection(self, 
                               collection_name: str,
                               config: Dict[str, Any]) -> bool:
        """Create a new collection/table/index"""
        pass
        
    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        pass
        
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        pass
        
    async def load_records(self, 
                          records: List[Dict[str, Any]], 
                          collection_name: str,
                          **kwargs) -> LoadResult:
        """
        Load records in batches
        """
        try:
            logger.info(f"Loading {len(records)} records to {collection_name}")
            
            # Validate records if enabled
            if self.config.validate_before_load:
                valid_records, validation_errors = self.validate_records(records)
                if validation_errors:
                    logger.warning(f"Found {len(validation_errors)} validation errors")
            else:
                valid_records = records
                validation_errors = []
                
            if not valid_records:
                return LoadResult(
                    success=False,
                    loaded_count=0,
                    failed_count=len(records),
                    errors=validation_errors + ["No valid records to load"],
                    metadata={},
                    loaded_at=datetime.utcnow(),
                    loader=self.loader_name
                )
                
            # Create collection if needed
            if self.config.create_collections:
                logger.info(f"Checking if collection '{collection_name}' exists...")
                collection_exists = await self.collection_exists(collection_name)
                
                if collection_exists:
                    logger.info(f"Collection '{collection_name}' already exists")
                    # Always call create_collection - it will handle recreate/upsert logic
                    collection_config = kwargs.get("collection_config", {})
                    success = await self.create_collection(collection_name, collection_config)
                    if not success:
                        logger.error(f"Failed to handle existing collection '{collection_name}'")
                else:
                    logger.info(f"Collection '{collection_name}' does not exist, creating it...")
                    collection_config = kwargs.get("collection_config", {})
                    success = await self.create_collection(collection_name, collection_config)
                    if success:
                        logger.info(f"Successfully created collection '{collection_name}'")
                    else:
                        logger.error(f"Failed to create collection '{collection_name}'")
                    
            # Load in batches
            total_loaded = 0
            total_failed = 0
            all_errors = validation_errors.copy()
            
            batch_size = kwargs.get("batch_size", self.config.batch_size)
            
            for i in range(0, len(valid_records), batch_size):
                batch = valid_records[i:i + batch_size]
                
                try:
                    batch_result = await self.load_batch(
                        batch, 
                        collection_name,
                        **kwargs
                    )
                    
                    total_loaded += batch_result.loaded_count
                    total_failed += batch_result.failed_count
                    all_errors.extend(batch_result.errors)
                    
                    self.load_stats["batches_processed"] += 1
                    
                    logger.info(f"Loaded batch {i//batch_size + 1}: "
                              f"{batch_result.loaded_count} success, "
                              f"{batch_result.failed_count} failed")
                    
                except Exception as e:
                    error_msg = f"Batch {i//batch_size + 1} failed: {e}"
                    logger.error(error_msg)
                    all_errors.append(error_msg)
                    total_failed += len(batch)
                    
            # Update stats
            self.load_stats["total_attempted"] += len(records)
            self.load_stats["total_loaded"] += total_loaded
            self.load_stats["total_failed"] += total_failed
            
            return LoadResult(
                success=total_loaded > 0,
                loaded_count=total_loaded,
                failed_count=total_failed,
                errors=all_errors,
                metadata={
                    "collection": collection_name,
                    "batches_processed": (len(valid_records) + batch_size - 1) // batch_size,
                    "validation_errors": len(validation_errors)
                },
                loaded_at=datetime.utcnow(),
                loader=self.loader_name
            )
            
        except Exception as e:
            logger.error(f"Load operation failed: {e}")
            return LoadResult(
                success=False,
                loaded_count=0,
                failed_count=len(records),
                errors=[str(e)],
                metadata={"collection": collection_name},
                loaded_at=datetime.utcnow(),
                loader=self.loader_name
            )
            
    def validate_records(self, 
                        records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate records before loading
        Override in subclasses for specific validation
        """
        valid_records = []
        errors = []
        
        for i, record in enumerate(records):
            try:
                # Basic validation
                if not isinstance(record, dict):
                    errors.append(f"Record {i}: Not a dictionary")
                    continue
                    
                if not record.get("id"):
                    errors.append(f"Record {i}: Missing required 'id' field")
                    continue
                    
                # Check for required vector field (if this is a vector database)
                if hasattr(self, '_requires_vector') and self._requires_vector:
                    if "embedding" not in record or not record["embedding"]:
                        errors.append(f"Record {i}: Missing required 'embedding' field")
                        continue
                        
                valid_records.append(record)
                
            except Exception as e:
                errors.append(f"Record {i}: Validation error - {e}")
                
        return valid_records, errors
        
    async def load_with_retry(self, 
                            records: List[Dict[str, Any]], 
                            collection_name: str,
                            **kwargs) -> LoadResult:
        """
        Load records with retry logic
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                result = await self.load_records(records, collection_name, **kwargs)
                
                if result.success or attempt == self.config.retry_attempts - 1:
                    return result
                    
                last_error = result.errors
                
            except Exception as e:
                last_error = [str(e)]
                logger.warning(f"Load attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    
        # All attempts failed
        return LoadResult(
            success=False,
            loaded_count=0,
            failed_count=len(records),
            errors=last_error or ["All retry attempts failed"],
            metadata={
                "attempts": self.config.retry_attempts,
                "collection": collection_name
            },
            loaded_at=datetime.utcnow(),
            loader=self.loader_name
        )
        
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection
        Override in subclasses for specific implementation
        """
        return {
            "collection": collection_name,
            "exists": await self.collection_exists(collection_name),
            "loader": self.loader_name
        }
        
    async def get_load_stats(self) -> Dict[str, Any]:
        """Get loading statistics"""
        stats = self.load_stats.copy()
        
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            stats["duration_seconds"] = duration.total_seconds()
            
            if stats["total_attempted"] > 0:
                stats["success_rate"] = stats["total_loaded"] / stats["total_attempted"]
                
            if duration.total_seconds() > 0:
                stats["records_per_second"] = stats["total_loaded"] / duration.total_seconds()
                
        return {
            "loader": self.loader_name,
            "statistics": stats,
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
        
    async def health_check(self) -> bool:
        """
        Check if the loader's target system is healthy
        Override in subclasses for specific health checks
        """
        try:
            return True
        except Exception as e:
            logger.error(f"Health check failed for {self.loader_name}: {e}")
            return False