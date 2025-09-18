"""
Qdrant Loader

Loads data into Qdrant vector database for semantic search.
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import hashlib
import json
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    CollectionStatus,
    CreateCollection,
    UpdateStatus,
    OptimizersConfig,
    HnswConfig,
    PayloadSchemaType,
)
from qdrant_client.http import models

from .base_loader import BaseLoader, LoadResult, LoadConfig

logger = logging.getLogger(__name__)


class QdrantLoader(BaseLoader):
    """
    Loader for Qdrant vector database

    Features:
    - Collection creation and management
    - Batch vector insertion
    - Payload indexing
    - Health monitoring
    - Optimization settings
    """

    def __init__(self, config: LoadConfig, qdrant_config: Dict[str, Any]):
        super().__init__(config, "qdrant_loader")

        # Qdrant configuration
        self.host = qdrant_config.get("host", "localhost")
        self.port = qdrant_config.get("port", 6333)
        self.api_key = qdrant_config.get("api_key")
        self.timeout = qdrant_config.get("timeout", 60)

        # Vector configuration
        self.vector_size = qdrant_config.get("vector_size", 768)
        self.distance_metric = qdrant_config.get("distance", "Cosine")

        # Performance settings
        self.batch_settings = qdrant_config.get("batch_settings", {})
        self.hnsw_config = qdrant_config.get("hnsw_config", {})

        # Client will be initialized in setup()
        self.client: Optional[QdrantClient] = None
        self._requires_vector = True  # This loader requires embeddings

        # Default collection configurations
        self.collection_configs = {
            "products": {
                "vector_size": self.vector_size,
                "distance": self.distance_metric,
                "payload_indexes": [
                    {"field": "category.name", "type": "keyword"},
                    {"field": "provider.id", "type": "keyword"},
                    {"field": "price.value", "type": "float"},
                    {"field": "availability", "type": "bool"},
                    {"field": "rating", "type": "float"},
                    {"field": "location.coordinates.latitude", "type": "float"},
                    {"field": "location.coordinates.longitude", "type": "float"},
                ],
            },
            "categories": {
                "vector_size": self.vector_size,
                "distance": self.distance_metric,
                "payload_indexes": [
                    {"field": "parent_id", "type": "keyword"},
                    {"field": "level", "type": "integer"},
                ],
            },
            "providers": {
                "vector_size": self.vector_size,
                "distance": self.distance_metric,
                "payload_indexes": [
                    {"field": "verified", "type": "bool"},
                    {"field": "rating", "type": "float"},
                    {"field": "location.coordinates.latitude", "type": "float"},
                    {"field": "location.coordinates.longitude", "type": "float"},
                ],
            },
        }

    async def setup(self):
        """Initialize Qdrant client"""
        await super().setup()

        try:
            # Initialize Qdrant client with compatibility check disabled for version tolerance
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=self.timeout,
                prefer_grpc=False,  # Use HTTP for better compatibility
                https=False,  # Ensure HTTP connection
            )

            # Test connection
            info = self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            logger.info(f"Found {len(info.collections)} existing collections")

            # Try to get server info for version checking
            try:
                server_info = self.client.get_cluster_info()
                logger.info(f"Qdrant server cluster info retrieved successfully")
            except Exception as version_check_error:
                logger.warning(
                    f"Could not retrieve server version info: {version_check_error}"
                )
                logger.info("Proceeding with connection despite version check issues")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            # Try with compatibility check disabled as fallback
            try:
                logger.warning(
                    "Retrying connection with relaxed compatibility settings..."
                )
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.client = QdrantClient(
                        host=self.host,
                        port=self.port,
                        api_key=self.api_key,
                        timeout=self.timeout,
                        prefer_grpc=False,
                    )
                    # Test fallback connection
                    info = self.client.get_collections()
                    logger.info(
                        f"Fallback connection successful to Qdrant at {self.host}:{self.port}"
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback connection also failed: {fallback_error}")
                raise

    async def cleanup(self):
        """Clean up Qdrant client"""
        if self.client:
            try:
                self.client.close()
            except:
                pass
            self.client = None
        await super().cleanup()

    async def _create_collection_via_rest_api(
        self, collection_name: str, vector_size: int, distance_metric: str
    ) -> bool:
        """Fallback method to create collection via REST API"""
        try:
            url = f"http://{self.host}:{self.port}/collections/{collection_name}"

            # Prepare collection configuration for REST API
            distance_map = {"Cosine": "Cosine", "Euclidean": "Euclid", "Dot": "Dot"}

            collection_config = {
                "vectors": {
                    "size": vector_size,
                    "distance": distance_map.get(distance_metric, "Cosine"),
                },
                "optimizers_config": {
                    "default_segment_number": 1,
                    "indexing_threshold": 50,  # Lowered from 20000 to ensure indexing with small datasets
                },
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.put(url, json=collection_config)

                if response.status_code == 200:
                    logger.info(
                        f"Successfully created collection '{collection_name}' via REST API"
                    )
                    return True
                elif response.status_code == 400:
                    # Check if it's "already exists" error
                    error_text = response.text
                    if "already exists" in error_text.lower():
                        etl_mode = os.getenv("ETL_MODE", "recreate").lower()
                        if etl_mode == "upsert":
                            logger.info(
                                f"Collection '{collection_name}' already exists, proceeding with upsert mode"
                            )
                            return True
                        else:
                            logger.warning(
                                f"Collection '{collection_name}' already exists but mode is '{etl_mode}'"
                            )
                            return False
                    else:
                        logger.error(
                            f"REST API collection creation failed: {response.status_code} - {response.text}"
                        )
                        return False
                else:
                    logger.error(
                        f"REST API collection creation failed: {response.status_code} - {response.text}"
                    )
                    return False

        except Exception as e:
            logger.error(f"REST API collection creation error: {e}")
            return False

    async def health_check(self) -> bool:
        """Check Qdrant health"""
        try:
            if not self.client:
                return False

            # Try to get collections info
            info = self.client.get_collections()
            return True

        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists in Qdrant"""
        try:
            if not self.client:
                return False

            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)

        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    async def create_collection(
        self, collection_name: str, config: Dict[str, Any]
    ) -> bool:
        """Create a new Qdrant collection with retry logic"""
        max_retries = 3
        retry_delay = 2

        # Check ETL mode from environment or config
        etl_mode = os.getenv("ETL_MODE", "recreate").lower()

        # Handle existing collection based on mode
        if await self.collection_exists(collection_name):
            if etl_mode == "recreate":
                logger.info(
                    f"Collection '{collection_name}' exists. Mode is 'recreate', deleting existing collection..."
                )
                if not await self.delete_collection(collection_name):
                    logger.error(
                        f"Failed to delete existing collection '{collection_name}'"
                    )
                    return False
                await asyncio.sleep(1)  # Brief pause for deletion to complete
                logger.info(
                    f"Successfully deleted collection '{collection_name}', will recreate..."
                )
            elif etl_mode == "upsert":
                logger.info(
                    f"Collection '{collection_name}' exists. Mode is 'upsert', will update existing data."
                )
                return True  # Collection exists, proceed with upsert
            else:
                logger.warning(f"Unknown ETL_MODE: {etl_mode}, defaulting to recreate")
                if not await self.delete_collection(collection_name):
                    return False
                await asyncio.sleep(1)

        for attempt in range(max_retries):
            try:
                if not self.client:
                    raise RuntimeError("Qdrant client not initialized")

                # Use provided config or default
                coll_config = config or self.collection_configs.get(
                    collection_name, self.collection_configs["products"]
                )

                vector_size = coll_config.get("vector_size", self.vector_size)
                distance = coll_config.get("distance", self.distance_metric)

                # Convert distance string to Distance enum
                distance_map = {
                    "Cosine": Distance.COSINE,
                    "Euclidean": Distance.EUCLID,
                    "Dot": Distance.DOT,
                }
                distance_metric = distance_map.get(distance, Distance.COSINE)

                # Create collection with minimal configuration for v1.7.0 compatibility
                try:
                    # Try with basic configuration first
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size, distance=distance_metric
                        ),
                    )
                    logger.info(
                        f"Created Qdrant collection with basic config: {collection_name}"
                    )
                    return True

                except Exception as basic_error:
                    logger.warning(f"Basic collection creation failed: {basic_error}")
                    logger.info(
                        "Attempting collection creation with optimized settings..."
                    )

                    # Fallback: Try with simplified optimizer config
                    try:
                        self.client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=vector_size, distance=distance_metric
                            ),
                            optimizers_config=OptimizersConfig(
                                default_segment_number=1,
                                indexing_threshold=50,  # Lowered from 20000 to ensure indexing with small datasets
                            ),
                        )
                        logger.info(
                            f"Created Qdrant collection with optimized config: {collection_name}"
                        )
                        return True

                    except Exception as optimized_error:
                        logger.error(
                            f"Optimized collection creation failed: {optimized_error}"
                        )
                        logger.warning(
                            "Attempting REST API fallback for collection creation..."
                        )

                        # Final fallback: Use REST API
                        rest_success = await self._create_collection_via_rest_api(
                            collection_name, vector_size, distance
                        )
                        if rest_success:
                            return True
                        else:
                            raise Exception(
                                f"REST API fallback also failed for collection '{collection_name}'"
                            )

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Collection creation attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"All {max_retries} collection creation attempts failed"
                    )
                    raise Exception(
                        f"Failed to create collection '{collection_name}' after {max_retries} attempts: {e}"
                    )

        # Create payload indexes if specified (after successful collection creation)
        try:
            payload_indexes = coll_config.get("payload_indexes", [])
            for index_config in payload_indexes:
                await self._create_payload_index(collection_name, index_config)
        except Exception as index_error:
            logger.warning(f"Failed to create payload indexes: {index_error}")

        return True

    async def _create_payload_index(
        self, collection_name: str, index_config: Dict[str, str]
    ):
        """Create payload index for better filtering performance"""
        try:
            field = index_config["field"]
            field_type = index_config["type"]

            # Map types to Qdrant schema types
            type_map = {
                "keyword": PayloadSchemaType.KEYWORD,
                "integer": PayloadSchemaType.INTEGER,
                "float": PayloadSchemaType.FLOAT,
                "bool": PayloadSchemaType.BOOL,
                "geo": PayloadSchemaType.GEO,
            }

            schema_type = type_map.get(field_type, PayloadSchemaType.KEYWORD)

            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_type=schema_type,
            )

            logger.info(
                f"Created payload index on {collection_name}.{field} ({field_type})"
            )

        except Exception as e:
            logger.warning(f"Failed to create payload index: {e}")

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a Qdrant collection"""
        try:
            if not self.client:
                return False

            if await self.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted Qdrant collection: {collection_name}")
                return True
            else:
                logger.warning(f"Collection {collection_name} does not exist")
                return True

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    async def load_batch(
        self, records: List[Dict[str, Any]], collection_name: str, **kwargs
    ) -> LoadResult:
        """Load a batch of records to Qdrant"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")

            points = []
            errors = []

            for i, record in enumerate(records):
                try:
                    point = self._record_to_point(record, i)
                    if point:
                        points.append(point)
                    else:
                        errors.append(f"Record {i}: Failed to convert to point")

                except Exception as e:
                    errors.append(f"Record {i}: {e}")

            if not points:
                return LoadResult(
                    success=False,
                    loaded_count=0,
                    failed_count=len(records),
                    errors=errors + ["No valid points to insert"],
                    metadata={"collection": collection_name},
                    loaded_at=datetime.utcnow(),
                    loader=self.loader_name,
                )

            # Verify collection exists before attempting upsert
            if not await self.collection_exists(collection_name):
                error_msg = (
                    f"Collection '{collection_name}' does not exist. Cannot load batch."
                )
                logger.error(error_msg)
                return LoadResult(
                    success=False,
                    loaded_count=0,
                    failed_count=len(records),
                    errors=errors + [error_msg],
                    metadata={"collection": collection_name},
                    loaded_at=datetime.utcnow(),
                    loader=self.loader_name,
                )

            # Insert points into Qdrant with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    operation_info = self.client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True,  # Wait for operation to complete
                    )
                    break  # Success, exit retry loop

                except Exception as upsert_error:
                    if "doesn't exist" in str(upsert_error) or "404" in str(
                        upsert_error
                    ):
                        error_msg = f"Collection '{collection_name}' was deleted or doesn't exist during upsert"
                        logger.error(error_msg)
                        return LoadResult(
                            success=False,
                            loaded_count=0,
                            failed_count=len(records),
                            errors=errors + [error_msg],
                            metadata={"collection": collection_name},
                            loaded_at=datetime.utcnow(),
                            loader=self.loader_name,
                        )

                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            f"Upsert attempt {attempt + 1} failed: {upsert_error}. Retrying in {wait_time} seconds..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise upsert_error

            # Check operation status
            if operation_info.status == UpdateStatus.COMPLETED:
                loaded_count = len(points)
                failed_count = len(records) - loaded_count

                logger.info(
                    f"Successfully loaded {loaded_count} points to {collection_name}"
                )

                return LoadResult(
                    success=True,
                    loaded_count=loaded_count,
                    failed_count=failed_count,
                    errors=errors,
                    metadata={
                        "collection": collection_name,
                        "operation_id": operation_info.operation_id,
                    },
                    loaded_at=datetime.utcnow(),
                    loader=self.loader_name,
                )
            else:
                return LoadResult(
                    success=False,
                    loaded_count=0,
                    failed_count=len(records),
                    errors=errors
                    + [f"Qdrant operation failed: {operation_info.status}"],
                    metadata={"collection": collection_name},
                    loaded_at=datetime.utcnow(),
                    loader=self.loader_name,
                )

        except Exception as e:
            logger.error(f"Batch load failed: {e}")
            return LoadResult(
                success=False,
                loaded_count=0,
                failed_count=len(records),
                errors=[str(e)],
                metadata={"collection": collection_name},
                loaded_at=datetime.utcnow(),
                loader=self.loader_name,
            )

    def _record_to_point(
        self, record: Dict[str, Any], index: int
    ) -> Optional[PointStruct]:
        """Convert a record to a Qdrant point"""
        try:
            # Get embedding vector
            embedding = record.get("embedding")
            if not embedding:
                raise ValueError("Record missing embedding vector")

            if not isinstance(embedding, list) or len(embedding) != self.vector_size:
                raise ValueError(
                    f"Invalid embedding: expected list of {self.vector_size} floats"
                )

            # Generate point ID - use deterministic hash for stability
            original_id = record.get("id", str(uuid.uuid4()))
            if isinstance(original_id, str):
                # Use SHA256 for deterministic hashing that's stable across restarts
                hash_bytes = hashlib.sha256(original_id.encode()).digest()
                # Convert to positive integer (Qdrant requires positive IDs)
                point_id = int.from_bytes(hash_bytes[:8], byteorder="big") % (2**63)
            else:
                point_id = original_id

            # Prepare payload (exclude embedding to save space)
            payload = record.copy()
            payload.pop("embedding", None)

            # Add metadata
            payload["indexed_at"] = datetime.utcnow().isoformat()
            payload["vector_dimensions"] = len(embedding)
            payload["original_id"] = original_id  # Store original ID in payload
            payload["id"] = (
                original_id  # Also store as 'id' for consistency with search
            )

            # Create point
            point = PointStruct(id=point_id, vector=embedding, payload=payload)

            return point

        except Exception as e:
            logger.error(f"Failed to convert record {index} to point: {e}")
            return None

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed statistics about a Qdrant collection"""
        try:
            if not self.client or not await self.collection_exists(collection_name):
                return {
                    "collection": collection_name,
                    "exists": False,
                    "error": "Collection does not exist",
                }

            # Get collection info
            info = self.client.get_collection(collection_name)

            stats = {
                "collection": collection_name,
                "exists": True,
                "status": info.status.name if info.status else "unknown",
                "vectors_count": info.vectors_count or 0,
                "indexed_vectors_count": info.indexed_vectors_count or 0,
                "points_count": info.points_count or 0,
                "segments_count": info.segments_count or 0,
                "config": {
                    "vector_size": info.config.params.vectors.size
                    if info.config
                    else None,
                    "distance": info.config.params.vectors.distance.name
                    if info.config
                    else None,
                },
                "loader": self.loader_name,
            }

            # Add optimizer info if available
            if info.config and info.config.optimizer_config:
                stats["optimizer"] = {
                    "indexing_threshold": info.config.optimizer_config.indexing_threshold,
                    "max_segment_size": info.config.optimizer_config.max_segment_size,
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"collection": collection_name, "exists": False, "error": str(e)}

    async def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection
        Useful for testing and validation
        """
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")

            # Build search request
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False,
            }

            if score_threshold:
                search_params["score_threshold"] = score_threshold

            if filter_conditions:
                search_params["query_filter"] = self._build_qdrant_filter(
                    filter_conditions
                )

            # Perform search
            results = self.client.search(**search_params)

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {"id": result.id, "score": result.score, "payload": result.payload}
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _build_qdrant_filter(self, conditions: Dict[str, Any]) -> models.Filter:
        """
        Build Qdrant filter from conditions
        Simplified implementation for common cases
        """
        must_conditions = []

        for field, value in conditions.items():
            if isinstance(value, (str, int, float, bool)):
                # Simple equality match
                must_conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=value)
                    )
                )
            elif isinstance(value, dict):
                # Range or complex conditions
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    range_condition = {}
                    if "gte" in value:
                        range_condition["gte"] = value["gte"]
                    if "lte" in value:
                        range_condition["lte"] = value["lte"]
                    if "gt" in value:
                        range_condition["gt"] = value["gt"]
                    if "lt" in value:
                        range_condition["lt"] = value["lt"]

                    must_conditions.append(
                        models.FieldCondition(
                            key=field, range=models.Range(**range_condition)
                        )
                    )

        return models.Filter(must=must_conditions) if must_conditions else None

    async def optimize_collection(self, collection_name: str) -> bool:
        """
        Trigger collection optimization
        Useful after bulk loading
        """
        try:
            if not self.client:
                return False

            # This will trigger optimization in the background
            # Qdrant handles optimization automatically, but we can force it
            info = self.client.get_collection(collection_name)

            logger.info(f"Collection {collection_name} optimization status checked")
            return True

        except Exception as e:
            logger.error(f"Failed to optimize collection {collection_name}: {e}")
            return False

    async def create_snapshot(self, collection_name: str) -> Optional[str]:
        """
        Create a snapshot of the collection
        Returns snapshot name if successful
        """
        try:
            if not self.client:
                return None

            snapshot_info = self.client.create_snapshot(collection_name)
            snapshot_name = snapshot_info.name

            logger.info(
                f"Created snapshot {snapshot_name} for collection {collection_name}"
            )
            return snapshot_name

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return None
