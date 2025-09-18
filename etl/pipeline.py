"""
Main ETL Pipeline Orchestrator

Coordinates the extraction, transformation, and loading of ONDC catalog data
into the Qdrant vector database.
"""

import asyncio
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from etl.extractors import HimiraExtractor, ONDCExtractor, FileExtractor, ExtractionConfig
from etl.transformers import (
    ProductTransformer, EmbeddingGenerator, MetadataEnricher, 
    TransformationConfig
)
from etl.loaders import QdrantLoader, LoadConfig
from etl.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    Main ETL Pipeline coordinator
    
    Orchestrates the complete data flow from extraction to vector database loading.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(project_root / "config" / "etl_config.yaml")
        self.config = self._load_config()
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_extracted": 0,
            "total_transformed": 0,
            "total_loaded": 0,
            "errors": []
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            "etl": {
                "batch_size": 100,
                "max_workers": 4,
                "sources": {
                    "himira_api": {"enabled": True, "priority": 1},
                    "file_system": {"enabled": True, "priority": 3}
                }
            },
            "transformers": {
                "embeddings": {
                    "provider": "gemini",
                    "model": "models/text-embedding-004",  # Use text-embedding-004 for compatibility
                    "dimensions": 768
                }
            },
            "loaders": {
                "qdrant": {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", 6333)),
                    "collections": {
                        "products": {"name": "himira_products"}
                    }
                }
            }
        }
        
    async def run_full_pipeline(self, 
                               data_types: Optional[List[str]] = None,
                               max_records: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline
        
        Args:
            data_types: Types of data to process (products, categories, providers)
            max_records: Maximum number of records to process (for testing)
        """
        self.stats["start_time"] = datetime.utcnow()
        logger.info("Starting ETL pipeline execution")
        
        try:
            data_types = data_types or ["products"]
            
            for data_type in data_types:
                logger.info(f"Processing {data_type} data")
                
                # Extract data
                extracted_data = await self._extract_data(data_type, max_records)
                if not extracted_data.success:
                    logger.error(f"Extraction failed for {data_type}: {extracted_data.errors}")
                    self.stats["errors"].extend(extracted_data.errors)
                    continue
                    
                self.stats["total_extracted"] += extracted_data.total_records
                logger.info(f"Extracted {extracted_data.total_records} {data_type} records")
                
                # Transform data
                transformed_data = await self._transform_data(extracted_data.data)
                if not transformed_data.success:
                    logger.error(f"Transformation failed for {data_type}: {transformed_data.errors}")
                    self.stats["errors"].extend(transformed_data.errors)
                    continue
                    
                self.stats["total_transformed"] += transformed_data.output_records
                logger.info(f"Transformed {transformed_data.output_records} {data_type} records")
                
                # Load data
                collection_name = self._get_collection_name(data_type)
                loaded_data = await self._load_data(transformed_data.data, collection_name)
                if not loaded_data.success:
                    logger.error(f"Loading failed for {data_type}: {loaded_data.errors}")
                    self.stats["errors"].extend(loaded_data.errors)
                    continue
                    
                self.stats["total_loaded"] += loaded_data.loaded_count
                logger.info(f"Loaded {loaded_data.loaded_count} {data_type} records to {collection_name}")
                
            self.stats["end_time"] = datetime.utcnow()
            duration = self.stats["end_time"] - self.stats["start_time"]
            
            summary = {
                "success": self.stats["total_loaded"] > 0,
                "duration_seconds": duration.total_seconds(),
                "extracted": self.stats["total_extracted"],
                "transformed": self.stats["total_transformed"], 
                "loaded": self.stats["total_loaded"],
                "errors": self.stats["errors"]
            }
            
            logger.info(f"ETL pipeline completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.stats["errors"].append(str(e))
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats
            }
            
    async def _extract_data(self, data_type: str, max_records: Optional[int] = None):
        """Extract data from configured sources"""
        
        # Determine which extractor to use
        himira_enabled = self.config["etl"]["sources"]["himira_api"]["enabled"]
        file_enabled = self.config["etl"]["sources"]["file_system"]["enabled"]
        
        extraction_config = ExtractionConfig(
            batch_size=self.config["etl"]["batch_size"],
            max_workers=self.config["etl"]["max_workers"]
        )
        
        # Try Himira API first (primary source)
        if himira_enabled:
            api_config = {
                "base_url": os.getenv("HIMIRA_BACKEND_URL"),
                "api_key": os.getenv("HIMIRA_API_KEY"),
                "user_id": os.getenv("HIMIRA_USER_ID", "guestUser"),
                "device_id": os.getenv("HIMIRA_DEVICE_ID", "etl_pipeline_001")
            }
            
            async with HimiraExtractor(extraction_config, api_config) as extractor:
                if await extractor.health_check():
                    if data_type == "products":
                        kwargs = {}
                        if max_records:
                            # Calculate pages needed
                            limit = 100
                            max_pages = max(1, (max_records + limit - 1) // limit)
                            kwargs = {"limit": limit, "max_pages": max_pages}
                        return await extractor.extract_products(**kwargs)
                    elif data_type == "categories":
                        return await extractor.extract_categories()
                    elif data_type == "providers":
                        return await extractor.extract_providers()
                        
        # Fallback to file system
        if file_enabled:
            file_config = {
                "data_path": os.getenv("DATA_DIR", str(project_root / "data")),
                "formats": ["json", "csv", "xlsx"]
            }
            
            async with FileExtractor(extraction_config, file_config) as extractor:
                if data_type == "products":
                    return await extractor.extract_products()
                elif data_type == "categories":
                    return await extractor.extract_categories()
                elif data_type == "providers":
                    return await extractor.extract_providers()
                    
        # If we get here, no extractors worked
        from etl.extractors.base_extractor import ExtractionResult
        return ExtractionResult(
            success=False,
            data=[],
            errors=["No working extractors available"],
            metadata={},
            extracted_at=datetime.utcnow(),
            source="pipeline",
            total_records=0
        )
        
    async def _transform_data(self, records: List[Dict[str, Any]]):
        """Transform extracted data - preserve original Himira structure, only add embeddings"""
        
        transformation_config = TransformationConfig(
            batch_size=50,
            max_workers=4
        )
        
        # Only Step: Generate embeddings while preserving original structure
        # Get embedding config from loaded config or use defaults
        embedding_config = self.config.get("transformers", {}).get("embeddings", {})
        ai_config = {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": embedding_config.get("model", "models/text-embedding-004"),  # Use text-embedding-004
            "dimensions": embedding_config.get("dimensions", 768),
            # Update fields to match extractor's comprehensive structure
            "fields_to_embed": ["name", "description", "category", "tags", "brand", "provider", "location"]
        }
        
        async with EmbeddingGenerator(transformation_config, ai_config) as transformer:
            embedding_result = await transformer.transform_batch(records)  # Pass original records directly
            
            if not embedding_result.success:
                return embedding_result
                
            return embedding_result
            
    async def _load_data(self, records: List[Dict[str, Any]], collection_name: str):
        """Load transformed data into Qdrant"""
        
        load_config = LoadConfig(
            batch_size=100,
            max_workers=2,
            create_collections=True
        )
        
        qdrant_config = {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", 6333)),
            "vector_size": 768,
            "distance": "Cosine"
        }
        
        async with QdrantLoader(load_config, qdrant_config) as loader:
            # Check health
            if not await loader.health_check():
                from etl.loaders.base_loader import LoadResult
                return LoadResult(
                    success=False,
                    loaded_count=0,
                    failed_count=len(records),
                    errors=["Qdrant health check failed"],
                    metadata={},
                    loaded_at=datetime.utcnow(),
                    loader="qdrant_loader"
                )
                
            # Load the data - pass collection_config to enable auto-creation
            result = await loader.load_records(records, collection_name, collection_config={})
            
            # Optimize collection to trigger index building
            # This ensures vectors are indexed immediately, not waiting for threshold
            if result.success and result.loaded_count > 0:
                logger.info(f"Optimizing collection {collection_name} to trigger indexing...")
                await loader.optimize_collection(collection_name)
                logger.info(f"Collection {collection_name} optimization triggered")
            
            return result
            
    def _get_collection_name(self, data_type: str) -> str:
        """Get collection name for data type"""
        collection_mapping = {
            "products": os.getenv("QDRANT_COLLECTION", "himira_products"),
            "categories": "himira_categories", 
            "providers": "himira_providers"
        }
        return collection_mapping.get(data_type, f"himira_{data_type}")
        
    async def test_pipeline(self) -> Dict[str, Any]:
        """Run a small test of the pipeline"""
        logger.info("Running pipeline test with limited data")
        
        return await self.run_full_pipeline(
            data_types=["products"],
            max_records=50  # Small test batch
        )
        
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all pipeline components"""
        logger.info("Running pipeline health check")
        
        results = {}
        
        # Check Himira API
        try:
            api_config = {
                "base_url": os.getenv("HIMIRA_BACKEND_URL"),
                "api_key": os.getenv("HIMIRA_API_KEY"),
                "user_id": "guestUser",
                "device_id": "health_check"
            }
            
            extraction_config = ExtractionConfig()
            async with HimiraExtractor(extraction_config, api_config) as extractor:
                results["himira_api"] = await extractor.health_check()
        except Exception as e:
            results["himira_api"] = False
            logger.error(f"Himira API health check failed: {e}")
            
        # Check Qdrant
        try:
            qdrant_config = {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", 6333)),
                "vector_size": 768
            }
            
            load_config = LoadConfig()
            async with QdrantLoader(load_config, qdrant_config) as loader:
                results["qdrant"] = await loader.health_check()
        except Exception as e:
            results["qdrant"] = False
            logger.error(f"Qdrant health check failed: {e}")
            
        # Check Gemini API
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Try a simple embedding
            result = genai.embed_content(
                model="models/text-embedding-004",
                content="test",
                task_type="retrieval_document"
            )
            results["gemini_api"] = len(result['embedding']) == 768
        except Exception as e:
            results["gemini_api"] = False
            logger.error(f"Gemini API health check failed: {e}")
            
        results["overall"] = all(results.values())
        logger.info(f"Health check results: {results}")
        
        return results


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ONDC Vector Database ETL Pipeline")
    parser.add_argument("--action", choices=["test", "health", "full"], default="test",
                       help="Action to perform")
    parser.add_argument("--data-types", nargs="+", default=["products"],
                       help="Data types to process")
    parser.add_argument("--max-records", type=int, help="Maximum records to process")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Create pipeline
    pipeline = ETLPipeline(args.config)
    
    try:
        if args.action == "health":
            result = await pipeline.health_check()
            logger.info("Health Check Results:")
            for component, status in result.items():
                status_str = "OK" if status else "FAILED"
                logger.info(f"{component}: {status_str}")
                
        elif args.action == "test":
            result = await pipeline.test_pipeline()
            logger.info("Test Pipeline Results:")
            logger.info(f"Success: {result.get('success')}")
            logger.info(f"Duration: {result.get('duration_seconds', 0):.2f}s")
            logger.info(f"Extracted: {result.get('extracted', 0)}")
            logger.info(f"Transformed: {result.get('transformed', 0)}")
            logger.info(f"Loaded: {result.get('loaded', 0)}")
            if result.get('errors'):
                logger.warning(f"Errors: {len(result['errors'])}")
                
        elif args.action == "full":
            result = await pipeline.run_full_pipeline(
                data_types=args.data_types,
                max_records=args.max_records
            )
            logger.info("Full Pipeline Results:")
            logger.info(f"Success: {result.get('success')}")
            logger.info(f"Duration: {result.get('duration_seconds', 0):.2f}s")
            logger.info(f"Extracted: {result.get('extracted', 0)}")
            logger.info(f"Transformed: {result.get('transformed', 0)}")
            logger.info(f"Loaded: {result.get('loaded', 0)}")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())