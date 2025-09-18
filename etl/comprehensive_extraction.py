"""
Comprehensive Catalog Extraction Strategy

Uses the same search patterns that work in the MCP server to ensure 
complete catalog coverage in the vector database.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from etl.extractors import HimiraExtractor, ExtractionConfig

logger = logging.getLogger(__name__)

class ComprehensiveCatalogExtractor:
    """
    Orchestrates comprehensive catalog extraction using proven search patterns
    """
    
    def __init__(self):
        self.successful_searches: Dict[str, int] = {}
        self.failed_searches: List[str] = []
        self.total_products: Set[str] = set()  # Track unique product IDs
        
        # Search terms that we know work from MCP testing
        self.proven_search_terms = [
            # Food & Beverages (known to work)
            "jam", "kiwi", "rice", "bread", "fruits", "vegetables",
            "food", "grocery", "snacks", "beverages",
            
            # Electronics (common categories)
            "phone", "laptop", "mobile", "electronics", "headset", 
            "speaker", "computer", "tablet", "camera",
            
            # Fashion & Apparel
            "shirt", "tshirt", "jeans", "shoes", "clothing", 
            "fashion", "apparel", "dress", "jacket", "trousers",
            
            # Home & Living
            "furniture", "home", "decor", "kitchen", "appliances",
            "bedding", "lighting", "storage",
            
            # Health & Beauty
            "health", "beauty", "cosmetics", "medicine", "skincare",
            "haircare", "supplements", "fitness",
            
            # Generic/Broad terms
            "product", "item", "sale", "offer", "new", "popular"
        ]
        
        # Category-based searches
        self.category_searches = [
            {"name": "electronics", "expected_min": 10},
            {"name": "food", "expected_min": 20}, 
            {"name": "clothing", "expected_min": 15},
            {"name": "home", "expected_min": 10}
        ]
    
    async def run_comprehensive_extraction(self, 
                                         max_products: int = None,
                                         dry_run: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive catalog extraction
        
        Args:
            max_products: Maximum products to extract (for testing)
            dry_run: If True, only test searches without saving to vector DB
            
        Returns:
            Extraction statistics and results
        """
        stats = {
            "start_time": datetime.utcnow().isoformat(),
            "searches_attempted": 0,
            "searches_successful": 0,
            "total_products_found": 0,
            "unique_products": 0,
            "search_results": {},
            "errors": []
        }
        
        # Initialize extractor with correct config
        api_config = {
            "base_url": os.getenv("BACKEND_ENDPOINT", "https://hp-buyer-backend-preprod.himira.co.in/clientApis"),
            "api_key": os.getenv("WIL_API_KEY", ""),
            "user_id": os.getenv("ETL_USER_ID", "guestUser"),
            "device_id": os.getenv("ETL_DEVICE_ID", "etl_pipeline_001")
        }
        
        extraction_config = ExtractionConfig(
            batch_size=int(os.getenv("ETL_BATCH_SIZE", "50")),
            timeout_seconds=int(os.getenv("ETL_TIMEOUT_SECONDS", "300"))
        )
        
        extractor = HimiraExtractor(extraction_config, api_config)
        
        try:
            await extractor.setup()
            
            # Test basic connectivity first
            logger.info("Testing API connectivity...")
            health_ok = await extractor.health_check()
            if not health_ok:
                raise Exception("Health check failed - cannot connect to API")
            
            logger.info(" API connectivity confirmed")
            
            # Phase 1: Test all search terms
            logger.info(f"Phase 1: Testing {len(self.proven_search_terms)} proven search terms")
            
            for search_term in self.proven_search_terms:
                try:
                    stats["searches_attempted"] += 1
                    
                    # Extract with this search term
                    result = await extractor.extract_products(
                        query=search_term,
                        limit=20,  # Start with smaller batches
                        max_pages=3 if not dry_run else 1
                    )
                    
                    if result.success and result.total_records > 0:
                        stats["searches_successful"] += 1
                        stats["total_products_found"] += result.total_records
                        stats["search_results"][search_term] = result.total_records
                        
                        # Track unique products
                        for product in result.data:
                            if product.get("id"):
                                self.total_products.add(product["id"])
                        
                        logger.info(f" '{search_term}': {result.total_records} products")
                    else:
                        logger.warning(f" '{search_term}': No products found")
                        self.failed_searches.append(search_term)
                        
                    # Add delay to be respectful to API
                    await asyncio.sleep(float(os.getenv("SEARCH_DELAY_SECONDS", "0.5")))
                    
                    # Stop if we hit max_products limit
                    if max_products and len(self.total_products) >= max_products:
                        logger.info(f"Reached max_products limit: {max_products}")
                        break
                        
                except Exception as e:
                    error_msg = f"Error searching '{search_term}': {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    self.failed_searches.append(search_term)
            
            # Phase 2: Category-based extraction for high-value categories
            if not dry_run:
                logger.info("Phase 2: Category-based deep extraction")
                
                for category in self.category_searches:
                    if max_products and len(self.total_products) >= max_products:
                        break
                        
                    try:
                        result = await extractor.extract_products(
                            query=category["name"],
                            limit=50,
                            max_pages=10  # Deep extraction for categories
                        )
                        
                        if result.success:
                            new_products = 0
                            for product in result.data:
                                if product.get("id") and product["id"] not in self.total_products:
                                    self.total_products.add(product["id"])
                                    new_products += 1
                            
                            logger.info(f"Category '{category['name']}': {new_products} new products")
                            
                    except Exception as e:
                        error_msg = f"Category extraction error for '{category['name']}': {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
            
            # Final statistics
            stats["unique_products"] = len(self.total_products)
            stats["end_time"] = datetime.utcnow().isoformat()
            
            # Success/failure analysis
            success_rate = (stats["searches_successful"] / stats["searches_attempted"]) * 100
            logger.info(f"""
 Extraction Complete:
   Searches Attempted: {stats['searches_attempted']}
   Searches Successful: {stats['searches_successful']} ({success_rate:.1f}%)
   Total Products Found: {stats['total_products_found']}
   Unique Products: {stats['unique_products']}
   Errors: {len(stats['errors'])}
            """)
            
            return stats
            
        except Exception as e:
            error_msg = f"Comprehensive extraction failed: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
            
        finally:
            await extractor.cleanup()
    
    async def verify_catalog_coverage(self) -> Dict[str, Any]:
        """
        Verify that we have good catalog coverage by testing key product categories
        """
        logger.info(" Verifying catalog coverage...")
        
        # Test searches we know should work
        test_searches = ["jam", "rice", "phone", "shirt"]
        coverage_results = {}
        
        # Same setup as comprehensive extraction
        api_config = {
            "base_url": os.getenv("BACKEND_ENDPOINT", "https://hp-buyer-backend-preprod.himira.co.in/clientApis"),
            "api_key": os.getenv("WIL_API_KEY", ""),
            "user_id": os.getenv("ETL_USER_ID", "guestUser"),
            "device_id": os.getenv("ETL_DEVICE_ID", "etl_coverage_test")
        }
        
        extraction_config = ExtractionConfig(batch_size=10, timeout_seconds=30)
        extractor = HimiraExtractor(extraction_config, api_config)
        
        try:
            await extractor.setup()
            
            for search_term in test_searches:
                result = await extractor.extract_products(
                    query=search_term,
                    limit=5,
                    max_pages=1
                )
                
                coverage_results[search_term] = {
                    "success": result.success,
                    "products_found": result.total_records,
                    "sample_products": [p.get("name", "Unknown") for p in result.data[:3]]
                }
                
                logger.info(f"Coverage test '{search_term}': {result.total_records} products")
                
        except Exception as e:
            logger.error(f"Coverage verification failed: {e}")
            
        finally:
            await extractor.cleanup()
        
        return coverage_results

if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env.etl")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run extraction
    async def main():
        extractor = ComprehensiveCatalogExtractor()
        
        # First, verify coverage with known working searches
        coverage = await extractor.verify_catalog_coverage()
        logger.info("Coverage Verification Results:")
        for search, result in coverage.items():
            status = "PASS" if result["success"] and result["products_found"] > 0 else "FAIL"
            logger.info(f"{status} {search}: {result['products_found']} products")
        
        # Then run comprehensive extraction
        logger.info("Starting comprehensive extraction...")
        stats = await extractor.run_comprehensive_extraction(
            max_products=500,  # Limit for initial test
            dry_run=False
        )
        
        # Save results
        results_file = project_root / "extraction_results.json"
        with open(results_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    asyncio.run(main())