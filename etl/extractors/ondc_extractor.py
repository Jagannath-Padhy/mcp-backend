"""
ONDC Protocol Extractor

Extracts data directly from ONDC protocol APIs.
This is primarily for learning and development purposes.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from urllib.parse import urljoin

from .base_extractor import BaseExtractor, ExtractionResult, ExtractionConfig

logger = logging.getLogger(__name__)


class ONDCExtractor(BaseExtractor):
    """
    Extractor for ONDC Protocol APIs
    
    This extractor demonstrates how to connect directly to ONDC
    protocol endpoints for learning purposes.
    """
    
    def __init__(self, config: ExtractionConfig, api_config: Dict[str, Any]):
        super().__init__(config, "ondc_protocol")
        
        # API Configuration
        self.base_url = api_config.get("base_url", "")
        self.api_key = api_config.get("api_key", "")
        
        # Headers for ONDC protocol requests
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "User-Agent": "ONDC-ETL/1.0"
        }
        
    async def health_check(self) -> bool:
        """Check if ONDC protocol API is accessible"""
        try:
            if not self.session:
                await self.setup()
                
            # Try to access a basic endpoint
            url = urljoin(self.base_url, "/health")
            
            async with self.session.get(url, headers=self.headers) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"ONDC protocol health check failed: {e}")
            return False
            
    async def extract_products(self, **kwargs) -> ExtractionResult:
        """
        Extract products from ONDC protocol
        
        This is a simplified implementation for learning purposes.
        Real ONDC integration requires proper authentication and protocol handling.
        """
        try:
            logger.info("Extracting products from ONDC protocol")
            
            # Build search request
            search_request = {
                "context": {
                    "domain": "retail",
                    "country": "IND",
                    "city": kwargs.get("city", "std:080"),  # Bangalore
                    "action": "search",
                    "core_version": "1.0.0"
                },
                "message": {
                    "intent": {
                        "item": {
                            "descriptor": {
                                "name": kwargs.get("query", "")
                            }
                        },
                        "fulfillment": {
                            "type": "Delivery"
                        }
                    }
                }
            }
            
            # Make search request
            url = urljoin(self.base_url, "/search")
            
            async with self.session.post(
                url, 
                json=search_request, 
                headers=self.headers
            ) as response:
                
                if response.status != 200:
                    error_msg = f"ONDC search failed with status {response.status}"
                    logger.error(error_msg)
                    return ExtractionResult(
                        success=False,
                        data=[],
                        errors=[error_msg],
                        metadata={},
                        extracted_at=datetime.utcnow(),
                        source=self.source_name,
                        total_records=0
                    )
                    
                data = await response.json()
                
                # Process ONDC response format
                products = self._process_ondc_response(data)
                
                return ExtractionResult(
                    success=len(products) > 0,
                    data=products,
                    errors=[],
                    metadata={"protocol_version": "1.0.0"},
                    extracted_at=datetime.utcnow(),
                    source=self.source_name,
                    total_records=len(products)
                )
                
        except Exception as e:
            logger.error(f"ONDC product extraction failed: {e}")
            return ExtractionResult(
                success=False,
                data=[],
                errors=[str(e)],
                metadata={},
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=0
            )
            
    async def extract_categories(self, **kwargs) -> ExtractionResult:
        """
        Extract categories from ONDC protocol
        
        This would typically involve fetching category taxonomies
        from ONDC registry or deriving from product data.
        """
        logger.info("ONDC category extraction not implemented - using mock data")
        
        # Mock categories for demonstration
        mock_categories = [
            {
                "id": "grocery",
                "name": "Grocery",
                "description": "Grocery and food items",
                "level": 1,
                "parent_id": None
            },
            {
                "id": "electronics",
                "name": "Electronics",
                "description": "Electronic devices and gadgets",
                "level": 1,
                "parent_id": None
            },
            {
                "id": "fashion",
                "name": "Fashion",
                "description": "Clothing and accessories",
                "level": 1,
                "parent_id": None
            }
        ]
        
        return ExtractionResult(
            success=True,
            data=mock_categories,
            errors=["Using mock data - real ONDC category API not implemented"],
            metadata={"mock": True},
            extracted_at=datetime.utcnow(),
            source=self.source_name,
            total_records=len(mock_categories)
        )
        
    async def extract_providers(self, **kwargs) -> ExtractionResult:
        """
        Extract providers from ONDC protocol
        
        This would involve fetching BPP (Business Provider Platform) data.
        """
        logger.info("ONDC provider extraction not implemented - using mock data")
        
        # Mock providers for demonstration
        mock_providers = [
            {
                "id": "bpp_001",
                "name": "Local Grocery Store",
                "description": "Fresh groceries and daily essentials",
                "location": {
                    "city": "Bangalore",
                    "state": "Karnataka",
                    "country": "India"
                },
                "rating": 4.2,
                "verified": True
            },
            {
                "id": "bpp_002", 
                "name": "Electronics Hub",
                "description": "Latest electronics and gadgets",
                "location": {
                    "city": "Bangalore",
                    "state": "Karnataka", 
                    "country": "India"
                },
                "rating": 4.5,
                "verified": True
            }
        ]
        
        return ExtractionResult(
            success=True,
            data=mock_providers,
            errors=["Using mock data - real ONDC provider API not implemented"],
            metadata={"mock": True},
            extracted_at=datetime.utcnow(),
            source=self.source_name,
            total_records=len(mock_providers)
        )
        
    def _process_ondc_response(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process ONDC protocol response into standardized format
        
        ONDC responses follow a specific protocol structure that needs
        to be normalized for our ETL pipeline.
        """
        products = []
        
        try:
            # ONDC response structure: context + message
            message = response_data.get("message", {})
            catalog = message.get("catalog", {})
            
            # Extract BPPs and their items
            bpps = catalog.get("bpp/providers", [])
            
            for bpp in bpps:
                bpp_id = bpp.get("id", "")
                bpp_name = bpp.get("descriptor", {}).get("name", "")
                
                items = bpp.get("items", [])
                
                for item in items:
                    try:
                        product = {
                            "id": f"{bpp_id}_{item.get('id', '')}",
                            "name": item.get("descriptor", {}).get("name", ""),
                            "description": item.get("descriptor", {}).get("short_desc", ""),
                            "price": {
                                "value": float(item.get("price", {}).get("value", 0)),
                                "currency": item.get("price", {}).get("currency", "INR")
                            },
                            "category": {
                                "id": item.get("category_id", ""),
                                "name": item.get("category_name", "")
                            },
                            "provider": {
                                "id": bpp_id,
                                "name": bpp_name,
                                "description": bpp.get("descriptor", {}).get("short_desc", "")
                            },
                            "images": [
                                {"url": img.get("url", ""), "type": "primary"}
                                for img in item.get("descriptor", {}).get("images", [])
                            ],
                            "availability": True,  # Assume available if listed
                            "tags": item.get("tags", []),
                            "extracted_at": datetime.utcnow().isoformat(),
                            "source": "ondc_protocol"
                        }
                        
                        products.append(product)
                        
                    except Exception as e:
                        logger.warning(f"Error processing ONDC item: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing ONDC response: {e}")
            
        return products