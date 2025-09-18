"""
Himira API Extractor

Extracts catalog data from the Himira ONDC backend API.
This is the primary data source for the ETL pipeline.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from urllib.parse import urljoin
import aiohttp

from .base_extractor import BaseExtractor, ExtractionResult, ExtractionConfig

logger = logging.getLogger(__name__)


class HimiraExtractor(BaseExtractor):
    """
    Extractor for Himira ONDC Backend API
    
    Connects to the Himira buyer backend to extract:
    - Product catalog data
    - Category hierarchies  
    - Provider information
    """
    
    def __init__(self, config: ExtractionConfig, api_config: Dict[str, Any]):
        super().__init__(config, "himira_api")
        
        # API Configuration
        self.base_url = api_config.get("base_url", "")
        self.api_key = api_config.get("api_key", "")
        self.user_id = api_config.get("user_id", "guestUser")
        self.device_id = api_config.get("device_id", "etl_pipeline_001")
        
        # Headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "wil-api-key": self.api_key,
            "User-Agent": "HimiraETL/1.0"
        }
        
        # Search parameters for different data types
        self.search_params = {
            "products": {
                "latitude": 12.9716,  # Bangalore coordinates
                "longitude": 77.5946,
                "page": 1,
                "limit": 100
            },
            "categories": {
                "page": 1,
                "limit": 50
            }
        }
        
    async def health_check(self) -> bool:
        """Check if Himira API is accessible"""
        try:
            if not self.session:
                await self.setup()
                
            # Try a simple search request (matching MCP server parameters)
            # Use proper URL construction - base_url already includes clientApis
            url = f"{self.base_url}/v2/search/{self.user_id}"
            params = {
                "name": "jam",  # Use name parameter like MCP server
                "page": 1,
                "limit": 1,
                "deviceId": self.device_id
            }
            
            logger.info(f"Health check URL: {url}")
            logger.info(f"Health check params: {params}")
            
            async with self.session.get(url, headers=self.headers, params=params) as response:
                logger.info(f"Health check response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    # Check if response has expected buyer backend structure
                    response_data = data.get("response", {})
                    has_data = response_data and "data" in response_data
                    logger.info(f"Health check success: found {len(response_data.get('data', []))} products")
                    return has_data
                else:
                    text = await response.text()
                    logger.error(f"Health check failed with status {response.status}: {text}")
                    return False
                
        except Exception as e:
            logger.error(f"Himira health check failed: {e}")
            return False
            
    async def extract_products(self, **kwargs) -> ExtractionResult:
        """
        Extract product data from Himira API
        
        Args:
            query (str): Search query (optional)
            category (str): Category filter (optional)
            latitude (float): User latitude (optional)
            longitude (float): User longitude (optional)
            limit (int): Results per page (optional)
            max_pages (int): Maximum pages to fetch (optional)
        """
        try:
            # Extract parameters
            query = kwargs.get("query", "")
            category = kwargs.get("category", "")
            latitude = kwargs.get("latitude", self.search_params["products"]["latitude"])
            longitude = kwargs.get("longitude", self.search_params["products"]["longitude"])
            limit = kwargs.get("limit", self.search_params["products"]["limit"])
            max_pages = kwargs.get("max_pages", 50)  # Increased to get more products
            
            all_products = []
            errors = []
            page = 1
            
            logger.info(f"Starting product extraction from Himira API")
            
            while page <= max_pages:
                try:
                    # Build search parameters (matching MCP server format)
                    params = {
                        "page": page,
                        "limit": limit,
                        "deviceId": self.device_id
                    }
                    
                    # Add location if provided (MCP server adds these conditionally)
                    if latitude and longitude:
                        params["latitude"] = latitude
                        params["longitude"] = longitude
                    
                    # Use 'name' parameter like MCP server (not 'query')
                    # For empty query, use empty string to get all products
                    params["name"] = query if query else ""
                    if category:
                        params["category"] = category
                        
                    # Make API request - use direct string formatting instead of urljoin
                    url = f"{self.base_url}/v2/search/{self.user_id}"
                    
                    async with self.session.get(url, headers=self.headers, params=params) as response:
                        if response.status != 200:
                            error_msg = f"API request failed with status {response.status}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                            break
                            
                        data = await response.json()
                        
                        # Parse response - using correct buyer backend structure
                        response_data = data.get("response", {})
                        if not response_data or not response_data.get("data"):
                            logger.info(f"No products in response: {data.get('message', 'Empty response')}")
                            break
                            
                        products = response_data.get("data", [])
                        
                        if not products:
                            logger.info(f"No more products found at page {page}")
                            break
                            
                        # Process products
                        processed_products = []
                        for product in products:
                            try:
                                processed_product = self._process_product(product)
                                if processed_product:
                                    processed_products.append(processed_product)
                            except Exception as e:
                                errors.append(f"Error processing product: {e}")
                                
                        all_products.extend(processed_products)
                        
                        logger.info(f"Extracted {len(processed_products)} products from page {page}")
                        
                        # Check if we should continue
                        if len(products) < limit:
                            logger.info("Reached end of results")
                            break
                            
                        page += 1
                        
                        # Add small delay between requests
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    error_msg = f"Error on page {page}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    break
                    
            # Validate extracted data
            valid_products, validation_errors = self.validate_data(all_products)
            errors.extend(validation_errors)
            
            return ExtractionResult(
                success=len(valid_products) > 0,
                data=valid_products,
                errors=errors,
                metadata={
                    "pages_fetched": page - 1,
                    "query": query,
                    "category": category,
                    "coordinates": {"lat": latitude, "lon": longitude}
                },
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=len(valid_products)
            )
            
        except Exception as e:
            logger.error(f"Product extraction failed: {e}")
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
        Extract category data from Himira API
        
        Currently, categories are extracted from product data
        since there's no dedicated categories endpoint.
        """
        try:
            logger.info("Extracting categories from Himira API")
            
            # First extract products to get category information
            products_result = await self.extract_products(limit=500, max_pages=5)
            
            if not products_result.success:
                return ExtractionResult(
                    success=False,
                    data=[],
                    errors=["Failed to extract products for category extraction"],
                    metadata={},
                    extracted_at=datetime.utcnow(),
                    source=self.source_name,
                    total_records=0
                )
                
            # Extract unique categories from products
            categories_dict = {}
            
            for product in products_result.data:
                category_data = product.get("category", {})
                
                if isinstance(category_data, dict) and category_data.get("id"):
                    cat_id = category_data["id"]
                    if cat_id not in categories_dict:
                        categories_dict[cat_id] = {
                            "id": cat_id,
                            "name": category_data.get("name", ""),
                            "description": category_data.get("description", ""),
                            "parent_id": category_data.get("parent_id"),
                            "level": category_data.get("level", 0),
                            "product_count": 1,
                            "extracted_at": datetime.utcnow().isoformat(),
                            "source": "himira_products"
                        }
                    else:
                        categories_dict[cat_id]["product_count"] += 1
                        
            categories = list(categories_dict.values())
            
            return ExtractionResult(
                success=len(categories) > 0,
                data=categories,
                errors=[],
                metadata={"derived_from": "products"},
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=len(categories)
            )
            
        except Exception as e:
            logger.error(f"Category extraction failed: {e}")
            return ExtractionResult(
                success=False,
                data=[],
                errors=[str(e)],
                metadata={},
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=0
            )
            
    async def extract_providers(self, **kwargs) -> ExtractionResult:
        """
        Extract provider data from Himira API
        
        Providers are extracted from product data since there's
        no dedicated providers endpoint.
        """
        try:
            logger.info("Extracting providers from Himira API")
            
            # Extract products to get provider information
            products_result = await self.extract_products(limit=500, max_pages=25)  # Get more data
            
            if not products_result.success:
                return ExtractionResult(
                    success=False,
                    data=[],
                    errors=["Failed to extract products for provider extraction"],
                    metadata={},
                    extracted_at=datetime.utcnow(),
                    source=self.source_name,
                    total_records=0
                )
                
            # Extract unique providers from products
            providers_dict = {}
            
            for product in products_result.data:
                provider_data = product.get("provider", {})
                
                if isinstance(provider_data, dict) and provider_data.get("id"):
                    provider_id = provider_data["id"]
                    if provider_id not in providers_dict:
                        providers_dict[provider_id] = {
                            "id": provider_id,
                            "name": provider_data.get("name", ""),
                            "description": provider_data.get("description", ""),
                            "location": provider_data.get("location", {}),
                            "contact": provider_data.get("contact", {}),
                            "rating": provider_data.get("rating", 0),
                            "verified": provider_data.get("verified", False),
                            "product_count": 1,
                            "categories": set(),
                            "extracted_at": datetime.utcnow().isoformat(),
                            "source": "himira_products"
                        }
                    else:
                        providers_dict[provider_id]["product_count"] += 1
                        
                    # Add category to provider
                    category = product.get("category", {})
                    if category.get("name"):
                        providers_dict[provider_id]["categories"].add(category["name"])
                        
            # Convert sets to lists for JSON serialization
            providers = []
            for provider in providers_dict.values():
                provider["categories"] = list(provider["categories"])
                providers.append(provider)
                
            return ExtractionResult(
                success=len(providers) > 0,
                data=providers,
                errors=[],
                metadata={"derived_from": "products"},
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=len(providers)
            )
            
        except Exception as e:
            logger.error(f"Provider extraction failed: {e}")
            return ExtractionResult(
                success=False,
                data=[],
                errors=[str(e)],
                metadata={},
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=0
            )
            
    def _process_product(self, raw_product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process and normalize a single product from Himira API response
        Handles the actual buyer backend structure with item_details, provider_details, location_details
        """
        try:
            # Get the full ONDC ID from top level
            full_ondc_id = raw_product.get("id", "")
            
            # Extract item_details from the response structure
            item_details = raw_product.get("item_details", {})
            if not item_details or not isinstance(item_details, dict):
                logger.warning(f"No valid item_details found in product: {raw_product}")
                return None
            
            # Extract descriptor information
            descriptor = item_details.get("descriptor", {})
            if not isinstance(descriptor, dict):
                descriptor = {}
            
            # Safely extract time information
            time_data = item_details.get("time", {})
            if isinstance(time_data, dict):
                created_at = time_data.get("timestamp", datetime.utcnow().isoformat())
            else:
                created_at = datetime.utcnow().isoformat()
            
            # Extract comprehensive provider details (from both provider_details and location_details)
            provider_data = self._extract_provider_from_item(
                raw_product.get("provider_details", {}), 
                raw_product.get("location_details", [])
            )
            
            # Extract comprehensive location data
            location_data = self._extract_location_comprehensive(
                raw_product.get("location_details", []), 
                item_details.get("location_id")
            )
            
            # Extract fulfillment data
            fulfillment_data = self._extract_fulfillment(raw_product.get("fulfillment_details", []))
            
            # Extract payment methods
            payment_methods = self._extract_payment_methods(raw_product.get("payment_details", []))
            
            # Build comprehensive product
            processed = {
                "id": full_ondc_id if full_ondc_id else str(item_details.get("id", "")),  # Use full ONDC ID when available
                "name": str(descriptor.get("name", "")),
                "description": str(descriptor.get("short_desc", "")) or str(descriptor.get("long_desc", "")),
                "price": self._extract_price(item_details.get("price", {})),
                "category": self._extract_category_comprehensive(item_details, descriptor),
                "provider": provider_data,
                "location": location_data,
                "images": self._extract_images(descriptor.get("images", [])),
                "availability": self._extract_availability(item_details),
                "rating": float(raw_product.get("rating", 0)) if isinstance(raw_product.get("rating"), (int, float)) else 0.0,
                "tags": self._extract_tags_comprehensive(item_details.get("tags", [])),
                "attributes": self._extract_attributes(item_details),
                "ondc_attributes": self._extract_ondc_attributes(item_details),
                "fulfillment": fulfillment_data,
                "payment_methods": payment_methods,
                "brand": descriptor.get("brand", ""),
                "model": descriptor.get("model", ""),
                "size": descriptor.get("size", ""),
                "color": descriptor.get("color", ""),
                "weight": descriptor.get("weight", ""),
                "created_at": created_at,
                "updated_at": datetime.utcnow().isoformat(),
                "extracted_at": datetime.utcnow().isoformat(),
                "source": "himira_api",
                "raw_data": raw_product  # Store complete raw data for reference
            }
            
            # Ensure required fields are present
            if not processed["id"] or not processed["name"]:
                logger.warning(f"Product missing required fields: id={processed['id']}, name={processed['name']}")
                return None
                
            return processed
            
        except Exception as e:
            logger.error(f"Error processing product {raw_product.get('item_details', {}).get('id', 'unknown')}: {e}")
            return None
            
    def _extract_price(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize price information"""
        if not isinstance(price_data, dict):
            return {"value": 0, "currency": "INR", "maximum_value": None, "minimum_value": None, "offered_value": None}
        
        return {
            "value": float(price_data.get("value", 0)) if price_data.get("value") else 0,
            "currency": str(price_data.get("currency", "INR")),
            "maximum_value": float(price_data.get("maximum_value")) if price_data.get("maximum_value") else None,
            "minimum_value": float(price_data.get("minimum_value")) if price_data.get("minimum_value") else None,
            "offered_value": float(price_data.get("offered_value")) if price_data.get("offered_value") else None
        }
        
    def _extract_category(self, category_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize category information"""
        return {
            "id": category_data.get("id", ""),
            "name": category_data.get("name", ""),
            "description": category_data.get("description", ""),
            "parent_id": category_data.get("parent_id"),
            "level": category_data.get("level", 0)
        }
        
    def _extract_provider(self, provider_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize provider information"""
        return {
            "id": provider_data.get("id", ""),
            "name": provider_data.get("name", ""),
            "description": provider_data.get("description", ""),
            "location": provider_data.get("location", {}),
            "contact": provider_data.get("contact", {}),
            "rating": provider_data.get("rating", 0),
            "verified": provider_data.get("verified", False)
        }
        
    def _extract_location(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize location information"""
        if not isinstance(location_data, dict):
            return {"latitude": None, "longitude": None, "address": "", "city": "", "state": "", "pincode": "", "country": "India"}
        
        return {
            "latitude": float(location_data.get("latitude")) if location_data.get("latitude") else None,
            "longitude": float(location_data.get("longitude")) if location_data.get("longitude") else None,
            "address": str(location_data.get("address", "")),
            "city": str(location_data.get("city", "")),
            "state": str(location_data.get("state", "")),
            "pincode": str(location_data.get("pincode", "")),
            "country": str(location_data.get("country", "India"))
        }
        
    def _extract_images(self, images_data: List[Any]) -> List[Dict[str, Any]]:
        """Extract and normalize image information"""
        if not isinstance(images_data, list):
            return []
        
        processed_images = []
        
        for img in images_data:
            if isinstance(img, str):
                processed_images.append({
                    "url": str(img),
                    "type": "primary" if len(processed_images) == 0 else "additional",
                    "alt_text": ""
                })
            elif isinstance(img, dict):
                processed_images.append({
                    "url": str(img.get("url", "")),
                    "type": str(img.get("type", "additional")),
                    "alt_text": str(img.get("alt_text", ""))
                })
                
        return processed_images
    
    def _extract_category_comprehensive(self, item_details: Dict[str, Any], descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive category information from item_details and descriptor"""
        category = {
            "id": item_details.get("category_id", ""),
            "name": item_details.get("category_id", ""),  # category_id often contains the name
            "description": "",
            "parent_id": None,
            "level": 0
        }
        
        # Try to extract category from tags
        tags = item_details.get("tags", [])
        for tag in tags:
            if isinstance(tag, dict):
                if tag.get("code") == "category":
                    tag_list = tag.get("list", [])
                    for item in tag_list:
                        if isinstance(item, dict):
                            if item.get("code") == "name":
                                category["name"] = item.get("value", category["name"])
                            elif item.get("code") == "id":
                                category["id"] = item.get("value", category["id"])
        
        # Extract from descriptor if available
        if descriptor:
            if descriptor.get("category"):
                category["name"] = descriptor.get("category", category["name"])
            if descriptor.get("category_id"):
                category["id"] = descriptor.get("category_id", category["id"])
        
        return category
    
    def _extract_provider_from_item(self, provider_details: Dict[str, Any], location_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract comprehensive provider information from provider_details and location_details"""
        provider = {
            "id": provider_details.get("id", ""),
            "name": provider_details.get("name", ""),
            "description": provider_details.get("description", ""),
            "rating": provider_details.get("rating", 0),
            "verified": provider_details.get("verified", False),
            "locations": [],
            "contact": {},
            "business_info": {}
        }
        
        # Extract descriptor if available
        descriptor = provider_details.get("descriptor", {})
        if descriptor:
            provider["name"] = descriptor.get("name", provider["name"])
            provider["short_desc"] = descriptor.get("short_desc", "")
            provider["long_desc"] = descriptor.get("long_desc", "")
            provider["images"] = descriptor.get("images", [])
            provider["symbol"] = descriptor.get("symbol", "")
        
        # Extract TTL (time to live) information
        if provider_details.get("ttl"):
            provider["ttl"] = provider_details.get("ttl")
        
        # Extract locations from location_details
        if isinstance(location_details, list):
            for loc in location_details:
                if isinstance(loc, dict):
                    location = {
                        "id": loc.get("id", ""),
                        "gps": loc.get("gps", ""),
                        "address": loc.get("address", {}),
                        "circle": loc.get("circle", {}),
                        "time": loc.get("time", {})
                    }
                    
                    # Parse GPS coordinates if available
                    if location["gps"]:
                        try:
                            lat, lon = location["gps"].split(",")
                            location["latitude"] = float(lat.strip())
                            location["longitude"] = float(lon.strip())
                        except:
                            pass
                    
                    # Extract address details
                    if isinstance(location["address"], dict):
                        addr = location["address"]
                        location["full_address"] = {
                            "door": addr.get("door", ""),
                            "name": addr.get("name", ""),
                            "building": addr.get("building", ""),
                            "street": addr.get("street", ""),
                            "locality": addr.get("locality", ""),
                            "ward": addr.get("ward", ""),
                            "city": addr.get("city", ""),
                            "state": addr.get("state", ""),
                            "country": addr.get("country", "IND"),
                            "area_code": addr.get("area_code", "")
                        }
                    
                    provider["locations"].append(location)
        
        # Extract categories served
        if provider_details.get("categories"):
            provider["categories"] = provider_details.get("categories", [])
        
        # Extract fulfillments offered
        if provider_details.get("fulfillments"):
            provider["fulfillments"] = provider_details.get("fulfillments", [])
        
        # Extract payment methods
        if provider_details.get("payments"):
            provider["payments"] = provider_details.get("payments", [])
        
        return provider
    
    def _extract_availability(self, item_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extract availability information from item_details"""
        quantity = item_details.get("quantity", {})
        return {
            "available": quantity.get("available", {}).get("count", "0") != "0",
            "count": quantity.get("available", {}).get("count", "0"),
            "maximum": quantity.get("maximum", {}).get("count", "0"),
            "measure": quantity.get("unitized", {}).get("measure", {})
        }
    
    def _extract_attributes(self, item_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all product attributes from item_details"""
        attributes = {}
        
        # Extract quantity information
        quantity = item_details.get("quantity", {})
        if quantity:
            unitized = quantity.get("unitized", {}).get("measure", {})
            if unitized:
                attributes["unit"] = unitized.get("unit", "")
                attributes["value"] = unitized.get("value", "")
        
        # Extract from tags (if available)
        tags = item_details.get("tags", [])
        for tag in tags:
            if isinstance(tag, dict):
                tag_code = tag.get("code", "")
                tag_list = tag.get("list", [])
                for tag_item in tag_list:
                    if isinstance(tag_item, dict):
                        key = tag_item.get("code", "")
                        value = tag_item.get("value", "")
                        if key and value:
                            attributes[f"{tag_code}_{key}"] = value
        
        return attributes
    
    def _extract_ondc_attributes(self, item_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ONDC-specific attributes"""
        return {
            "returnable": item_details.get("@ondc/org/returnable", False),
            "cancellable": item_details.get("@ondc/org/cancellable", False),
            "available_on_cod": item_details.get("@ondc/org/available_on_cod", False),
            "time_to_ship": item_details.get("@ondc/org/time_to_ship", ""),
            "contact_details_consumer_care": item_details.get("@ondc/org/contact_details_consumer_care", "")
        }
    
    def _extract_location_comprehensive(self, location_details: List[Dict[str, Any]], location_id: str = None) -> Dict[str, Any]:
        """Extract comprehensive location information from location_details"""
        location = {
            "id": location_id or "",
            "latitude": None,
            "longitude": None,
            "address": "",
            "city": "",
            "state": "",
            "pincode": "",
            "country": "India",
            "all_locations": []
        }
        
        # Process all location details
        if isinstance(location_details, list):
            for loc in location_details:
                if isinstance(loc, dict):
                    loc_data = {
                        "id": loc.get("id", ""),
                        "gps": loc.get("gps", "")
                    }
                    
                    # Parse GPS coordinates
                    if loc_data["gps"]:
                        try:
                            lat, lon = loc_data["gps"].split(",")
                            loc_data["latitude"] = float(lat.strip())
                            loc_data["longitude"] = float(lon.strip())
                            
                            # Use first location as primary
                            if location["latitude"] is None:
                                location["latitude"] = loc_data["latitude"]
                                location["longitude"] = loc_data["longitude"]
                        except:
                            pass
                    
                    # Extract address
                    if loc.get("address"):
                        addr = loc["address"]
                        if isinstance(addr, dict):
                            loc_data["address"] = {
                                "door": addr.get("door", ""),
                                "name": addr.get("name", ""),
                                "building": addr.get("building", ""),
                                "street": addr.get("street", ""),
                                "locality": addr.get("locality", ""),
                                "ward": addr.get("ward", ""),
                                "city": addr.get("city", ""),
                                "state": addr.get("state", ""),
                                "country": addr.get("country", "IND"),
                                "area_code": addr.get("area_code", "")
                            }
                            
                            # Use first location address as primary
                            if not location["city"]:
                                location["city"] = addr.get("city", "")
                                location["state"] = addr.get("state", "")
                                location["pincode"] = addr.get("area_code", "")
                                location["address"] = f"{addr.get('building', '')} {addr.get('street', '')} {addr.get('locality', '')}".strip()
                    
                    location["all_locations"].append(loc_data)
        
        return location
    
    def _extract_fulfillment(self, fulfillment_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract fulfillment information"""
        fulfillment = {
            "types": [],
            "tracking": False,
            "same_day": False,
            "next_day": False,
            "standard": True,
            "providers": []
        }
        
        if isinstance(fulfillment_details, list):
            for f in fulfillment_details:
                if isinstance(f, dict):
                    f_type = f.get("type", "")
                    if f_type:
                        fulfillment["types"].append(f_type)
                    
                    # Check for tracking
                    if f.get("tracking"):
                        fulfillment["tracking"] = True
                    
                    # Extract TAT (turnaround time)
                    if f.get("@ondc/org/TAT"):
                        tat = f.get("@ondc/org/TAT", "")
                        if "PT0H" in tat or "PT1H" in tat:
                            fulfillment["same_day"] = True
                        elif "PT24H" in tat:
                            fulfillment["next_day"] = True
                    
                    # Extract provider info
                    if f.get("provider_name"):
                        fulfillment["providers"].append(f.get("provider_name"))
        
        return fulfillment
    
    def _extract_payment_methods(self, payment_details: List[Dict[str, Any]]) -> List[str]:
        """Extract available payment methods"""
        methods = set()
        
        if isinstance(payment_details, list):
            for p in payment_details:
                if isinstance(p, dict):
                    p_type = p.get("type", "")
                    if p_type:
                        methods.add(p_type)
                    
                    # Check for specific payment types
                    if p.get("@ondc/org/buyer_app_finder_fee_type"):
                        methods.add("BUYER_APP_FEE")
                    if p.get("@ondc/org/settlement_details"):
                        methods.add("SETTLEMENT")
        
        # Add default if no methods found
        if not methods:
            methods = {"ON-FULFILLMENT", "PRE-FULFILLMENT"}
        
        return list(methods)
    
    def _extract_tags_comprehensive(self, tags: List[Any]) -> List[Dict[str, Any]]:
        """Extract comprehensive tag information"""
        processed_tags = []
        
        if not isinstance(tags, list):
            return processed_tags
        
        for tag in tags:
            if isinstance(tag, dict):
                tag_data = {
                    "code": tag.get("code", ""),
                    "name": tag.get("name", ""),
                    "display": tag.get("display", True),
                    "values": {}
                }
                
                # Extract tag list values
                tag_list = tag.get("list", [])
                if isinstance(tag_list, list):
                    for item in tag_list:
                        if isinstance(item, dict):
                            code = item.get("code", "")
                            value = item.get("value", "")
                            if code:
                                tag_data["values"][code] = value
                
                if tag_data["code"] or tag_data["values"]:
                    processed_tags.append(tag_data)
            elif isinstance(tag, str):
                processed_tags.append({"code": tag, "name": tag, "display": True, "values": {}})
        
        return processed_tags