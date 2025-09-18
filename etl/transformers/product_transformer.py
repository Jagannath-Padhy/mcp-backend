"""
Product Transformer

Normalizes and standardizes product data for consistent storage and search.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from urllib.parse import urlparse
import uuid

from .base_transformer import BaseTransformer, TransformationConfig

logger = logging.getLogger(__name__)


class ProductTransformer(BaseTransformer):
    """
    Transformer for normalizing product data
    
    Handles:
    - Data type conversions
    - Price normalization
    - Image URL validation
    - Category standardization
    - Location processing
    - Text cleanup
    """
    
    def __init__(self, 
                 config: TransformationConfig,
                 normalization_config: Dict[str, Any]):
        super().__init__(config, "product_transformer")
        
        # Normalization settings
        self.price_config = normalization_config.get("price", {})
        self.location_config = normalization_config.get("location", {})
        self.text_config = normalization_config.get("text", {})
        
        # Default values
        self.default_currency = self.price_config.get("default_currency", "INR")
        self.default_country = self.location_config.get("default_country", "India")
        self.max_name_length = self.text_config.get("max_name_length", 200)
        self.max_description_length = self.text_config.get("max_description_length", 1000)
        
        # Category mapping (simple implementation)
        self.category_mapping = {
            # Food & Beverages
            "grocery": "Food & Beverages",
            "food": "Food & Beverages", 
            "beverages": "Food & Beverages",
            "snacks": "Food & Beverages",
            
            # Electronics
            "electronics": "Electronics",
            "mobile": "Electronics",
            "laptop": "Electronics",
            "computer": "Electronics",
            
            # Fashion
            "clothing": "Fashion",
            "fashion": "Fashion",
            "apparel": "Fashion",
            "shoes": "Fashion",
            
            # Home & Living
            "furniture": "Home & Living",
            "home": "Home & Living",
            "decor": "Home & Living",
            
            # Health & Beauty
            "health": "Health & Beauty",
            "beauty": "Health & Beauty",
            "cosmetics": "Health & Beauty",
            "medicine": "Health & Beauty"
        }
        
    async def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single product record
        """
        try:
            if not self.validate_input(record):
                raise ValueError("Invalid input record")
                
            # Start with clean record
            transformed = {
                "id": self._normalize_id(record.get("id")),
                "name": self._normalize_name(record.get("name", "")),
                "description": self._normalize_description(record.get("description", "")),
                "price": self._normalize_price(record.get("price")),
                "category": self._normalize_category(record.get("category")),
                "provider": self._normalize_provider(record.get("provider")),
                "location": self._normalize_location(record.get("location")),
                "images": self._normalize_images(record.get("images", [])),
                "availability": self._normalize_availability(record.get("availability", True)),
                "rating": self._normalize_rating(record.get("rating", 0)),
                "tags": self._normalize_tags(record.get("tags", [])),
                "attributes": self._normalize_attributes(record.get("attributes", {})),
                "dates": self._normalize_dates(record),
                "source": record.get("source", "unknown"),
                "transformed_at": datetime.utcnow().isoformat()
            }
            
            # Add derived fields
            transformed["search_text"] = self._generate_search_text(transformed)
            transformed["keywords"] = self.extract_keywords(transformed["search_text"])
            transformed["price_category"] = self._categorize_price(transformed["price"])
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming product record: {e}")
            # Return minimal record with error
            return {
                "id": record.get("id", str(uuid.uuid4())),
                "name": record.get("name", "Unknown Product"),
                "transformation_error": str(e),
                "original_record": record,
                "transformed_at": datetime.utcnow().isoformat()
            }
            
    def _normalize_id(self, id_value: Any) -> str:
        """Normalize product ID"""
        if not id_value:
            return str(uuid.uuid4())
            
        # Convert to string and clean
        id_str = str(id_value).strip()
        
        # Remove special characters, keep alphanumeric, hyphens, underscores
        id_str = re.sub(r'[^a-zA-Z0-9\-_]', '_', id_str)
        
        return id_str[:50]  # Limit length
        
    def _normalize_name(self, name: Any) -> str:
        """Normalize product name"""
        if not name:
            return "Unknown Product"
            
        name_str = self.normalize_text(str(name))
        
        # Truncate if too long
        if len(name_str) > self.max_name_length:
            name_str = name_str[:self.max_name_length].rsplit(' ', 1)[0]
            
        return name_str
        
    def _normalize_description(self, description: Any) -> str:
        """Normalize product description"""
        if not description:
            return ""
            
        desc_str = self.normalize_text(str(description))
        
        # Truncate if too long
        if len(desc_str) > self.max_description_length:
            desc_str = desc_str[:self.max_description_length].rsplit(' ', 1)[0]
            
        return desc_str
        
    def _normalize_price(self, price_data: Any) -> Dict[str, Any]:
        """Normalize price information"""
        if price_data is None:
            return {
                "value": 0.0,
                "currency": self.default_currency,
                "formatted": f"₹0",
                "valid": False
            }
            
        # Handle different price formats
        if isinstance(price_data, (int, float)):
            value = float(price_data)
            return {
                "value": value,
                "currency": self.default_currency,
                "formatted": f"₹{value:,.2f}",
                "valid": value > 0
            }
            
        elif isinstance(price_data, dict):
            value = float(price_data.get("value", 0))
            currency = price_data.get("currency", self.default_currency)
            
            # Handle currency symbols
            currency_symbols = {"INR": "₹", "USD": "$", "EUR": "€"}
            symbol = currency_symbols.get(currency, currency)
            
            return {
                "value": value,
                "currency": currency,
                "formatted": f"{symbol}{value:,.2f}",
                "maximum_value": price_data.get("maximum_value"),
                "minimum_value": price_data.get("minimum_value"),
                "offered_value": price_data.get("offered_value"),
                "valid": value > 0
            }
            
        else:
            # Try to parse string prices
            try:
                # Remove currency symbols and extract number
                price_str = str(price_data).replace("₹", "").replace(",", "")
                value = float(re.findall(r'\d+\.?\d*', price_str)[0])
                
                return {
                    "value": value,
                    "currency": self.default_currency,
                    "formatted": f"₹{value:,.2f}",
                    "valid": value > 0
                }
            except:
                return {
                    "value": 0.0,
                    "currency": self.default_currency,
                    "formatted": "₹0",
                    "valid": False,
                    "parse_error": str(price_data)
                }
                
    def _normalize_category(self, category_data: Any) -> Dict[str, Any]:
        """Normalize category information"""
        if not category_data:
            return {
                "id": "uncategorized",
                "name": "Uncategorized",
                "level": 0,
                "path": ["Uncategorized"]
            }
            
        if isinstance(category_data, str):
            category_name = category_data.strip()
            standardized_name = self._standardize_category_name(category_name)
            
            return {
                "id": category_name.lower().replace(" ", "_"),
                "name": standardized_name,
                "original_name": category_name,
                "level": 1,
                "path": [standardized_name]
            }
            
        elif isinstance(category_data, dict):
            name = category_data.get("name", "Uncategorized")
            standardized_name = self._standardize_category_name(name)
            
            return {
                "id": category_data.get("id", name.lower().replace(" ", "_")),
                "name": standardized_name,
                "original_name": name,
                "description": category_data.get("description", ""),
                "parent_id": category_data.get("parent_id"),
                "level": category_data.get("level", 1),
                "path": category_data.get("path", [standardized_name])
            }
            
        return {
            "id": "uncategorized",
            "name": "Uncategorized",
            "level": 0,
            "path": ["Uncategorized"]
        }
        
    def _standardize_category_name(self, name: str) -> str:
        """Standardize category name using mapping"""
        if not name:
            return "Uncategorized"
            
        name_lower = name.lower().strip()
        
        # Check for exact matches first
        if name_lower in self.category_mapping:
            return self.category_mapping[name_lower]
            
        # Check for partial matches
        for key, standard_name in self.category_mapping.items():
            if key in name_lower or name_lower in key:
                return standard_name
                
        # Return title case version of original
        return name.title()
        
    def _normalize_provider(self, provider_data: Any) -> Dict[str, Any]:
        """Normalize provider information"""
        if not provider_data:
            return {
                "id": "unknown",
                "name": "Unknown Provider",
                "verified": False
            }
            
        if isinstance(provider_data, str):
            return {
                "id": provider_data.lower().replace(" ", "_"),
                "name": provider_data.strip(),
                "verified": False
            }
            
        elif isinstance(provider_data, dict):
            return {
                "id": provider_data.get("id", "unknown"),
                "name": provider_data.get("name", "Unknown Provider"),
                "description": self.normalize_text(provider_data.get("description", "")),
                "verified": bool(provider_data.get("verified", False)),
                "rating": self._normalize_rating(provider_data.get("rating", 0)),
                "location": self._normalize_location(provider_data.get("location")),
                "contact": provider_data.get("contact", {})
            }
            
        return {
            "id": "unknown",
            "name": str(provider_data),
            "verified": False
        }
        
    def _normalize_location(self, location_data: Any) -> Dict[str, Any]:
        """Normalize location information"""
        default_location = {
            "country": self.default_country,
            "state": "",
            "city": "",
            "address": "",
            "pincode": "",
            "coordinates": {
                "latitude": None,
                "longitude": None
            },
            "formatted_address": ""
        }
        
        if not location_data:
            return default_location
            
        if isinstance(location_data, dict):
            location = default_location.copy()
            location.update({
                "country": location_data.get("country", self.default_country),
                "state": self.normalize_text(location_data.get("state", "")),
                "city": self.normalize_text(location_data.get("city", "")),
                "address": self.normalize_text(location_data.get("address", "")),
                "pincode": str(location_data.get("pincode", "")).strip(),
                "coordinates": {
                    "latitude": self._normalize_coordinate(location_data.get("latitude")),
                    "longitude": self._normalize_coordinate(location_data.get("longitude"))
                }
            })
            
            # Generate formatted address
            address_parts = [
                location["address"],
                location["city"],
                location["state"],
                location["country"]
            ]
            location["formatted_address"] = ", ".join([p for p in address_parts if p])
            
            return location
            
        elif isinstance(location_data, str):
            # Parse address string
            return {
                **default_location,
                "address": self.normalize_text(location_data),
                "formatted_address": self.normalize_text(location_data)
            }
            
        return default_location
        
    def _normalize_coordinate(self, coord: Any) -> Optional[float]:
        """Normalize coordinate value"""
        if coord is None:
            return None
            
        try:
            coord_float = float(coord)
            # Basic sanity check for lat/lon ranges
            if -180 <= coord_float <= 180:
                return coord_float
        except (ValueError, TypeError):
            pass
            
        return None
        
    def _normalize_images(self, images_data: Any) -> List[Dict[str, Any]]:
        """Normalize image information"""
        if not images_data:
            return []
            
        if isinstance(images_data, str):
            # Single image URL
            return [self._normalize_single_image(images_data, "primary")]
            
        elif isinstance(images_data, list):
            normalized_images = []
            
            for i, img in enumerate(images_data):
                img_type = "primary" if i == 0 else "additional"
                normalized_img = self._normalize_single_image(img, img_type)
                
                if normalized_img:
                    normalized_images.append(normalized_img)
                    
            return normalized_images
            
        return []
        
    def _normalize_single_image(self, img_data: Any, img_type: str) -> Optional[Dict[str, Any]]:
        """Normalize a single image"""
        if not img_data:
            return None
            
        if isinstance(img_data, str):
            url = img_data.strip()
            if self._is_valid_url(url):
                return {
                    "url": url,
                    "type": img_type,
                    "alt_text": "",
                    "valid": True
                }
            else:
                return {
                    "url": url,
                    "type": img_type,
                    "alt_text": "",
                    "valid": False,
                    "error": "Invalid URL"
                }
                
        elif isinstance(img_data, dict):
            url = img_data.get("url", "")
            return {
                "url": url,
                "type": img_data.get("type", img_type),
                "alt_text": img_data.get("alt_text", ""),
                "valid": self._is_valid_url(url)
            }
            
        return None
        
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
            
    def _normalize_availability(self, availability: Any) -> bool:
        """Normalize availability status"""
        if isinstance(availability, bool):
            return availability
        elif isinstance(availability, str):
            return availability.lower() in ["true", "available", "yes", "in stock", "1"]
        elif isinstance(availability, (int, float)):
            return availability > 0
        else:
            return True  # Default to available
            
    def _normalize_rating(self, rating: Any) -> float:
        """Normalize rating value"""
        try:
            rating_float = float(rating)
            # Clamp between 0 and 5
            return max(0.0, min(5.0, rating_float))
        except (ValueError, TypeError):
            return 0.0
            
    def _normalize_tags(self, tags_data: Any) -> List[str]:
        """Normalize tags/keywords"""
        if not tags_data:
            return []
            
        if isinstance(tags_data, str):
            # Split comma/semicolon separated tags
            tags = re.split(r'[,;]', tags_data)
            return [self.normalize_text(tag) for tag in tags if tag.strip()]
            
        elif isinstance(tags_data, list):
            normalized_tags = []
            for tag in tags_data:
                if isinstance(tag, str) and tag.strip():
                    normalized_tags.append(self.normalize_text(tag))
                    
            return normalized_tags
            
        return []
        
    def _normalize_attributes(self, attributes: Any) -> Dict[str, Any]:
        """Normalize product attributes"""
        if not isinstance(attributes, dict):
            return {}
            
        normalized = {}
        for key, value in attributes.items():
            if key and value is not None:
                # Normalize key
                clean_key = str(key).strip().lower().replace(" ", "_")
                
                # Normalize value
                if isinstance(value, str):
                    clean_value = self.normalize_text(value)
                else:
                    clean_value = value
                    
                normalized[clean_key] = clean_value
                
        return normalized
        
    def _normalize_dates(self, record: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Normalize date fields"""
        dates = {
            "created_at": None,
            "updated_at": None,
            "extracted_at": None
        }
        
        date_fields = ["created_at", "updated_at", "extracted_at"]
        
        for field in date_fields:
            date_value = record.get(field)
            dates[field] = self._normalize_date(date_value)
            
        return dates
        
    def _normalize_date(self, date_value: Any) -> Optional[str]:
        """Normalize a single date value"""
        if not date_value:
            return None
            
        try:
            # Handle different date formats
            if isinstance(date_value, datetime):
                return date_value.isoformat()
            elif isinstance(date_value, str):
                # Try to parse ISO format
                if 'T' in date_value or 'Z' in date_value:
                    return datetime.fromisoformat(date_value.replace('Z', '+00:00')).isoformat()
                else:
                    # Assume it's already a valid ISO string
                    return date_value
        except:
            pass
            
        return None
        
    def _generate_search_text(self, record: Dict[str, Any]) -> str:
        """Generate combined search text for full-text search"""
        text_parts = [
            record.get("name", ""),
            record.get("description", ""),
            record.get("category", {}).get("name", ""),
            record.get("provider", {}).get("name", ""),
        ]
        
        # Add tags
        tags = record.get("tags", [])
        if tags:
            text_parts.extend(tags)
            
        # Add attributes values
        attributes = record.get("attributes", {})
        if attributes:
            text_parts.extend([str(v) for v in attributes.values()])
            
        return " ".join([str(part) for part in text_parts if part]).strip()
        
    def _categorize_price(self, price_data: Dict[str, Any]) -> str:
        """Categorize price into ranges"""
        value = price_data.get("value", 0)
        
        if value == 0:
            return "free"
        elif value < 100:
            return "budget"
        elif value < 1000:
            return "affordable"
        elif value < 10000:
            return "moderate"
        elif value < 50000:
            return "premium"
        else:
            return "luxury"