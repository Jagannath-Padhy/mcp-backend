"""
Metadata Enricher

Enriches product records with additional metadata for improved search and analytics.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import Counter

from .base_transformer import BaseTransformer, TransformationConfig

logger = logging.getLogger(__name__)


class MetadataEnricher(BaseTransformer):
    """
    Transformer that enriches product records with additional metadata
    
    Adds:
    - Search keywords and features
    - Product quality scores
    - Popularity indicators
    - Market insights
    - Semantic tags
    """
    
    def __init__(self, 
                 config: TransformationConfig,
                 enrichment_config: Dict[str, Any]):
        super().__init__(config, "metadata_enricher")
        
        # Enrichment settings
        self.add_search_metadata = enrichment_config.get("add_search_metadata", True)
        self.generate_keywords = enrichment_config.get("generate_keywords", True)
        self.extract_features = enrichment_config.get("extract_features", True)
        self.analyze_sentiment = enrichment_config.get("sentiment_analysis", False)
        
        # Keyword extraction settings
        self.max_keywords = enrichment_config.get("max_keywords", 15)
        self.min_keyword_length = enrichment_config.get("min_keyword_length", 3)
        
        # Feature extraction patterns
        self.feature_patterns = {
            "brand": [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?=\s+(?:brand|Brand))',
                r'(?:by|Brand|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            ],
            "color": [
                r'\b(red|blue|green|yellow|black|white|gray|grey|brown|pink|purple|orange|gold|silver|navy|maroon|lime|cyan|magenta)\b',
                r'\b([a-z]+-colored?)\b',
                r'\b(multi-colored?|multicolored?)\b'
            ],
            "size": [
                r'\b(small|medium|large|extra\s*large|xl|xxl|xs|s|m|l)\b',
                r'\b(\d+(?:\.\d+)?\s*(?:inch|inches|cm|mm|meter|metres?|feet|ft))\b',
                r'\b(\d+(?:\.\d+)?\s*(?:kg|gram|grams|lb|pound|oz|ounce))\b'
            ],
            "material": [
                r'\b(cotton|silk|wool|polyester|leather|plastic|wood|metal|glass|ceramic|rubber)\b',
                r'\b(stainless\s*steel|aluminum|brass|copper|iron)\b',
                r'\b(organic|natural|synthetic|artificial)\b'
            ],
            "technology": [
                r'\b(bluetooth|wifi|wireless|usb|hdmi|4k|hd|full\s*hd|smart|digital)\b',
                r'\b(android|ios|windows|linux)\b',
                r'\b(led|lcd|oled|amoled)\b'
            ]
        }
        
        # Quality indicators
        self.quality_keywords = {
            "high": ["premium", "luxury", "high-quality", "professional", "deluxe", "superior"],
            "medium": ["standard", "regular", "classic", "basic", "normal"],
            "low": ["budget", "economy", "cheap", "basic", "entry-level"]
        }
        
        # Stop words for keyword extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'us', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they',
            'them', 'their', 'can', 'may', 'might', 'must', 'shall', 'will',
            'from', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'now'
        }
        
    def _safe_get_price_value(self, price_field: Any) -> tuple[float, bool]:
        """
        Safely extract price value and validity from various price field formats
        
        Args:
            price_field: Could be dict, int, float, str, or None
            
        Returns:
            tuple: (price_value, is_valid)
        """
        if price_field is None:
            return 0.0, False
            
        if isinstance(price_field, dict):
            # Standard format: {"value": 100, "valid": True}
            value = price_field.get("value", 0)
            valid = price_field.get("valid", False)
            return float(value) if value is not None else 0.0, valid
            
        if isinstance(price_field, (int, float)):
            # Direct numeric value
            return float(price_field), price_field > 0
            
        if isinstance(price_field, str):
            # String representation
            try:
                value = float(price_field)
                return value, value > 0
            except (ValueError, TypeError):
                return 0.0, False
                
        # Unknown format
        return 0.0, False
        
    def _safe_get_nested_value(self, record: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Safely get nested dictionary values without assuming field types
        
        Args:
            record: The record to search in
            path: Dot-separated path like "price.value" or "category.name"
            default: Default value if path doesn't exist or field is wrong type
            
        Returns:
            The value at the path or default
        """
        try:
            keys = path.split('.')
            value = record
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
                    
            return value
        except (AttributeError, TypeError):
            return default
        
    async def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single product record with metadata
        """
        try:
            # Start with the original record
            enriched = record.copy()
            
            # Add search metadata
            if self.add_search_metadata:
                enriched["search_metadata"] = self._generate_search_metadata(record)
                
            # Generate keywords
            if self.generate_keywords:
                enriched["generated_keywords"] = self._generate_advanced_keywords(record)
                
            # Extract features
            if self.extract_features:
                enriched["extracted_features"] = self._extract_product_features(record)
                
            # Add quality score
            enriched["quality_score"] = self._calculate_quality_score(record)
            
            # Add popularity indicators
            enriched["popularity_indicators"] = self._analyze_popularity(record)
            
            # Add market insights
            enriched["market_insights"] = self._generate_market_insights(record)
            
            # Add semantic tags
            enriched["semantic_tags"] = self._generate_semantic_tags(record)
            
            # Add completeness score
            enriched["data_completeness"] = self._calculate_completeness_score(record)
            
            # Add metadata timestamp
            enriched["metadata_enriched_at"] = datetime.utcnow().isoformat()
            
            return enriched
            
        except Exception as e:
            logger.error(f"Error enriching record metadata: {e}")
            # Return original record with error info
            record["enrichment_error"] = str(e)
            return record
            
    def _generate_search_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata to improve search performance"""
        
        name = record.get("name", "")
        description = record.get("description", "")
        
        return {
            "searchable_text": self._create_searchable_text(record),
            "normalized_name": self._normalize_for_search(name),
            "name_length": len(name),
            "description_length": len(description),
            "has_images": len(record.get("images", [])) > 0,
            "has_price": self._safe_get_price_value(record.get("price"))[1],
            "has_rating": record.get("rating", 0) > 0,
            "word_count": len((name + " " + description).split()),
            "character_count": len(name + description)
        }
        
    def _create_searchable_text(self, record: Dict[str, Any]) -> str:
        """Create comprehensive searchable text"""
        text_parts = []
        
        # Basic fields
        text_parts.extend([
            record.get("name", ""),
            record.get("description", ""),
        ])
        
        # Category information
        category = record.get("category", {})
        if isinstance(category, dict):
            text_parts.extend([
                category.get("name", ""),
                category.get("description", "")
            ])
            
        # Provider information
        provider = record.get("provider", {})
        if isinstance(provider, dict):
            text_parts.extend([
                provider.get("name", ""),
                provider.get("description", "")
            ])
            
        # Tags
        tags = record.get("tags", [])
        if isinstance(tags, list):
            text_parts.extend(tags)
            
        # Attributes
        attributes = record.get("attributes", {})
        if isinstance(attributes, dict):
            text_parts.extend([str(v) for v in attributes.values()])
            
        return " ".join([str(part) for part in text_parts if part]).strip()
        
    def _normalize_for_search(self, text: str) -> str:
        """Normalize text for better search matching"""
        if not text:
            return ""
            
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove special characters but keep spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
        
    def _generate_advanced_keywords(self, record: Dict[str, Any]) -> List[str]:
        """Generate keywords using advanced techniques"""
        
        # Get all text content
        searchable_text = self._create_searchable_text(record)
        
        if not searchable_text:
            return []
            
        # Extract keywords using multiple methods
        keywords = set()
        
        # Method 1: Simple word frequency
        keywords.update(self._extract_by_frequency(searchable_text))
        
        # Method 2: N-gram extraction
        keywords.update(self._extract_ngrams(searchable_text))
        
        # Method 3: Pattern-based extraction
        keywords.update(self._extract_by_patterns(searchable_text))
        
        # Method 4: Category-specific keywords
        keywords.update(self._extract_category_keywords(record))
        
        # Filter and rank keywords
        filtered_keywords = self._filter_keywords(list(keywords))
        
        return filtered_keywords[:self.max_keywords]
        
    def _extract_by_frequency(self, text: str) -> Set[str]:
        """Extract keywords by frequency analysis"""
        # Clean and split text
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        
        # Remove stop words and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) >= self.min_keyword_length
        ]
        
        # Count word frequency
        word_freq = Counter(filtered_words)
        
        # Return most common words
        return {word for word, count in word_freq.most_common(10) if count > 1}
        
    def _extract_ngrams(self, text: str, n: int = 2) -> Set[str]:
        """Extract n-gram keywords"""
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        
        # Generate n-grams
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            
            # Filter out n-grams with stop words
            if not any(word in self.stop_words for word in words[i:i + n]):
                if len(ngram) >= self.min_keyword_length * n:
                    ngrams.add(ngram)
                    
        return ngrams
        
    def _extract_by_patterns(self, text: str) -> Set[str]:
        """Extract keywords using regex patterns"""
        keywords = set()
        
        # Brand names (capitalized words)
        brand_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        keywords.update(re.findall(brand_pattern, text))
        
        # Model numbers
        model_pattern = r'\b[A-Z0-9]+[-_]?[A-Z0-9]+\b'
        keywords.update(re.findall(model_pattern, text))
        
        # Technical specifications
        spec_pattern = r'\b\d+(?:\.\d+)?\s*(?:gb|mb|ghz|mhz|mp|inch|kg|g|ml|l)\b'
        keywords.update(re.findall(spec_pattern, text.lower()))
        
        return keywords
        
    def _extract_category_keywords(self, record: Dict[str, Any]) -> Set[str]:
        """Extract keywords based on product category"""
        keywords = set()
        
        category = record.get("category", {})
        if isinstance(category, dict):
            category_name = category.get("name", "").lower()
            
            # Add category-specific keywords based on common terms
            category_keywords_map = {
                "electronics": ["device", "gadget", "tech", "digital", "electronic"],
                "fashion": ["style", "wear", "clothing", "apparel", "fashion"],
                "food": ["edible", "consumable", "nutrition", "dietary", "meal"],
                "home": ["household", "domestic", "interior", "living", "home"],
                "beauty": ["cosmetic", "skincare", "beauty", "grooming", "care"]
            }
            
            for cat_key, cat_keywords in category_keywords_map.items():
                if cat_key in category_name:
                    keywords.update(cat_keywords)
                    
        return keywords
        
    def _filter_keywords(self, keywords: List[str]) -> List[str]:
        """Filter and rank keywords by relevance"""
        
        # Remove duplicates and empty strings
        unique_keywords = list(set(k.strip() for k in keywords if k and k.strip()))
        
        # Sort by length and frequency (preference for longer, more specific terms)
        ranked_keywords = sorted(
            unique_keywords,
            key=lambda k: (len(k.split()), len(k)),
            reverse=True
        )
        
        return ranked_keywords
        
    def _extract_product_features(self, record: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract product features using pattern matching"""
        
        text = self._create_searchable_text(record)
        features = {}
        
        for feature_type, patterns in self.feature_patterns.items():
            matches = set()
            
            for pattern in patterns:
                found_matches = re.findall(pattern, text, re.IGNORECASE)
                
                if isinstance(found_matches[0], tuple) if found_matches else False:
                    # Handle grouped matches
                    matches.update([match[0] if isinstance(match, tuple) else match 
                                  for match in found_matches])
                else:
                    matches.update(found_matches)
                    
            features[feature_type] = list(matches)
            
        return features
        
    def _calculate_quality_score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a quality score based on various factors"""
        
        score = 0.0
        max_score = 100.0
        factors = {}
        
        # Name quality (20 points)
        name = record.get("name", "")
        if name:
            name_score = min(20, len(name) / 5)  # Up to 20 points for good length
            score += name_score
            factors["name_quality"] = name_score
            
        # Description quality (25 points)
        description = record.get("description", "")
        if description:
            desc_score = min(25, len(description) / 10)  # Up to 25 points
            score += desc_score
            factors["description_quality"] = desc_score
            
        # Image availability (15 points)
        images = record.get("images", [])
        image_score = min(15, len(images) * 5)  # 5 points per image, max 15
        score += image_score
        factors["image_quality"] = image_score
        
        # Price validity (10 points)
        price_value, price_valid = self._safe_get_price_value(record.get("price"))
        if price_valid:
            score += 10
            factors["price_quality"] = 10
        else:
            factors["price_quality"] = 0
            
        # Rating (15 points)
        rating = record.get("rating", 0)
        rating_score = (rating / 5.0) * 15 if rating > 0 else 0
        score += rating_score
        factors["rating_quality"] = rating_score
        
        # Data completeness (15 points)
        completeness = self._calculate_completeness_score(record)["score"]
        completeness_score = (completeness / 100.0) * 15
        score += completeness_score
        factors["completeness_quality"] = completeness_score
        
        return {
            "score": min(100.0, score),
            "grade": self._score_to_grade(score),
            "factors": factors
        }
        
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
            
    def _analyze_popularity(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze popularity indicators"""
        
        indicators = {
            "has_rating": record.get("rating", 0) > 0,
            "rating_level": "high" if record.get("rating", 0) >= 4 else "medium" if record.get("rating", 0) >= 3 else "low",
            "has_multiple_images": len(record.get("images", [])) > 1,
            "detailed_description": len(record.get("description", "")) > 100,
            "brand_presence": bool(self._extract_brand(record)),
            "price_competitive": self._is_price_competitive(record)
        }
        
        # Calculate popularity score
        score_map = {
            "has_rating": 20,
            "rating_level": {"high": 25, "medium": 15, "low": 5}.get(indicators["rating_level"], 0),
            "has_multiple_images": 15,
            "detailed_description": 20,
            "brand_presence": 10,
            "price_competitive": 10
        }
        
        total_score = sum(
            score_map.get(k, 0) if v is True 
            else score_map.get(k, {}).get(v, 0) if isinstance(v, str)
            else 0
            for k, v in indicators.items()
        )
        
        indicators["popularity_score"] = total_score
        return indicators
        
    def _extract_brand(self, record: Dict[str, Any]) -> Optional[str]:
        """Extract brand name from product data"""
        
        # Check if explicitly provided in attributes
        attributes = record.get("attributes", {})
        if "brand" in attributes:
            return attributes["brand"]
            
        # Extract from name using patterns
        name = record.get("name", "")
        
        # Look for capitalized words at the beginning
        brand_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', name)
        if brand_match:
            return brand_match.group(1)
            
        return None
        
    def _is_price_competitive(self, record: Dict[str, Any]) -> bool:
        """Determine if price appears competitive (simplified)"""
        
        value, _ = self._safe_get_price_value(record.get("price"))
        
        # Simple heuristic based on price range
        # This could be enhanced with market data
        if value == 0:
            return False
        elif value < 1000:
            return True  # Budget items are competitive
        else:
            # Check for round numbers (often promotional)
            return value % 100 == 0 or value % 500 == 0
            
    def _generate_market_insights(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market-related insights"""
        
        price, _ = self._safe_get_price_value(record.get("price"))
        category = self._safe_get_nested_value(record, "category.name", "")
        
        insights = {
            "price_segment": self._categorize_price_segment(price),
            "category_popularity": self._estimate_category_popularity(category),
            "seasonal_relevance": self._assess_seasonal_relevance(record),
            "target_demographic": self._identify_target_demographic(record)
        }
        
        return insights
        
    def _categorize_price_segment(self, price: float) -> str:
        """Categorize price into market segments"""
        if price == 0:
            return "free"
        elif price < 500:
            return "economy"
        elif price < 2000:
            return "budget"
        elif price < 10000:
            return "mid-range"
        elif price < 50000:
            return "premium"
        else:
            return "luxury"
            
    def _estimate_category_popularity(self, category: str) -> str:
        """Estimate category popularity (simplified)"""
        
        popular_categories = [
            "electronics", "fashion", "food", "mobile", "clothing"
        ]
        
        if any(pop_cat in category.lower() for pop_cat in popular_categories):
            return "high"
        else:
            return "medium"
            
    def _assess_seasonal_relevance(self, record: Dict[str, Any]) -> List[str]:
        """Assess seasonal relevance of product"""
        
        text = self._create_searchable_text(record).lower()
        
        seasonal_keywords = {
            "summer": ["summer", "cooling", "ac", "fan", "swimwear", "shorts"],
            "winter": ["winter", "heating", "warm", "jacket", "sweater", "heater"],
            "monsoon": ["rain", "umbrella", "waterproof", "monsoon"],
            "festival": ["diwali", "christmas", "eid", "holi", "gift", "decoration"]
        }
        
        relevant_seasons = []
        for season, keywords in seasonal_keywords.items():
            if any(keyword in text for keyword in keywords):
                relevant_seasons.append(season)
                
        return relevant_seasons if relevant_seasons else ["year-round"]
        
    def _identify_target_demographic(self, record: Dict[str, Any]) -> List[str]:
        """Identify likely target demographic"""
        
        text = self._create_searchable_text(record).lower()
        price = record.get("price", {}).get("value", 0)
        
        demographics = []
        
        # Age-based
        if any(keyword in text for keyword in ["kid", "child", "baby", "infant"]):
            demographics.append("children")
        elif any(keyword in text for keyword in ["teen", "youth", "young"]):
            demographics.append("teenagers")
        elif any(keyword in text for keyword in ["senior", "elderly"]):
            demographics.append("seniors")
        else:
            demographics.append("adults")
            
        # Gender-based
        if any(keyword in text for keyword in ["men", "male", "man's", "boys"]):
            demographics.append("male")
        elif any(keyword in text for keyword in ["women", "female", "woman's", "girls", "ladies"]):
            demographics.append("female")
        else:
            demographics.append("unisex")
            
        # Income-based (simplified by price)
        if price < 1000:
            demographics.append("budget-conscious")
        elif price > 10000:
            demographics.append("affluent")
        else:
            demographics.append("middle-income")
            
        return demographics
        
    def _generate_semantic_tags(self, record: Dict[str, Any]) -> List[str]:
        """Generate semantic tags for enhanced search"""
        
        tags = set()
        
        # Add category-based semantic tags
        category = record.get("category", {}).get("name", "").lower()
        
        semantic_mapping = {
            "electronics": ["technology", "digital", "gadget", "device"],
            "food": ["consumable", "edible", "nutrition", "grocery"],
            "clothing": ["wearable", "fashion", "apparel", "textile"],
            "home": ["household", "domestic", "living", "interior"],
            "beauty": ["cosmetic", "personal-care", "grooming", "wellness"],
            "sports": ["fitness", "athletic", "exercise", "recreation"],
            "books": ["educational", "knowledge", "literature", "reading"],
            "toys": ["entertainment", "play", "recreation", "fun"]
        }
        
        for key, sem_tags in semantic_mapping.items():
            if key in category:
                tags.update(sem_tags)
                
        # Add feature-based tags
        features = self._extract_product_features(record)
        for feature_type, feature_values in features.items():
            if feature_values:
                tags.add(f"has-{feature_type}")
                
        # Add price-based tags
        price_value, _ = self._safe_get_price_value(record.get("price"))
        price_segment = self._categorize_price_segment(price_value)
        tags.add(f"price-{price_segment}")
        
        # Add availability tags
        if record.get("availability", True):
            tags.add("available")
        else:
            tags.add("out-of-stock")
            
        # Add quality tags
        quality_score = self._calculate_quality_score(record)["score"]
        if quality_score >= 80:
            tags.add("high-quality")
        elif quality_score >= 60:
            tags.add("good-quality")
        else:
            tags.add("basic-quality")
            
        return list(tags)
        
    def _calculate_completeness_score(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data completeness score"""
        
        required_fields = ["id", "name", "price"]
        important_fields = ["description", "category", "provider", "images"]
        optional_fields = ["rating", "tags", "attributes", "location"]
        
        scores = {}
        total_score = 0
        
        # Required fields (40 points)
        required_score = 0
        for field in required_fields:
            if field in record and record[field]:
                if field == "price":
                    _, price_valid = self._safe_get_price_value(record.get(field))
                    if price_valid:
                        required_score += 40 / len(required_fields)
                else:
                    required_score += 40 / len(required_fields)
        scores["required"] = required_score
        total_score += required_score
        
        # Important fields (40 points)
        important_score = 0
        for field in important_fields:
            if field in record and record[field]:
                if field == "images":
                    if len(record[field]) > 0:
                        important_score += 40 / len(important_fields)
                else:
                    important_score += 40 / len(important_fields)
        scores["important"] = important_score
        total_score += important_score
        
        # Optional fields (20 points)
        optional_score = 0
        for field in optional_fields:
            if field in record and record[field]:
                optional_score += 20 / len(optional_fields)
        scores["optional"] = optional_score
        total_score += optional_score
        
        return {
            "score": min(100, total_score),
            "breakdown": scores,
            "missing_required": [f for f in required_fields if not record.get(f)],
            "missing_important": [f for f in important_fields if not record.get(f)]
        }