"""Query Analysis for Intelligent Search Result Sizing

Analyzes search queries to determine search intent and optimal result configuration.
"""

import re
from typing import Dict, Any, Optional, List
from enum import Enum
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SearchIntent(Enum):
    """Search intent categories"""
    SPECIFIC = "specific"        # Specific product search: "organic basmati rice"
    BROAD = "broad"             # Category/brand search: "spices", "amul products"  
    DISCOVERY = "discovery"     # Exploratory search: "healthy snacks", "breakfast items"
    GENERIC = "generic"         # Very broad search: "food", "groceries"


class QueryConfig:
    """Configuration for search query execution"""
    
    def __init__(self, 
                 intent: SearchIntent,
                 min_results: int,
                 max_results: int, 
                 relevance_threshold: float,
                 description: str):
        self.intent = intent
        self.min_results = min_results
        self.max_results = max_results
        self.relevance_threshold = relevance_threshold
        self.description = description


class QueryAnalyzer:
    """Intelligent query analysis for dynamic search configuration"""
    
    def __init__(self):
        """Initialize analyzer with pattern matching and configuration"""
        
        # Brand patterns for specific searches
        self.brand_patterns = [
            r'\b(amul|britannia|parle|haldirams|mtr|everest|mdh|aashirvaad|fortune|sundrop|dalda)\b',
            r'\b(tata|reliance|patanjali|organic|dabur|himalaya|nestle|maggi)\b'
        ]
        
        # Specific product indicators
        self.specific_indicators = [
            r'\b\d+\s*(kg|gm|gram|liter|litre|ml|pack|packet)\b',  # Quantity specified
            r'\borganics?\b',                                       # Organic products
            r'\b(brand|premium|pure|natural|fresh)\b',              # Quality indicators
            r'\b[a-z]+ rice\b',                                     # Specific rice types
            r'\boil (coconut|olive|mustard|sunflower)\b'            # Specific oils
        ]
        
        # Broad category patterns
        self.category_patterns = [
            r'\b(spices?|masalas?|seasonings?)\b',
            r'\b(dairy|milk|cheese|yogurt|paneer)\b',
            r'\b(rice|grains?|cereals?)\b',
            r'\b(oils?|ghee|butter)\b',
            r'\b(snacks?|namkeen|chips)\b',
            r'\b(tea|coffee|beverages?)\b'
        ]
        
        # Discovery patterns
        self.discovery_patterns = [
            r'\b(healthy|nutritious|diet|fitness)\b',
            r'\b(breakfast|lunch|dinner|meal)\b',
            r'\b(kids?|children|baby|infant)\b',
            r'\b(festival|party|celebration)\b',
            r'\b(regional|traditional|authentic)\b',
            r'\b(quick|instant|ready)\b'
        ]
        
        # Generic/very broad patterns  
        self.generic_patterns = [
            r'^(food|grocery|groceries|items?|products?|stuff)$',
            r'^(indian|south indian|north indian)$',
            r'^(vegetarian|vegan|non.?veg)$'
        ]
        
        # Default configurations for each intent type
        self.intent_configs = {
            SearchIntent.SPECIFIC: QueryConfig(
                intent=SearchIntent.SPECIFIC,
                min_results=2,
                max_results=8,
                relevance_threshold=0.8,
                description="Specific product search - high relevance required"
            ),
            SearchIntent.BROAD: QueryConfig(
                intent=SearchIntent.BROAD,
                min_results=5,
                max_results=15,
                relevance_threshold=0.7,
                description="Category/brand search - balanced results"
            ),
            SearchIntent.DISCOVERY: QueryConfig(
                intent=SearchIntent.DISCOVERY,
                min_results=8,
                max_results=25,
                relevance_threshold=0.6,
                description="Discovery search - comprehensive results"
            ),
            SearchIntent.GENERIC: QueryConfig(
                intent=SearchIntent.GENERIC,
                min_results=10,
                max_results=30,
                relevance_threshold=0.5,
                description="Generic search - maximum coverage"
            )
        }
    
    def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryConfig:
        """
        Analyze a search query to determine optimal search configuration
        
        Args:
            query: Search query string
            context: Optional context like user preferences, session history
            
        Returns:
            QueryConfig with optimal search parameters
        """
        if not query or not query.strip():
            return self.intent_configs[SearchIntent.GENERIC]
            
        query_lower = query.lower().strip()
        intent = self._determine_intent(query_lower)
        config = self.intent_configs[intent]
        
        # Apply context-based adjustments if available
        if context:
            config = self._apply_context_adjustments(config, context)
        
        logger.info(f"[QueryAnalyzer] '{query}' -> {intent.value} "
                   f"(results: {config.min_results}-{config.max_results}, "
                   f"threshold: {config.relevance_threshold})")
        
        return config
    
    def _determine_intent(self, query: str) -> SearchIntent:
        """Determine search intent from query patterns"""
        
        # Check for specific product indicators
        if self._matches_patterns(query, self.specific_indicators):
            return SearchIntent.SPECIFIC
            
        # Check for brand names (also indicates specific intent)
        if self._matches_patterns(query, self.brand_patterns):
            return SearchIntent.SPECIFIC
        
        # Check for generic patterns first (most restrictive)
        if self._matches_patterns(query, self.generic_patterns):
            return SearchIntent.GENERIC
            
        # Check for discovery patterns
        if self._matches_patterns(query, self.discovery_patterns):
            return SearchIntent.DISCOVERY
            
        # Check for category patterns
        if self._matches_patterns(query, self.category_patterns):
            return SearchIntent.BROAD
        
        # Additional heuristics based on query characteristics
        words = query.split()
        
        # Single word queries are usually broad categories
        if len(words) == 1:
            return SearchIntent.BROAD
            
        # Very long queries (>5 words) are often discovery-oriented
        if len(words) > 5:
            return SearchIntent.DISCOVERY
            
        # 2-3 word queries are often specific products
        if 2 <= len(words) <= 3:
            return SearchIntent.SPECIFIC
            
        # Default to broad for 4-5 word queries
        return SearchIntent.BROAD
    
    def _matches_patterns(self, query: str, patterns: List[str]) -> bool:
        """Check if query matches any of the given patterns"""
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _apply_context_adjustments(self, config: QueryConfig, context: Dict[str, Any]) -> QueryConfig:
        """Apply context-based adjustments to the base configuration"""
        
        # Create a copy to avoid modifying the original
        adjusted_config = QueryConfig(
            intent=config.intent,
            min_results=config.min_results,
            max_results=config.max_results,
            relevance_threshold=config.relevance_threshold,
            description=config.description
        )
        
        # User preference adjustments
        user_prefs = context.get('user_preferences', {})
        result_density = user_prefs.get('result_density', 'balanced')  # minimal, balanced, comprehensive
        
        if result_density == 'minimal':
            adjusted_config.max_results = min(adjusted_config.max_results, adjusted_config.min_results + 3)
            adjusted_config.relevance_threshold = min(0.9, adjusted_config.relevance_threshold + 0.1)
        elif result_density == 'comprehensive':
            adjusted_config.max_results = min(30, int(adjusted_config.max_results * 1.5))
            adjusted_config.relevance_threshold = max(0.4, adjusted_config.relevance_threshold - 0.1)
        
        # Session history adjustments
        session_history = context.get('session_history', {})
        recent_searches = session_history.get('recent_search_count', 0)
        
        # If user has been searching frequently, they might want more comprehensive results
        if recent_searches > 3:
            adjusted_config.max_results = min(25, adjusted_config.max_results + 5)
        
        # Cart context - if cart has items, user might want more specific results
        cart_items = context.get('cart_items', 0)
        if cart_items > 0:
            adjusted_config.relevance_threshold = min(0.85, adjusted_config.relevance_threshold + 0.05)
        
        return adjusted_config
    
    def get_search_suggestions(self, query: str, intent: SearchIntent) -> List[str]:
        """Generate search improvement suggestions based on intent"""
        
        suggestions = []
        
        if intent == SearchIntent.GENERIC:
            suggestions = [
                "Try being more specific: 'basmati rice' instead of 'rice'",
                "Add brand names: 'amul butter' instead of 'butter'",
                "Include product type: 'organic spices' instead of 'spices'"
            ]
        elif intent == SearchIntent.BROAD:
            suggestions = [
                "Add specific variety: 'jeera rice' instead of 'rice'", 
                "Include quantity: '1kg flour' instead of 'flour'",
                "Try brand preference: 'aashirvaad atta' instead of 'atta'"
            ]
        elif intent == SearchIntent.DISCOVERY:
            suggestions = [
                "Browse by category to see all options",
                "Try related terms for more variety",
                "Consider filtering by price or brand"
            ]
        
        return suggestions[:2]  # Return max 2 suggestions


# Global instance
_query_analyzer = None


def get_query_analyzer() -> QueryAnalyzer:
    """Get singleton QueryAnalyzer instance"""
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer()
    return _query_analyzer