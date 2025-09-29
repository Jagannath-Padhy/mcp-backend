"""Product Enrichment Service - DEPRECATED"""

from typing import Dict, List, Optional, Any
import logging

# CartItem removed - product enrichment now handled in pure backend mode
from ..buyer_backend_client import BuyerBackendClient, get_buyer_backend_client
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProductEnrichmentService:
    """
    DEPRECATED: Product enrichment now handled in pure backend mode
    This service is kept for backward compatibility only
    """
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """Initialize service"""
        self.buyer_app = buyer_backend_client or get_buyer_backend_client()
        logger.warning("ProductEnrichmentService is deprecated - enrichment now done in backend")
    
    async def enrich_cart_items(self, cart_items: List[Dict[str, Any]], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """DEPRECATED: Return items unchanged"""
        logger.warning("enrich_cart_items is deprecated - returning items unchanged")
        return cart_items
    
    async def _enrich_single_item(self, item: Dict[str, Any], batch_data: Optional[Dict]) -> Dict[str, Any]:
        """DEPRECATED: Return item unchanged"""
        return item
    
    def _find_item_in_batch_data(self, item: Dict[str, Any], batch_data: List[Dict]) -> Optional[Dict]:
        """DEPRECATED: Return None"""
        return None
    
    def _apply_enrichment_data(self, item: Dict[str, Any], enriched_data: Dict) -> Dict[str, Any]:
        """DEPRECATED: Return item unchanged"""
        return item
    
    def _create_provider_structure(self, provider_details: Dict, location_details: Dict, 
                                   item: Dict[str, Any], bpp_id: str) -> Dict:
        """DEPRECATED: Return empty dict"""
        return {}


# Singleton instance
_enrichment_service: Optional[ProductEnrichmentService] = None


def get_product_enrichment_service() -> ProductEnrichmentService:
    """Get singleton ProductEnrichmentService instance"""
    global _enrichment_service
    if _enrichment_service is None:
        _enrichment_service = ProductEnrichmentService()
    return _enrichment_service