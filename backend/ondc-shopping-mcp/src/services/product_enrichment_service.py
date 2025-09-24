"""Product Enrichment Service - BIAP Compatible"""

from typing import Dict, List, Optional, Any
import logging

from ..models.session import CartItem
from ..buyer_backend_client import BuyerBackendClient, get_buyer_backend_client
from ..utils.logger import get_logger
# Removed fake data imports - using only real data from cart items

logger = get_logger(__name__)


class ProductEnrichmentService:
    """
    Service for enriching cart items with full product data from BIAP APIs
    Matches the Node.js selectOrder.service.js enrichment logic
    """
    
    def __init__(self, buyer_backend_client: Optional[BuyerBackendClient] = None):
        """Initialize product enrichment service"""
        self.buyer_app = buyer_backend_client or get_buyer_backend_client()
        logger.info("ProductEnrichmentService initialized")
    
    async def enrich_cart_items(self, cart_items: List[CartItem], user_id: Optional[str] = None) -> List[CartItem]:
        """
        Enrich cart items with full product data from BIAP APIs
        Matches BIAP selectOrder.service.js logic
        
        Args:
            cart_items: List of cart items to enrich
            user_id: User ID for API calls
            
        Returns:
            List of enriched cart items
        """
        if not cart_items:
            return cart_items
            
        logger.info(f"[ProductEnrichment] Enriching {len(cart_items)} cart items")
        
        try:
            # Step 1: Collect item and provider IDs for batch API call
            product_ids = []
            provider_ids = []
            
            for item in cart_items:
                if item.local_id:
                    product_ids.append(item.local_id)
                if item.provider and isinstance(item.provider, dict):
                    provider_id = item.provider.get('id') or item.provider.get('local_id')
                    if provider_id:
                        provider_ids.append(provider_id)
            
            # Remove duplicates
            product_ids = list(set(product_ids))
            provider_ids = list(set(provider_ids))
            
            logger.info(f"[ProductEnrichment] Calling protocolGetItemList with {len(product_ids)} products, {len(provider_ids)} providers")
            
            # Step 2: Call BIAP protocolGetItemList for batch enrichment (may fail with 404)
            enriched_data = None
            if product_ids or provider_ids:
                try:
                    enriched_data = await self.buyer_app.get_item_list({
                        "itemIds": ",".join(product_ids) if product_ids else "",
                        "providerIds": ",".join(provider_ids) if provider_ids else ""
                    })
                    # Check if response indicates an error
                    if enriched_data and 'error' in enriched_data:
                        logger.warning(f"[ProductEnrichment] Batch API error: {enriched_data.get('error')}")
                        enriched_data = None
                except Exception as e:
                    logger.warning(f"[ProductEnrichment] Batch enrichment failed (will use cart data): {e}")
                    enriched_data = None
            
            # Step 3: Enrich each cart item
            enriched_items = []
            for item in cart_items:
                try:
                    enriched_item = await self._enrich_single_item(item, enriched_data)
                    enriched_items.append(enriched_item)
                except Exception as e:
                    logger.error(f"[ProductEnrichment] Failed to enrich item {item.id}: {e}")
                    # Use original item if enrichment fails
                    enriched_items.append(item)
            
            logger.info(f"[ProductEnrichment] Successfully enriched {len(enriched_items)} items")
            return enriched_items
            
        except Exception as e:
            logger.error(f"[ProductEnrichment] Failed to enrich cart items: {e}")
            # Return original items if enrichment completely fails
            return cart_items
    
    async def _enrich_single_item(self, item: CartItem, batch_data: Optional[Dict]) -> CartItem:
        """
        Enrich a single cart item with product data
        
        Args:
            item: Cart item to enrich
            batch_data: Data from protocolGetItemList batch call
            
        Returns:
            Enriched cart item
        """
        enriched_data = None
        
        # Step 1: Try to find item in batch data
        if batch_data and isinstance(batch_data, dict) and batch_data.get('data'):
            enriched_data = self._find_item_in_batch_data(item, batch_data['data'])
        
        # Step 2: Fallback to individual API call if not found in batch
        if not enriched_data or not enriched_data.get('item_details'):
            logger.debug(f"[ProductEnrichment] Item {item.local_id} not found in batch, trying individual API")
            try:
                individual_response = await self.buyer_app.get_item_details({'id': item.id})
                if individual_response and 'error' not in individual_response:
                    enriched_data = individual_response
                else:
                    logger.debug(f"[ProductEnrichment] Individual API also failed for {item.id}")
            except Exception as e:
                logger.debug(f"[ProductEnrichment] Individual enrichment failed for {item.id}: {e}")
        
        # Step 3: Apply enrichment if data available, otherwise return item as-is
        if enriched_data:
            return self._apply_enrichment_data(item, enriched_data)
        else:
            # Cart item already has real provider data from search results
            logger.debug(f"[ProductEnrichment] API enrichment not available, using existing real data for item {item.id}")
            return item
    
    def _find_item_in_batch_data(self, item: CartItem, batch_data: List[Dict]) -> Optional[Dict]:
        """
        Find item data in batch response
        
        Args:
            item: Cart item to find
            batch_data: List of item data from batch API
            
        Returns:
            Matching item data or None
        """
        if not isinstance(batch_data, list):
            return None
            
        for item_data in batch_data:
            if (item_data.get('item_details', {}).get('id') == item.local_id or
                item_data.get('item_details', {}).get('id') == item.id):
                return item_data
        
        return None
    
    def _apply_enrichment_data(self, item: CartItem, enriched_data: Dict) -> CartItem:
        """
        Apply enriched data to cart item - matches BIAP logic
        
        Args:
            item: Original cart item
            enriched_data: Enriched data from API
            
        Returns:
            New enriched cart item
        """
        try:
            # Extract enriched fields like BIAP does
            context = enriched_data.get('context', {})
            item_details = enriched_data.get('item_details', {})
            provider_details = enriched_data.get('provider_details', {})
            location_details = enriched_data.get('location_details', {})
            
            # Calculate subtotal like BIAP
            subtotal = item_details.get('price', {}).get('value', item.price)
            
            # Create enriched item - matches BIAP structure
            return CartItem(
                # Keep original basic fields
                id=item.id,
                name=item.name,
                price=item.price,
                quantity=item.quantity,
                local_id=item.local_id,
                category=item.category,
                image_url=item.image_url,
                description=item.description,
                
                # Update with enriched BIAP fields
                bpp_id=context.get('bpp_id', item.bpp_id),
                bpp_uri=context.get('bpp_uri', item.bpp_uri),
                contextCity=context.get('city', item.contextCity),
                
                # Enriched product details
                # CRITICAL: Ensure location_id is always set for backend transformation
                product={
                    'subtotal': subtotal,
                    **item_details,
                    'location_id': location_details.get('id') or location_details.get('local_id')
                },
                
                # Enriched provider details with proper BIAP structure
                provider=self._create_provider_structure(
                    provider_details, 
                    location_details, 
                    item,
                    context.get('bpp_id', item.bpp_id)
                ),
                
                # Keep existing optional fields
                fulfillment_id=item.fulfillment_id,
                parent_item_id=item.parent_item_id,
                tags=item.tags,
                customisations=item.customisations
            )
            
        except Exception as e:
            logger.error(f"[ProductEnrichment] Failed to apply enrichment data: {e}")
            return item
    
    def _create_provider_structure(self, provider_details: Dict, location_details: Dict, 
                                   item: CartItem, bpp_id: str) -> Dict:
        """
        Preserve real provider structure from API response - NO fake data generation
        
        Args:
            provider_details: Real provider data from API
            location_details: Real location data from API
            item: CartItem to update
            bpp_id: BPP ID from context
            
        Returns:
            Real provider structure from API response
            
        Raises:
            ValueError: If provider data is missing or invalid
        """
        if not provider_details or not provider_details.get('id'):
            error_msg = f"Missing real provider data for item {item.name}"
            logger.error(f"[ProductEnrichment] {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"[ProductEnrichment] Using REAL provider data from API")
        logger.info(f"  - Provider ID: {provider_details.get('id')}")
        logger.info(f"  - Location ID: {location_details.get('id') if location_details else 'None'}")
        
        # Use real provider structure directly from API
        provider_structure = provider_details
        
        # Extract location_id for cart_service
        location_id = None
        if location_details:
            location_id = location_details.get('local_id') or location_details.get('id')
        elif provider_structure.get('locations') and len(provider_structure['locations']) > 0:
            first_location = provider_structure['locations'][0]
            location_id = first_location.get('local_id') or first_location.get('id')
        
        if not location_id:
            error_msg = f"Missing location data for item {item.name}"
            logger.error(f"[ProductEnrichment] {error_msg}")
            raise ValueError(error_msg)
        
        # Set location_id on item for extraction by cart_service
        item.location_id = location_id
        
        logger.info(f"[ProductEnrichment] Real provider structure preserved - Provider: {provider_structure['id']}, Location: {location_id}")
        return provider_structure


# Singleton instance
_product_enrichment_service: Optional[ProductEnrichmentService] = None


def get_product_enrichment_service() -> ProductEnrichmentService:
    """Get singleton ProductEnrichmentService instance"""
    global _product_enrichment_service
    if _product_enrichment_service is None:
        _product_enrichment_service = ProductEnrichmentService()
    return _product_enrichment_service