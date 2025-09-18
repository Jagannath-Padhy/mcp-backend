"""BIAP Validation Service - matches Node.js validation logic"""

from typing import List, Optional, Dict, Any
from ..models.session import CartItem
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BiapValidationService:
    """
    BIAP validation service matching Node.js implementation
    Prevents checkout with items from multiple BPPs or providers
    """
    
    def are_multiple_bpp_items_selected(self, items: Optional[List[CartItem]]) -> bool:
        """
        Check if items from multiple BPPs are selected
        Matches BIAP areMultipleBppItemsSelected function
        
        Args:
            items: List of cart items to validate
            
        Returns:
            True if items from multiple BPPs are selected
        """
        if not items:
            return False
            
        try:
            # Get unique BPP IDs - matches BIAP logic: [...new Set(items.map(item => item.bpp_id))].length > 1
            bpp_ids = set()
            for item in items:
                if item.bpp_id:
                    bpp_ids.add(item.bpp_id)
            
            result = len(bpp_ids) > 1
            if result:
                logger.warning(f"[BiapValidation] Multiple BPPs detected: {list(bpp_ids)}")
            
            return result
            
        except Exception as e:
            logger.error(f"[BiapValidation] Error checking multiple BPPs: {e}")
            return False
    
    def are_multiple_provider_items_selected(self, items: Optional[List[CartItem]]) -> bool:
        """
        Check if items from multiple providers are selected  
        Matches BIAP areMultipleProviderItemsSelected function
        
        Args:
            items: List of cart items to validate
            
        Returns:
            True if items from multiple providers are selected
        """
        if not items:
            return False
            
        try:
            # Get unique provider IDs - matches BIAP logic: [...new Set(items.map(item => item.provider.id))].length > 1
            provider_ids = set()
            for item in items:
                if item.provider and isinstance(item.provider, dict):
                    provider_id = item.provider.get('id') or item.provider.get('local_id')
                    if provider_id:
                        provider_ids.add(provider_id)
            
            result = len(provider_ids) > 1
            if result:
                logger.warning(f"[BiapValidation] Multiple providers detected: {list(provider_ids)}")
            
            return result
            
        except Exception as e:
            logger.error(f"[BiapValidation] Error checking multiple providers: {e}")
            return False
    
    def validate_order_items(self, items: Optional[List[CartItem]], operation: str = "select") -> Dict[str, Any]:
        """
        Validate order items for BIAP compliance
        Matches BIAP validation flow used in select, init, and confirm operations
        
        Args:
            items: List of cart items to validate
            operation: Operation being performed (select, init, confirm)
            
        Returns:
            Validation result with success/error info
        """
        if not items:
            return {
                "success": False,
                "error": {"message": "Empty order received"}
            }
        
        # Check for multiple BPP items - matches BIAP logic
        if self.are_multiple_bpp_items_selected(items):
            return {
                "success": False,
                "error": {"message": "More than one BPP's item(s) selected/initialized"}
            }
        
        # Check for multiple provider items - matches BIAP logic
        if self.are_multiple_provider_items_selected(items):
            return {
                "success": False, 
                "error": {"message": "More than one Provider's item(s) selected/initialized"}
            }
        
        logger.info(f"[BiapValidation] {operation.capitalize()} validation passed for {len(items)} items")
        return {"success": True}
    
    def get_order_bpp_info(self, items: Optional[List[CartItem]]) -> Optional[Dict[str, str]]:
        """
        Get BPP information from validated order items
        Used after validation passes to extract BPP details
        
        Args:
            items: List of cart items (already validated)
            
        Returns:
            Dictionary with bpp_id and bpp_uri or None
        """
        if not items:
            return None
            
        for item in items:
            if item.bpp_id:
                return {
                    "bpp_id": item.bpp_id,
                    "bpp_uri": item.bpp_uri or ""
                }
        
        return None
    
    def get_order_provider_info(self, items: Optional[List[CartItem]]) -> Optional[Dict[str, Any]]:
        """
        Get provider information from validated order items
        Used after validation passes to extract provider details
        
        Args:
            items: List of cart items (already validated)
            
        Returns:
            Provider info dictionary or None
        """
        if not items:
            return None
            
        for item in items:
            if item.provider and isinstance(item.provider, dict):
                return item.provider
        
        return None


# Singleton instance
_biap_validation_service: Optional[BiapValidationService] = None


def get_biap_validation_service() -> BiapValidationService:
    """Get singleton BiapValidationService instance"""
    global _biap_validation_service
    if _biap_validation_service is None:
        _biap_validation_service = BiapValidationService()
    return _biap_validation_service