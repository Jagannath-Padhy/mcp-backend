"""BIAP Validation Service - DEPRECATED"""

from typing import List, Optional, Dict, Any
# CartItem removed - validation now uses backend cart data
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BiapValidationService:
    """
    DEPRECATED: BIAP validation now handled in pure backend mode
    This service is kept for backward compatibility only
    """
    
    def are_multiple_bpp_items_selected(self, items: Optional[List[Dict[str, Any]]]) -> bool:
        """
        DEPRECATED: Check if items from multiple BPPs are selected
        Now returns False as validation is done in backend
        """
        logger.warning("are_multiple_bpp_items_selected is deprecated - returning False")
        return False
    
    def are_multiple_provider_items_selected(self, items: Optional[List[Dict[str, Any]]]) -> bool:
        """
        DEPRECATED: Check if items from multiple providers are selected
        Now returns False as validation is done in backend
        """
        logger.warning("are_multiple_provider_items_selected is deprecated - returning False")
        return False
    
    def validate_order_items(self, items: Optional[List[Dict[str, Any]]], stage: str = "select") -> Dict[str, Any]:
        """
        DEPRECATED: Validate order items for BIAP compliance
        Now always returns success as validation is done in backend
        """
        logger.warning(f"validate_order_items is deprecated for stage '{stage}' - returning success")
        return {
            'success': True,
            'message': 'Validation passed (backend mode)'
        }
    
    def get_order_bpp_info(self, items: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: Get BPP info from order items
        Now returns None as info is managed in backend
        """
        logger.warning("get_order_bpp_info is deprecated - returning None")
        return None


# Singleton instance
_biap_validation_service: Optional[BiapValidationService] = None


def get_biap_validation_service() -> BiapValidationService:
    """Get singleton BiapValidationService instance"""
    global _biap_validation_service
    if _biap_validation_service is None:
        _biap_validation_service = BiapValidationService()
    return _biap_validation_service