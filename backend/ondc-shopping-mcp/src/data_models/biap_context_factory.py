"""BIAP-compatible Context Factory matching Node.js implementation"""

import os
from typing import Dict, Optional
from datetime import datetime
import uuid


class BiapContextFactory:
    """
    BIAP-compatible context factory that matches the Node.js ContextFactory
    Uses proper ONDC protocol values from environment variables
    """
    
    def __init__(self):
        """Initialize with BIAP configuration from .env"""
        self.domain = os.getenv("DOMAIN", "ONDC:RET10")  #  Fixed to match Himira constants
        self.country = os.getenv("COUNTRY", "IND") 
        self.bap_id = os.getenv("BAP_ID", "hp-buyer-preprod.himira.co.in")
        self.bap_url = os.getenv("BAP_URL", "https://hp-buyer-backend-preprod.himira.co.in/protocol/v1")
        self.city_default = os.getenv("CITY", "std:080")
        self.timestamp = datetime.utcnow()
    
    def get_city_by_pincode(self, pincode: Optional[str], city: Optional[str] = None, action: Optional[str] = None) -> str:
        """
        Get city code - ALWAYS use pincode directly for ONDC compliance
        
        CRITICAL: Backend cart data provides area_code which should be used as city
        No conversion to std: format - use pincode directly as per working curl examples
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"[BiapContext] get_city_by_pincode called: pincode={pincode}, city={city}, action={action}")
        
        # ALWAYS use pincode directly - no std: format conversion
        if pincode:
            logger.info(f"[BiapContext] Using pincode as city directly (no conversion): {pincode}")
            return pincode
        
        # If no pincode provided, try city parameter
        if city:
            logger.info(f"[BiapContext] Using city parameter directly: {city}")
            return city
            
        # Final fallback to default
        logger.warning(f"[BiapContext] No pincode or city provided, using default: {self.city_default}")
        return self.city_default
    
    def get_transaction_id(self, transaction_id: Optional[str] = None) -> str:
        """Get or generate transaction ID"""
        if transaction_id:
            return transaction_id
        else:
            return str(uuid.uuid4())
    
    def create(self, context_params: Dict) -> Dict:
        """
        Create BIAP-compatible ONDC context structure
        Returns simplified format for SELECT operations to match backend expectations
        
        Args:
            context_params: Dictionary containing:
                - action: ONDC action (select, init, confirm)
                - transaction_id: Transaction ID (optional)
                - message_id: Message ID (optional) 
                - bpp_id: BPP ID (optional)
                - bpp_uri: BPP URI (optional)
                - city: City (optional)
                - state: State (optional)
                - pincode: Pincode for city mapping (optional)
                - domain: Domain override (optional)
        
        Returns:
            Simplified context for SELECT, full context for other operations
        """
        # Extract parameters
        action = context_params.get('action', 'select')
        transaction_id = context_params.get('transactionId') or context_params.get('transaction_id')
        message_id = context_params.get('messageId') or context_params.get('message_id', str(uuid.uuid4()))
        bpp_id = context_params.get('bppId') or context_params.get('bpp_id')
        bpp_uri = context_params.get('bpp_uri')
        city = context_params.get('city')
        state = context_params.get('state') 
        pincode = context_params.get('pincode')
        domain = context_params.get('domain', self.domain)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[BiapContext] create() params: action={action}, city={city}, pincode={pincode}")
        
        # Get city value based on action and pincode
        final_city = self.get_city_by_pincode(pincode, city, action)  # Pass action for proper city formatting
        logger.info(f"[BiapContext] Final city value for {action}: {final_city}")
        
        # For SELECT operations: Return simplified context to match backend expectations
        if action == 'select':
            logger.info("[BiapContext] SELECT operation - returning simplified context")
            return {
                "domain": domain,
                "city": final_city
            }
        
        # Full ONDC context structure for other operations (init, confirm, etc.)
        context = {
            "domain": domain,
            "country": self.country,
            "city": final_city,
            "action": action,
            "core_version": "1.2.0",  # PROTOCOL_VERSION.v_1_2_0
            "bap_id": self.bap_id,
            "bap_uri": self.bap_url,
            "transaction_id": self.get_transaction_id(transaction_id),
            "message_id": message_id,
            "timestamp": self.timestamp.isoformat() + "Z",
            "ttl": "PT30S"  # Protocol requirement
        }
        
        # Add BPP details if available
        if bpp_uri:
            context["bpp_uri"] = bpp_uri
        if bpp_id:
            context["bpp_id"] = bpp_id
            
        return context


# Singleton instance
_biap_context_factory = None

def get_biap_context_factory() -> BiapContextFactory:
    """Get singleton BiapContextFactory instance"""
    global _biap_context_factory
    if _biap_context_factory is None:
        _biap_context_factory = BiapContextFactory()
    return _biap_context_factory


def create_biap_context(action: str, transaction_id: Optional[str] = None, **kwargs) -> Dict:
    """
    Convenience function to create BIAP context
    
    Args:
        action: ONDC action
        transaction_id: Transaction ID (optional)
        **kwargs: Additional context parameters
    
    Returns:
        BIAP-compatible ONDC context
    """
    factory = get_biap_context_factory()
    context_params = {
        'action': action,
        'transaction_id': transaction_id,
        **kwargs
    }
    return factory.create(context_params)