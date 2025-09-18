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
        Get proper city code by pincode - matches BIAP Node.js implementation
        
        CRITICAL: For Himira backend SELECT requests, always use just the pincode as city
        This matches what the Himira documentation shows
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"[BiapContext] get_city_by_pincode called: pincode={pincode}, city={city}, action={action}")
        
        # For SELECT action, always use pincode directly (Himira requirement)
        if action == 'select' and pincode:
            logger.info(f"[BiapContext] SELECT ACTION: Using pincode as city directly: {pincode}")
            return pincode
        
        # For Himira area pincodes, use just the pincode (not std: format)
        himira_pincodes = ['140301', '140401', '140501', '160001', '160002']
        
        if pincode and pincode in himira_pincodes:
            return pincode  # Return just the pincode for Himira areas
        
        # For other areas, use standard city code mapping
        from ..utils.city_code_mapping import get_city_code_by_pincode, get_city_code_by_name
        
        if pincode:
            return get_city_code_by_pincode(pincode)
        elif city:
            # Try to get by city name if pincode not available
            return get_city_code_by_name(city)
        else:
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
        Matches the Node.js ContextFactory.create() method
        
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
            BIAP-compatible ONDC context
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
        
        # Create context structure matching BIAP Node.js
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