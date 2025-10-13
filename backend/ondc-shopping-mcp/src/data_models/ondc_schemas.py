"""
Centralized ONDC Schema Definitions
====================================

Single source of truth for all ONDC data structures.
These schemas are used across MCP tools, services, and backend integrations.

Based on actual Himira backend requirements from Postman collection.
"""

from typing import Dict, Any, List, Optional
import uuid


class ONDCSchemas:
    """Centralized ONDC schema definitions following DRY principle"""
    
    @staticmethod
    def get_provider_location_schema() -> Dict[str, Any]:
        """
        Provider location structure required for SELECT operations
        
        Returns:
            Dict containing location schema
        """
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Full location ID (bpp_domain_provider_location format)"
                },
                "local_id": {
                    "type": "string", 
                    "description": "Local location UUID"
                }
            },
            "required": ["id", "local_id"]
        }
    
    @staticmethod
    def get_provider_schema() -> Dict[str, Any]:
        """
        Provider structure required by backend
        
        Returns:
            Dict containing provider schema
        """
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Full provider ID (bpp_domain_provider format)"
                },
                "local_id": {
                    "type": "string",
                    "description": "Local provider UUID"
                },
                "locations": {
                    "type": "array",
                    "description": "Provider locations (required for SELECT)",
                    "items": ONDCSchemas.get_provider_location_schema()
                }
            },
            "required": ["id", "local_id", "locations"]
        }
    
    @staticmethod
    def get_quantity_schema() -> Dict[str, Any]:
        """
        Quantity structure for cart items
        
        Returns:
            Dict containing quantity schema
        """
        return {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Item quantity",
                    "minimum": 1
                }
            },
            "required": ["count"]
        }
    
    @staticmethod
    def get_cart_item_schema(minimal: bool = False) -> Dict[str, Any]:
        """
        Cart item structure for SELECT/INIT/CONFIRM operations
        
        Args:
            minimal: If True, returns minimal schema for user input
                    If False, returns full backend schema
        
        Returns:
            Dict containing cart item schema
        """
        if minimal:
            # Minimal schema for user input (Langflow compatible)
            return {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Product name (required)"
                    },
                    "price": {
                        "type": "number",
                        "description": "Product price"
                    },
                    "id": {
                        "type": "string",
                        "description": "Product ID (optional, auto-detected from search)"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity to add (default: 1)",
                        "default": 1
                    }
                },
                "required": ["name"]
            }
        else:
            # Full backend schema
            return {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Full ONDC item ID (bpp_domain_provider_item format)"
                    },
                    "local_id": {
                        "type": "string",
                        "description": "Local item UUID"
                    },
                    "bpp_id": {
                        "type": "string",
                        "description": "Backend provider platform ID"
                    },
                    "bpp_uri": {
                        "type": "string",
                        "description": "Backend provider API URI"
                    },
                    "contextCity": {
                        "type": "string",
                        "description": "City/area code (e.g., std:0172)"
                    },
                    "provider": ONDCSchemas.get_provider_schema(),
                    "quantity": ONDCSchemas.get_quantity_schema(),
                    "customisations": {
                        "type": ["array", "null"],
                        "description": "Item customisations",
                        "default": []
                    },
                    "hasCustomisations": {
                        "type": "boolean",
                        "description": "Whether item has customisations",
                        "default": False
                    },
                    "customisationState": {
                        "type": "object",
                        "description": "Customisation state tracking",
                        "default": {}
                    }
                },
                "required": ["id", "local_id", "provider", "quantity"]
            }
    
    @staticmethod
    def get_select_request_schema() -> Dict[str, Any]:
        """
        SELECT request structure for backend API
        
        Returns:
            Dict containing SELECT request schema
        """
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "default": "ONDC:RET10"
                            },
                            "city": {
                                "type": "string",
                                "description": "Area code"
                            }
                        },
                        "required": ["domain", "city"]
                    },
                    "message": {
                        "type": "object",
                        "properties": {
                            "cart": {
                                "type": "object",
                                "properties": {
                                    "items": {
                                        "type": "array",
                                        "items": ONDCSchemas.get_cart_item_schema(minimal=False)
                                    }
                                },
                                "required": ["items"]
                            },
                            "fulfillments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "end": {
                                            "type": "object",
                                            "properties": {
                                                "location": {
                                                    "type": "object",
                                                    "properties": {
                                                        "gps": {"type": "string"},
                                                        "address": {
                                                            "type": "object",
                                                            "properties": {
                                                                "area_code": {"type": "string"}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "required": ["cart"]
                    },
                    "userId": {"type": "string"},
                    "deviceId": {"type": "string"}
                },
                "required": ["context", "message"]
            }
        }
    
    @staticmethod
    def get_search_item_schema() -> Dict[str, Any]:
        """
        Search result item structure
        
        Returns:
            Dict containing search item schema
        """
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "price": {"type": "number"},
                "description": {"type": "string"},
                "category": {"type": "string"},
                "seller": {"type": "string"},
                "image_url": {"type": "string"},
                "in_stock": {"type": "boolean"},
                "rating": {"type": "number"},
                "provider": ONDCSchemas.get_provider_schema(),
                "raw": {
                    "type": "object",
                    "description": "Complete product data (for backend compatibility)"
                }
            }
        }


class ONDCDataFactory:
    """Factory for creating ONDC-compliant data structures"""
    
    @staticmethod
    def create_cart_item(item_data: Dict[str, Any], auto_enrich: bool = True) -> Dict[str, Any]:
        """
        Create ONDC-compliant cart item from user input
        
        Args:
            item_data: User provided item data (can be minimal)
            auto_enrich: Whether to auto-enrich with provider data
            
        Returns:
            ONDC-compliant cart item
        """
        from ..utils.himira_provider_constants import (
            enrich_cart_item_with_provider,
            HIMIRA_PROVIDER_LOCAL_ID
        )
        
        # Handle minimal input
        if auto_enrich and not item_data.get('provider'):
            # Auto-enrich with Himira provider data
            item_data = enrich_cart_item_with_provider(item_data)
        
        # Extract or generate local_id
        local_id = item_data.get('local_id')
        if not local_id:
            if item_data.get('id'):
                # Try to extract from full ONDC ID
                id_parts = item_data['id'].split('_')
                if len(id_parts) >= 4:
                    local_id = id_parts[-1]
                else:
                    local_id = str(uuid.uuid4())
            else:
                local_id = str(uuid.uuid4())
        
        # Build cart item
        cart_item = {
            "id": item_data.get('id', ''),
            "local_id": local_id,
            "quantity": {
                "count": item_data.get('quantity', 1)
            },
            "provider": item_data.get('provider', {}),
            "customisations": item_data.get('customisations', None),
            "hasCustomisations": item_data.get('hasCustomisations', False),
            "customisationState": item_data.get('customisationState', {})
        }
        
        # Add backend-specific fields if present
        if item_data.get('bpp_id'):
            cart_item['bpp_id'] = item_data['bpp_id']
        if item_data.get('bpp_uri'):
            cart_item['bpp_uri'] = item_data['bpp_uri']
        if item_data.get('contextCity'):
            cart_item['contextCity'] = item_data['contextCity']
        
        return cart_item
    
    @staticmethod
    def create_select_payload(cart_items: List[Dict[str, Any]], 
                            city: str = "140301",
                            user_id: str = None,
                            device_id: str = None) -> List[Dict[str, Any]]:
        """
        Create SELECT request payload for backend
        
        Args:
            cart_items: List of cart items
            city: Area code
            user_id: User ID
            device_id: Device ID
            
        Returns:
            SELECT request payload
        """
        return [{
            "context": {
                "domain": "ONDC:RET10",
                "city": city
            },
            "message": {
                "cart": {
                    "items": cart_items
                },
                "fulfillments": [{
                    "end": {
                        "location": {
                            "gps": "30.745765,76.653633",
                            "address": {
                                "area_code": city
                            }
                        }
                    }
                }]
            },
            "userId": user_id or "anonymous",
            "deviceId": device_id or str(uuid.uuid4())
        }]


# Export convenience functions
def get_minimal_item_schema() -> Dict[str, Any]:
    """Get minimal item schema for MCP tools"""
    return ONDCSchemas.get_cart_item_schema(minimal=True)


def get_full_item_schema() -> Dict[str, Any]:
    """Get full item schema for backend"""
    return ONDCSchemas.get_cart_item_schema(minimal=False)


def create_cart_item(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create ONDC cart item from data"""
    return ONDCDataFactory.create_cart_item(data)


def create_select_payload(items: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Create SELECT payload for backend"""
    return ONDCDataFactory.create_select_payload(items, **kwargs)