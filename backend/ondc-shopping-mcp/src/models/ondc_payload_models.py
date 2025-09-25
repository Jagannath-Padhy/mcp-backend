"""
ONDC Payload Models - Distinct structures for SELECT vs INIT requests

These models ensure proper payload format and prevent confusion between SELECT and INIT stages.
Based on working curl examples and ONDC specification.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass 
class ONDCContext:
    """Common ONDC context structure"""
    domain: str = "ONDC:RET10"
    transaction_id: Optional[str] = None
    city: str = ""  # Use pincode for city in context


@dataclass
class SelectItem:
    """Item structure for SELECT requests"""
    id: str  # Full ONDC item ID
    local_id: str  # Local item UUID
    customisationState: Dict = None
    quantity: Dict[str, int] = None  # {"count": 1}
    provider: Dict = None  # Provider with id, local_id, locations
    customisations: Optional[Any] = None
    hasCustomisations: bool = False


@dataclass
class SelectMessage:
    """Message structure for SELECT requests - uses cart.items format"""
    cart: Dict[str, List[Dict]]  # {"items": [...]}
    fulfillments: List[Dict]  # GPS and area_code info


@dataclass 
class SelectPayload:
    """Complete SELECT request payload"""
    context: Dict[str, str]  # Context with domain, city, transaction_id
    message: SelectMessage
    userId: str
    deviceId: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            'context': self.context,
            'message': {
                'cart': self.message.cart,
                'fulfillments': self.message.fulfillments
            },
            'userId': self.userId,
            'deviceId': self.deviceId
        }


@dataclass
class InitItem:
    """Item structure for INIT requests - different from SELECT"""
    id: str  # Full ONDC item ID  
    local_id: str  # Local item UUID
    quantity: Dict[str, int]  # {"count": 1}
    provider: Dict  # Provider info with id, local_id


@dataclass
class InitBillingInfo:
    """Billing information for INIT requests"""
    address: Dict[str, str]  # name, building, street, locality, city, state, etc.
    phone: str
    name: str
    email: str


@dataclass 
class InitDeliveryInfo:
    """Delivery information for INIT requests"""
    end: Dict[str, Any]  # location with gps, address details
    type: str = "Delivery"


@dataclass
class InitPayment:
    """Payment structure for INIT requests"""
    type: str = "ON-ORDER"  # Static value - COD not enabled


@dataclass
class InitMessage:
    """Message structure for INIT requests - items at root level (NOT cart.items)"""
    items: List[Dict]  # Items directly under message
    billing_info: InitBillingInfo
    delivery_info: InitDeliveryInfo  
    payment: InitPayment


@dataclass
class InitPayload:
    """Complete INIT request payload"""
    context: Dict[str, str]  # Context with transaction_id from SELECT
    message: InitMessage
    deviceId: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            'context': self.context,
            'message': {
                'items': self.message.items,  # Direct under message (NOT cart.items)
                'billing_info': self.message.billing_info.__dict__,
                'delivery_info': self.message.delivery_info.__dict__,
                'payment': self.message.payment.__dict__
            },
            'deviceId': self.deviceId
        }


# Validation functions
def validate_select_payload(payload: Dict[str, Any]) -> bool:
    """Validate SELECT payload has correct structure"""
    try:
        # Must have message.cart.items format
        return (
            'message' in payload and
            'cart' in payload['message'] and
            'items' in payload['message']['cart']
        )
    except (KeyError, TypeError):
        return False


def validate_init_payload(payload: Dict[str, Any]) -> bool:
    """Validate INIT payload has correct structure"""
    try:
        # Must have message.items format (NOT message.cart.items)
        return (
            'message' in payload and
            'items' in payload['message'] and
            'cart' not in payload['message']  # Should NOT have cart wrapper
        )
    except (KeyError, TypeError):
        return False


def create_select_payload(
    context: Dict[str, str],
    cart_items: List[Dict],
    fulfillments: List[Dict],
    user_id: str,
    device_id: str
) -> Dict[str, Any]:
    """Create properly formatted SELECT payload"""
    return {
        'context': context,
        'message': {
            'cart': {'items': cart_items},  # SELECT uses cart.items
            'fulfillments': fulfillments
        },
        'userId': user_id,
        'deviceId': device_id
    }


def create_init_payload(
    context: Dict[str, str],
    items: List[Dict],
    billing_info: Dict[str, Any],
    delivery_info: Dict[str, Any],
    device_id: str
) -> Dict[str, Any]:
    """Create properly formatted INIT payload"""
    return {
        'context': context,
        'message': {
            'items': items,  # INIT uses direct items (NOT cart.items)
            'billing_info': billing_info,
            'delivery_info': delivery_info,
            'payment': {'type': 'ON-ORDER'}  # Static payment type
        },
        'deviceId': device_id
    }


# Constants for payload validation
SELECT_REQUIRED_FIELDS = ['context', 'message', 'userId', 'deviceId']
SELECT_MESSAGE_REQUIRED_FIELDS = ['cart', 'fulfillments'] 
SELECT_CART_REQUIRED_FIELDS = ['items']

INIT_REQUIRED_FIELDS = ['context', 'message', 'deviceId']
INIT_MESSAGE_REQUIRED_FIELDS = ['items', 'billing_info', 'delivery_info', 'payment']

# Error messages
SELECT_VALIDATION_ERROR = "SELECT payload must have message.cart.items structure"
INIT_VALIDATION_ERROR = "INIT payload must have message.items structure (NOT message.cart.items)"