"""
ONDC Payload Models - Distinct structures for SELECT vs INIT requests

These models ensure proper payload format and prevent confusion between SELECT and INIT stages.
Based on working curl examples and ONDC specification.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass 
class ONDCContext:
    """
    Common ONDC context structure for all operations
    
    Attributes:
        domain: ONDC domain (e.g., "ONDC:RET10" for retail)
        transaction_id: Unique transaction identifier across ONDC operations
        city: City code - use user's delivery pincode (not std: format)
    """
    domain: str = "ONDC:RET10"
    transaction_id: Optional[str] = None
    city: str = ""  # Use user's delivery pincode directly


@dataclass
class SelectItem:
    """
    Item structure for SELECT requests
    
    Attributes:
        id: Full ONDC item ID (format: provider_domain_provider-uuid_item-uuid)  
        local_id: Local item UUID for internal reference
        customisationState: Item customization state (empty dict if none)
        quantity: Quantity requested (e.g., {"count": 1})
        provider: Complete provider object from cart data with locations
        customisations: Available customizations (null if none)
        hasCustomisations: Boolean indicating if item has customizations
    """
    id: str  
    local_id: str  
    customisationState: Dict = None
    quantity: Dict[str, int] = None  
    provider: Dict = None  
    customisations: Optional[Any] = None
    hasCustomisations: bool = False


@dataclass
class SelectMessage:
    """
    Message structure for SELECT requests - CRITICAL: uses cart.items format
    
    Attributes:
        cart: Cart wrapper containing items array {"items": [...]}
        fulfillments: Delivery fulfillments with user's GPS and area_code
    
    Note: 
        SELECT uses message.cart.items (nested)
        INIT uses message.items (direct) - different structure!
    """
    cart: Dict[str, List[Dict]]  
    fulfillments: List[Dict]  


@dataclass 
class SelectPayload:
    """
    Complete SELECT request payload for ONDC backend
    
    Attributes:
        context: ONDC context with user's delivery pincode as city
        message: Message containing cart.items and fulfillments
        userId: User identifier from session (e.g., "EUSJ0ypAJJVdo3gXrUJe4uIBwDB2")
        deviceId: Device identifier from session (e.g., "ed0bda0dd8c167a73721be5bb142dfc9")
    """
    context: Dict[str, str]  
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
    """
    Item structure for INIT requests - DIFFERENT from SELECT structure
    
    Attributes:
        id: Full ONDC item ID (same format as SELECT)
        local_id: Local item UUID for internal reference  
        quantity: Quantity being ordered (e.g., {"count": 1})
        provider: Provider info containing id and local_id (simplified vs SELECT)
    
    Note:
        INIT items are simpler than SELECT items (no customizations, etc.)
    """
    id: str  
    local_id: str  
    quantity: Dict[str, int]  
    provider: Dict  


@dataclass
class InitBillingInfo:
    """
    Billing information for INIT requests
    
    Attributes:
        address: Complete billing address with all fields
                (name, building, street, locality, city, state, area_code)
        phone: Customer phone number (e.g., "9999999999")
        name: Customer full name (e.g., "John Doe")
        email: Customer email address (e.g., "john@example.com")
    """
    address: Dict[str, str]  
    phone: str
    name: str
    email: str


@dataclass 
class InitDeliveryInfo:
    """
    Delivery information for INIT requests
    
    Attributes:
        end: Delivery location with GPS coordinates and complete address
        type: Delivery type - always "Delivery" (COD not supported)
    """
    end: Dict[str, Any]  
    type: str = "Delivery"


@dataclass
class InitPayment:
    """
    Payment structure for INIT requests
    
    Attributes:
        type: Payment type - always "ON-ORDER" (prepayment required, no COD)
    """
    type: str = "ON-ORDER"  


@dataclass
class InitMessage:
    """
    Message structure for INIT requests - CRITICAL: different from SELECT
    
    Attributes:
        items: Items directly under message (NOT message.cart.items like SELECT)
        billing_info: Customer billing information
        delivery_info: Delivery location and preferences
        payment: Payment method (always ON-ORDER)
    
    Note:
        INIT uses message.items (direct)
        SELECT uses message.cart.items (nested) - different structure!
    """
    items: List[Dict]  
    billing_info: InitBillingInfo
    delivery_info: InitDeliveryInfo  
    payment: InitPayment


@dataclass
class InitPayload:
    """
    Complete INIT request payload for ONDC backend
    
    Attributes:
        context: ONDC context with transaction_id from SELECT response
        message: Message with direct items and customer info
        deviceId: Device identifier (no userId in INIT requests)
    """
    context: Dict[str, str]  
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


# Response Models for ONDC Operations

@dataclass
class QuoteItem:
    """Item in a SELECT response quote"""
    id: str
    local_id: str
    quantity: Dict[str, int]
    price: Dict[str, Any]  # Contains value, currency
    fulfillment_id: str
    provider_id: str
    

@dataclass
class DeliveryOption:
    """Delivery option from SELECT response"""
    fulfillment_id: str
    type: str  # "Delivery", "Self-Pickup"
    tracking: bool
    delivery_time: Optional[str] = None
    delivery_charge: Optional[float] = None
    

@dataclass
class ProviderQuote:
    """Quote from a single provider in SELECT response"""
    provider_id: str
    provider_name: str
    items: List[QuoteItem]
    fulfillments: List[DeliveryOption]
    total_value: float
    delivery_charges: float
    

@dataclass
class SelectResponse:
    """Complete response structure for SELECT operations"""
    success: bool
    message: str
    message_id: Optional[str] = None
    transaction_id: Optional[str] = None
    providers: List[ProviderQuote] = field(default_factory=list)
    total_items: int = 0
    grand_total: float = 0.0
    stage: str = "select"
    next_step: Optional[str] = None
    

@dataclass
class OrderSummary:
    """Order summary from INIT response"""
    order_id: str
    total_value: float
    delivery_charges: float
    taxes: float
    grand_total: float
    estimated_delivery: Optional[str] = None
    

@dataclass
class PaymentDetails:
    """Payment information from INIT response"""
    type: str  # "ON-ORDER", "COD", etc.
    status: str
    collected_by: str
    payment_methods: List[str] = field(default_factory=list)
    

@dataclass
class InitResponse:
    """Complete response structure for INIT operations"""
    success: bool
    message: str
    message_id: Optional[str] = None
    transaction_id: Optional[str] = None
    order_summary: Optional[OrderSummary] = None
    payment_details: Optional[PaymentDetails] = None
    stage: str = "init"
    next_step: Optional[str] = None


@dataclass
class ConfirmResponse:
    """Complete response structure for CONFIRM operations"""
    success: bool
    message: str
    order_id: Optional[str] = None
    tracking_id: Optional[str] = None
    payment_status: str = "PENDING"
    estimated_delivery: Optional[str] = None
    stage: str = "confirm"


# Response factory functions

def create_select_response(
    success: bool,
    message: str,
    providers: List[ProviderQuote] = None,
    message_id: str = None,
    transaction_id: str = None
) -> SelectResponse:
    """Create a standardized SELECT response"""
    providers = providers or []
    total_items = sum(len(p.items) for p in providers)
    grand_total = sum(p.total_value + p.delivery_charges for p in providers)
    
    return SelectResponse(
        success=success,
        message=message,
        message_id=message_id,
        transaction_id=transaction_id,
        providers=providers,
        total_items=total_items,
        grand_total=grand_total,
        next_step="initialize_order" if success else None
    )


def create_init_response(
    success: bool,
    message: str,
    order_summary: OrderSummary = None,
    payment_details: PaymentDetails = None,
    message_id: str = None,
    transaction_id: str = None
) -> InitResponse:
    """Create a standardized INIT response"""
    return InitResponse(
        success=success,
        message=message,
        message_id=message_id,
        transaction_id=transaction_id,
        order_summary=order_summary,
        payment_details=payment_details,
        next_step="confirm_order" if success else None
    )


def create_confirm_response(
    success: bool,
    message: str,
    order_id: str = None,
    tracking_id: str = None,
    payment_status: str = "PENDING"
) -> ConfirmResponse:
    """Create a standardized CONFIRM response"""
    return ConfirmResponse(
        success=success,
        message=message,
        order_id=order_id,
        tracking_id=tracking_id,
        payment_status=payment_status
    )