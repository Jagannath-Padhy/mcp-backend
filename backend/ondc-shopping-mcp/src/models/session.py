"""Data models for session management with proper typing and validation"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import uuid


class CheckoutStage(Enum):
    """ONDC-compatible checkout stages - simplified to match protocol"""
    NONE = "none"
    SELECT = "select"           # ONDC SELECT step - items selected, quote received
    INIT = "init"              # ONDC INIT step - order initialized with delivery info
    PAYMENT_PENDING = "payment_pending"  # Payment order created, waiting for user payment
    CONFIRMED = "confirmed"     # ONDC CONFIRM step - order confirmed


@dataclass
class CartItem:
    """BIAP-compatible cart item with full ONDC fields matching BIAP Node.js structure"""
    id: str
    name: str
    price: float
    quantity: int
    local_id: str                                 # BIAP requirement - used in SELECT API
    bpp_id: str                                  # BIAP requirement - from product enrichment
    bpp_uri: str                                 # BIAP requirement - from product enrichment
    contextCity: Optional[str] = None            # BIAP requirement - from enrichment
    category: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    product: Optional[Dict[str, Any]] = None     # BIAP enriched product details with subtotal
    provider: Optional[Dict[str, Any]] = None    # BIAP provider structure with locations array
    location_id: Optional[str] = None            # BIAP requirement - provider location ID
    fulfillment_id: Optional[str] = None         # Required for INIT operation
    parent_item_id: Optional[str] = None         # For complex/bundled items
    tags: Optional[List[Dict[str, Any]]] = None  # Item metadata and type information
    customisations: Optional[List[Dict[str, Any]]] = None  # Item customization options
    
    @property
    def provider_id(self) -> str:
        """Extract provider ID from provider object for backward compatibility"""
        try:
            if self.provider:
                if isinstance(self.provider, dict):
                    return self.provider.get('id') or self.provider.get('local_id', 'unknown')
                elif isinstance(self.provider, list) and len(self.provider) > 0:
                    # Handle corrupted case where provider became a list
                    first_provider = self.provider[0]
                    if isinstance(first_provider, dict):
                        return first_provider.get('id') or first_provider.get('local_id', 'unknown')
                    # If it's a list of strings, use the first one
                    elif isinstance(first_provider, str):
                        return first_provider
                # Handle any other corrupted type
                elif isinstance(self.provider, str):
                    return self.provider
        except Exception:
            # Completely defensive - if anything goes wrong, return unknown
            pass
        return 'unknown'
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal for this item"""
        return self.price * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'price': self.price,
            'quantity': self.quantity,
            'local_id': self.local_id,
            'bpp_id': self.bpp_id,
            'bpp_uri': self.bpp_uri,
            'contextCity': self.contextCity,
            'category': self.category,
            'image_url': self.image_url,
            'description': self.description,
            'product': self.product,
            'provider': self.provider,
            'location_id': self.location_id,
            'fulfillment_id': self.fulfillment_id,
            'parent_item_id': self.parent_item_id,
            'tags': self.tags,
            'customisations': self.customisations,
            'subtotal': self.subtotal,
            'provider_id': self.provider_id  # Computed property
        }
    
    def to_biap_select_format(self) -> Dict[str, Any]:
        """Convert to BIAP SELECT API format"""
        return {
            'id': self.id,
            'local_id': self.local_id,
            'bpp_id': self.bpp_id,
            'bpp_uri': self.bpp_uri,
            'contextCity': self.contextCity,
            'product': self.product,
            'provider': self.provider,
            'quantity': {'count': self.quantity}
        }
    
    def to_biap_init_format(self) -> Dict[str, Any]:
        """Convert to BIAP INIT API format"""
        item_data = {
            'id': self.local_id,
            'quantity': self.quantity,
            'location_id': self.provider.get('locations', [{}])[0].get('local_id') if self.provider else None,
            'fulfillment_id': self.fulfillment_id,
            'product': self.product
        }
        
        # Add parent_item_id if present
        if self.parent_item_id:
            item_data['parent_item_id'] = self.parent_item_id
        
        # Add tags if present
        if self.tags:
            item_data['tags'] = self.tags
            
        return item_data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CartItem':
        """Create CartItem from dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            price=float(data['price']),
            quantity=int(data['quantity']),
            local_id=data['local_id'],
            bpp_id=data['bpp_id'],
            bpp_uri=data['bpp_uri'],
            contextCity=data.get('contextCity'),
            category=data.get('category'),
            image_url=data.get('image_url'),
            description=data.get('description'),
            product=data.get('product'),
            provider=data.get('provider'),
            location_id=data.get('location_id'),
            fulfillment_id=data.get('fulfillment_id'),
            parent_item_id=data.get('parent_item_id'),
            tags=data.get('tags'),
            customisations=data.get('customisations')
        )
    
    @classmethod
    def create_from_enriched_data(cls, base_item: 'CartItem', enriched_data: Dict[str, Any]) -> 'CartItem':
        """Create CartItem with enriched data from BIAP protocolGetItemList/protocolGetItemDetails"""
        return cls(
            id=base_item.id,
            name=base_item.name,
            price=base_item.price,
            quantity=base_item.quantity,
            local_id=base_item.local_id,
            # Update with enriched data from BIAP
            bpp_id=enriched_data.get('context', {}).get('bpp_id', base_item.bpp_id),
            bpp_uri=enriched_data.get('context', {}).get('bpp_uri', base_item.bpp_uri),
            contextCity=enriched_data.get('context', {}).get('city'),
            category=base_item.category,
            image_url=base_item.image_url,
            description=base_item.description,
            # Enriched product details
            product={
                'subtotal': enriched_data.get('item_details', {}).get('price', {}).get('value', base_item.price),
                **enriched_data.get('item_details', {})
            },
            # Enriched provider details
            provider={
                'locations': [enriched_data.get('location_details', {})],
                **enriched_data.get('provider_details', {})
            }
        )


@dataclass
class Cart:
    """Shopping cart with items and calculations"""
    items: List[CartItem] = field(default_factory=list)
    
    @property
    def total_items(self) -> int:
        """Total number of items in cart"""
        return sum(item.quantity for item in self.items)
    
    @property
    def total_value(self) -> float:
        """Total value of cart"""
        return sum(item.subtotal for item in self.items)
    
    def add_item(self, item: CartItem) -> None:
        """Add item to cart or update quantity if exists"""
        existing = self.find_item(item.id)
        if existing:
            existing.quantity += item.quantity
        else:
            self.items.append(item)
    
    def remove_item(self, item_id: str) -> bool:
        """Remove item from cart"""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                del self.items[i]
                return True
        return False
    
    def update_quantity(self, item_id: str, quantity: int) -> bool:
        """Update item quantity"""
        item = self.find_item(item_id)
        if item and quantity > 0:
            item.quantity = quantity
            return True
        elif item and quantity == 0:
            return self.remove_item(item_id)
        return False
    
    def find_item(self, item_id: str) -> Optional[CartItem]:
        """Find item in cart by ID"""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def clear(self) -> None:
        """Clear all items from cart"""
        self.items.clear()
    
    def is_empty(self) -> bool:
        """Check if cart is empty"""
        return len(self.items) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'items': [item.to_dict() for item in self.items],
            'total_items': self.total_items,
            'total_value': self.total_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cart':
        """Create Cart from dictionary"""
        items = [CartItem.from_dict(item) for item in data.get('items', [])]
        return cls(items=items)
    
    # Dictionary compatibility methods for backward compatibility with existing code
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like get method for backward compatibility
        
        Enables Cart objects to be used where dictionary access is expected.
        Maintains object-oriented functionality while supporting legacy dict patterns.
        
        Args:
            key: Dictionary key to access ('items', 'total_items', 'total_value')
            default: Default value if key not found
            
        Returns:
            Requested cart data in dictionary-compatible format
        """
        if key == 'items':
            return [item.to_dict() for item in self.items]  # Return dict format for compatibility
        elif key == 'total_items':
            return self.total_items
        elif key == 'total_value':
            return self.total_value
        return default
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-like item access for backward compatibility
        
        Enables cart['items'] syntax for legacy code compatibility.
        
        Args:
            key: Dictionary key to access
            
        Returns:
            Requested cart data
            
        Raises:
            KeyError: If key is not a valid cart attribute
        """
        result = self.get(key)
        if result is None and key in ['items', 'total_items', 'total_value']:
            raise KeyError(f"'{key}'")
        return result
    
    def keys(self):
        """Dictionary-like keys method for backward compatibility
        
        Returns:
            List of available cart data keys
        """
        return ['items', 'total_items', 'total_value']
    
    def __contains__(self, key: str) -> bool:
        """Dictionary-like 'in' operator support for backward compatibility
        
        Enables 'key in cart' syntax for legacy code.
        
        Args:
            key: Key to check for existence
            
        Returns:
            True if key is a valid cart attribute
        """
        return key in ['items', 'total_items', 'total_value']


@dataclass
class DeliveryInfo:
    """Delivery information for checkout"""
    address: str
    phone: str
    email: str
    name: Optional[str] = None
    city: Optional[str] = None
    pincode: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'address': self.address,
            'phone': self.phone,
            'email': self.email,
            'name': self.name,
            'city': self.city,
            'pincode': self.pincode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeliveryInfo':
        """Create DeliveryInfo from dictionary"""
        return cls(
            address=data['address'],
            phone=data['phone'],
            email=data['email'],
            name=data.get('name'),
            city=data.get('city'),
            pincode=data.get('pincode')
        )


@dataclass
class CheckoutState:
    """State management for checkout flow"""
    stage: CheckoutStage = CheckoutStage.NONE
    transaction_id: Optional[str] = None
    delivery_info: Optional[DeliveryInfo] = None
    payment_method: Optional[str] = None
    payment_status: str = "none"  # none, pending, success, failed
    payment_id: Optional[str] = None  # Mock Razorpay payment ID (e.g., pay_RFWPuAV50T2Qnj)
    order_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'stage': self.stage.value,
            'transaction_id': self.transaction_id,
            'delivery_info': self.delivery_info.to_dict() if self.delivery_info else None,
            'payment_method': self.payment_method,
            'payment_status': self.payment_status,
            'payment_id': self.payment_id,
            'order_id': self.order_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckoutState':
        """Create CheckoutState from dictionary"""
        return cls(
            stage=CheckoutStage(data.get('stage', 'none')),
            transaction_id=data.get('transaction_id'),
            delivery_info=DeliveryInfo.from_dict(data['delivery_info']) if data.get('delivery_info') else None,
            payment_method=data.get('payment_method'),
            payment_status=data.get('payment_status', 'none'),
            payment_id=data.get('payment_id'),
            order_id=data.get('order_id')
        )


@dataclass
class UserPreferences:
    """User preferences and settings"""
    language: str = "en"
    currency: str = "INR"
    location: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'language': self.language,
            'currency': self.currency,
            'location': self.location,
            'categories': self.categories
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create UserPreferences from dictionary"""
        return cls(
            language=data.get('language', 'en'),
            currency=data.get('currency', 'INR'),
            location=data.get('location'),
            categories=data.get('categories', [])
        )


@dataclass
class Session:
    """Complete session model with all components"""
    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:16]}")
    user_id: Optional[str] = None  # Will be set from authenticated user
    device_id: str = field(default_factory=lambda: f"mcp_{uuid.uuid4().hex[:16]}")
    cart: Cart = field(default_factory=Cart)
    checkout_state: CheckoutState = field(default_factory=CheckoutState)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    history: List[Dict[str, Any]] = field(default_factory=list)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    # Authentication fields
    auth_token: Optional[str] = None
    user_authenticated: bool = False
    user_profile: Optional[Dict[str, Any]] = None
    demo_mode: bool = False  # Always real backend authentication
    
    def update_access_time(self) -> None:
        """Update last accessed time"""
        self.last_accessed = datetime.utcnow()
    
    def add_to_history(self, action: str, data: Dict[str, Any]) -> None:
        """Add action to session history"""
        self.history.append({
            'action': action,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'device_id': self.device_id,
            'cart': self.cart.to_dict(),
            'checkout_state': self.checkout_state.to_dict(),
            'preferences': self.preferences.to_dict(),
            'history': self.history,
            'search_history': self.search_history,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'auth_token': self.auth_token,
            'user_authenticated': self.user_authenticated,
            'user_profile': self.user_profile,
            'demo_mode': self.demo_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create Session from dictionary"""
        return cls(
            session_id=data['session_id'],
            user_id=data.get('user_id'),  # No default, must be from auth
            device_id=data.get('device_id'),
            cart=Cart.from_dict(data.get('cart', {})),
            checkout_state=CheckoutState.from_dict(data.get('checkout_state', {})),
            preferences=UserPreferences.from_dict(data.get('preferences', {})),
            history=data.get('history', []),
            search_history=data.get('search_history', []),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow(),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if 'last_accessed' in data else datetime.utcnow(),
            auth_token=data.get('auth_token'),
            user_authenticated=data.get('user_authenticated', False),
            user_profile=data.get('user_profile'),
            demo_mode=data.get('demo_mode', False)  # Default to real backend
        )