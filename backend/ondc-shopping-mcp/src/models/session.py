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
    CONFIRMED = "confirmed"     # ONDC CONFIRM step - order confirmed


# CartItem class removed - cart data now stored purely in backend


# Cart class removed - cart data now stored purely in backend


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
    
    # Enhanced debugging and recovery system
    operation_responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Cache ONDC operation responses
    last_error: Optional[Dict[str, Any]] = None  # Store detailed error info for debugging
    force_fresh_execution: bool = False  # Flag to bypass AI agent caching
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'stage': self.stage.value,
            'transaction_id': self.transaction_id,
            'delivery_info': self.delivery_info.to_dict() if self.delivery_info else None,
            'payment_method': self.payment_method,
            'payment_status': self.payment_status,
            'payment_id': self.payment_id,
            'order_id': self.order_id,
            'operation_responses': self.operation_responses,
            'last_error': self.last_error,
            'force_fresh_execution': self.force_fresh_execution
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
            order_id=data.get('order_id'),
            operation_responses=data.get('operation_responses', {}),
            last_error=data.get('last_error'),
            force_fresh_execution=data.get('force_fresh_execution', False)
        )
    
    def cache_operation_response(self, operation: str, response: Dict[str, Any], message_id: Optional[str] = None) -> None:
        """Cache operation response for debugging and recovery"""
        from datetime import datetime
        
        self.operation_responses[operation] = {
            'response': response,
            'timestamp': datetime.utcnow().isoformat(),
            'message_id': message_id,
            'stage': operation
        }
    
    def get_cached_response(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get cached operation response if available"""
        return self.operation_responses.get(operation)
    
    def set_last_error(self, error_info: Dict[str, Any]) -> None:
        """Store detailed error information for debugging"""
        from datetime import datetime
        
        self.last_error = {
            **error_info,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _extract_message_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract message_id from ONDC response for caching"""
        if isinstance(response, list) and len(response) > 0:
            first_item = response[0]
            if 'context' in first_item and 'message_id' in first_item['context']:
                return first_item['context']['message_id']
        elif isinstance(response, dict):
            if 'context' in response and 'message_id' in response['context']:
                return response['context']['message_id']
        return None


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
    # cart removed - now stored purely in backend
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
            # cart removed - stored in backend only
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
            # cart removed - stored in backend only
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