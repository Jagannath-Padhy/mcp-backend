"""ONDC checkout flow operations for MCP adapters"""

from typing import Dict, Any, Optional
import re
from src.adapters.utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Get services
services = get_services()
checkout_service = services['checkout_service']


def validate_gps_coordinates(gps_string: Optional[str]) -> Dict[str, Any]:
    """
    Validate GPS coordinates format and return validation result
    
    Args:
        gps_string: GPS coordinates in "lat,lng" format
        
    Returns:
        Dict with validation result and parsed coordinates
    """
    if not gps_string:
        return {
            'valid': False,
            'error': 'GPS coordinates required',
            'lat': None,
            'lng': None
        }
    
    # Check format: "lat,lng" with optional spaces
    gps_pattern = r'^(-?\d+\.?\d*),\s*(-?\d+\.?\d*)$'
    match = re.match(gps_pattern, gps_string.strip())
    
    if not match:
        return {
            'valid': False,
            'error': 'Invalid GPS format. Use: latitude,longitude',
            'lat': None,
            'lng': None
        }
    
    try:
        lat = float(match.group(1))
        lng = float(match.group(2))
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90):
            return {
                'valid': False,
                'error': 'Latitude must be between -90 and 90',
                'lat': lat,
                'lng': lng
            }
        
        if not (-180 <= lng <= 180):
            return {
                'valid': False,
                'error': 'Longitude must be between -180 and 180',
                'lat': lat,
                'lng': lng
            }
        
        return {
            'valid': True,
            'error': None,
            'lat': lat,
            'lng': lng
        }
        
    except ValueError:
        return {
            'valid': False,
            'error': 'Invalid numeric values in GPS coordinates',
            'lat': None,
            'lng': None
        }


def get_gps_help_message(delivery_city: str, delivery_pincode: str) -> str:
    """Generate helpful GPS coordinates message for user"""
    examples = {
        'bangalore': '12.9716,77.5946',
        'chandigarh': '30.745765,76.653633',
        'delhi': '28.6139,77.2090',
        'mumbai': '19.0760,72.8777',
        'pune': '18.5204,73.8567',
        'hyderabad': '17.3850,78.4867'
    }
    
    city_lower = delivery_city.lower()
    city_example = examples.get(city_lower, '12.9716,77.5946')  # Default to Bangalore
    
    return f"""📍 **GPS Coordinates Required**

For accurate delivery to {delivery_city} {delivery_pincode}, please provide GPS coordinates:

**Format:** latitude,longitude
**Example for {delivery_city}:** {city_example}

**How to get GPS coordinates:**
1. Open Google Maps
2. Right-click your delivery location
3. Click the coordinates that appear to copy them
4. Paste here in format: lat,lng

**Common Examples:**
• Bangalore: 12.9716,77.5946
• Chandigarh: 30.745765,76.653633  
• Delhi: 28.6139,77.2090
• Mumbai: 19.0760,72.8777

**Please provide GPS coordinates for your exact delivery location.**"""


async def select_items_for_order(
    delivery_city: str,
    delivery_state: str,
    delivery_pincode: str,
    delivery_gps: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    ONDC SELECT stage - Get delivery quotes and options
    
    UX Flow:
    User: "checkout my cart"
    System: " I need delivery location..."
    User: provides city, state, pincode  
    System: [Calls this function] " Delivery available! Quotes ready."
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="select_items_for_order", **kwargs)
        
        # Validate cart exists
        if session_obj.cart.is_empty():
            return format_mcp_response(
                False,
                ' Cart is empty. Please add items first.',
                session_obj.session_id
            )
        
        # Check if delivery location is available in session or parameters
        session_location = getattr(session_obj, 'delivery_location', None)
        
        if session_location and all([
            session_location.get('city'),
            session_location.get('state'),
            session_location.get('pincode')
        ]):
            # Use delivery location from session
            delivery_city = session_location['city']
            delivery_state = session_location['state'] 
            delivery_pincode = session_location['pincode']
            
            logger.info(f"Using delivery location from session: {delivery_city}, {delivery_state}, {delivery_pincode}")
            
        elif not all([delivery_city, delivery_state, delivery_pincode]):
            # No location in session and parameters not provided
            return format_mcp_response(
                False,
                """ **Delivery Location Required**

To get delivery quotes, I need to collect your address first.

 **Please provide:**
• city='Bangalore' 
• state='Karnataka' 
• pincode='560001'
• delivery_gps='12.9716,77.5946'

Or provide delivery location directly:
• Format: city='Bangalore', state='Karnataka', pincode='560001', delivery_gps='12.9716,77.5946'""",
                session_obj.session_id
            )
        
        # Validate GPS coordinates
        gps_validation = validate_gps_coordinates(delivery_gps)
        if not gps_validation['valid']:
            # GPS missing or invalid - provide helpful error message
            if delivery_gps:
                error_msg = f" **GPS Validation Error**\n\n{gps_validation['error']}\n\n{get_gps_help_message(delivery_city, delivery_pincode)}"
            else:
                error_msg = get_gps_help_message(delivery_city, delivery_pincode)
            
            return format_mcp_response(
                False,
                error_msg,
                session_obj.session_id
            )
        
        # Call consolidated checkout service with validated GPS
        result = await checkout_service.select_items_for_order(
            session_obj, delivery_city, delivery_state, delivery_pincode, delivery_gps
        )
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            result['success'],
            result['message'],
            session_obj.session_id,
            stage=result.get('stage'),
            quote_data=result.get('quote_data'),
            next_step=result.get('next_step')
        )
        
    except Exception as e:
        logger.error(f"Failed to select items for order: {e}")
        return format_mcp_response(
            False,
            f' Failed to get delivery quotes: {str(e)}',
            session_id or 'unknown'
        )


async def initialize_order(
    customer_name: str,
    delivery_address: str,
    phone: str,
    email: str,
    payment_method: str = 'razorpay',
    city: Optional[str] = None,
    state: Optional[str] = None,
    pincode: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    BIAP-compatible ONDC INIT stage - Initialize order with complex delivery structure
    
    UX Flow:
    System: " I need complete customer and delivery details..."
    User: provides all details including customer name
    System: [Calls this function] " Order initialized with BIAP validation!"
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="initialize_order", **kwargs)
        
        # AUTHENTICATED MODE: Users must have provided real Himira credentials
        # All operations require valid userId and deviceId from initialize_shopping
        logger.info(f"[AUTHENTICATED MODE] Proceeding with authenticated order initialization")
        
        # Validate user credentials exist (set in initialize_shopping)
        if not session_obj.user_id or not session_obj.device_id:
            return format_mcp_response(
                False,
                "❌ **Authentication Required**\n\n"
                "Please provide your Himira credentials first:\n\n"
                "`initialize_shopping(userId='your_user_id', deviceId='your_device_id')`\n\n"
                "This ensures proper access to your cart and order history.",
                session_obj.session_id
            )
        
        # Validate session is in SELECT stage
        if session_obj.checkout_state.stage.value != 'select':
            return format_mcp_response(
                False,
                ' Please select delivery location first using select_items_for_order.',
                session_obj.session_id
            )
        
        # Check for required customer information and guide through proper flow
        missing = []
        if not customer_name: missing.append("customer_name")
        if not delivery_address: missing.append("delivery_address")
        if not phone: missing.append("phone")
        if not email: missing.append("email")
        
        if missing:
            return format_mcp_response(
                False,
                f""" **Customer Details Required**

To initialize your order, I need your complete information:

 **Missing Details:** {', '.join(missing)}

 **Please provide:**
• **Customer Name:** Your full name
• **Delivery Address:** Complete street address
• **Phone:** Contact number (e.g., 9876543210)
• **Email:** Email address

 **Or call this tool with all details:**
• Format: customer_name='John Doe', delivery_address='123 Main St Apartment 4B', phone='9876543210', email='user@example.com'""",
                session_obj.session_id
            )
        
        # Call enhanced BIAP-compatible checkout service
        result = await checkout_service.initialize_order(
            session_obj, customer_name, delivery_address, phone, email, 
            payment_method, city, state, pincode
        )
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            result['success'],
            result['message'],
            session_obj.session_id,
            stage=result.get('stage'),
            order_summary=result.get('order_summary'),
            init_data=result.get('init_data'),
            next_step=result.get('next_step')
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize order: {e}")
        return format_mcp_response(
            False,
            f' Failed to initialize order: {str(e)}',
            session_id or 'unknown'
        )


async def create_payment(
    payment_method: str,
    amount: float,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    MOCK PAYMENT CREATION - Create mock payment between INIT and CONFIRM
    
    This creates a mock payment using values from the Himira Order Postman collection.
    This step simulates the Razorpay payment creation that would happen between INIT and CONFIRM.
    
    UX Flow:
    System: " Creating payment... Please wait."
    System: " Payment created successfully! Payment ID: pay_RFWPuAV50T2Qnj"
    System: "Ready for order confirmation. Use 'confirm_order' next."
    
    Args:
        session: User session (must be in INIT stage)
        payment_method: Payment method (default: razorpay)
        
    Returns:
        Mock payment creation response
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="create_payment", **kwargs)
        
        # MOCK PAYMENT CREATION - Clear labeling
        logger.info(f"[MCP ADAPTER] Creating mock payment for session: {session_id}")
        result = await checkout_service.create_payment(session_obj, payment_method)
        
        if result.get('success'):
            # Log mock payment creation with indicators
            payment_id = result['data']['payment_id']
            logger.info(f"[MCP ADAPTER] Mock payment created: {payment_id}")
            
            # Save enhanced session with conversation tracking
            save_persistent_session(session_obj, conversation_manager)
            
            return format_mcp_response(
                True,
                f" [MOCK] Payment created successfully!\n"
                f"Payment ID: {payment_id}\n"
                f"Amount: ₹{result['data']['amount']} INR\n"
                f"Status: {result['data']['status']}\n\n"
                f" Ready for order confirmation. Use 'confirm_order' next.",
                session_obj.session_id,
                {
                    'payment_data': result['data'],
                    'next_step': result['next_step'],
                    '_mock_indicators': result['data'].get('_mock_indicators', {})
                }
            )
        else:
            # Save session even on failure to preserve state
            save_persistent_session(session_obj, conversation_manager)
            
            return format_mcp_response(
                False,
                result.get('message', 'Payment creation failed'),
                session_obj.session_id
            )
            
    except Exception as e:
        logger.error(f"[MCP ADAPTER] Payment creation failed: {str(e)}")
        return format_mcp_response(
            False,
            f" Payment creation failed: {str(e)}",
            session_id or 'unknown'
        )


async def confirm_order(
    payment_status: str = 'PAID',
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    BIAP-compatible ONDC CONFIRM stage - Finalize the order with payment validation
    
    UX Flow:
    System: " Final Order Summary... Payment status? Confirm? (yes/no)"
    User: provides payment_status and confirms
    System: [Calls this function] " Order confirmed with BIAP validation! Order ID: ABC123"
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="confirm_order", **kwargs)
        
        # AUTHENTICATED MODE: Users must have provided real Himira credentials
        # This confirms orders for authenticated users only
        logger.info(f"[AUTHENTICATED MODE] Proceeding with authenticated order confirmation")
        
        # Validate user credentials exist (set in initialize_shopping)
        if not session_obj.user_id or not session_obj.device_id:
            return format_mcp_response(
                False,
                "❌ **Authentication Required**\n\n"
                "Please provide your Himira credentials first:\n\n"
                "`initialize_shopping(userId='your_user_id', deviceId='your_device_id')`\n\n"
                "Orders can only be confirmed with valid user credentials.",
                session_obj.session_id
            )
        
        # Validate session is in INIT stage
        if session_obj.checkout_state.stage.value != 'init':
            return format_mcp_response(
                False,
                ' Please complete delivery and payment details first using initialize_order.',
                session_obj.session_id
            )
        
        # Validate payment status for non-COD orders
        payment_method = session_obj.checkout_state.payment_method or 'cod'
        if payment_method.lower() != 'cod' and payment_status.upper() not in ['PAID', 'CAPTURED', 'SUCCESS']:
            return format_mcp_response(
                False,
                f" Payment verification required. Current status: {payment_status}\\n" +
                f"For {payment_method.upper()} payments, status must be 'PAID', 'CAPTURED', or 'SUCCESS'\\n" +
                f"Format: payment_status='PAID'",
                session_obj.session_id
            )
        
        # Call enhanced BIAP-compatible checkout service
        result = await checkout_service.confirm_order(session_obj, payment_status)
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            result['success'],
            result['message'],
            session_obj.session_id,
            order_id=result.get('order_id'),
            order_details=result.get('order_details'),
            confirm_data=result.get('confirm_data'),
            next_actions=result.get('next_actions')
        )
        
    except Exception as e:
        logger.error(f"Failed to confirm order: {e}")
        return format_mcp_response(
            False,
            f' Failed to confirm order: {str(e)}',
            session_id or 'unknown'
        )