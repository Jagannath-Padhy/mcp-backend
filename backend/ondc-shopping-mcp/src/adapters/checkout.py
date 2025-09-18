"""ONDC checkout flow operations for MCP adapters"""

from typing import Dict, Any, Optional
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Get services
services = get_services()
checkout_service = services['checkout_service']


async def select_items_for_order(
    session_id: Optional[str] = None,
    delivery_city: Optional[str] = None,
    delivery_state: Optional[str] = None,
    delivery_pincode: Optional[str] = None,
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

 **Please use the 'set_delivery_address' tool first:**
• This will ask for your city, state, and pincode
• Then you can proceed with getting delivery quotes

Or provide delivery location directly:
• Format: city='Bangalore', state='Karnataka', pincode='560001'""",
                session_obj.session_id
            )
        
        # Call consolidated checkout service
        result = await checkout_service.select_items_for_order(
            session_obj, delivery_city, delivery_state, delivery_pincode
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
    session_id: Optional[str] = None,
    customer_name: Optional[str] = None,
    delivery_address: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    payment_method: Optional[str] = 'cod',
    city: Optional[str] = None,
    state: Optional[str] = None,
    pincode: Optional[str] = None,
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
        
        # GUEST MODE: Authentication check removed for guest journey
        # Guest users can proceed without authentication
        logger.info(f"[GUEST MODE] Proceeding with guest order initialization")
        
        # Ensure guest mode is active
        if not session_obj.user_id:
            session_obj.user_id = "guestUser"
        if not session_obj.device_id:
            from ..config import config
            session_obj.device_id = config.guest.device_id
        
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
    session_id: Optional[str] = None,
    payment_method: Optional[str] = 'razorpay',
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
    session_id: Optional[str] = None, 
    payment_status: Optional[str] = 'PENDING',
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
        
        # GUEST MODE: Mock confirmation allowed without authentication
        # This is a MOCK confirmation for testing only
        logger.info(f"[GUEST MODE] Proceeding with MOCK order confirmation")
        
        # Ensure guest mode is active
        if not session_obj.user_id:
            session_obj.user_id = "guestUser"
        if not session_obj.device_id:
            from ..config import config
            session_obj.device_id = config.guest.device_id
        
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