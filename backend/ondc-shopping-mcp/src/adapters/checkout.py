"""ONDC checkout flow operations for MCP adapters"""

from typing import Dict, Any, Optional
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services,
    send_raw_data_to_frontend
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Get services
services = get_services()
checkout_service = services['checkout_service']


async def _fetch_user_addresses(user_id: str, session_id: str) -> Dict[str, Any]:
    """Helper function to fetch user addresses via address adapter"""
    try:
        from .address import get_delivery_addresses
        result = await get_delivery_addresses(user_id, session_id=session_id)
        
        if result.get('success', False) and result.get('addresses'):
            return {
                'success': True,
                'addresses': result['addresses'],
                'count': result.get('address_count', 0)
            }
        else:
            return {'success': False, 'addresses': [], 'count': 0}
    except Exception as e:
        logger.error(f"[Checkout] Failed to fetch addresses: {e}")
        return {'success': False, 'addresses': [], 'count': 0}


def _extract_delivery_location(addresses: list) -> Optional[Dict[str, str]]:
    """Extract city, state, pincode from addresses (prefer default address)"""
    if not addresses:
        return None
        
    # Find default address first
    default_address = None
    for addr in addresses:
        if addr.get('defaultAddress', False):
            default_address = addr
            break
    
    # Use first address if no default
    if not default_address and addresses:
        default_address = addresses[0]
    
    if default_address and default_address.get('address'):
        addr_data = default_address['address']
        return {
            'city': addr_data.get('city', ''),
            'state': addr_data.get('state', ''), 
            'pincode': addr_data.get('areaCode', '')
        }
    
    return None


def _extract_customer_details(addresses: list) -> Optional[Dict[str, str]]:
    """Extract customer details from addresses (prefer default address)"""
    if not addresses:
        return None
        
    # Find default address first
    default_address = None
    for addr in addresses:
        if addr.get('defaultAddress', False):
            default_address = addr
            break
    
    # Use first address if no default
    if not default_address and addresses:
        default_address = addresses[0]
    
    if default_address:
        descriptor = default_address.get('descriptor', {})
        address_data = default_address.get('address', {})
        
        # Format full address
        address_parts = []
        if address_data.get('building'): address_parts.append(address_data['building'])
        if address_data.get('street'): address_parts.append(address_data['street'])
        if address_data.get('locality'): address_parts.append(address_data['locality'])
        if address_data.get('city'): address_parts.append(address_data['city'])
        if address_data.get('state'): address_parts.append(address_data['state'])
        if address_data.get('areaCode'): address_parts.append(address_data['areaCode'])
        
        return {
            'customer_name': descriptor.get('name', ''),
            'phone': descriptor.get('phone', ''),
            'email': descriptor.get('email', ''),
            'delivery_address': ', '.join(address_parts) if address_parts else '',
            'city': address_data.get('city', ''),
            'state': address_data.get('state', ''),
            'pincode': address_data.get('areaCode', '')
        }
    
    return None


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
        
        # Validate cart exists - with auto-sync for authenticated users
        if session_obj.cart.is_empty():
            # AUTO-SYNC: If user is authenticated, try to sync backend cart to local session
            if session_obj.user_authenticated and session_obj.user_id and session_obj.device_id:
                logger.info(f"[SMART CHECKOUT DEBUG] Local cart empty, attempting backend sync for user: {session_obj.user_id}, device: {session_obj.device_id}")
                
                try:
                    # Import cart service for sync operation
                    from ..services.cart_service import get_cart_service
                    cart_service = get_cart_service()
                    
                    # Sync backend cart to local session
                    sync_success = await cart_service.sync_backend_to_local_cart(session_obj)
                    logger.info(f"[SMART CHECKOUT DEBUG] Backend sync result: {sync_success}")
                    
                    if sync_success and not session_obj.cart.is_empty():
                        logger.info(f"[SMART CHECKOUT DEBUG] Backend sync successful - found {len(session_obj.cart.items)} items in backend cart")
                        # Continue with checkout flow
                    else:
                        logger.info(f"[SMART CHECKOUT DEBUG] Backend sync failed or still empty - returning empty cart message")
                        return format_mcp_response(
                            False,
                            ' Cart is empty. Please add items first.',
                            session_obj.session_id
                        )
                except Exception as e:
                    logger.error(f"[SMART CHECKOUT DEBUG] Backend sync failed: {e}")
                    return format_mcp_response(
                        False,
                        ' Cart is empty. Please add items first.',
                        session_obj.session_id
                    )
            else:
                # Not authenticated or missing credentials - return empty cart message
                logger.info(f"[SMART CHECKOUT DEBUG] Not authenticated or missing credentials - cannot sync backend cart")
                return format_mcp_response(
                    False,
                    ' Cart is empty. Please add items first.',
                    session_obj.session_id
                )
        
        # Check if delivery location is available in session or parameters
        session_location = getattr(session_obj, 'delivery_location', None)
        
        # SMART AUTOMATION DEBUG LOGGING
        logger.info(f"[SMART CHECKOUT DEBUG] Starting select_items_for_order")
        logger.info(f"[SMART CHECKOUT DEBUG] user_id: {session_obj.user_id}")
        logger.info(f"[SMART CHECKOUT DEBUG] session_location: {session_location}")
        logger.info(f"[SMART CHECKOUT DEBUG] manual params - city: {delivery_city}, state: {delivery_state}, pincode: {delivery_pincode}")
        
        if session_location and all([
            session_location.get('city'),
            session_location.get('state'),
            session_location.get('pincode')
        ]):
            # Use delivery location from session
            delivery_city = session_location['city']
            delivery_state = session_location['state'] 
            delivery_pincode = session_location['pincode']
            
            logger.info(f"[SMART CHECKOUT DEBUG] Using delivery location from session: {delivery_city}, {delivery_state}, {delivery_pincode}")
            
        elif not all([delivery_city, delivery_state, delivery_pincode]):
            # Try to auto-fetch addresses for intelligent checkout
            logger.info(f"[SMART CHECKOUT DEBUG] No manual location provided, starting auto-fetch for user: {session_obj.user_id}")
            
            if session_obj.user_id and session_obj.user_id != "guestUser":
                logger.info(f"[SMART CHECKOUT DEBUG] User is authenticated, fetching addresses...")
                addresses_result = await _fetch_user_addresses(session_obj.user_id, session_obj.session_id)
                logger.info(f"[SMART CHECKOUT DEBUG] Address fetch result: success={addresses_result.get('success', False)}, count={addresses_result.get('address_count', 0)}")
                
                if addresses_result['success'] and addresses_result['addresses']:
                    # AUTO-PATH: Extract delivery location from saved address
                    logger.info(f"[SMART CHECKOUT DEBUG] AUTO-PATH: Found addresses, extracting location...")
                    location = _extract_delivery_location(addresses_result['addresses'])
                    logger.info(f"[SMART CHECKOUT DEBUG] Extracted location: {location}")
                    
                    if location and all([location['city'], location['state'], location['pincode']]):
                        delivery_city = location['city']
                        delivery_state = location['state']
                        delivery_pincode = location['pincode']
                        
                        logger.info(f"[SMART CHECKOUT DEBUG] AUTO-PATH SUCCESS: Using auto-extracted location: {delivery_city}, {delivery_state}, {delivery_pincode}")
                    else:
                        # Address found but incomplete location data
                        logger.warning(f"[SMART CHECKOUT DEBUG] AUTO-PATH FAILED: Address found but incomplete location data: {location}")
                        return format_mcp_response(
                            False,
                            "📍 Found saved address but missing location details. Please provide: city, state, pincode",
                            session_obj.session_id
                        )
                else:
                    # MANUAL-PATH: No addresses found, ask user
                    logger.info(f"[SMART CHECKOUT DEBUG] MANUAL-PATH: No addresses found, requesting manual input")
                    return format_mcp_response(
                        False,
                        """📍 **Delivery Location Required**

No saved addresses found. Please provide:
• City (e.g., 'Bangalore')
• State (e.g., 'Karnataka')  
• Pincode (e.g., '560001')

Format: select_items_for_order(delivery_city='Bangalore', delivery_state='Karnataka', delivery_pincode='560001')""",
                        session_obj.session_id
                    )
            else:
                # Guest user or no user_id
                logger.info(f"[SMART CHECKOUT DEBUG] MANUAL-PATH: Guest user or no user_id, requesting manual input")
                return format_mcp_response(
                    False,
                    "📍 **Delivery Location Required**\n\nPlease provide: city, state, pincode",
                    session_obj.session_id
                )
        
        # Store delivery location in session for reuse
        session_obj.delivery_location = {
            'city': delivery_city,
            'state': delivery_state,
            'pincode': delivery_pincode
        }
        
        # Determine if address was auto-fetched for smart next_step guidance
        address_auto_fetched = False
        if session_location and all([session_location.get('city'), session_location.get('state'), session_location.get('pincode')]):
            # Address from session storage
            address_auto_fetched = True
            logger.info(f"[SMART CHECKOUT DEBUG] Address source: session storage (auto)")
        elif delivery_city and delivery_state and delivery_pincode:
            # Check if we went through the auto-fetch path above
            if session_obj.user_id and session_obj.user_id != "guestUser":
                address_auto_fetched = True
                logger.info(f"[SMART CHECKOUT DEBUG] Address source: backend auto-fetch (auto)")
            else:
                logger.info(f"[SMART CHECKOUT DEBUG] Address source: manual parameters (manual)")
        
        # Call consolidated checkout service
        result = await checkout_service.select_items_for_order(
            session_obj, delivery_city, delivery_state, delivery_pincode, address_auto_fetched
        )
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        # Send raw checkout data to frontend via SSE (Universal Pattern)
        if result['success'] and result.get('quote_data'):
            raw_data_for_sse = {
                'stage': result.get('stage'),
                'quote_data': result.get('quote_data'),
                'next_step': result.get('next_step'),
                'biap_specifications': True
            }
            send_raw_data_to_frontend(session_obj.session_id, 'select_items_for_order', raw_data_for_sse)
        
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
        
        # AUTO-PATH: Try to auto-populate customer details from saved addresses
        auto_details = None
        if session_obj.user_id and session_obj.user_id != "guestUser":
            addresses_result = await _fetch_user_addresses(session_obj.user_id, session_obj.session_id)
            if addresses_result['success'] and addresses_result.get('addresses'):
                auto_details = _extract_customer_details(addresses_result['addresses'])
                if auto_details:
                    # AUTO-PATH: Use saved customer details
                    customer_name = customer_name or auto_details.get('customer_name')
                    delivery_address = delivery_address or auto_details.get('delivery_address')
                    phone = phone or auto_details.get('phone')
                    email = email or auto_details.get('email')
                    logger.info(f"[AUTO-PATH] Populated customer details from saved addresses")
        
        # Check for required customer information and guide through proper flow
        missing = []
        if not customer_name: missing.append("customer_name")
        if not delivery_address: missing.append("delivery_address")
        if not phone: missing.append("phone")
        if not email: missing.append("email")
        
        if missing:
            # MANUAL-PATH: Ask for missing customer details
            field_list = ''.join([f'• {field.replace("_", " ").title()}\n' for field in missing])
            if auto_details:
                message = f"📍 **Additional Details Required**\n\nFound saved address but missing:\n{field_list}Please provide the missing details to proceed."
            else:
                message = f"📍 **Customer Details Required**\n\nTo initialize your order, I need:\n{field_list}Provide these details to proceed with order initialization."
            
            return format_mcp_response(
                False,
                message,
                session_obj.session_id
            )
        
        # Call enhanced BIAP-compatible checkout service
        result = await checkout_service.initialize_order(
            session_obj, customer_name, delivery_address, phone, email, 
            payment_method, city, state, pincode
        )
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        # Send raw checkout data to frontend via SSE (Universal Pattern)
        if result['success'] and result.get('init_data'):
            raw_data_for_sse = {
                'stage': result.get('stage'),
                'order_summary': result.get('order_summary'),
                'init_data': result.get('init_data'),
                'next_step': result.get('next_step'),
                'biap_specifications': True
            }
            send_raw_data_to_frontend(session_obj.session_id, 'initialize_order', raw_data_for_sse)
        
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
    amount: Optional[float] = None,
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
        result = await checkout_service.create_payment(session_obj, payment_method, amount)
        
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
                f"🔄 **Next Step**: Complete payment and call verify_payment with status='PAID' to proceed.",
                session_obj.session_id,
                stage=result.get('stage'),  # Add stage field from checkout service
                payment_data=result['data'],
                next_step=result['next_step'],
                _mock_indicators=result['data'].get('_mock_indicators', {})
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
        
        # Validate session is in INIT or PAYMENT_PENDING stage
        if session_obj.checkout_state.stage.value not in ['init', 'payment_pending']:
            return format_mcp_response(
                False,
                ' Please complete delivery and payment details first using initialize_order.',
                session_obj.session_id
            )
        
        # Validate payment status for non-COD orders
        payment_method = session_obj.checkout_state.payment_method or 'cod'
        if payment_method.lower() != 'cod' and payment_status and payment_status.upper() not in ['PAID', 'CAPTURED', 'SUCCESS']:
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
        
        # Send raw checkout data to frontend via SSE (Universal Pattern)
        if result['success'] and result.get('confirm_data'):
            raw_data_for_sse = {
                'order_id': result.get('order_id'),
                'order_details': result.get('order_details'),
                'confirm_data': result.get('confirm_data'),
                'next_actions': result.get('next_actions'),
                'biap_specifications': True
            }
            send_raw_data_to_frontend(session_obj.session_id, 'confirm_order', raw_data_for_sse)
        
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