"""Order management operations for MCP adapters"""

from typing import Dict, Any, Optional
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    get_services
)
from ..models.session import CheckoutStage
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Get services
services = get_services()
cart_service = services['cart_service']
payment_service = services['payment_service']
order_service = services['order_service']
session_service = services['session_service']


async def initiate_payment(session_id: Optional[str] = None, payment_method: str = None, 
                              amount: float = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for payment with demo mode"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="initiate_payment", **kwargs)
        
        # Check order initialized
        if session_obj.checkout_state.stage != CheckoutStage.INIT:
            return format_mcp_response(False,
                " Please initialize order first using 'initialize_order' tool.",
                session_obj.session_id,
                required_action="initialize_order")
        
        # Calculate amount from cart if not provided
        if not amount:
            cart_summary = cart_service.get_cart_summary(session_obj)
            amount = cart_summary['total_value']
        
        # Show payment options if method not selected
        if not payment_method:
            real_methods = payment_service.get_available_payment_methods(amount)
            
            payment_options = "\n".join([f"[{i+1}] {method['display_name']} (Fee: ₹{method['processing_fee']:.2f})" for i, method in enumerate(real_methods)])
            
            return format_mcp_response(False,
                f" **SELECT PAYMENT METHOD**\n **Order Amount:** ₹{amount:.2f}\n\n**Available Payment Methods:**\n{payment_options}\n\n Reply with payment method number (e.g., '1' for the first option)",
                session_obj.session_id,
                amount=amount,
                payment_methods=real_methods,
                input_required="payment_method")
        
        # Real payment processing using PaymentService
        success, message, payment_result = await payment_service.initiate_payment(
            session_obj, payment_method, amount, session_obj.auth_token
        )
        
        if success:
            session_obj.checkout_state.payment_method = payment_method
            session_obj.checkout_state.payment_status = "success"
            
            # Save enhanced session with conversation tracking
            save_persistent_session(session_obj, conversation_manager)
            
            return format_mcp_response(True,
                f"{message}\n\n **Next Step:** Use 'confirm_order' tool to complete the order",
                session_obj.session_id,
                payment_result=payment_result,
                next_step="confirm_order")
        else:
            # Save session even on failure
            save_persistent_session(session_obj, conversation_manager)
            return format_mcp_response(False, message, session_obj.session_id)
            
    except Exception as e:
        logger.error(f"Failed to initiate payment: {e}")
        return format_mcp_response(
            False,
            f' Failed to initiate payment: {str(e)}',
            session_id or 'unknown'
        )


async def confirm_order_simple(session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Simple MCP adapter for ONDC CONFIRM step (deprecated - use confirm_order)"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="confirm_order_simple", **kwargs)
        
        # Check payment completed
        if session_obj.checkout_state.stage != CheckoutStage.INIT or session_obj.checkout_state.payment_status not in ["success", "pending"]:
            return format_mcp_response(False,
                " Please complete payment first using 'initiate_payment' tool.",
                session_obj.session_id,
                required_action="initiate_payment")
        
        # Prepare payment info
        payment_info = {
            "method": session_obj.checkout_state.payment_method,
            "status": session_obj.checkout_state.payment_status,
            "amount": cart_service.get_cart_summary(session_obj)['total_value']
        }
        
        # ONDC CONFIRM using OrderService
        success, message, confirm_result = await order_service.confirm_order(
            session_obj, payment_info, session_obj.auth_token
        )
        
        if success:
            session_obj.checkout_state.stage = CheckoutStage.CONFIRMED
            
            # Save enhanced session with conversation tracking
            save_persistent_session(session_obj, conversation_manager)
            
            order_id = confirm_result.get('order_id', 'N/A')
            return format_mcp_response(True,
                f"{message}\n\n **ORDER PLACED SUCCESSFULLY!**\n\n **Order ID:** {order_id}\n **Status:** Confirmed\n **Amount:** ₹{payment_info['amount']:.2f}\n **Payment:** {payment_info['method']}\n\n **Next Steps:**\n• Use 'track_order' to track your order\n• Use 'get_order_status' to check order status",
                session_obj.session_id,
                order_placed=True,
                order_id=order_id,
                confirm_result=confirm_result)
        else:
            # Save session even on failure
            save_persistent_session(session_obj, conversation_manager)
            return format_mcp_response(False, message, session_obj.session_id)
            
    except Exception as e:
        logger.error(f"Failed to confirm order (simple): {e}")
        return format_mcp_response(
            False,
            f' Failed to confirm order: {str(e)}',
            session_id or 'unknown'
        )


async def get_order_status(session_id: Optional[str] = None, order_id: str = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for getting order status"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="get_order_status", **kwargs)
        
        # Check authentication - both flag and token required
        if not session_obj.user_authenticated or not session_obj.auth_token:
            return format_mcp_response(False,
                " **Authentication Required**\n\nPlease login to check order status.\n\n **Use:** `phone_login phone='9876543210'`",
                session_obj.session_id,
                required_action="phone_login")
        
        # Get order status
        success, message, status_result = await order_service.get_order_status(
            session_obj, order_id, session_obj.auth_token
        )
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        if success:
            return format_mcp_response(True, message, session_obj.session_id, 
                                     order_status=status_result)
        else:
            return format_mcp_response(False, message, session_obj.session_id)
            
    except Exception as e:
        logger.error(f"Failed to get order status: {e}")
        return format_mcp_response(
            False,
            f' Failed to get order status: {str(e)}',
            session_id or 'unknown'
        )


async def track_order(session_id: Optional[str] = None, order_id: str = None, **kwargs) -> Dict[str, Any]:
    """MCP adapter for order tracking"""
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="track_order", **kwargs)
        
        # Check authentication - both flag and token required  
        if not session_obj.user_authenticated or not session_obj.auth_token:
            return format_mcp_response(False,
                " **Authentication Required**\n\nPlease login to track your orders.\n\n **Use:** `phone_login phone='9876543210'`",
                session_obj.session_id,
                required_action="phone_login")
        
        # Track order
        success, message, tracking_result = await order_service.track_order(
            session_obj, session_obj.auth_token, order_id
        )
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        if success:
            return format_mcp_response(True, message, session_obj.session_id,
                                     tracking_info=tracking_result)
        else:
            return format_mcp_response(False, message, session_obj.session_id)
            
    except Exception as e:
        logger.error(f"Failed to track order: {e}")
        return format_mcp_response(
            False,
            f' Failed to track order: {str(e)}',
            session_id or 'unknown'
        )