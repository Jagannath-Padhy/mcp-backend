"""Payment verification and management operations for MCP adapters"""

from typing import Dict, Any, Optional
from .utils import (
    get_persistent_session, 
    save_persistent_session, 
    extract_session_id, 
    format_mcp_response,
    send_raw_data_to_frontend
)
from ..utils.logger import get_logger
from ..models.session import CheckoutStage

logger = get_logger(__name__)


async def verify_payment(
    session_id: Optional[str] = None,
    payment_id: Optional[str] = None,
    payment_status: Optional[str] = None,
    razorpay_payment_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Verify payment status after user completes payment via frontend Razorpay SDK
    
    This tool allows the frontend to send payment completion status back to the system
    after the user completes payment using Razorpay SDK.
    
    UX Flow:
    Frontend: User completes payment via Razorpay SDK
    Frontend: Calls verify_payment with payment status
    System: Updates session payment status
    System: "Payment verified! Ready for order confirmation."
    
    Args:
        session_id: User session ID
        payment_id: Payment ID (from create_payment response)
        payment_status: Payment status ('PAID', 'SUCCESS', 'FAILED', 'PENDING')
        razorpay_payment_id: Actual Razorpay payment ID from frontend
        
    Returns:
        Payment verification response
    """
    try:
        # Get enhanced session with conversation tracking
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="verify_payment", **kwargs)
        
        # Validate session is in PAYMENT_PENDING stage
        if session_obj.checkout_state.stage != CheckoutStage.PAYMENT_PENDING:
            return format_mcp_response(
                False,
                f' Cannot verify payment. Session is in {session_obj.checkout_state.stage.value} stage, expected PAYMENT_PENDING.',
                session_obj.session_id
            )
        
        # Validate required parameters
        if not payment_status:
            return format_mcp_response(
                False,
                ' Payment status is required. Provide payment_status ("PAID", "SUCCESS", "FAILED", etc.)',
                session_obj.session_id
            )
        
        # Normalize payment status
        payment_status = payment_status.upper()
        
        # Validate payment ID matches (optional check for mock mode)
        if payment_id and session_obj.checkout_state.payment_id:
            if payment_id != session_obj.checkout_state.payment_id:
                logger.warning(f"[Payment Verification] Payment ID mismatch: provided={payment_id}, session={session_obj.checkout_state.payment_id}")
        
        # Update session with payment verification
        session_obj.checkout_state.payment_status = payment_status.lower()
        
        # Store Razorpay payment ID if provided (for real integration)
        if razorpay_payment_id:
            session_obj.checkout_state.payment_id = razorpay_payment_id
        
        # Check if payment was successful
        successful_statuses = ['PAID', 'SUCCESS', 'CAPTURED', 'AUTHORIZED']
        if payment_status in successful_statuses:
            # Payment successful - ready for order confirmation
            success_message = f"✅ Payment verified successfully! Status: {payment_status}\n\n🚀 **Proceeding to order confirmation automatically...**"
            next_step = "confirm_order"
            user_action = "Order confirmation will proceed automatically."
            
            # Send payment success data to frontend via SSE
            raw_data_for_sse = {
                'payment_verification': 'success',
                'payment_status': payment_status,
                'payment_id': session_obj.checkout_state.payment_id,
                'next_step': next_step,
                'biap_specifications': True
            }
            send_raw_data_to_frontend(session_obj.session_id, 'verify_payment', raw_data_for_sse)
            
        else:
            # Payment failed or pending
            success_message = f" Payment status updated: {payment_status}"
            if payment_status in ['FAILED', 'CANCELLED']:
                next_step = "create_payment"
                user_action = "Payment failed. Please try payment again or use a different payment method."
            else:
                next_step = "verify_payment"
                user_action = f"Payment status: {payment_status}. Please complete payment or verify status."
        
        # Save enhanced session with conversation tracking
        save_persistent_session(session_obj, conversation_manager)
        
        return format_mcp_response(
            True,
            success_message,
            session_obj.session_id,
            payment_status=payment_status,
            payment_id=session_obj.checkout_state.payment_id,
            next_step=next_step,
            user_action_required=user_action,
            stage=session_obj.checkout_state.stage.value
        )
        
    except Exception as e:
        logger.error(f"Failed to verify payment: {e}")
        return format_mcp_response(
            False,
            f' Failed to verify payment: {str(e)}',
            session_id or 'unknown'
        )


async def get_payment_status(
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get current payment status for the session
    
    UX Flow:
    User: "what's my payment status?"
    System: [Calls this function] "Payment status: PENDING, waiting for completion"
    
    Args:
        session_id: User session ID
        
    Returns:
        Current payment status information
    """
    try:
        # Get enhanced session with conversation tracking  
        session_obj, conversation_manager = get_persistent_session(session_id, tool_name="get_payment_status", **kwargs)
        
        # Get payment information
        payment_status = session_obj.checkout_state.payment_status or "none"
        payment_id = session_obj.checkout_state.payment_id
        stage = session_obj.checkout_state.stage.value
        
        # Determine next action based on current state
        if stage == "payment_pending":
            if payment_status in ["paid", "success", "captured"]:
                next_action = "Ready for order confirmation. Use 'confirm_order'."
            else:
                next_action = "Complete payment or verify payment status."
        elif stage == "init":
            next_action = "Create payment using 'create_payment'."
        elif stage == "confirmed":
            next_action = "Order already confirmed."
        else:
            next_action = "Continue with checkout flow."
        
        message = f" Payment Status: {payment_status.upper()}"
        if payment_id:
            message += f"\nPayment ID: {payment_id}"
        message += f"\nStage: {stage}"
        message += f"\nNext Action: {next_action}"
        
        return format_mcp_response(
            True,
            message,
            session_obj.session_id,
            payment_status=payment_status,
            payment_id=payment_id,
            stage=stage,
            next_action=next_action
        )
        
    except Exception as e:
        logger.error(f"Failed to get payment status: {e}")
        return format_mcp_response(
            False,
            f' Failed to get payment status: {str(e)}',
            session_id or 'unknown'
        )